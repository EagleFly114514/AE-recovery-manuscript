# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Rebuild clinical tables, cohort flow and recovery paths from accepted inputs."""
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from evaluate import COHORTS, wilson


def percent(k, n, denominator=True):
    if not n:
        return "—"
    return f"{k}/{n} ({100*k/n:.1f}%)" if denominator else f"{k} ({100*k/n:.1f}%)"


def iqr(s, round_half_up=False):
    lo, mid, hi = s.quantile([.25, .5, .75])
    if round_half_up:
        lo, mid, hi = (Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) for v in (lo, mid, hi))
    return f"{mid:.1f} ({lo:.1f}–{hi:.1f})"


def six_month_tiers(f):
    return (f.mrs6.between(0, 2), f.mrs6.eq(3), f.mrs6.between(4, 5))


def clinical_groups(f):
    cohorts = {name: f.loc[f.cohort.eq(name)] for name in COHORTS}
    eligible = {name: s.loc[s.mrs6.between(0, 5)]
                for name, s in cohorts.items()}
    observed = {name: s.loc[s.mrs12.notna()] for name, s in eligible.items()}
    return cohorts, eligible, observed


def table_one(observed):
    def describe(s):
        result = [iqr(s.age)]
        for k in ("impaired_consciousness", "movement_abnormality", "treatment_delay_days", "preceding_icu"):
            if k == "treatment_delay_days":
                result.append(iqr(s[k], round_half_up=True))
                continue
            n = s[k].notna().sum()
            value = percent(s[k].eq(1).sum(), n)
            result.append(value + (f"; missing {len(s)-n}" if n < len(s) else ""))
        return result + [percent(s[k].eq(1).sum(), len(s)) for k in ("female", "nmdar", "lgi1")] + [
            int(a.sum()) for a in six_month_tiers(s)] + [percent(s.mrs12.ge(3).sum(), len(s))]

    labels = ["Age, years", "Impaired consciousness by admission", "Movement abnormalities by admission",
              "Onset to qualifying treatment, days", "Available preceding ICU admission", "Female", "NMDAR positive", "LGI1 positive",
              "Six-month mRS 0–2", "Six-month mRS 3", "Six-month mRS 4–5", "Twelve-month dependence or death"]
    return pd.DataFrame(dict(Characteristic=labels, **{f"{k.capitalize()} n={len(v)}": describe(v) for k, v in observed.items()}))


def table_two(eligible):
    rows = []
    for co, f in eligible.items():
        tiers = list(zip(["0–2", "3", "4–5"], six_month_tiers(f)))
        if co == "development":
            tiers.append(("3–5", f.mrs6.between(3, 5)))
        for label, take in tiers:
            outcome = f.loc[take, "mrs12"]
            n = outcome.notna().sum()
            p = [percent(mask.sum(), n) for mask in (outcome.le(2), outcome.between(3, 5), outcome.eq(6), outcome.ge(3))]
            rows.append(["Dependent subtotal" if label == "3–5" else co.capitalize(), label, f"{len(outcome)}/{n}", len(outcome)-n, *p])
    return pd.DataFrame(rows, columns=["Cohort", "Six-month mRS", "Source/ observed", "Not\navailable", "Independent",
                                      "Alive dependent", "Death", "Dependent or dead"])


def recovery(dev):
    dependent = dev.loc[dev.mrs6.between(3, 5)]
    records = []
    for month in (12, 24):
        for label, key, mask in (("3", "mRS6_3", dependent.mrs6.eq(3)), ("4–5", "mRS6_4_5", dependent.mrs6.ge(4)),
                                 ("3–5", "mRS6_3_5", dependent.mrs6.between(3, 5))):
            outcome = dependent.loc[mask, f"mrs{month}"]
            n, k, source_n = int(outcome.notna().sum()), int(outcome.le(2).sum()), len(outcome)
            lo, hi = wilson(k, n)
            records.append(dict(month=month, label=label, six_month_stratum=key, observed_n=n, independent_n=k,
                                eligible_n=source_n, observed_recovery=k/n, wilson_95_lower=lo, wilson_95_upper=hi))
    data = pd.DataFrame(records)
    rows = []
    for r in data.itertuples():
        k, n, total = r.independent_n, r.observed_n, r.eligible_n
        rows.append([r.month, r.label, f"{n}/{total}", percent(k, n), f"{100*r.wilson_95_lower:.1f}–{100*r.wilson_95_upper:.1f}",
                     total-n, f"{100*k/total:.1f}–{100*(k+total-n)/total:.1f}"])
    table = pd.DataFrame(rows, columns=["Month", "Six-month mRS", "Observed/ original", "Independent n/N (%)", "95% CI (%)", "Missing", "Fixed bounds (%)"])
    mod, severe = data.iloc[0], data.iloc[1]
    p, q = mod.observed_recovery, severe.observed_recovery
    diff = p-q
    lower = diff - np.hypot(p-mod.wilson_95_lower, severe.wilson_95_upper-q)
    upper = diff + np.hypot(mod.wilson_95_upper-p, q-severe.wilson_95_lower)
    return table, data, 100*np.array([diff, lower, upper])


def transition_table(dev):
    """Tabulate the two recorded states in the fixed six-month dependent cohort."""
    dependent = dev.loc[dev.mrs6.between(3, 5)]
    states = ["Independent", "Dependent", "Death", "Missing"]

    def classify(scores):
        assert scores.dropna().isin(range(7)).all()
        labels = np.select([scores.le(2), scores.between(3, 5), scores.eq(6)], states[:3], default=states[3])
        return pd.Categorical(labels, categories=states, ordered=True)

    counts = pd.crosstab(classify(dependent.mrs12), classify(dependent.mrs24), dropna=False)
    assert counts.to_numpy().sum() == len(dependent)
    counts["Total"] = counts.sum(axis=1)
    counts.loc["Total"] = counts.sum(axis=0)
    return counts.rename_axis("12-month status").reset_index()


def longitudinal_sources(dev, transitions):
    """Reconcile individual-record-derived full-cohort flow and export source data.

    Every interval is calculated from the supplied individual scores.
    Previously documented death remains an absorbing state in the full flow.
    """
    states = ["I", "D3", "D45", "X", "M"]
    intervals = ["worst_to_six", "six_to_twelve", "twelve_to_twentyfour"]
    visits = ["worst_inpatient", "six_month", "twelve_month", "twentyfour_month"]
    total = len(dev)
    matrices = []
    for interval in intervals:
        data = transitions.loc[transitions.interval.eq(interval)]
        assert len(data) == 25 and data.n.ge(0).all() and data.n.eq(data.n.astype(int)).all()
        matrix = data.pivot(index="from_state", columns="to_state", values="n").loc[states, states]
        assert matrix.to_numpy().sum() == total
        assert matrix.loc["X", ["I", "D3", "D45", "M"]].sum() == 0
        matrices.append(matrix)
    for first, second in zip(matrices, matrices[1:]):
        assert np.array_equal(first.sum(axis=0), second.sum(axis=1))

    def classify(scores):
        return pd.Series(np.select([scores.le(2), scores.eq(3), scores.between(4, 5), scores.eq(6)],
                                   states[:4], default="M"), index=scores.index)

    observed = [classify(dev.worst_mrs), classify(dev.mrs6), classify(dev.mrs12)]
    observed[2].loc[dev.mrs6.eq(6)] = "X"
    for i in range(2):
        current = pd.crosstab(observed[i], observed[i+1]).reindex(index=states, columns=states, fill_value=0)
        assert np.array_equal(current, matrices[i]), intervals[i]
    counts = [matrices[0].sum(axis=1)] + [matrix.sum(axis=0) for matrix in matrices]
    occupancy = pd.DataFrame([
        dict(timepoint=visit, state=state, n=int(count[state]), denominator_n=total)
        for visit, count in zip(visits, counts) for state in states
    ])
    subset = dev.loc[dev.mrs6.between(3, 5)]
    confirmed = subset.loc[subset.mrs12.between(3, 5) & subset.mrs24.between(0, 2)]
    recovered = int(matrices[2].loc[["D3", "D45"], "I"].sum())
    assert len(confirmed) <= recovered
    for state, mask in [("D3", confirmed.mrs12.eq(3)), ("D45", confirmed.mrs12.between(4, 5))]:
        assert int(mask.sum()) <= matrices[2].loc[state, "I"]
    second_year = pd.DataFrame([
        dict(population="Full development cohort", denominator_n=total, recovered_n=recovered),
        dict(population="Dependent at six months", denominator_n=len(subset), recovered_n=len(confirmed)),
        dict(population="Six-month mRS 3", denominator_n=int(subset.mrs6.eq(3).sum()), recovered_n=int(confirmed.mrs6.eq(3).sum())),
        dict(population="Six-month mRS 4-5", denominator_n=int(subset.mrs6.between(4, 5).sum()), recovered_n=int(confirmed.mrs6.between(4, 5).sum())),
    ])
    return occupancy, transitions.copy(), second_year


def screening(cohorts, eligible, observed, original):
    n_reg = original["internal_screened"]-original["no_immunotherapy"]-original["infectious_encephalitis"]
    n_age = n_reg-original["below_age_14"]
    n_timeline = n_age-original["inconsistent_internal_timeline"]
    assert len(cohorts["development"])+len(cohorts["temporal"]) == n_timeline-original["missing_qualifying_treatment_date"]
    rows = [
        ["Screened internal", f'{original["internal_screened"]} total', "Same source"],
        ["Registry after infection and no-treatment exclusions", f"{n_reg} total", "Same source"],
        ["Age at least 14", f"{n_age} total", "Same source"],
        ["Interpretable internal timeline", f"{n_timeline} total", "Same source"],
        ["Eligible treatment cohort", len(cohorts["development"]), len(cohorts["temporal"])],
        ["Six-month deaths", cohorts["development"].mrs6.eq(6).sum(), cohorts["temporal"].mrs6.eq(6).sum()],
        ["Missing six-month assessment", cohorts["development"].mrs6.isna().sum(), cohorts["temporal"].mrs6.isna().sum()],
        ["Living six-month score", len(eligible["development"]), len(eligible["temporal"])],
        ["Six-month timing basis", "Nominal; actual dates not recorded", "Nominal; actual dates not recorded"],
        ["Due twelve-month target", *map(len, eligible.values())],
        ["Eligible observed twelve-month outcome", *map(len, observed.values())],
        ["Unavailable twelve-month endpoint", *[len(eligible[c])-len(observed[c]) for c in COHORTS]],
        ["Six-month dependence with observed endpoint", *[s.mrs6.between(3, 5).sum() for s in observed.values()]],
    ]
    return pd.DataFrame(rows, columns=["Stage", "Development", "Temporal"])


def antibodies(s):
    return s.antibody_codes.fillna("").map(lambda x: set(map(int, str(x).split("|"))) if x else set())


def clinical_descriptives(cohorts, eligible, codes):
    s2, s3, s4 = {}, {}, {}
    for co in COHORTS[:2]:
        s = cohorts[co]
        a = antibodies(s)
        flags = [s.female.eq(1), s.nmdar.eq(1), s.lgi1.eq(1), a.str.len().ge(2), s.neoplasm.eq(1), s.worst_mrs.between(3, 5),
                 s.icu_admission.eq(1), s.respiratory_failure.eq(1), s.tx_steroid.eq(1), s.tx_ivig.eq(1), s.tx_plasma_exchange.eq(1)]
        counts = [percent(int(mask.sum()), len(s)) for mask in flags]
        s2[f"Development cohort (n = {len(s)})" if co == "development" else f"Temporal cohort (n = {len(s)})"] = [
            iqr(s.age), *counts[:4], iqr(s.treatment_delay_days, round_half_up=True), *counts[4:]]
        fields = list(dict.fromkeys(codes.values()))
        s3[f"Development (n={len(s)})" if co == "development" else f"Temporal (n={len(s)})"] = [
            percent(a.map(lambda x: bool(x & {int(k) for k, v in codes.items() if v == label})).sum(), len(s), False) for label in fields
        ] + [percent(mask.sum(), len(s), False) for mask in (s.mog.eq(1), a.str.len().ge(2), s.legacy_multiple.eq(1))]
    for label, mask in (("Moderate dependence, mRS 3", eligible["development"].mrs6.eq(3)),
                        ("Severe dependence, mRS 4–5", eligible["development"].mrs6.between(4, 5))):
        s = eligible["development"].loc[mask]
        a = antibodies(s)
        b = [a.map(lambda x: k in x) for k in range(1, 8)]
        b += [a.map(lambda x: bool(x-set(range(1, 8)))), a.str.len().ge(2), s.neoplasm.eq(1), s.tx_steroid.eq(1), s.tx_ivig.eq(1),
              s.tx_plasma_exchange.eq(1), s.tx_immunosuppressant.eq(1)]
        s4[f"{label}, n={len(s)}"] = [iqr(s.age), percent(s.female.eq(1).sum(), len(s), False), iqr(s.treatment_delay_days),
                                       *[percent(flag.sum(), len(s), False) for flag in b]]
    return {
        "Supplementary_Table_2": pd.DataFrame(dict(Characteristic=["Age at onset, years", "Female sex", "NMDAR antibody", "LGI1 antibody",
            "Two or more specified antibodies", "Onset-to-treatment interval, days", "Recorded neoplasm", "Worst inpatient mRS 3–5",
            "Recorded ICU-admission flag", "Recorded respiratory-failure occurrence", "Corticosteroid date eligible for T0",
            "Intravenous immunoglobulin date eligible for T0", "Plasma-exchange date eligible for T0"], **s2)),
        "Supplementary_Table_3": pd.DataFrame({"Recorded antibody specificity": [*fields, "MOG (separate registry field)",
            "Two or more classified antibodies, excluding MOG", "Original registry multiple-antibody flag"], **s3}),
        "Supplementary_Table_4": pd.DataFrame(dict(Characteristic=["Age at disease onset, years", "Female sex", "Onset to treatment initiation, days",
            *[f"{codes[str(k)]} antibody" for k in range(1, 8)], "Other recorded specificities", "Two or more classified antibodies", "Associated neoplasm",
            "Corticosteroids", "Intravenous immunoglobulin", "Plasma exchange", "Immunosuppressant"], **s4)),
    }


def run(frame, spec, dictionary, where, longitudinal=None):
    cohorts, eligible, observed = clinical_groups(frame)
    recovery_table, estimates, contrast = recovery(eligible["development"])
    tables = dict(Table_1=table_two(eligible), Table_2=transition_table(eligible["development"]),
                  Supplementary_Table_1=screening(cohorts, eligible, observed, spec["screening"]), Supplementary_Table_5=dictionary,
                  Supplementary_Table_23=table_one(observed), Supplementary_Table_24=recovery_table)
    tables.update(clinical_descriptives(cohorts, eligible, spec["antibody_codes"]))
    for name, data in tables.items():
        data.to_csv(where / f"{name}.csv", index=False, encoding="utf-8")
    estimates.to_csv(where / "recovery_estimates.csv", index=False, encoding="utf-8")
    pd.DataFrame([contrast], columns=["difference_pp", "lower_95_pp", "upper_95_pp"]).to_csv(where / "recovery_contrast.csv", index=False)
    if longitudinal is not None:
        flow_sources = longitudinal_sources(cohorts["development"], longitudinal)
        for name, data in zip(("longitudinal_state_counts", "longitudinal_transition_counts", "second_year_recovery_counts"), flow_sources):
            data.to_csv(where / f"{name}.csv", index=False, encoding="utf-8")
