# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Paired outcome sensitivity with fixed scores and common outcome assignments."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from common import state

COARSE = ["mRS 0–2", "mRS 3", "mRS 4–5"]


def squared_losses(p):
    return np.stack([p ** 2, (1 - p) ** 2], axis=-1)


def scenario_rows(target, cohort, records, loss, g, plan):
    x, y, availability = (records[k] for k in ("x", "y", "availability"))
    tier = state(x)
    eligible = len(y)
    observed = np.isfinite(y)
    unseen = ~observed
    count = int(unseen.sum())
    yint = y[observed].astype(int)
    rows = np.arange(eligible)[observed]
    observed_loss = loss[rows, :, yint]
    totals = observed_loss.sum(axis=0)
    event = int(yint.sum())
    observed_by_group = np.bincount(tier[observed], minlength=3)
    events_by_group = np.bincount(tier[observed], weights=yint, minlength=3)
    mass = plan["reference_risk_prior_count"]
    rate = (events_by_group + mass * g) / (observed_by_group + mass)
    expected = {"all_missing_favourable": np.zeros(count), "all_missing_adverse": np.ones(count),
                "development_G_reference": g[tier[unseen]]}
    for ratio in plan["missing_case_odds_ratios"]:
        expected[f"missing_odds_{ratio:g}"] = expit(logit(rate[tier[unseen]]) + np.log(ratio))
    all_rows = []
    ident = dict(target=target, cohort=cohort, eligible=eligible, observed=int(observed.sum()), missing=count, events=event)

    def result(scenario, losses, n, events):
        stats = losses / n
        return dict(ident, scenario=scenario, scored_n=n, adverse_rate=events / n,
                    H=float(stats[0]), G=float(stats[1]), HG=float(stats[2]),
                    H_minus_HG=float(stats[0] - stats[2]), G_minus_HG=float(stats[1] - stats[2]))

    all_rows.append(result("observed_outcomes", totals, int(observed.sum()), event))
    m = loss[unseen]
    for scenario, q in expected.items():
        losses = (1 - q[:, None]) * m[:, :, 0] + q[:, None] * m[:, :, 1]
        all_rows.append(result(scenario, totals + losses.sum(axis=0), eligible, event + q.sum()))
    bounds = []
    for j, label in enumerate(("H_minus_HG", "G_minus_HG")):
        observed_gain = (observed_loss[:, j] - observed_loss[:, 2]).sum()
        d = m[:, j] - m[:, 2]
        low, high = (observed_gain + d.min(axis=1).sum()) / eligible, (observed_gain + d.max(axis=1).sum()) / eligible
        assert all(low - 1e-14 <= r[label] <= high + 1e-14 for r in all_rows[1:])
        bounds.append(dict(ident, comparison=label, observed_delta=observed_gain / observed.sum(),
                           lower=low, upper=high, all_completions_positive=bool(low > 0)))
    ascertainment, reasons = [], []
    for level, name in [(-1, "All"), *enumerate(COARSE)]:
        ix = np.ones(eligible, bool) if level == -1 else tier == level
        n = int(ix.sum())
        e = int(y[ix & observed].sum())
        complete = int((ix & observed).sum())
        ascertainment.append(dict(target=target, cohort=cohort, tier=name, eligible=n, observed=complete,
            unavailable=n - complete, adverse=e, complete_percent=100 * complete / n if n else None,
            adverse_lower=e / n if n else None, adverse_upper=(e + n - complete) / n if n else None,
            reference_risk=float(rate[level]) if level >= 0 else None))
        for key, k in pd.Series(availability[ix]).value_counts().items():
            reasons.append(dict(target=target, cohort=cohort, tier=name, reason=key, count=int(k)))
    return all_rows, bounds, ascertainment, reasons
