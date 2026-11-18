# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Supplementary Table 11 from the legacy nested estimator and paired SE draws."""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from functional_baseline import stable_seed
from legacy_validation import _one_se_choice, make_analysis_ridge_pipeline, make_splits


OMISSIONS = {
    "SE timing proxy": "hmax_status_epilepticus_admission",
    "Numerical consciousness category": "hmax_consciousness_ordinal_admission",
}


def feature_pair(train, test, spec):
    """Recompute all data-dependent domain and item transformations in a fold."""
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    fit, query = train.copy(), test.copy()
    for domain, items in spec["domains"].items():
        keys = []
        for item in items:
            source = item["column"]
            a, b = train[source], test[source]
            if item["transform"] == "log1p":
                a, b = np.log1p(a.clip(lower=0)), np.log1p(b.clip(lower=0))
            observed = a.dropna()
            if observed.empty:
                lo, hi, median, scale = 0., 0., 0., 1.
            else:
                if observed.nunique() <= 2:
                    lo, hi = observed.min(), observed.max()
                else:
                    lo, hi = observed.quantile(.01), observed.quantile(.99)
                clipped = observed.clip(lo, hi)
                median = clipped.median()
                iqr = clipped.quantile(.75) - clipped.quantile(.25)
                scale = iqr / 1.349 if iqr > 0 else clipped.std(ddof=0)
                if not np.isfinite(scale) or scale <= 1e-8:
                    scale = 1.
            column = "hmax_item_" + item["key"]
            keys.append(column)
            fit[column] = ((a.clip(lo, hi) - median) / scale) * item["direction"]
            query[column] = ((b.clip(lo, hi) - median) / scale) * item["direction"]
        for matrix in (fit, query):
            matrix[domain] = matrix[keys].mean(axis=1, skipna=True).where(matrix[keys].notna().mean(axis=1).ge(spec["coverage"]))
    return fit, query


def outer_fit(frame, y, groups, tr, te, index, seed, schemas, spec):
    """One-SE penalty selection using the original grouped inner partitions."""
    train = frame.iloc[tr]
    inner_seed = stable_seed(seed, "outer_inner", index)
    grid = spec["validation"]["c_grid"]
    no_indicator = set(spec["no_indicator"])
    splits, _ = make_splits(y[tr], groups[tr], folds=spec["validation"]["inner_folds"], repeats=1, seed=inner_seed)
    scores = {name: {c: [] for c in grid} for name in schemas}
    with threadpool_limits(limits=1):
        for j, (fit, valid) in enumerate(splits):
            a, b = feature_pair(train.iloc[fit], train.iloc[valid], spec)
            for name, columns in schemas.items():
                template = make_analysis_ridge_pipeline(columns, seed=stable_seed(inner_seed, "inner_preprocess", index, j, name), no_indicator=no_indicator)
                prep = template.named_steps["preprocess"]
                z, v = prep.fit_transform(a), prep.transform(b)
                for c in grid:
                    model = template.named_steps["model"].set_params(C=c, random_state=stable_seed(inner_seed, "inner_model", index, j, name, c))
                    p = model.fit(z, y[tr[fit]]).predict_proba(v)[:, 1]
                    scores[name][c].append(np.mean((y[tr[valid]]-p)**2))
        a, b = feature_pair(train, frame.iloc[te], spec)
        predictions = []
        for name, columns in schemas.items():
            c, _ = _one_se_choice(scores[name])
            model = make_analysis_ridge_pipeline(columns, seed=stable_seed(seed, "outer_model", index, name), c=c, no_indicator=no_indicator)
            predictions.append(model.fit(a, y[tr]).predict_proba(b)[:, 1])
    return te, np.array(predictions)


def scores(frame, y, family, seed, labels, spec, pool=None):
    schemas = {}
    for label in labels:
        h = [name for name in spec["history_columns"] if name != OMISSIONS[label]]
        if label == "SE timing proxy":
            names = ("HexpandedNoSE12", "HexpandedNoSEG12") if family == "dynamic" else ("R_HexpandedNoSE", "R_HexpandedNoSEG")
        else:
            names = ("Omit_H", "Omit_HG")
        schemas[names[0]] = h
        schemas[names[1]] = [*h, *spec["state_columns"][family]]
    group = frame.study_id.to_numpy(int)
    validation = spec["validation"]
    splits, _ = make_splits(y, group, folds=validation["outer_folds"], repeats=validation["outer_repeats"], seed=seed)
    args = ((frame, y, group, tr, te, i, seed, schemas, spec) for i, (tr, te) in enumerate(splits))
    fits = pool(delayed(outer_fit)(*row) for row in args) if pool is not None else (outer_fit(*row) for row in args)
    total, count = np.zeros((len(schemas), len(y))), np.zeros(len(y), int)
    for te, p in fits:
        total[:, te] += p
        count[te] += 1
    assert (count == validation["outer_repeats"]).all()
    return ((total / count - y) ** 2).mean(axis=1)


def bootstrap_draw(frame, y, family, index, spec):
    seed = spec["validation"]["seed"]
    choice = np.random.default_rng(stable_seed(seed, "targeted_v5_bootstrap_sample", family, index)).choice(len(frame), len(frame), replace=True)
    refit = stable_seed(seed, "targeted_v5_bootstrap_outcome", family, index)
    a, b = scores(frame.iloc[choice].reset_index(drop=True), y[choice], family, refit, ["SE timing proxy"], spec)
    return a-b


def load_data(cohort, data):
    spec = json.loads((data / "legacy_spec.json").read_text(encoding="utf-8"))
    dynamic = pd.read_csv(data / "legacy_history.csv", float_precision="round_trip")
    mrs = cohort.set_index("study_id").mrs12
    recovery = dynamic.loc[dynamic.recovery_order.notna()].sort_values("recovery_order").reset_index(drop=True)
    return spec, {"dynamic": (dynamic, mrs.loc[dynamic.study_id].ge(3).to_numpy(int)),
                  "recovery": (recovery, mrs.loc[recovery.study_id].le(2).to_numpy(int))}


def run(cohort, data, where, jobs, refit_se=False):
    spec, frames = load_data(cohort, data)
    archive = pd.read_csv(data / "legacy_se_paired_draws.csv", float_precision="round_trip")
    assert len(archive) == spec["bootstrap_draws"]
    rows = []
    with Parallel(n_jobs=jobs, verbose=5) as pool:
        for family, (frame, y) in frames.items():
            seed = stable_seed(spec["validation"]["seed"], f"targeted_v5_{family}_point")
            values = scores(frame, y, family, seed, list(OMISSIONS), spec, pool)
            if refit_se:
                print(f"Legacy {family}: {spec['bootstrap_draws']:,} repeated nested bootstrap refits", flush=True)
                draws = pool(delayed(bootstrap_draw)(frame, y, family, i, spec) for i in range(spec["bootstrap_draws"]))
                np.testing.assert_allclose(draws, archive[family], rtol=0, atol=1e-10)
            else:
                draws = archive[family]
            for i, omission in enumerate(OMISSIONS):
                a, b = values[2*i:2*i+2]
                lo, hi = np.quantile(draws, [.025, .975]) if i == 0 else (None, None)
                rows.append(dict(omission=omission, outcome="Dependence or death" if family == "dynamic" else "Recovery",
                                 H_brier=a, HG_brier=b, gain=a-b, lower=lo, upper=hi))
    pd.DataFrame(rows).to_csv(where / "legacy_omissions.csv", index=False, encoding="utf-8")
