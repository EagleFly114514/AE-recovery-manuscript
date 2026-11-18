# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Development-only estimation and frozen temporal evaluation with paired resampling."""
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

import conditional_history
import history_marginalized
from common import state
from functional_baseline import age_function_baseline, stable_seed, strict_splits, strong_g
from missing_outcomes import scenario_rows, squared_losses


CORE_ARMS = ("H", "G", "HG")
ARMS = (*CORE_ARMS, "B")
COHORTS = ("development", "temporal")
FEATURES = ("age", "impaired_consciousness", "movement_abnormality", "log1p_treatment_delay", "preceding_icu")
GAINS = ("H_minus_HG", "G_minus_HG")


def matrix(frame):
    mrs = frame.mrs6.to_numpy()
    assert np.isin(mrs, range(6)).all()
    return np.column_stack([frame[list(FEATURES)].to_numpy(float), mrs == 3, mrs >= 4])


def outcome(frame):
    return np.where(frame.mrs12.notna(), frame.mrs12.ge(3).astype(float), np.nan)


def rosters(frame, target):
    assert frame.cohort.isin(COHORTS).all(), 'This report accepts internal cohorts only'
    subset = frame if target == "overall_dynamic" else frame.loc[frame.nmdar.eq(1)]
    groups = {name: subset.loc[subset.cohort.eq(name) & subset.mrs6.between(0, 5)].copy() for name in COHORTS}
    eligible = {name: dict(x=matrix(f), y=outcome(f), availability=f.availability.to_numpy()) for name, f in groups.items()}
    observed = {}
    for cohort in COHORTS:
        rows = groups[cohort].loc[groups[cohort].mrs12.notna()].sort_values("analysis_order")
        ids = rows.study_id.to_numpy(int)
        assert rows.analysis_order.notna().all() and len(np.unique(ids)) == len(ids)
        observed[cohort] = dict(x=matrix(rows), y=outcome(rows).astype(int), ids=ids)
    return eligible, observed


def fitted(data, tr, query, policy, seed, kind, engine=history_marginalized, clinical_c=None):
    """Select G and estimate the fixed policy using development training rows only.

    Query rows supply predictors for scoring, never outcomes or selection criteria.
    """
    x, y, ids = (data[k][tr] for k in ("x", "y", "ids"))
    with threadpool_limits(limits=1):
        g, candidate = strong_g(x, y, ids, seed)
        task = dict(train=x, y=y, ids=ids, test=query, kind=kind, baseline=g, baseline_spec=candidate)
        result = engine.fit_batch(task, [policy])
        p = result["predictions"][0]
        if clinical_c is not None:
            p = np.vstack([p, age_function_baseline(x, y, query, clinical_c)])
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p[1], g[state(query)], rtol=0, atol=2e-12)
    return p, result["models"].get(policy["name"]), bool(result["nonage"][0]), candidate


def auc(p, y):
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return np.full(p.shape[:-1], np.nan)
    ranks = rankdata(p, axis=-1)
    return (ranks[..., y == 1].sum(axis=-1) - positive * (positive + 1) / 2) / (positive * negative)


def log_loss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean(axis=-1)


def nested(target, data, missing_x, policy, plan, pool):
    y, x, ids = (data[k] for k in ("y", "x", "ids"))
    folds, repeats = plan["outer_folds"], plan["outer_repeats"]
    seed = plan["outer_seed"]
    split = strict_splits(y, ids, folds=folds, repeats=repeats, seed=stable_seed(seed, target, "outer"))
    clinical_c = plan["clinical_baseline"]["ridge_c"]
    results = pool(delayed(fitted)(data, tr, np.vstack([x[te], missing_x]), policy, stable_seed(seed, "inner", target, k), "cv", clinical_c=clinical_c)
                   for k, (tr, te) in enumerate(split))
    oof = np.full((len(ARMS), repeats, len(x)), np.nan)
    auc_sum = np.zeros(len(ARMS))
    total_pairs = 0
    missing_loss = np.zeros((len(missing_x), len(CORE_ARMS), 2))
    for k, ((_, te), (p, _, _, _)) in enumerate(zip(split, results, strict=True)):
        held_out = p[:, :len(te)]
        oof[:, k // folds, te] = held_out
        labels = y[te]
        n_pairs = labels.sum() * (1 - labels).sum()
        auc_sum += n_pairs * auc(held_out, labels)
        total_pairs += n_pairs
        missing_loss += squared_losses(p[:len(CORE_ARMS), len(te):].T)
    assert np.isfinite(oof).all()
    losses = (oof - y) ** 2
    average = oof.mean(axis=1)
    brier = losses.mean(axis=(1, 2))
    points, estimands = [], []
    for j, arm in enumerate(ARMS):
        points.append(dict(target=target, cohort="development", model=arm, n=len(y), events=int(y.sum()),
                           Brier=brier[j], AUROC=auc_sum[j] / total_pairs, log_loss=log_loss(oof[j], y).mean(), mean_risk=oof[j].mean()))
        estimands.append(dict(target=target, model=arm, within_fold_pair_weighted=auc_sum[j] / total_pairs,
                              pooled_repeat_mean=auc(oof[j], y).mean(), mean_prediction_AUROC=auc(average[j], y),
                              mean_individual_loss=brier[j], loss_of_mean_prediction=((y - average[j]) ** 2).mean()))
    cv_gains = (losses[:2] - losses[2]).mean(axis=-1)
    # Evaluate both possible outcome assignments before aggregating held-out losses.
    core_loss = squared_losses(oof[:len(CORE_ARMS)].transpose(2, 0, 1)).mean(axis=2)
    return points, estimands, average, cv_gains, core_loss, missing_loss / len(split)


def draw_metrics(p, y, target, cohort, kind, index):
    loss = (p - y) ** 2
    arms = ARMS[:len(p)]
    values = dict(zip(arms, loss.mean(axis=1), strict=True))
    values.update({f"{arm}_minus_HG": (loss[j] - loss[2]).mean() for j, arm in enumerate(arms) if arm != "HG"})
    values.update({f"risk_{name}": value for name, value in zip(arms, p.mean(axis=1))})
    values.update({f"AUROC_{name}": value for name, value in zip(arms, auc(p, y))})
    return dict(target=target, cohort=cohort, kind=kind, index=index, n=len(y), events=int(y.sum()), **values)


def refit_draw(index, target, observed, query, spans, policy, seed, precision, clinical_c):
    dev = observed["development"]
    n = len(dev["y"])
    random_seed = stable_seed(seed, "bootstrap", target, index)
    tr = np.random.default_rng(random_seed).integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(tr))
    assert len(oob) and not np.intersect1d(dev["ids"][oob], dev["ids"][tr]).size
    engine = conditional_history if precision else history_marginalized
    p, _, nonage, candidate = fitted(dev, tr, query, policy, random_seed, "bootstrap", engine, clinical_c)
    rows = [draw_metrics(p[:, oob], dev["y"][oob], target, "development", "oob", index)]
    for cohort in COHORTS[1:]:
        rows.append(draw_metrics(p[:, spans[cohort]], observed[cohort]["y"], target, cohort, "development_refit", index))
    return rows, nonage, candidate


def temporal_samples(data, cohort, target, plan):
    y = data["y"]
    n, draws = len(y), plan["validation_patient_draws"]
    rng = np.random.default_rng(stable_seed(plan["validation_patient_seed"], cohort, target))
    ordinary = rng.integers(0, n, (draws, n))
    strata = [np.flatnonzero(y == k) for k in (0, 1)]
    stratified = np.column_stack([rng.choice(v, (draws, len(v)), replace=True) for v in strata])
    samples = dict(patient=ordinary, stratified=stratified)
    return samples


def statistic(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(evaluable=0, mean=None, lower=None, upper=None, positive=None)
    lower, upper = np.quantile(v, [.025, .975])
    return dict(evaluable=len(v), mean=v.mean(), lower=lower, upper=upper, positive=int((v > 0).sum()))




def resampling(target, observed, query, spans, model_p, policy, plan, pool, precision=False, clinical_c=None):
    print(f"{target}: {plan['development_refits']:,} complete development refits", flush=True)
    refits = pool(delayed(refit_draw)(i, target, observed, query, spans, policy, plan["development_seed"], precision, clinical_c)
                  for i in range(plan["development_refits"]))
    rows = [row for sample, _, _ in refits for row in sample]
    for cohort in COHORTS[1:]:
        p, y = model_p[:, spans[cohort]], observed[cohort]["y"]
        for kind, samples in temporal_samples(observed[cohort], cohort, target, plan).items():
            if precision and kind == "stratified":
                continue
            rows.extend(draw_metrics(p[:, ix], y[ix], target, cohort, kind, i) for i, ix in enumerate(samples))
    draws = pd.DataFrame(rows)
    keys = ["target", "cohort", "kind"]
    value_columns = [name for name in draws if name not in (*keys, "index", "n", "events")]
    summary, classes = [], []
    for key, group in draws.groupby(keys, sort=False):
        tag = dict(zip(keys, key))
        assert group["index"].nunique() == len(group)
        classes.append(dict(tag, draws=len(group), event_free=int(group.events.eq(0).sum()), no_nonevents=int(group.events.eq(group.n).sum())))
        for name in ("G_minus_HG",) if precision else value_columns:
            summary.append(dict(tag, estimand=name, draws=len(group), **statistic(group[name])))
    return draws, pd.DataFrame(summary), pd.DataFrame(classes), np.mean([row[1] for row in refits])


def wilson(events, n):
    z = 1.959963984540054
    p = events / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., c - e), min(1., c + e)


def risk_intervals(target, observed, avg, points, summary, plan):
    n = len(observed["development"]["y"])
    seed = stable_seed(plan["validation_patient_seed"], "descriptive_mean_risk", target)
    ids = np.random.default_rng(seed).integers(0, n, (plan["validation_patient_draws"], n))
    values = avg.T[ids].mean(axis=1)
    rows = []
    for co in COHORTS:
        y = observed[co]["y"]
        lo, hi = wilson(int(y.sum()), len(y))
        rows.append(dict(cohort=co, model="Observed", mean=y.mean(), lower=lo, upper=hi))
        for j, arm in enumerate(ARMS):
            point = points.loc[(target, co, arm), "mean_risk"]
            if co == "development":
                lo, hi = np.quantile(values[:, j], [.025, .975])
            else:
                r = summary.loc[(target, co, "patient", f"risk_{arm}")]
                lo, hi = r.lower, r.upper
            rows.append(dict(cohort=co, model=arm, mean=point, lower=lo, upper=hi))
    return pd.DataFrame(rows)


def assessment(target, rows, point, summary, cv_gains, nonage):
    # Descriptive repeated-fold diagnostics, never model-qualification criteria.
    rows[2].extend(dict(target=target,control=arm,cv_delta=cv_gains[j].mean(),
        cv_positive=int((cv_gains[j]>0).sum()),nonage_retention=nonage)
        for j,arm in enumerate(CORE_ARMS[:2]))


def target_results(frame, target, plan, pool):
    # Temporal records supply query predictors and outcomes for scoring after
    # all fitting in development under the fixed specification.
    config = plan["policy"]
    clinical_c = plan["clinical_baseline"]["ridge_c"]
    eligible, observed = rosters(frame, target)
    dev = observed["development"]
    incomplete = ~np.isfinite(eligible["development"]["y"])
    missing_x = eligible["development"]["x"][incomplete]
    points, estimands, average, cv_gains, observed_loss, missing_loss = nested(target, dev, missing_x, config, plan, pool)
    query = np.vstack([observed[co]["x"] for co in COHORTS])
    edges = np.r_[0, np.cumsum([len(observed[co]["y"]) for co in COHORTS])]
    spans = {co: slice(edges[i], edges[i + 1]) for i, co in enumerate(COHORTS)}
    seed = stable_seed(plan["full_fit_seed"], "full", target, 0)
    p, models, _, _ = fitted(dev, np.arange(len(dev["y"])), query, config, seed, "full", clinical_c=clinical_c)
    # The full-development fit is frozen before temporal outcomes are scored.
    if plan.get("frozen_output"):
        from pathlib import Path
        import pickle, json, hashlib
        folder=Path(plan["frozen_output"]); folder.mkdir(parents=True, exist_ok=True)
        blob=pickle.dumps(dict(models=models,training_ids=dev['ids'],policy=config,seed=seed),protocol=5)
        (folder/f'{target}_frozen.pkl').write_bytes(blob)
        (folder/f'{target}_freeze.json').write_text(json.dumps(dict(target=target,sha256=hashlib.sha256(blob).hexdigest(),
            training_n=len(dev['ids']),validation_feedback=False,stage='full development fit saved before temporal scoring'),indent=2),encoding='utf-8')
        prediction_rows=[]
        for cohort_name in COHORTS:
            q=average if cohort_name=='development' else p[:,spans[cohort_name]].T
            # average has shape arms x patients in the existing estimator.
            if cohort_name=='development':q=average.T
            for ii,record_id in enumerate(observed[cohort_name]['ids']):
                prediction_rows.append(dict(study_id=int(record_id),cohort=cohort_name,y=int(observed[cohort_name]['y'][ii]),**dict(zip(ARMS,map(float,q[ii])))))
        pd.DataFrame(prediction_rows).to_csv(folder/f'{target}_predictions.csv',index=False)
    directions, exact, leave = [], [], []
    for co in COHORTS[1:]:
        d = observed[co]
        q, y = p[:, spans[co]], d["y"]
        for j, arm in enumerate(ARMS):
            points.append(dict(target=target, cohort=co, model=arm, n=len(y), events=int(y.sum()), Brier=((q[j] - y) ** 2).mean(),
                               AUROC=auc(q[j], y), log_loss=log_loss(q[j], y), mean_risk=q[j].mean()))
        for tier in np.unique(state(d["x"])):
            subset = state(d["x"]) == tier
            difference = q[2, subset] - q[1, subset]
            directions.append(dict(target=target, cohort=co, tier=tier, n=int(subset.sum()), events=int(y[subset].sum()),
                                   risk_up=int((difference > 1e-12).sum()), risk_down=int((difference < -1e-12).sum()),
                                   mean_risk_change=difference.mean()))
    draws, summary, classes, nonage = resampling(target, observed, query, spans, p, config, plan["primary"], pool, clinical_c=clinical_c)
    index = summary.set_index(["target", "cohort", "kind", "estimand"])
    point = pd.DataFrame(points).set_index(["target", "cohort", "model"])
    assessments = []
    assessment(target, (exact, leave, assessments), point, index, cv_gains, nonage)
    scenarios, bounds, ascertainment, reasons = [], [], [], []
    g = np.asarray(models["G"]["baseline"])
    for co, records in eligible.items():
        if co == "development":
            # The development roster and the fitted rows retain their original order.
            np.testing.assert_array_equal(records["x"][~incomplete], dev["x"])
            loss = np.empty((len(incomplete), 3, 2))
            loss[~incomplete], loss[incomplete] = observed_loss, missing_loss
        else:
            q = np.column_stack([history_marginalized.predict(models[arm], records["x"]) for arm in CORE_ARMS])
            np.testing.assert_allclose(q[np.isfinite(records["y"])].T, p[:len(CORE_ARMS), spans[co.removesuffix("_closed")]], rtol=0, atol=2e-12)
            loss = squared_losses(q)
        a, b, c, d = scenario_rows(target, co, records, loss, g, plan["missing"])
        scenarios.extend(a)
        bounds.extend(b)
        ascertainment.extend(c)
        reasons.extend(d)
    fitted_hg = models["HG"]
    base = fitted_hg["base"]
    constants = [dict(target=target, tier=k, unique_n=int((state(dev["x"]) == k).sum()), G_probability=g[k],
                      weight=fitted_hg["taper"][k], anchor=base["fine" if k == 1 else "table"]["intercepts"][k]) for k in range(3)]
    result = dict(model_points=point.reset_index(), OOF_estimands=estimands, bootstrap_draws=draws, bootstrap_summary=summary,
                  bootstrap_class_audit=classes, bootstrap_assessment=assessments, temporal_tier_risk_updates=directions, model_constants=constants,
                  missing_outcome_scenarios=scenarios, missing_outcome_sharp_bounds=bounds, outcome_ascertainment=ascertainment,
                  outcome_ascertainment_reasons=reasons)
    if target == "overall_dynamic":
        result["risk_intervals"] = risk_intervals(target, observed, average, point, index, plan["primary"])
    return {name: pd.DataFrame(value) for name, value in result.items()}


def run(frame, plan, where, jobs):
    results = {}
    with Parallel(n_jobs=jobs, verbose=6) as pool:
        for target in plan["populations"]:
            for name, table in target_results(frame, target, plan, pool).items():
                results.setdefault(name, []).append(table)
    for name, tables in results.items():
        pd.concat(tables, ignore_index=True).to_csv(where / f"{name}.csv", index=False, encoding="utf-8")


def run_precision(frame, plan, where, jobs):
    """Recompute the earlier numerical precision comparison for Supplementary Table 12."""
    target = "overall_dynamic"
    _, observed = rosters(frame, target)
    dev = observed["development"]
    query = np.vstack([observed[co]["x"] for co in COHORTS])
    edges = np.r_[0, np.cumsum([len(observed[co]["y"]) for co in COHORTS])]
    spans = {co: slice(edges[i], edges[i + 1]) for i, co in enumerate(COHORTS)}
    seed = stable_seed(plan["full_fit_seed"], "full", target, 0)
    p, _, _, _ = fitted(dev, np.arange(len(dev["y"])), query, plan["policy"], seed, "full", conditional_history)
    with Parallel(n_jobs=jobs, verbose=6) as pool:
        _, summary, _, _ = resampling(target, observed, query, spans, p, plan["policy"], plan["precision"], pool, precision=True)
    summary.to_csv(where / "precision_summary.csv", index=False, encoding="utf-8")
