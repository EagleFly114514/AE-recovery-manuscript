# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""The original soft-percentile conditional density path."""
from itertools import product

import numpy as np
from scipy.special import expit, softmax

from common import baseline_design, state
from risk_refinement import normalized_link, predict as predict_tilt

BINARY = [1, 2, 4]
CONTINUOUS = [0, 3]
PATTERNS = np.array(list(product((0, 1), repeat=3)))
PATTERN_CODES = np.array([4, 2, 1])
BURDEN = PATTERNS.sum(axis=1)


def adjacent_matrix(k, smoothing):
    result = np.eye(k)
    if k > 1:
        for step in (-1, 1):
            i = np.arange(k)
            j = np.clip(i + step, 0, k - 1)
            result[i, i] -= smoothing / 2
            result[i, j] += smoothing / 2
    return result


def table(codes, k, group, y, n_groups):
    available = codes >= 0
    index = ((group * 2 + y) * k + codes)[available]
    return np.bincount(index, minlength=n_groups * 2 * k).reshape(n_groups, 2, k).astype(float)


def burden_smooth(counts):
    marginal = counts.sum(axis=1) + .5
    result = counts.copy()
    for burden in range(4):
        select = BURDEN == burden
        prior = marginal[:, select] / marginal[:, select].sum(axis=1, keepdims=True)
        same_burden = counts[:, :, select].sum(axis=2)
        result[:, :, select] = .5 * counts[:, :, select] + .5 * same_burden[:, :, None] * prior[:, None, :]
    return result


def density(counts, baseline, prior, pooling):
    marginal = counts.sum(axis=1) + .5
    marginal /= marginal.sum(axis=1, keepdims=True)
    n = counts.sum(axis=2)
    q = (counts + prior * marginal[:, None, :]) / (n[:, :, None] + prior)
    r = np.log(q[:, 1]) - np.log(q[:, 0])
    effective = np.divide(n[:, 0] * n[:, 1], n.sum(axis=1), out=np.zeros(len(n)), where=n.sum(axis=1) > 0)
    if pooling == "shared_outcome" and len(baseline) == 3 and effective.sum() > 0:
        common = np.average(r, axis=0, weights=effective)
        r = .5 * r + .5 * common
    if pooling == "shared_boundary_change" and len(baseline) == 3:
        edges = np.array([0, 2])
        sign = np.array([1., -1.])
        if effective[edges].sum() > 0:
            shared = np.average(r[edges] * sign[:, None], axis=0, weights=effective[edges])
            r[edges] = .5 * r[edges] + .5 * sign[:, None] * shared
    tilt = np.column_stack([-baseline, 1 - baseline])
    return softmax(np.log(marginal)[:, None] + tilt[:, :, None] * r[:, None], axis=2)


def percentile_fit(x):
    result = []
    grid = np.linspace(.025, .975, 21)
    for col in CONTINUOUS:
        observed = x[np.isfinite(x[:, col]), col]
        if not len(observed):
            result.append(dict(values=[], quantiles=[]))
            continue
        values, position = np.unique(np.quantile(observed, grid), return_inverse=True)
        quantile = np.bincount(position, weights=grid) / np.bincount(position)
        result.append(dict(values=values.tolist(), quantiles=quantile.tolist()))
    return result


def hats(percentile, cells):
    position = np.clip(percentile * cells - .5, 0., cells - 1)
    left = position.astype(int)
    right = np.minimum(left + 1, cells - 1)
    fraction = position - left
    p = np.zeros((len(percentile), cells))
    ix = np.arange(len(percentile))
    p[ix, left] = 1 - fraction
    p[ix, right] += fraction
    return p


def memberships(x, fit, cells):
    result = []
    for col, param in zip(CONTINUOUS, fit):
        seen = np.isfinite(x[:, col])
        u = np.full(len(x), .5)
        if len(param["values"]) > 1:
            u[seen] = np.interp(x[seen, col], param["values"], param["quantiles"], left=0., right=1.)
        result.append((hats(u, cells), seen))
    return result


def pattern_compatibility(x):
    acute = x[:, BINARY]
    available = np.isfinite(acute)
    assert np.isin(acute[available], [0, 1]).all()
    match = np.all((acute[:, None, :] == PATTERNS[None]) | ~available[:, None], axis=2)
    return acute, available, match


def fractional_pattern_counts(complete, match, available, group, y):
    counts = np.zeros_like(complete)
    for k in range(len(complete)):
        marginal = complete[k].sum(axis=0) + .5
        pseudo = 8 * marginal / marginal.sum()
        for value in (0, 1):
            observed = match[(group == k) & (y == value) & available.any(axis=1)]
            if not len(observed):
                continue
            q = pseudo / pseudo.sum()
            for _ in range(150):
                posterior = observed * q
                posterior /= posterior.sum(axis=1, keepdims=True)
                expected = posterior.sum(axis=0)
                update = (expected + pseudo) / (len(observed) + pseudo.sum())
                if np.max(abs(update - q)) <= 1e-11:
                    break
                q = update
            else:
                raise RuntimeError("Fractional pattern counts did not converge")
            np.testing.assert_allclose(expected.sum(), len(observed), atol=1e-10, rtol=0)
            counts[k, value] = expected
    return counts


def log_evidence(group, continuous, patterns, p):
    result = []
    for (weights, observed), probabilities in zip(continuous, p[:2]):
        mass = np.sum(probabilities[group] * weights[:, None], axis=2)
        result.append(np.where(observed, np.log(mass[:, 1]) - np.log(mass[:, 0]), 0.))
    mass = np.sum(p[2][group] * patterns[:, None], axis=2)
    result.append(np.log(mass[:, 1]) - np.log(mass[:, 0]))
    return np.column_stack(result)


def predict(model, x):
    if model["model"] == "G":
        return predict_tilt(model, x)
    group = np.zeros(len(x), int) if model["model"] == "H" else state(x)
    c = model["config"]
    continuous = memberships(x, model["percentiles"], c["cells"])
    _, _, patterns = pattern_compatibility(x)
    p = [np.asarray(v) for v in model["densities"]]
    score = log_evidence(group, continuous, patterns, p).sum(axis=1)
    return expit(np.asarray(model["intercepts"])[group] + c["temperature"] * score)


def fit_batch(task, configs):
    predictions = np.empty((len(configs), 3, len(task["test"])))
    nonage = np.zeros(len(configs), bool)
    models = {}
    raw, test, y = task["train"], task["test"], task["y"]
    ranks = percentile_fit(raw)
    cells = set(c["cells"] for c in configs)
    continuous = {k: memberships(raw, ranks, k) for k in cells}
    continuous_v = {k: memberships(test, ranks, k) for k in cells}
    acute, available, compatibility = pattern_compatibility(raw)
    _, _, compatibility_v = pattern_compatibility(test)
    observed_code = np.where(available.all(axis=1), np.where(available, acute, 0) @ PATTERN_CODES, -1).astype(int)
    max_error = 0.
    n_fits = 0
    for slot, name in ((0, "H"), (2, "HG")):
        group, gv, baseline = baseline_design(task, name)
        complete = table(observed_code, 8, group, y, len(baseline))
        counts = {"complete_patterns": burden_smooth(complete),
                  "fractional_patterns": burden_smooth(fractional_pattern_counts(complete, compatibility, available, group, y))}
        cell_counts, fitted, final = {}, {}, {}
        for i, c in enumerate(configs):
            k = c["cells"]
            raw_key = k, c["smoothing"]
            if raw_key not in cell_counts:
                smooth = adjacent_matrix(k, c["smoothing"])
                terms = []
                for weights, observed in continuous[k]:
                    rows = [weights[observed & (group == level) & (y == value)].sum(axis=0)
                            for level in range(len(baseline)) for value in (0, 1)]
                    terms.append(np.asarray(rows).reshape(len(baseline), 2, k) @ smooth)
                cell_counts[raw_key] = terms
            pool = c["pool"] if name == "HG" else "within_tier"
            fit_key = (*raw_key, c["prior"], c["acute"], pool)
            if fit_key not in fitted:
                p = [density(v, baseline, c["prior"], pool) for v in cell_counts[raw_key] + [counts[c["acute"]]]]
                train = log_evidence(group, continuous[k], compatibility, p)
                query = log_evidence(gv, continuous_v[k], compatibility_v, p).sum(axis=1)
                fitted[fit_key] = p, train, query
                n_fits += 1
            p, e, query = fitted[fit_key]
            key = (*fit_key, c["temperature"])
            if key not in final:
                temperature = c["temperature"]
                pred, intercepts, _ = normalized_link(temperature * e.sum(axis=1), group, baseline)
                max_error = max(max_error, max(abs(pred[group == level].mean() - baseline[level]) for level in np.unique(group)))
                other = e[:, 1:].sum(axis=1)
                active = any(other[group == level].std() > 1e-10 for level in np.unique(group))
                final[key] = expit(intercepts[gv] + temperature * query), intercepts, active
            predictions[i, slot], intercepts, active = final[key]
            if name == "HG":
                nonage[i] = active
            if task["kind"] == "full":
                models.setdefault(c["name"], {})[name] = dict(model=name, config=c, percentiles=ranks,
                    baseline=baseline.tolist(), densities=[v.tolist() for v in p], intercepts=intercepts.tolist())
    assert max_error < 1e-11
    g = dict(model="G", baseline=task["baseline"].tolist(), baseline_spec=task["baseline_spec"])
    predictions[:, 1] = predict(g, task["test"])
    for candidate in models.values():
        candidate["G"] = g
    return dict(predictions=predictions, nonage=nonage, models=models,
                audit=dict(conditional_density_fits=n_fits, maximum_tier_mean_error=max_error))
