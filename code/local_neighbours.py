# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""The original local history estimator."""
import hashlib

import numpy as np
from scipy.special import expit, logit

from common import baseline_design, history_design, history_transform, state
from risk_refinement import normalized_link, predict as predict_tilt


def kernel_weights(x, z, mask_x, mask_z, group_x, group_z, bandwidth, kind, columns):
    shape = (len(x), len(z))
    coverage = np.zeros(shape)
    distance = np.zeros(shape)
    for j in columns:
        common = mask_x[:, j, None] & mask_z[None, :, j]
        squared = (x[:, j, None] - z[None, :, j]) ** 2
        coverage += common
        if kind == "radial":
            distance += np.where(common, squared, 0.)
        else:
            distance += np.where(common, np.exp(-.5 * squared / bandwidth ** 2), 0.)
    average = distance / np.maximum(coverage, 1)
    weight = np.exp(-.5 * average / bandwidth ** 2) if kind == "radial" else average
    return weight * np.sqrt(coverage / len(columns)) * (group_x[:, None] == group_z[None, :])


def log_evidence(events, support, baseline, count):
    local = (events + count * baseline) / (support + count)
    local = np.clip(local, 1e-14, 1 - 1e-14)
    return logit(local) - logit(baseline)


def predict(model, x):
    if model["model"] == "G":
        return predict_tilt(model, x)
    g = np.zeros(len(x), int) if model["model"] == "H" else state(x)
    p0 = np.asarray(model["baseline"])[g]
    z = history_design(x, g, model["transform"])
    c, memory = model["config"], model["memory"]
    weights = kernel_weights(z, memory["history"], np.isfinite(x[:, :5]), memory["available"], g, memory["group"],
                             c["bandwidth"], c["kernel"], range(5))
    score = log_evidence(weights @ memory["outcome"], weights.sum(axis=1), p0, c["count"])
    return expit(np.array(model["intercepts"])[g] + c["temperature"] * score)


def fit_batch(task, configs):
    predictions = np.empty((len(configs), 3, len(task["test"])))
    nonage = np.zeros(len(configs), bool)
    models = {}
    y = task["y"]
    identities = task["ids"]
    distinct = identities[:, None] != identities[None, :]
    fit_count = iterations = 0
    max_error = 0.0
    for slot, name in ((0, "H"), (2, "HG")):
        g, gv, base = baseline_design(task, name)
        assert np.all((base > 0) & (base < 1))
        transform = history_transform(task["train"], g, "neutral_winsor", len(base))
        z = history_design(task["train"], g, transform)
        v = history_design(task["test"], gv, transform)
        observed = np.isfinite(task["train"][:, :5])
        observed_v = np.isfinite(task["test"][:, :5])
        memory = dict(history=z, available=observed, group=g, outcome=y)
        for bandwidth, kind in dict.fromkeys((c["bandwidth"], c["kernel"]) for c in configs):
            train_weight = kernel_weights(z, z, observed, observed, g, g, bandwidth, kind, range(5))
            pred_weight = kernel_weights(v, z, observed_v, observed, gv, g, bandwidth, kind, range(5))
            pred_e, pred_n = pred_weight @ y, pred_weight.sum(axis=1)
            support = {"resubstitution": (train_weight @ y, train_weight.sum(axis=1))}
            train_weight *= distinct
            support["leave_patient_out"] = (train_weight @ y, train_weight.sum(axis=1))
            if name == "HG":
                age_weight = kernel_weights(z, z, observed, observed, g, g, bandwidth, kind, [0]) * distinct
                age_e, age_n = age_weight @ y, age_weight.sum(axis=1)
            for i, c in enumerate(configs):
                if (c["bandwidth"], c["kernel"]) != (bandwidth, kind):
                    continue
                e, n = support[c["reference"]]
                train_score = c["temperature"] * log_evidence(e, n, base[g], c["count"])
                probability, intercept, used = normalized_link(train_score, g, base)
                score = c["temperature"] * log_evidence(pred_e, pred_n, base[gv], c["count"])
                predictions[i, slot] = expit(intercept[gv] + score)
                error = max(abs(probability[g == k].mean() - p) for k, p in enumerate(base) if (g == k).any())
                max_error = max(max_error, float(error))
                if name == "HG":
                    eo, no = support["leave_patient_out"]
                    change = log_evidence(eo, no, base[g], c["count"]) - log_evidence(age_e, age_n, base[g], c["count"])
                    nonage[i] = any(change[g == k].std() > 1e-10 for k in np.unique(g))
                if task["kind"] == "full":
                    models.setdefault(c["name"], {})[name] = dict(model=name, transform=transform, baseline=base.tolist(),
                        memory=memory, config=c, intercepts=intercept.tolist())
                fit_count += 1
                iterations += used
    g = dict(model="G", baseline=task["baseline"].tolist(), baseline_spec=task["baseline_spec"])
    predictions[:, 1] = predict(g, task["test"])
    for candidate in models.values():
        candidate["G"] = g
    return dict(predictions=predictions, nonage=nonage, models=models,
                audit=dict(fits=fit_count, iterations=iterations, maximum_gradient=0.0, maximum_tier_mean_error=max_error))
