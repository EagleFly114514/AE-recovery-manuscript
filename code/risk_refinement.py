# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Tier risk and mean-preserving intercept solver."""
import numpy as np
from scipy.special import expit, logit
from common import state, history_design, effective_slopes


def normalized_link(score, group, baseline):
    """Solve each intercept from mean(expit(intercept + score)) = baseline."""
    n_groups = len(baseline)
    count = np.bincount(group, minlength=n_groups)
    observed = count > 0
    interior = (baseline > 0) & (baseline < 1)
    safe_baseline = np.clip(baseline, 1e-14, 1 - 1e-14)
    anchor = logit(safe_baseline)
    smallest = np.full(n_groups, np.inf)
    largest = np.full(n_groups, -np.inf)
    np.minimum.at(smallest, group, score)
    np.maximum.at(largest, group, score)
    smallest[~observed] = largest[~observed] = 0
    lower = anchor - largest
    upper = anchor - smallest
    intercept = anchor - np.bincount(group, weights=score, minlength=n_groups) / np.maximum(count, 1)
    active = observed & interior
    for iteration in range(80):
        prob = expit(intercept[group] + score)
        prob = np.where(interior[group], prob, baseline[group])
        residual = np.bincount(group, weights=prob, minlength=n_groups) - baseline * count
        live = active & (np.abs(residual) > 1e-13 * count)
        if not live.any():
            return prob, intercept, iteration + 1
        lower = np.where(live & (residual < 0), intercept, lower)
        upper = np.where(live & (residual > 0), intercept, upper)
        derivative = np.bincount(group, weights=prob * (1 - prob), minlength=n_groups)
        step = intercept - residual / np.maximum(derivative, 1e-30)
        candidate = np.where((step > lower) & (step < upper), step, 0.5 * (lower + upper))
        intercept = np.where(live, candidate, intercept)
    raise RuntimeError("Mean-preserving intercept root did not converge")


def predict(model, x):
    group = np.zeros(len(x), int) if model["model"] == "H" else state(x)
    baseline = np.asarray(model["baseline"])
    if model["model"] == "G":
        return baseline[group]
    z = history_design(x, group, model["transform"])
    slopes = effective_slopes(model["coefficients"], len(baseline))
    probability = expit(np.asarray(model["intercepts"])[group] + np.einsum("ij,ij->i", z, slopes[group]))
    return np.where((baseline[group] > 0) & (baseline[group] < 1), probability, baseline[group])
