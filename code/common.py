# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Training-only transforms shared by the conditional history estimators."""
import numpy as np


def state(x):
    return x[:, 5].astype(int) + 2 * x[:, 6].astype(int)


def history_transform(train, group, mode, n_groups):
    """Fit training-only within-tier history deviations and robust scalings."""
    raw = train[:, :5].copy()
    missing = ~np.isfinite(raw)
    lower, upper = [None] * 5, [None] * 5
    if mode == "neutral_winsor":
        for column in (0, 3):
            observed = raw[~missing[:, column], column]
            if observed.size:
                lower[column], upper[column] = np.quantile(observed, [0.05, 0.95]).tolist()
                raw[:, column] = np.clip(raw[:, column], lower[column], upper[column])
    fill = np.array([
        np.median(raw[~missing[:, j], j]) if (~missing[:, j]).any() else 0.0 for j in range(5)
    ])
    median_filled = np.where(missing, fill, raw)
    centres = np.tile(median_filled.mean(axis=0), (n_groups, 1))
    if mode == "neutral_winsor":
        for j in range(5):
            observed = ~missing[:, j]
            if observed.any():
                centres[:, j] = raw[observed, j].mean()
            for level in range(n_groups):
                subset = observed & (group == level)
                if subset.any():
                    centres[level, j] = raw[subset, j].mean()
        filled = np.where(missing, centres[group], raw)
    else:
        filled = median_filled
        for level in range(n_groups):
            if (group == level).any():
                centres[level] = filled[group == level].mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return dict(mode=mode, median=fill.tolist(), centres=centres.tolist(), scale=scale.tolist(),
                lower=lower, upper=upper)


def history_design(data, group, transform):
    raw = data[:, :5].copy()
    for j, (lo, hi) in enumerate(zip(transform["lower"], transform["upper"])):
        if lo is not None:
            raw[:, j] = np.clip(raw[:, j], lo, hi)
    centres = np.array(transform["centres"])[group]
    fill = centres if transform["mode"] == "neutral_winsor" else np.array(transform["median"])
    return (np.where(np.isfinite(raw), raw, fill) - centres) / np.array(transform["scale"])


def effective_slopes(beta, n_groups):
    beta = np.asarray(beta, float).reshape(-1, 5)
    if len(beta) == 1:
        return np.repeat(beta, n_groups, axis=0)
    return beta[0] + beta[1:]


def baseline_design(task, model):
    if model == "H":
        return np.zeros(len(task["train"]), int), np.zeros(len(task["test"]), int), np.array([task["y"].mean()])
    return state(task["train"]), state(task["test"]), task["baseline"]
