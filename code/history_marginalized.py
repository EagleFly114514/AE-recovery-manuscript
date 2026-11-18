# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Evaluate the same conditional risk given observed function or history alone."""
import warnings

import numpy as np
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from common import history_design, history_transform, state
import conditional_history as conditional


def fit_function_probabilities(x, precision):
    group = state(x)
    pooled = np.zeros(len(group), int)
    transform = history_transform(x, pooled, "neutral_winsor", 1)
    z = history_design(x, pooled, transform)
    levels = np.unique(group)
    if len(levels) == 1:
        beta, intercept = np.zeros((1, z.shape[1])), np.zeros(1)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(C=1 / precision, solver="lbfgs", max_iter=1200, tol=1e-10).fit(z, group)
        beta, intercept = model.coef_, model.intercept_
        np.testing.assert_array_equal(levels, model.classes_)
    return dict(transform=transform, levels=levels.tolist(), coefficients=beta.tolist(), intercepts=intercept.tolist())


def function_probabilities(model, x):
    z = history_design(x, np.zeros(len(x), int), model["transform"])
    levels = np.asarray(model["levels"])
    score = z @ np.asarray(model["coefficients"]).T + np.asarray(model["intercepts"])
    if len(levels) == 2:
        p = expit(score[:, 0])
        probabilities = np.column_stack([1 - p, p])
    else:
        probabilities = softmax(score, axis=1)
    full = np.zeros((len(z), 3))
    full[:, levels] = probabilities
    return full


def within_tier_query(x, tier):
    query = x.copy()
    query[:, 5] = tier == 1
    query[:, 6] = tier == 2
    return query


def predict(model, x):
    if model["model"] != "H":
        return conditional.predict(model, x)
    p = function_probabilities(model["function_given_history"], x)
    q = np.column_stack([conditional.predict(model["conditional_HG"], within_tier_query(x, k)) for k in range(3)])
    return np.sum(p * q, axis=1)


def fit_batch(task, configs):
    x = task["test"]
    original_n = len(x)
    query = np.vstack([x] + [within_tier_query(x, k) for k in range(3)])
    result = conditional.fit_batch(dict(task, test=query), configs)
    predictions = result["predictions"][..., :original_n].copy()
    gates = {}
    for i, c in enumerate(configs):
        penalty = c["gate_precision"]
        if penalty not in gates:
            gate = fit_function_probabilities(task["train"], penalty)
            gates[penalty] = (gate, function_probabilities(gate, x))
        gate, weights = gates[penalty]
        components = result["predictions"][i, 2, original_n:].reshape(3, original_n).T
        predictions[i, 0] = np.sum(components * weights, axis=1)
        if task["kind"] == "full":
            selected = result["models"][c["name"]]
            selected["H"] = dict(model="H", function_given_history=gate, conditional_HG=selected["HG"])
    return dict(result, predictions=predictions, audit=dict(result["audit"], function_models=len(gates)))
