# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Combine the two active historical estimators and apply development support tapering."""
import numpy as np

import local_neighbours as local
import soft_percentile_evidence as soft
from common import state
from risk_refinement import predict as predict_tilt

ENGINES = {"soft": soft, "local": local}


def mixture_predict(model, x):
    if model["model"] == "G":
        return predict_tilt(model, x)
    group = np.zeros(len(x), int) if model["model"] == "H" else state(x)
    weight = np.asarray(model["weight"])[group]
    table = ENGINES[model["table_family"]].predict(model["table"], x)
    fine = ENGINES[model["fine_family"]].predict(model["fine"], x)
    return weight * table + (1 - weight) * fine


def mixture_fit_batch(task, configs):
    groups = state(task["train"])
    permuted = task["train"].copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        permuted[rows, 1:5] = task["train"][np.roll(rows, 1), 1:5]
    query = np.vstack([task["test"], task["train"], permuted])
    query_groups = state(query)
    cfgs = {family: {} for family in ENGINES}
    for c in configs:
        for component in ("table", "fine"):
            cfgs[c[f"{component}_family"]][c[component]["name"]] = c[component]
    fitted = {key: ENGINES[key].fit_batch(dict(task, test=query), list(values.values())) for key, values in cfgs.items() if values}
    positions = {key: {name: i for i, name in enumerate(values)} for key, values in cfgs.items()}
    n, test_n = len(groups), len(task["test"])
    predictions = np.empty((len(configs), 3, test_n))
    nonage = np.zeros(len(configs), bool)
    models = {}
    g = dict(model="G", baseline=task["baseline"].tolist(), baseline_spec=task["baseline_spec"])
    for i, c in enumerate(configs):
        families = {k: c[f"{k}_family"] for k in ("table", "fine")}
        names = {k: c[k]["name"] for k in families}
        f = {k: fitted[v] for k, v in families.items()}
        components = {k: f[k]["predictions"][positions[v][names[k]]] for k, v in families.items()}
        for slot, arm in ((0, "H"), (2, "HG")):
            w = [c["table_weight"][1]] if arm == "H" else c["table_weight"]
            group = np.zeros(len(query), int) if arm == "H" else query_groups
            weight = np.asarray(w)[group]
            p = weight * components["table"][slot] + (1 - weight) * components["fine"][slot]
            predictions[i, slot] = p[:test_n]
            if arm == "HG":
                nonage[i] = np.max(abs(p[test_n:test_n+n] - p[test_n+n:])) > 1e-10
            if task["kind"] == "full":
                models.setdefault(c["name"], {})[arm] = dict(model=arm, weight=w,
                    **{f"{k}_family": v for k, v in families.items()},
                    **{k: f[k]["models"][names[k]][arm] for k in families})
        predictions[i, 1] = mixture_predict(g, task["test"])
        if task["kind"] == "full":
            models[c["name"]]["G"] = g
    return dict(predictions=predictions, nonage=nonage, models=models,
                audit=dict(component_policy_count=sum(map(len, cfgs.values())),
                           maximum_tier_mean_error=max(v["audit"]["maximum_tier_mean_error"] for v in fitted.values())))


def predict(model, x):
    if model["model"] == "G":
        return predict_tilt(model, x)
    group = np.zeros(len(x), int) if model["model"] == "H" else state(x)
    p0 = np.asarray(model["baseline"])[group]
    fitted = mixture_predict(model["base"], x)
    return p0 + np.asarray(model["taper"])[group] * (fitted - p0)


def fit_batch(task, configs):
    definitions = {c["base"]["name"]: c["base"] for c in configs}
    indices = {name: i for i, name in enumerate(definitions)}
    base = mixture_fit_batch(task, list(definitions.values()))
    group = state(task["test"])
    _, unique = np.unique(task["ids"], return_index=True)
    counts = np.bincount(state(task["train"])[unique], minlength=3)
    predictions = np.empty((len(configs), 3, len(group)))
    nonage = np.zeros(len(configs), bool)
    models = {}
    p_g = task["baseline"]
    p_h = float(task["y"].mean())
    for i, c in enumerate(configs):
        if c["fixed"] is None:
            weight = counts / (counts + c["prior"])
            history_weight = len(unique) / (len(unique) + c["prior"])
        else:
            weight = np.asarray(c["fixed"])
            history_weight = float(weight[1])
        name = c["base"]["name"]
        index = indices[name]
        p = base["predictions"][index]
        predictions[i, 0] = p_h + history_weight * (p[0] - p_h)
        predictions[i, 1] = p[1]
        predictions[i, 2] = p[1] + weight[group] * (p[2] - p[1])
        nonage[i] = base["nonage"][index]
        if task["kind"] == "full":
            fitted = base["models"][name]
            models[c["name"]] = dict(G=fitted["G"],
                H=dict(model="H", base=fitted["H"], baseline=[p_h], taper=[history_weight]),
                HG=dict(model="HG", base=fitted["HG"], baseline=p_g.tolist(), taper=weight.tolist()))
    return dict(predictions=predictions, nonage=nonage, models=models,
                audit=dict(distinct_training_patients=int(len(unique)), maximum_tier_mean_error=base["audit"]["maximum_tier_mean_error"]))
