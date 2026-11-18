# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Original numeric preprocessing, grouped splits and one-SE tuning rule."""
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from functional_baseline import stable_seed



def make_analysis_ridge_pipeline(
    columns: Sequence[str], *, seed: int, no_indicator: set[str], c: float = 1.0
) -> Pipeline:
    """Fold-local numeric pipeline with non-duplicated missingness encoding.

    D2--D4 already have explicit blood/CSF availability, and the gated course
    block already has ``course_verified``.  Automatic missing indicators for
    those columns would duplicate the same availability information and alter
    ridge-penalty allocation.  Other numeric predictors retain training-fold
    indicators exactly as in the maintained Tier-1 pipeline.
    """

    with_indicator = [
        column for column in columns if column not in no_indicator
    ]
    without_indicator = [
        column for column in columns if column in no_indicator
    ]

    def imputer(*, add_indicator: bool) -> SimpleImputer:
        try:
            return SimpleImputer(
                strategy="median", add_indicator=add_indicator, keep_empty_features=True
            )
        except TypeError:  # pragma: no cover - older sklearn compatibility
            return SimpleImputer(strategy="median", add_indicator=add_indicator)

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if with_indicator:
        transformers.append(
            (
                "numeric_with_indicator",
                Pipeline(
                    [
                        ("impute", imputer(add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                with_indicator,
            )
        )
    if without_indicator:
        transformers.append(
            (
                "numeric_with_explicit_availability",
                Pipeline(
                    [
                        ("impute", imputer(add_indicator=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                without_indicator,
            )
        )
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(
        penalty="l2",
        C=float(c),
        solver="lbfgs",
        max_iter=3000,
        random_state=int(seed),
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    records: list[tuple[np.ndarray, np.ndarray]] = []
    digest = hashlib.sha256()
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=stable_seed(seed, "fold", repeat)
        )
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            if np.intersect1d(np.unique(groups[train]), np.unique(groups[test])).size:
                raise RuntimeError("Physical patient crossed a validation fold")
            if len(np.unique(y[train])) < 2:
                raise RuntimeError("A training fold contained one outcome class")
            records.append((train, test))
            token = f"{repeat}:{fold}:" + ",".join(sorted(str(x) for x in np.unique(groups[test])))
            digest.update(token.encode("utf-8"))
    return records, digest.hexdigest().upper()


def _one_se_choice(
    fold_scores: Mapping[float, Sequence[float]],
) -> tuple[float, list[dict[str, Any]]]:
    summaries = []
    for ridge_c in sorted(fold_scores):
        values = np.asarray(fold_scores[ridge_c], dtype=float)
        mean = float(np.mean(values))
        se = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        summaries.append(
            {
                "ridge_c": float(ridge_c),
                "mean_inner_brier": mean,
                "se_inner_brier": se,
                "inner_fold_n": len(values),
            }
        )
    best = min(
        summaries,
        key=lambda row: (row["mean_inner_brier"], row["ridge_c"]),
    )
    threshold = best["mean_inner_brier"] + best["se_inner_brier"]
    eligible = [
        row for row in summaries if row["mean_inner_brier"] <= threshold + 1e-15
    ]
    chosen = min(eligible, key=lambda row: row["ridge_c"])["ridge_c"]
    for row in summaries:
        row["minimum_mean_brier_c"] = best["ridge_c"]
        row["one_se_threshold"] = threshold
        row["one_se_eligible"] = row["mean_inner_brier"] <= threshold + 1e-15
        row["selected_c"] = float(chosen)
    return float(chosen), summaries
