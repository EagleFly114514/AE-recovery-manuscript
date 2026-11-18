# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Numerically equivalent execution of the fixed legacy nested estimator.

Only Python/DataFrame orchestration overhead is removed. The sklearn L-BFGS
solver, independent C fits, grouped partitions, one-SE choice, and every seed
are retained. See preparation/legacy_execution_parity.json for comparisons
against legacy_omissions.py. No sampling count or scientific rule is changed.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model._logistic import _logistic_regression_path
from threadpoolctl import ThreadpoolController

import legacy_validation as lv
import legacy_omissions as reference
from functional_baseline import stable_seed

# Cache the loaded-library inventory instead of enumerating Windows DLLs in
# each of 400,000 outer fits. Every fit still enters a single-thread limit.
THREADS = ThreadpoolController()


class VectorizedGroupKFold(StratifiedGroupKFold):
    """Same greedy decisions, with all candidate std calculations in one array."""
    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        k = self.n_splits
        trial = np.broadcast_to(y_counts_per_fold, (k, *y_counts_per_fold.shape)).copy()
        trial[np.arange(k), np.arange(k)] += group_y_counts
        evaluations = np.mean(np.std(trial / y_cnt.reshape(1, 1, -1), axis=1), axis=1)
        sizes = np.sum(y_counts_per_fold, axis=1)
        best, minimum, size = 0, np.inf, np.inf
        for i in range(k):
            v = evaluations[i]
            # np.isclose defaults, with the same strict-first / tie-second rule.
            close = abs(v-minimum) <= 1e-8 + 1e-5*abs(minimum) if np.isfinite(minimum) else False
            if v < minimum or (close and sizes[i] < size):
                best, minimum, size = i, v, sizes[i]
        return best


def make_splits(y, groups, *, folds, repeats, seed):
    # Keep the original splitting routine, including group safety checks/digest.
    original = lv.StratifiedGroupKFold
    lv.StratifiedGroupKFold = VectorizedGroupKFold
    try:
        return lv.make_splits(y, groups, folds=folds, repeats=repeats, seed=seed)
    finally:
        lv.StratifiedGroupKFold = original


def feature_arrays(train, test, spec):
    fit = {key: train[key].to_numpy(dtype=float) for key in train.columns}
    query = {key: test[key].to_numpy(dtype=float) for key in test.columns}
    for domain, items in spec['domains'].items():
        fit_items, query_items = [], []
        for item in items:
            a = fit[item['column']]
            b = query[item['column']]
            if item['transform'] == 'log1p':
                a, b = np.log1p(np.clip(a, 0, None)), np.log1p(np.clip(b, 0, None))
            observed = a[~np.isnan(a)]
            if not len(observed):
                lo, hi, median, scale = 0., 0., 0., 1.
            else:
                if len(np.unique(observed)) <= 2:
                    lo, hi = observed.min(), observed.max()
                else:
                    lo, hi = np.quantile(observed, [.01, .99])
                clipped = np.clip(observed, lo, hi)
                median = np.median(clipped)
                q = np.quantile(clipped, [.25, .75]); iqr = q[1]-q[0]
                scale = iqr/1.349 if iqr > 0 else np.std(clipped, ddof=0)
                if not np.isfinite(scale) or scale <= 1e-8: scale = 1.
            a = ((np.clip(a, lo, hi)-median)/scale)*item['direction']
            b = ((np.clip(b, lo, hi)-median)/scale)*item['direction']
            column = 'hmax_item_'+item['key']; fit[column], query[column] = a, b
            fit_items.append(a); query_items.append(b)
        for frame, arrays in [(fit, fit_items), (query, query_items)]:
            # items x patients follows pandas' column-block reduction orientation.
            matrix = np.array(arrays)
            count = (~np.isnan(matrix)).sum(axis=0)
            mean = np.divide(np.nansum(matrix, axis=0), count,
                             out=np.full(matrix.shape[1], np.nan), where=count > 0)
            mean[count/len(items) < spec['coverage']] = np.nan
            frame[domain] = mean
    return fit, query


def feature_pair(train, test, spec):
    a, b = feature_arrays(train, test, spec)
    return pd.DataFrame(a), pd.DataFrame(b)


def preprocess(a, b, columns, no_indicator):
    blocks_a, blocks_b = [], []
    for indicator in [True, False]:
        names = [c for c in columns if (c not in no_indicator) == indicator]
        if not names: continue
        imputer = SimpleImputer(strategy='median', add_indicator=indicator, keep_empty_features=True)
        # DataFrame extraction retains the same feature order as ColumnTransformer.
        x = imputer.fit_transform(np.asfortranarray(np.column_stack([a[n] for n in names])))
        v = imputer.transform(np.asfortranarray(np.column_stack([b[n] for n in names])))
        scale = StandardScaler()
        blocks_a.append(scale.fit_transform(x)); blocks_b.append(scale.transform(v))
    return np.ascontiguousarray(np.hstack(blocks_a)), np.ascontiguousarray(np.hstack(blocks_b))


def predict(z, y, v, c, seed):
    coef, _, _ = _logistic_regression_path(
        z, y, pos_class=1, Cs=[c], fit_intercept=True, max_iter=3000,
        tol=1e-4, solver='lbfgs', penalty='l2', multi_class='ovr',
        random_state=seed, check_input=False, n_threads=1,
    )
    return expit(v @ coef[0, :-1] + coef[0, -1])


def outer_fit(frame, y, groups, tr, te, index, seed, schemas, spec):
    train = frame.iloc[tr]
    inner_seed = stable_seed(seed, 'outer_inner', index)
    grid = spec['validation']['c_grid']; no_indicator = set(spec['no_indicator'])
    splits, _ = make_splits(y[tr], groups[tr], folds=spec['validation']['inner_folds'], repeats=1, seed=inner_seed)
    scores = {name: {c: [] for c in grid} for name in schemas}
    with THREADS.limit(limits=1):
        for j, (fit, valid) in enumerate(splits):
            a, b = feature_arrays(train.iloc[fit], train.iloc[valid], spec)
            for name, columns in schemas.items():
                z, v = preprocess(a, b, columns, no_indicator)
                for c in grid:
                    p = predict(z, y[tr[fit]], v, c, stable_seed(inner_seed, 'inner_model', index, j, name, c))
                    scores[name][c].append(np.mean((y[tr[valid]]-p)**2))
        a, b = feature_arrays(train, frame.iloc[te], spec); predictions = []
        for name, columns in schemas.items():
            c, _ = lv._one_se_choice(scores[name]); z, v = preprocess(a, b, columns, no_indicator)
            predictions.append(predict(z, y[tr], v, c, stable_seed(seed, 'outer_model', index, name)))
    return te, np.array(predictions)


def scores(frame, y, family, seed, labels, spec):
    schemas = {}
    for label in labels:
        h = [name for name in spec['history_columns'] if name != reference.OMISSIONS[label]]
        if label == 'SE timing proxy':
            names = ('HexpandedNoSE12','HexpandedNoSEG12') if family=='dynamic' else ('R_HexpandedNoSE','R_HexpandedNoSEG')
        else: names = ('Omit_H','Omit_HG')
        schemas[names[0]] = h; schemas[names[1]] = [*h, *spec['state_columns'][family]]
    group = frame.study_id.to_numpy(int); validation = spec['validation']
    splits, _ = make_splits(y, group, folds=validation['outer_folds'], repeats=validation['outer_repeats'], seed=seed)
    total = np.zeros((len(schemas),len(y))); count = np.zeros(len(y),int)
    for i, (tr, te) in enumerate(splits):
        _, p = outer_fit(frame, y, group, tr, te, i, seed, schemas, spec)
        total[:,te] += p; count[te] += 1
    assert (count == validation['outer_repeats']).all()
    return ((total/count-y)**2).mean(axis=1)


def bootstrap_draw(frame, y, family, index, spec):
    seed = spec['validation']['seed']
    choice = np.random.default_rng(stable_seed(seed,'targeted_v5_bootstrap_sample',family,index)).choice(len(frame),len(frame),replace=True)
    refit = stable_seed(seed,'targeted_v5_bootstrap_outcome',family,index)
    a, b = scores(frame.iloc[choice].reset_index(drop=True),y[choice],family,refit,['SE timing proxy'],spec)
    return a-b
