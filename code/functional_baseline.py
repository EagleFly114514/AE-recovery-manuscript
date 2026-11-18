# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Development-only 13-candidate G library and grouped resampling."""
from dataclasses import dataclass
import hashlib
from typing import Any
import warnings

import numpy as np
from scipy.linalg import qr
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import state

G_GRID = (.01, .03, .1, .3, 1., 3., 10., 30., 100., 300., 1000.)


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join(str(x) for x in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little") % (2**32 - 1)


def strict_splits(y, groups, *, folds, repeats, seed):
    """Stratify unique patients, then expand copies without patient leakage.

    Valid only for a single consistent binary endpoint per physical patient.
    For bootstrap copies this balances unique-patient class counts; row counts
    need not be balanced. Never silently reduce folds or retry until favourable.
    """
    y, groups = np.asarray(y, int), np.asarray(groups)
    u, inverse = np.unique(groups, return_inverse=True)
    uy = np.empty(len(u), int)
    for k in range(len(u)):
        labels = np.unique(y[inverse == k])
        if len(labels) != 1:
            raise ValueError("One physical patient has inconsistent outcomes")
        uy[k] = labels[0]
    if set(np.unique(uy)) != {0, 1} or np.bincount(uy, minlength=2).min() < folds:
        raise ValueError("Too few unique patients in either class for requested folds")
    splits = []
    for repeat in range(repeats):
        cv = StratifiedKFold(folds, shuffle=True, random_state=stable_seed(seed, "fold", repeat))
        coverage = np.zeros(len(y), int)
        for tr_u, te_u in cv.split(np.zeros(len(u)), uy):
            tr = np.flatnonzero(np.isin(inverse, tr_u))
            te = np.flatnonzero(np.isin(inverse, te_u))
            if np.intersect1d(groups[tr], groups[te]).size:
                raise AssertionError("Patient leakage")
            if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
                raise AssertionError("A class is absent from a fold")
            coverage[te] += 1
            splits.append((tr, te))
        assert np.all(coverage == 1)
    return splits


@dataclass(frozen=True)
class Candidate:
    method:str
    family:str
    cg:float=0.
    ch:float=0.
    @property
    def key(self):return f'{self.method}|{self.family}|{self.cg:g}|{self.ch:g}'


G = [Candidate('ridge', 'G', cg) for cg in G_GRID] + [Candidate('jeffreys', 'G'), Candidate('flic', 'G')]


class Firth:
    """Jeffreys-penalized logistic fit, optionally corrected at the intercept."""
    def fit(self,z,y,correct=True):
        x=np.column_stack([np.ones(len(z)),z])
        centered=x[:,1:]-x[:,1:].mean(axis=0)
        if centered.shape[1]:
            _,r,piv=qr(centered,mode='economic',pivoting=True)
            diag=np.abs(np.diag(r)); rank=int((diag>max(diag.max(initial=0)*1e-9,1e-10)).sum())
            keep=np.r_[0,1+np.sort(piv[:rank])]
        else:keep=np.array([0])
        x=x[:,keep]; beta=np.zeros(x.shape[1]);beta[0]=np.log((y.sum()+.5)/(len(y)-y.sum()+.5))
        def calc(b):
            eta=x@b;p=expit(eta);w=np.maximum(p*(1-p),1e-12)
            info=x.T@(w[:,None]*x)
            sign,logdet=np.linalg.slogdet(info)
            if sign<=0:return -np.inf,p,w,info
            value=float(np.sum(y*eta-np.logaddexp(0,eta))+.5*logdet)
            return value,p,w,info
        converged=False
        for it in range(500):
            objective,p,w,info=calc(beta)
            inv_xt=np.linalg.solve(info,x.T)
            h=w*np.einsum('ij,ji->i',x,inv_xt)
            score=x.T@(y-p+h*(.5-p))
            step=np.linalg.solve(info,score)
            scale=1.
            for _ in range(35):
                proposed=beta+scale*step
                if calc(proposed)[0]>=objective-1e-10:break
                scale*=.5
            beta=proposed
            if np.max(np.abs(scale*step))<1e-7 or np.max(np.abs(score))<1e-7:
                converged=True;break
        # Fisher scoring uses an approximate Hessian. Rare bootstrap designs can
        # converge very slowly; optimize the SAME Jeffreys objective by BFGS.
        # This fallback changes only the numerical solver, not the model or data.
        def value_score(b):
            objective,p,w,info=calc(b)
            inv_xt=np.linalg.solve(info,x.T)
            h=w*np.einsum('ij,ji->i',x,inv_xt)
            score=x.T@(y-p+h*(.5-p))
            return -objective,-score
        self.fallback=False
        if not converged or np.max(np.abs(value_score(beta)[1]))>1e-5:
            self.fallback=True
            opt=minimize(value_score,beta,method='BFGS',jac=True,
                         options={'gtol':1e-8,'maxiter':2000})
            beta=opt.x
        self.score_norm=float(np.max(np.abs(value_score(beta)[1])))
        if self.score_norm>1e-5 or not np.all(np.isfinite(beta)):
            raise RuntimeError(f'Firth solver residual too large: {self.score_norm}')
        self.uncorrected=beta.copy()
        if correct:
            eta=x@beta
            beta[0]+=brentq(lambda delta:float(np.sum(expit(eta+delta)-y)),-100,100)
        self.beta=beta;self.keep=keep;self.iterations=it+1
        return self
    def predict(self,z):return expit(np.column_stack([np.ones(len(z)),z])[:,self.keep]@self.beta)


def fit_predict(candidate,design):
    y=design['y'];gt,gv=design['gt'],design['gv']
    if candidate.method=='jeffreys':
        probs={k:(y[design['gc']==k].sum()+.5)/((design['gc']==k).sum()+1) for k in np.unique(design['gc'])}
        return np.array([probs.get(k,(y.sum()+.5)/(len(y)+1)) for k in design['gcv']]),probs
    zt, zv = gt, gv
    if candidate.method=='flic':
        model=Firth().fit(zt,y,correct=True)
        return model.predict(zv),model
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=candidate.cg,penalty='l2',solver='lbfgs',max_iter=3000,tol=1e-6)
        model.fit(zt,y)
    return model.predict_proba(zv)[:,1],model


def age_function_baseline(x, y, query, ridge_c):
    """Fit a fixed additive ridge logistic model on age and the observed mRS tier."""
    train = x[:, [0, 5, 6]].copy()
    test = query[:, [0, 5, 6]].copy()
    age = train[np.isfinite(train[:, 0]), 0]
    fill = float(np.median(age)) if age.size else 0.
    for design in (train, test):
        design[~np.isfinite(design[:, 0]), 0] = fill
    scale = StandardScaler().fit(train)
    inputs = dict(y=y, gt=scale.transform(train), gv=scale.transform(test))
    prediction, _ = fit_predict(Candidate("ridge", "G", ridge_c), inputs)
    return prediction


def strong_g(x, y, ids, seed):
    """Select G by development-only inner Brier loss; no external results enter."""
    candidates = G
    assert len(candidates) == 13
    group = state(x)
    features = x[:, 5:7]
    canonical = np.array([[0, 0], [1, 0], [0, 1]])
    scores = np.zeros(len(candidates))
    for tr, te in strict_splits(y, ids, folds=4, repeats=2, seed=seed):
        scaler = StandardScaler().fit(features[tr])
        inputs = dict(y=y[tr], gt=scaler.transform(features[tr]), gv=scaler.transform(features[te]),
                      gc=group[tr], gcv=group[te])
        for k, spec in enumerate(candidates):
            pred, _ = fit_predict(spec, inputs)
            scores[k] += np.sum((y[te] - pred) ** 2)
    index = min(range(len(candidates)), key=lambda k: (scores[k], candidates[k].key))
    scaler = StandardScaler().fit(features)
    inputs = dict(y=y, gt=scaler.transform(features), gv=scaler.transform(canonical), gc=group, gcv=np.arange(3))
    pred, _ = fit_predict(candidates[index], inputs)
    return pred, candidates[index].key
