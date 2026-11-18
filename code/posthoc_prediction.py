# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""S20-S22 diagnostics and independently seeded matched prediction evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import evaluate
from functional_baseline import strict_splits, stable_seed


def calibration(p,y):
    z=logit(np.clip(p,1e-12,1-1e-12));y=np.asarray(y,int)
    intercept=brentq(lambda a:float(np.sum(expit(z+a)-y)),-100,100)
    z0,z1=z[y==0],z[y==1]
    separated=(not len(z0) or not len(z1) or z0.max()<=z1.min() or z1.max()<=z0.min())
    if separated or np.ptp(z)<1e-12:return intercept,np.nan,np.nan,np.nan,'separation or constant score'
    x=np.column_stack([np.ones(len(z)),z])
    def fn(b):
        eta=x@b
        return float(np.sum(np.logaddexp(0,eta)-y*eta)),x.T@(expit(eta)-y)
    opt=minimize(fn,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':2000})
    if np.max(np.abs(fn(opt.x)[1]))>1e-5:raise RuntimeError('Calibration likelihood did not converge')
    pr=expit(x@opt.x);info=x.T@((pr*(1-pr))[:,None]*x)
    if np.linalg.cond(info)>1e12:return intercept,np.nan,np.nan,np.nan,'ill-conditioned information'
    se=np.sqrt(np.linalg.inv(info)[1,1]);b=opt.x[1]
    return intercept,b,b-1.959963984540054*se,b+1.959963984540054*se,'finite maximum likelihood'


def diagnostics(frame,predictions,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=predictions.merge(frame[['study_id','cohort','mrs6']],on=['study_id','cohort'],validate='one_to_one')
    cal=[];metrics=[];changes=[];decision=[]
    for cohort,rows in p.groupby('cohort',sort=False):
        y=rows.y.to_numpy(int);preds={k:rows[k].to_numpy() for k in ['H','G','HG','B']}
        preds['Persistence']=rows.mrs6.ge(3).to_numpy(float)
        for model,score in preds.items():
            metrics.append(dict(cohort=cohort,model=model,n=len(y),events=int(y.sum()),Brier=float(np.mean((score-y)**2)),AUROC=float(evaluate.auc(score,y))))
            if model!='Persistence':
                a,b,lo,hi,status=calibration(score,y)
                cal.append(dict(cohort=cohort,model=model,n=len(y),events=int(y.sum()),intercept=a,slope=b,lower=lo,upper=hi,status=status))
            for threshold in np.arange(.05,.501,.01):
                positive=score>=threshold
                nb=np.mean(positive*y)-np.mean(positive*(1-y))*threshold/(1-threshold)
                decision.append(dict(cohort=cohort,model=model,threshold=threshold,net_benefit=nb))
        changes.append(dict(cohort=cohort,mean_absolute_probability_change_pp=100*np.mean(np.abs(rows.HG-rows.G))))
    for name,rows in [('S20_calibration',cal),('S21_saved_score_metrics',metrics),('S21_probability_changes',changes),('decision_curves',decision)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)


def direct(train,y,test,columns):
    x=train[:,columns].copy();q=test[:,columns].copy()
    med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in x.T])
    x=np.where(np.isfinite(x),x,med);q=np.where(np.isfinite(q),q,med)
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,tol=1e-6)
    model.fit(scaler.transform(x),y)
    return model.predict_proba(scaler.transform(q))[:,1]


ARMS=['H','G','HG','B','H_direct','HG_additive','Persistence','Prevalence']


def fold_job(k,tr,te,data,policy,seed):
    with threadpool_limits(limits=1):
        x,y=data['x'],data['y']
        pred,_,_,candidate=evaluate.fitted(data,tr,x[te],policy,stable_seed(seed,'audit_inner',k),'cv',clinical_c=1.)
        pred=np.vstack([pred,direct(x[tr],y[tr],x[te],list(range(5))),
                        direct(x[tr],y[tr],x[te],list(range(7))),
                        (x[te,5]+x[te,6]>0).astype(float),np.full(len(te),y[tr].mean())])
    return pred,str(candidate)


def nested(frame,plan,out,jobs=4,seed=20261115,repeats=20,draws=10000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    _,observed=evaluate.rosters(frame,'overall_dynamic');data=observed['development']
    x,y,ids=data['x'],data['y'],data['ids'];folds=5
    splits=strict_splits(y,ids,folds=folds,repeats=repeats,seed=seed)
    fitted=Parallel(n_jobs=jobs)(delayed(fold_job)(k,tr,te,data,plan['policy'],seed) for k,(tr,te) in enumerate(splits))
    oof=np.full((len(ARMS),repeats,len(y)),np.nan);auc_sums=np.zeros((2,len(ARMS)));pairs=np.zeros(2)
    records=[];folds_info=[];groups=[np.ones(len(y),bool),(x[:,5]+x[:,6])>0]
    for k,((tr,te),(p,candidate)) in enumerate(zip(splits,fitted)):
        rep,fold=divmod(k,folds);oof[:,rep,te]=p
        assert not np.intersect1d(ids[tr],ids[te]).size
        folds_info.append(dict(repetition=rep,fold=fold,inner_seed=stable_seed(seed,'audit_inner',k),train_n=len(tr),test_n=len(te),candidate=candidate))
        for ix,subset in enumerate(groups):
            keep=subset[te];labels=y[te][keep];n_pairs=int(labels.sum()*(len(labels)-labels.sum()))
            if n_pairs:auc_sums[ix]+=n_pairs*evaluate.auc(p[:,keep],labels);pairs[ix]+=n_pairs
        for j,idx in enumerate(te):
            records.append(dict(study_id=int(ids[idx]),repetition=rep,fold=fold,y=int(y[idx]),dependent=bool(groups[1][idx]),**dict(zip(ARMS,p[:,j]))))
    assert np.isfinite(oof).all()
    loss=(oof-y)**2;rows=[]
    for ix,subset in enumerate(groups):
        for j,arm in enumerate(ARMS):
            rows.append(dict(population='All' if ix==0 else 'Six-month dependent',model=arm,n=int(subset.sum()),
                             Brier=float(loss[j][:,subset].mean()),AUROC=float(auc_sums[ix,j]/pairs[ix])))
    pd.DataFrame(rows).to_csv(out/'S22_independent_evaluation.csv',index=False)
    pd.DataFrame(records).to_csv(out/'controlled_audit_oof_predictions.csv',index=False)
    pd.DataFrame(folds_info).to_csv(out/'audit_folds.csv',index=False)
    gain=(loss[3]-loss[5]).mean(axis=0)
    indices=np.random.default_rng(20261116).integers(0,len(y),(draws,len(y)))
    boot=gain[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
    pd.DataFrame({'draw':np.arange(draws),'B_minus_HG_additive':boot}).to_csv(out/'audit_conditional_bootstrap.csv',index=False)
    metadata=dict(implementation='nested clinical comparator evaluation',outer_seed=seed,inner_seed_rule='stable_seed(seed, audit_inner, fold_index)',
                  bootstrap_seed=20261116,folds=folds,repeats=repeats,draws=draws,
                  conditional_B_minus_HG_additive=dict(point=float(gain.mean()),lower=float(lo),upper=float(hi)),
                  partitions_recorded=True)
    (out/'audit_run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return rows
