# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Clinical sensitivity analyses and linked functional transitions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit
from scipy.stats import chi2
from joblib import Parallel, delayed
from functional_baseline import Firth
from evaluate import wilson


def design(frame, fields):
    z = {'moderate': frame.mrs6.eq(3).astype(float).to_numpy(),
         'age': (frame.age.to_numpy(float)-frame.age.mean())/10,
         'neoplasm': frame.neoplasm.to_numpy(float),
         'female': frame.female.to_numpy(float),
         'nmdar': frame.nmdar.to_numpy(float),
         'delay': np.log1p(frame.treatment_delay_days.to_numpy(float)),
         'worst': frame.worst_mrs.to_numpy(float)}
    x=np.column_stack([z[k] for k in fields])
    if not np.isfinite(x).all():
        raise ValueError('Post hoc clinical design contains missing covariates; specify their handling explicitly.')
    return x


def penalised(beta, x, y):
    eta=x@beta;p=expit(eta);w=np.maximum(p*(1-p),1e-12)
    info=x.T@(w[:,None]*x)
    sign, ld=np.linalg.slogdet(info)
    if sign<=0:return np.inf,np.zeros(len(beta))
    inv=np.linalg.solve(info,x.T)
    h=w*np.einsum('ij,ji->i',x,inv)
    val=np.sum(y*eta-np.logaddexp(0,eta))+.5*ld
    score=x.T@(y-p+h*(.5-p))
    return -val,-score


def profile_interval(model,z,y,column=1):
    x=np.column_stack([np.ones(len(z)),z])[:,model.keep]
    j=list(model.keep).index(column);b=model.beta.copy()
    best=penalised(b,x,y)[0];free=np.arange(len(b))!=j
    cutoff=chi2.ppf(.95,1)
    def objective(value):
        def fn(nuisance):
            bb=b.copy();bb[j]=value;bb[free]=nuisance
            val,score=penalised(bb,x,y)
            return val,score[free]
        opt=minimize(fn,b[free],jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':2000})
        if np.max(np.abs(fn(opt.x)[1]))>1e-5:raise RuntimeError('Profile nuisance fit did not converge')
        return 2*(opt.fun-best)-cutoff
    ends=[]
    for sign in [-1,1]:
        step=.5
        for _ in range(30):
            edge=b[j]+sign*step
            if objective(edge)>0:break
            step*=1.6
        else:raise RuntimeError('Profile interval could not be bracketed')
        ends.append(brentq(objective,min(b[j],edge),max(b[j],edge),xtol=1e-7))
    return np.exp(ends)


def standardised(model,z):
    moderate=z.copy();severe=z.copy();moderate[:,0]=1;severe[:,0]=0
    return float(np.mean(model.predict(moderate)-model.predict(severe)))


def boot_one(z,y,ix):
    zb=z[ix];yb=y[ix]
    model=Firth().fit(zb,yb,correct=False)
    return standardised(model,zb),model.score_norm


def risk_difference(a,na,b,nb):
    pa,pb=a/na,b/nb;la,ua=wilson(a,na);lb,ub=wilson(b,nb)
    d=pa-pb
    return d,d-np.hypot(pa-la,ub-pb),d+np.hypot(ua-pa,pb-lb)


def run(frame, out, draws=2000, seed=20261115, jobs=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dev=frame.loc[frame.cohort.eq('development')].copy()
    risk=dev.loc[dev.mrs6.between(3,5)].copy()
    assert len(risk)>0 and risk.study_id.is_unique
    specs=[('Unadjusted',['moderate']),('Age only',['moderate','age']),
           ('Age + recorded neoplasm',['moderate','age','neoplasm']),
           ('+ sex, NMDAR, delay',['moderate','age','neoplasm','female','nmdar','delay']),
           ('+ worst inpatient mRS',['moderate','age','neoplasm','female','nmdar','delay','worst'])]
    rows=[];draw_rows=[];ipw=[];extremes=[];solver=[]
    # A single declared stream is consumed in endpoint order (12, then 24).
    # Exact index arrays are exported so this convention is explicit.
    rng=np.random.default_rng(seed)
    for month in [12,24]:
        observed=risk.loc[risk[f'mrs{month}'].notna()].copy()
        y=observed[f'mrs{month}'].le(2).to_numpy(int)
        for label, fields in specs:
            z=design(observed,fields);model=Firth().fit(z,y,correct=False)
            lo,hi=profile_interval(model,z,y)
            rd=standardised(model,z);rdlo=rdhi=np.nan
            if label=='Age + recorded neoplasm' and draws:
                indices=rng.integers(0,len(y),(draws,len(y)))
                vals=Parallel(n_jobs=jobs)(delayed(boot_one)(z,y,ix) for ix in indices)
                rdlo,rdhi=np.quantile([v[0] for v in vals],[.025,.975])
                draw_rows.extend(dict(month=month,draw=i,rd=v[0],score_norm=v[1]) for i,v in enumerate(vals))
                np.savez_compressed(out/f'clinical_bootstrap_indices_{month}.npz',indices=indices)
            rows.append(dict(month=month,model=label,n=len(y),odds_ratio=float(np.exp(model.beta[1])),
                             or_lower=lo,or_upper=hi,rd=rd,rd_lower=rdlo,rd_upper=rdhi))
            solver.append(dict(month=month,model=label,score_norm=model.score_norm))
        z=design(risk,['moderate','age','neoplasm']);seen=risk[f'mrs{month}'].notna().to_numpy(int)
        availability=Firth().fit(z,seen,correct=False).predict(z)
        weight=1/availability[seen==1]
        for rule in ['untruncated','p95_truncated']:
            w=weight.copy()
            if rule=='p95_truncated':w=np.minimum(w,np.quantile(w,.95))
            rates=[];ess=[]
            for moderate in [True,False]:
                ix=observed.mrs6.eq(3).to_numpy()==moderate
                rates.append(float(np.average(y[ix],weights=w[ix])))
                ess.append(float(w[ix].sum()**2/(w[ix]@w[ix])))
            ipw.append(dict(month=month,weight_rule=rule,moderate=rates[0],severe=rates[1],difference=rates[0]-rates[1],ess_moderate=ess[0],ess_severe=ess[1]))
        counts=[]
        for mask in [risk.mrs6.eq(3),risk.mrs6.ge(4)]:
            s=risk.loc[mask,f'mrs{month}'];counts.append((int(s.le(2).sum()),int(s.isna().sum()),len(s)))
        (a,am,na),(b,bm,nb)=counts
        for scenario,aa,bb in [('least_favourable',a,b+bm),('most_favourable',a+am,b)]:
            d,lo,hi=risk_difference(aa,na,bb,nb)
            extremes.append(dict(month=month,scenario=scenario,moderate_independent=aa,moderate_n=na,severe_independent=bb,severe_n=nb,difference=d,lower=lo,upper=hi))
    selection=[]
    groups={'observed_living':dev.loc[dev.mrs6.between(0,5)],'unobserved':dev.loc[dev.mrs6.isna()],'dead':dev.loc[dev.mrs6.eq(6)]}
    fields=['age','female','nmdar','lgi1','neoplasm','treatment_delay_days','worst_mrs','icu_admission','respiratory_failure','tx_steroid','tx_ivig','tx_plasma_exchange','tx_immunosuppressant']
    for key in fields:
        a=groups['observed_living'][key].dropna();b=groups['unobserved'][key].dropna();binary=key not in ['age','treatment_delay_days','worst_mrs']
        denominator=np.sqrt((a.mean()*(1-a.mean())+b.mean()*(1-b.mean()))/2) if binary else np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        row=dict(characteristic=key,smd=(a.mean()-b.mean())/denominator if denominator else np.nan)
        for name,g in groups.items():
            v=g[key].dropna()
            row[name]=f'{int(v.sum())}/{len(v)} ({100*v.mean():.1f}%)' if binary else f'{v.median():.1f} ({v.quantile(.25):.1f}-{v.quantile(.75):.1f})'
        selection.append(row)
    late=risk.loc[risk.mrs12.between(3,5)&risk.mrs24.le(2),['study_id','mrs6','mrs12','mrs24']].copy()
    late['audit_label']=[f'LR{i+1:02}' for i in range(len(late))];late['improvement']=late.mrs12-late.mrs24
    assert late.study_id.is_unique and late.improvement.ge(1).all()
    for name,values in [('S16_adjusted',rows),('S17_selection',selection),('S18_weighting',ipw),('extreme_completions',extremes),('clinical_bootstrap_draws',draw_rows),('clinical_solver_checks',solver)]:
        pd.DataFrame(values).to_csv(out/f'{name}.csv',index=False)
    late.drop(columns='study_id').to_csv(out/'S19_late_transitions.csv',index=False)
    late.to_csv(out/'controlled_late_transition_mapping.csv',index=False)
    meta=dict(implementation='clinical sensitivity analyses',n=len(risk),draws_per_endpoint=draws,seed=seed,
              seed_convention='one default_rng(seed), consumed for month 12 then month 24',success=True)
    (out/'clinical_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return rows
