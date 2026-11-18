# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
import os
"""Fixed-risk-set analyses supporting the clinical manuscript, not model selection."""
from pathlib import Path
import json,sys
import pandas as pd,numpy as np
ROOT=Path(os.environ['AE_RUN_ROOT']).resolve();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from evaluate import wilson

def four(s):
    return dict(independent=int(s.le(2).sum()),dependent=int(s.between(3,5).sum()),death=int(s.eq(6).sum()),missing=int(s.isna().sum()))

def summaries(f,label):
    f=f.loc[f.cohort.eq('development')&f.mrs6.between(3,5)]
    rows=[];contrasts=[]
    for month in [12,24]:
        for group,mask in [('3',f.mrs6.eq(3)),('4-5',f.mrs6.ge(4)),('3-5',f.mrs6.ge(3))]:
            x=f.loc[mask,f'mrs{month}'];n=len(x);c=four(x);obs=n-c['missing'];p=c['independent']/obs
            lo,hi=wilson(c['independent'],obs)
            rows.append(dict(population=label,month=month,stratum=group,n=n,observed=obs,**c,p=p,lower=lo,upper=hi,
                fixed_lower=c['independent']/n,fixed_upper=(c['independent']+c['missing'])/n))
        a,b=rows[-3:-1];d=a['p']-b['p']
        contrasts.append(dict(population=label,month=month,n=len(f),moderate_n=a['n'],severe_n=b['n'],difference=d,
            lower=d-np.hypot(a['p']-a['lower'],b['upper']-b['p']),upper=d+np.hypot(a['upper']-a['p'],b['p']-b['lower']),
            fixed_lower=a['fixed_lower']-b['fixed_upper'],fixed_upper=a['fixed_upper']-b['fixed_lower']))
    dep=f.loc[f.mrs12.between(3,5)]
    paired=dict(population=label,original_n=len(f),twelve_dependent=len(dep),**four(dep.mrs24),
        late_from_moderate=int((dep.mrs6.eq(3)&dep.mrs24.le(2)).sum()),late_from_severe=int((dep.mrs6.ge(4)&dep.mrs24.le(2)).sum()))
    return rows,contrasts,paired

def main():
    f=pd.read_csv(ROOT/'data/cohort.csv',dtype={'antibody_codes':str});multi=pd.read_csv(ROOT/'data/cohort_multiple_episode_sensitivity.csv',dtype={'antibody_codes':str})
    lin=pd.read_csv(ROOT/'preparation/record_lineage.csv') if (ROOT/'preparation/record_lineage.csv').exists() else None
    # The two registry-based subsets are diagnostic-set sensitivity analyses, not adjudicated diagnoses.
    antibodies=f.antibody_codes.fillna('').map(lambda s:{int(v) for v in s.split('|') if v})
    surface=antibodies.map(lambda a:bool(a&{1,2,3,5,7}))
    cases=[('Primary',f),('Multiple episodes retained',multi),('NMDAR positive',f.loc[f.nmdar.eq(1)]),
        ('Neuronal-surface antibody subset',f.loc[surface])]
    rows=[];contrasts=[];pairs=[]
    for label,frame in cases:
        a,b,c=summaries(frame,label);rows.extend(a);contrasts.extend(b);pairs.append(c)
    pd.DataFrame(rows).to_csv(ROOT/'results/clinical_four_state.csv',index=False)
    pd.DataFrame(contrasts).to_csv(ROOT/'results/clinical_contrasts.csv',index=False)
    pd.DataFrame(pairs).to_csv(ROOT/'results/clinical_paired_sensitivity.csv',index=False)
    # Complete cohort counts preserve each distinct target population.
    roster=[]
    for co in ['development','temporal']:
        x=f.loc[f.cohort.eq(co)];alive=x.loc[x.mrs6.between(0,5)]
        due=alive
        obs=due.loc[due.mrs12.notna()]
        roster.append(dict(cohort=co,treatment_cohort_n=len(x),dated_six_month_n=None,
            six_month_alive=len(alive),six_month_death=int(x.mrs6.eq(6).sum()),
            six_month_missing=int(x.mrs6.isna().sum()),due_twelve=len(due),observed_twelve=len(obs),
            missing_twelve=int(due.mrs12.isna().sum()),six_month_dependent=int(due.mrs6.between(3,5).sum()),
            observed_dependent=int(obs.mrs6.between(3,5).sum()),events=int(obs.mrs12.ge(3).sum()),
            age_median=obs.age.median(),age_q25=obs.age.quantile(.25),age_q75=obs.age.quantile(.75),female=int(obs.female.eq(1).sum()),nmdar=int(obs.nmdar.eq(1).sum())))
    pd.DataFrame(roster).to_csv(ROOT/'results/clinical_rosters.csv',index=False)
    dev=f.loc[f.cohort.eq('development')];late=dev.loc[dev.mrs6.between(3,5)&dev.mrs12.between(3,5)&dev.mrs24.le(2)]
    late[['study_id','mrs6','mrs12','mrs24']].to_csv(ROOT/'provenance/confirmed_late_transitions.csv',index=False)
    assert late.study_id.is_unique
    assert not (f.mrs6.eq(6)&f.mrs12.between(0,5)).any()
    assert not (f.mrs12.eq(6)&f.mrs24.between(0,5)).any()
    # Source-record linkage, when supplied, is managed separately under controlled access.
    print(pd.DataFrame(contrasts).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))

if __name__=='__main__':main()
