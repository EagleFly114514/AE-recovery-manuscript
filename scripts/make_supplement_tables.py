# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
import os
"""Create publication tables from completed analysis stages."""
from pathlib import Path
import json,sys,platform
import numpy as np,pandas as pd,scipy,sklearn,joblib
ROOT=Path(os.environ['AE_RUN_ROOT']).resolve();R=ROOT/'results';O=R/'publication_tables';O.mkdir(exist_ok=True)
CO={'development':'Development','temporal':'Temporal'}
KIND={'oob':'Out-of-bag refit','development_refit':'Development refit','patient':'Patient resampling','stratified':'Outcome-stratified'}
ARM={'H':'History-marginalised (H)','G':'Function (G)','HG':'History + function (HG)','B':'Age + function (B)'}
def read(n):return pd.read_csv(R/f'{n}.csv')
def f(x,d=5):return '—' if pd.isna(x) else f'{x:.{d}f}'
def interval(r,d=5):return '—' if pd.isna(r.lower) or pd.isna(r.upper) else f'{f(r.lower,d)} to {f(r.upper,d)}'
def save(n,frame):frame.to_csv(O/f'S{n}.csv',index=False,encoding='utf-8-sig')

def main():
    for stage in ['primary','precision','legacy']:
        if not (R/f'completed_{stage}.json').exists():raise RuntimeError(f'{stage} not complete')
    for stage in ['multiple','icu']:
        if not (ROOT/'sensitivity'/stage/f'completed_{stage}.json').exists():raise RuntimeError(f'{stage} not complete')
    for n in range(1,6):save(n,read(f'Supplementary_Table_{n}'))
    plan=json.loads((ROOT/'analysis_plan.json').read_text(encoding='utf-8'))
    definitions=[['Outcome','Twelve-month mRS 3–6 among six-month mRS 0–5 survivors'],['H','Shared conditional risk marginalised over history-predicted functional strata'],['G','Six-month mRS strata 0–2, 3 and 4–5; selected within development training'],['HG','G plus the frozen within-stratum historical update'],['B','Age plus mRS 3 and 4–5 indicators; additive ridge logistic regression; C = 1'],['G library','11 ridge C values: 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000; Jeffreys-smoothed; Firth intercept-corrected'],['G selection','Minimum Brier loss in 4-fold validation repeated twice; development training only'],['HG independent/severe strata','Eight soft percentile cells; smoothing 0.25; prior 512; temperature 0.5; joint binary clinical patterns; shared boundary-change evidence'],['HG moderate stratum','Additive local similarity; bandwidth 0.75; support count 64; temperature 0.25; leave-patient-out reference'],['Support shrinkage','n / (n + 32), using distinct development training patients in each stratum'],['History-to-function probabilities','Multinomial ridge logistic regression; precision 16 (C = 1/16)'],['Repeated evaluation','Five outer folds × 20 repetitions; patient-grouped duplicates in resampling'],['Primary uncertainty','1,000 development refits; 1,000 ordinary and 1,000 outcome-stratified temporal patient samples'],['Precision analysis','10,000 development refits; 20,000 temporal patient samples; separate fixed seeds'],['Validation role','Same-centre temporal scoring after freezing; no temporal feedback to selection'],['Missing-risk odds scenarios','0.25, 0.5, 1, 2 and 4; cohort/stratum risk shrunk by 2 pseudo-observations toward development G']]
    save(6,pd.DataFrame(definitions,columns=['Component','Specification']))
    pts=read('model_points');bs=read('bootstrap_summary')
    def metrics(target):
        result=[]
        for p in pts.loc[pts.target.eq(target)].itertuples():
            kind='oob' if p.cohort=='development' else 'patient';z=bs.loc[bs.target.eq(target)&bs.cohort.eq(p.cohort)&bs.kind.eq(kind)]
            b=z.loc[z.estimand.eq(p.model)].iloc[0];a=z.loc[z.estimand.eq('AUROC_'+p.model)].iloc[0]
            result.append([CO[p.cohort],p.model,f'{p.n} / {p.events}',f(p.Brier),interval(b),f(p.AUROC,3),interval(a,3),int(a.evaluable)])
        return pd.DataFrame(result,columns=['Cohort','Model','n / events','Brier','Brier interval','AUROC','AUROC interval','AUROC draws'])
    save(7,metrics('overall_dynamic'));save('9a',metrics('nmdar_dynamic'))
    def gains(target):
        result=[];selected=bs.loc[bs.target.eq(target)&bs.estimand.isin(['H_minus_HG','G_minus_HG','B_minus_HG'])]
        for r in selected.itertuples():
            p=pts.loc[pts.target.eq(target)&pts.cohort.eq(r.cohort)].set_index('model');delta=p.loc[r.estimand.split('_')[0],'Brier']-p.loc['HG','Brier']
            result.append([CO[r.cohort],KIND[r.kind],r.estimand.replace('_minus_',' − '),f(delta,7),interval(r,7),f(100*r.positive/r.draws,1)])
        return pd.DataFrame(result,columns=['Cohort','Uncertainty procedure','Comparison','Point gain','95% interval','Positive draws (%)'])
    save(8,gains('overall_dynamic'));save('9b',gains('nmdar_dynamic'))
    bounds=read('missing_outcome_sharp_bounds');data=[]
    for r in bounds.itertuples():data.append(['All' if r.target=='overall_dynamic' else 'NMDAR',CO[r.cohort],f'{r.eligible} / {r.observed} / {r.missing}',r.comparison.replace('_minus_',' − '),f(r.observed_delta,7),f'{f(r.lower,7)} to {f(r.upper,7)}','Yes' if r.all_completions_positive else 'No'])
    save('10a',pd.DataFrame(data,columns=['Set','Roster','Due / observed / missing','Comparison','Observed gain','Sharp bounds','All positive']))
    scenarios=read('missing_outcome_scenarios');selected=scenarios.loc[scenarios.target.eq('overall_dynamic')&scenarios.cohort.eq('development')];data=[]
    for r in selected.itertuples():data.append([CO[r.cohort],r.scenario.replace('_',' '),r.scored_n,f(r.adverse_rate,3),f(r.H_minus_HG,7),f(r.G_minus_HG,7)])
    save('10b',pd.DataFrame(data,columns=['Roster','Missing-outcome scenario','n','Adverse risk','H − HG','G − HG']))
    legacy=read('legacy_omissions');data=[]
    cohort=pd.read_csv(ROOT/'data/cohort.csv'); observed=cohort.loc[cohort.cohort.eq('development')&cohort.mrs6.between(0,5)&cohort.mrs12.notna()]
    for r in legacy.itertuples():data.append([r.omission,r.outcome,len(observed) if r.outcome=='Dependence or death' else int(observed.mrs6.between(3,5).sum()),f(r.H_brier),f(r.HG_brier),f(r.gain),interval(r)])
    save(11,pd.DataFrame(data,columns=['Omitted history item','Outcome','n','History Brier','History + function','Gain','95% interval']))
    high=read('precision_summary');primary=bs.loc[bs.target.eq('overall_dynamic')&bs.estimand.eq('G_minus_HG')&bs.kind.ne('stratified')]
    data=[]
    for r in primary.itertuples():
        q=high.loc[high.cohort.eq(r.cohort)&high.kind.eq(r.kind)].iloc[0]
        data.append([CO[r.cohort],KIND[r.kind],int(r.draws),f(r.lower*1e6,3),f(r.upper*1e6,3),int(q.draws),f(q.lower*1e6,3),f(q.upper*1e6,3)])
    save(12,pd.DataFrame(data,columns=['Cohort','Procedure','Primary draws','Lower ×10⁻⁶','Upper ×10⁻⁶','Precision draws','Lower ×10⁻⁶ ','Upper ×10⁻⁶ ']))
    data=[]
    for name,folder in [('Primary',R),('Episodes retained',ROOT/'sensitivity/multiple'),('ICU interval masked',ROOT/'sensitivity/icu')]:
        a=pd.read_csv(folder/'model_points.csv');b=pd.read_csv(folder/'bootstrap_summary.csv')
        for target in ['overall_dynamic','nmdar_dynamic']:
            for co in ['development','temporal']:
                p=a.loc[a.target.eq(target)&a.cohort.eq(co)].set_index('model');kind='oob' if co=='development' else 'patient'
                q=b.loc[b.target.eq(target)&b.cohort.eq(co)&b.kind.eq(kind)&b.estimand.eq('G_minus_HG')].iloc[0]
                data.append([name,'All' if target=='overall_dynamic' else 'NMDAR',CO[co],int(p.loc['HG','n']),f(p.loc['HG','Brier']),f(p.loc['H','Brier']-p.loc['HG','Brier'],7),f(p.loc['G','Brier']-p.loc['HG','Brier'],7),interval(q,7)])
    save(13,pd.DataFrame(data,columns=['Analysis','Set','Cohort','n','HG Brier','H − HG','G − HG','G − HG interval']))
    c=read('clinical_contrasts');data=[]
    for r in c.itertuples():data.append([r.population,r.month,f'{r.moderate_n} / {r.severe_n}',f(r.difference*100,1),f'{f(r.lower*100,1)} to {f(r.upper*100,1)}',f'{f(r.fixed_lower*100,1)} to {f(r.fixed_upper*100,1)}'])
    save('14a',pd.DataFrame(data,columns=['Clinical risk set','Month','mRS 3 / 4–5 n','Difference (pp)','95% CI (pp)','Missing-outcome bounds (pp)']))
    c=read('clinical_paired_sensitivity');c=c[['population','original_n','twelve_dependent','independent','dependent','death','missing','late_from_moderate','late_from_severe']]
    c.columns=['Risk set','6-mo n','12-month dependent','Later independent','Still dependent','Dead','Missing','From mRS 3','From mRS 4–5'];save('14b',c)
    flow=read('longitudinal_state_counts').pivot(index='timepoint',columns='state',values='n').reindex(['worst_inpatient','six_month','twelve_month','twentyfour_month'])[['I','D3','D45','X','M']].reset_index();flow.columns=['Timepoint','Independent','mRS 3','mRS 4–5','Death','Missing']
    flow['Timepoint']=flow.Timepoint.map({'worst_inpatient':'Worst inpatient','six_month':'Nominal 6 months','twelve_month':'Nominal 12 months','twentyfour_month':'Nominal 24 months'});save('14c',flow)
    asc=read('outcome_ascertainment');a=asc.loc[asc.target.eq('overall_dynamic')][['cohort','tier','eligible','observed','unavailable','adverse']].copy();a['cohort']=a.cohort.map(CO);a.columns=['Roster','Six-month tier','Eligible','Observed','Unavailable','Adverse'];save(15,a)
    env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__,joblib=joblib.__version__)
    (ROOT/'provenance/software_environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    print('Publication tables complete',len(list(O.glob('S*.csv'))))
if __name__=='__main__':main()
