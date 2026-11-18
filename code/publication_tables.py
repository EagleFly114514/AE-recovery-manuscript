# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Format manuscript Tables 1-3 and post hoc Tables S16-S22."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def main_tables(frame,run):
    run=Path(run);out=run/'results/publication_tables';out.mkdir(exist_ok=True,parents=True)
    roster=pd.read_csv(run/'results/clinical_rosters.csv').set_index('cohort')
    screening=json.loads((run/'data/clinical_spec.json').read_text(encoding='utf-8'))['screening']
    columns=['development','temporal']
    fields=[('Eligible treatment records','treatment_cohort_n'),('Eligible living six-month assessment','six_month_alive'),
            ('Death by six months','six_month_death'),('Six-month score missing','six_month_missing'),
            ('Twelve-month target due','due_twelve'),('Observed twelve-month outcome','observed_twelve'),
            ('Unavailable twelve-month outcome','missing_twelve'),('Six-month dependence in due roster','six_month_dependent'),
            ('Dependent patients with observed outcome','observed_dependent'),('Observed dependence/death at twelve months','events')]
    rows=[]
    for label,key in fields:
        values=[]
        for co in columns:
            val=roster.loc[co,key]
            values.append('—' if pd.isna(val) else str(int(val)))
        rows.append([label,*values])
    pd.DataFrame(rows,columns=['Population or outcome','Development','Temporal']).to_csv(out/'Table_1.csv',index=False)
    four=pd.read_csv(run/'results/clinical_four_state.csv');rows=[]
    for r in four.loc[four.population.eq('Primary')].itertuples():
        rows.append([r.month,f'{r.stratum.replace("-","–")} (n={r.n})',*[f'{getattr(r,k)} ({100*getattr(r,k)/r.n:.1f}%)' for k in ['independent','dependent','death','missing']],
                     f'{r.independent}/{r.observed}\n{100*r.p:.1f}% ({100*r.lower:.1f}–{100*r.upper:.1f})'])
    pd.DataFrame(rows,columns=['Month','Six-month mRS','Independent','Dependent','Death','Missing','Observed independence\n(95% CI)']).to_csv(out/'Table_2.csv',index=False)
    f=frame.loc[frame.cohort.eq('development')&frame.mrs6.between(3,5)]
    def state(v):return np.select([v.le(2),v.between(3,5),v.eq(6)],['Independent','Dependent','Death'],default='Missing')
    order=['Independent','Dependent','Death','Missing'];matrix=pd.crosstab(state(f.mrs12),state(f.mrs24)).reindex(index=order,columns=order,fill_value=0)
    matrix['Total']=matrix.sum(axis=1);matrix.loc['Total']=matrix.sum(axis=0)
    matrix.index.name='12-month state\n→24-month state';matrix.to_csv(out/'Table_3.csv')
    late=f.loc[f.mrs12.between(3,5)];pairs=[]
    for label,subset in [('Six-month mRS 3',late.loc[late.mrs6.eq(3)]),('Six-month mRS 4–5',late.loc[late.mrs6.ge(4)]),('All',late)]:
        v=subset.mrs24;pairs.append([label,len(subset),int(v.le(2).sum()),int(v.between(3,5).sum()),int(v.eq(6).sum()),int(v.isna().sum())])
    (run/'figures').mkdir(exist_ok=True)
    pd.DataFrame(pairs,columns=['prior','n','independent','dependent','death','missing']).to_csv(run/'figures/source_paired_by_prior_severity.csv',index=False)


def posthoc_tables(run):
    run=Path(run);src=run/'results/posthoc';out=run/'results/publication_tables';out.mkdir(exist_ok=True,parents=True)
    a=pd.read_csv(src/'S16_adjusted.csv');rows=[]
    for r in a.itertuples():
        rd=f'{100*r.rd:.1f}'+(f' ({100*r.rd_lower:.1f}–{100*r.rd_upper:.1f})' if pd.notna(r.rd_lower) else '')
        rows.append([r.month,r.model,r.n,f'{r.odds_ratio:.2f} ({r.or_lower:.2f}–{r.or_upper:.2f})',rd])
    pd.DataFrame(rows,columns=['Month','Model','Known n','Moderate vs severe OR (95% CI)','Standardised RD, pp (95% CI)']).to_csv(out/'S16.csv',index=False)
    a=pd.read_csv(src/'S17_selection.csv');a['smd']=a.smd.map(lambda v:f'{v:.2f}')
    labels={'age':'Age, years','female':'Female sex','nmdar':'NMDAR antibody','lgi1':'LGI1 antibody',
        'neoplasm':'Recorded neoplasm','treatment_delay_days':'Onset-to-treatment interval, days',
        'worst_mrs':'Worst inpatient mRS','icu_admission':'Recorded ICU admission',
        'respiratory_failure':'Respiratory failure','tx_steroid':'Corticosteroids',
        'tx_ivig':'Intravenous immunoglobulin','tx_plasma_exchange':'Plasma exchange',
        'tx_immunosuppressant':'Immunosuppressant'}
    a['characteristic']=a.characteristic.map(lambda value:labels.get(value,value))
    a=a[['characteristic','observed_living','unobserved','dead','smd']]
    a.columns=['Characteristic','Observed living','State unobserved','Dead','SMD living − unobserved'];a.to_csv(out/'S17.csv',index=False)
    a=pd.read_csv(src/'S18_weighting.csv')[['month','weight_rule','moderate','severe','difference']]
    a['weight_rule']=a.weight_rule.map({'untruncated':'Uncapped','p95_truncated':'Capped at 95th percentile'})
    for k in ['moderate','severe','difference']:a[k]=a[k].map(lambda v:f'{100*v:.1f}')
    a.columns=['Month','Weight rule','Moderate independent (%)','Severe independent (%)','Difference (pp)'];a.to_csv(out/'S18.csv',index=False)
    a=pd.read_csv(src/'S19_late_transitions.csv');a=a[['audit_label','mrs6','mrs12','mrs24','improvement']]
    for c in a.columns[1:]:a[c]=a[c].astype(int)
    a.columns=['Participant label','6-month mRS','12-month mRS','24-month mRS','12-to-24 improvement'];a.to_csv(out/'S19.csv',index=False)
    a=pd.read_csv(src/'S20_calibration.csv');rows=[]
    for r in a.itertuples():rows.append([r.cohort,r.model,f'{r.n}/{r.events}',f'{r.intercept:.3f}', 'Not estimable' if pd.isna(r.slope) else f'{r.slope:.3f} ({r.lower:.3f}–{r.upper:.3f})'])
    pd.DataFrame(rows,columns=['Cohort','Model','n/events','Calibration intercept','Slope (conditional 95% CI)']).to_csv(out/'S20.csv',index=False)
    a=pd.read_csv(src/'S21_saved_score_metrics.csv');a=a.loc[a.model.ne('H'),['cohort','model','n','Brier','AUROC']]
    a['Brier']=a.Brier.map(lambda v:f'{v:.5f}');a['AUROC']=a.AUROC.map(lambda v:f'{v:.3f}');a.columns=['Cohort','Model','n','Brier','AUROC'];a.to_csv(out/'S21.csv',index=False)
    a=pd.read_csv(src/'S22_independent_evaluation.csv');rows=[]
    for model in a.model.unique():
        one=a.loc[a.model.eq(model)].set_index('population');b=one.loc['All'];d=one.loc['Six-month dependent']
        label={'H_direct':'Direct history','HG_additive':'Additive history + function'}.get(model,model)
        rows.append([label,f'{b.Brier:.5f}',f'{b.AUROC:.3f}',f'{d.Brier:.5f}',f'{d.AUROC:.3f}'])
    pd.DataFrame(rows,columns=['Model','All Brier','All AUROC','Dependent Brier','Dependent AUROC']).to_csv(out/'S22.csv',index=False)
