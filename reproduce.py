# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Run functional recovery analyses using authorised inputs."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLBACKEND']='Agg';sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code'))

def script(name,run,*args):
    env=os.environ.copy();env['AE_RUN_ROOT']=str(run)
    subprocess.run([sys.executable,str(ROOT/'scripts'/name),*map(str,args)],env=env,check=True)

def prepare(run,inputs):
    source=Path(inputs).resolve()
    if source==ROOT or source.is_relative_to(ROOT):
        raise ValueError('Authorised inputs must be outside the software repository.')
    if source==run or source.is_relative_to(run) or run.is_relative_to(source):
        raise ValueError('Input and output directories must be separate.')
    for rel in ['data','code','results','results/publication_tables','results/posthoc','preparation','provenance','qa','figures','sensitivity']:
        (run/rel).mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'code').glob('*.py'):
        shutil.copy2(path,run/'code'/path.name)
    shutil.copy2(ROOT/'configs/analysis_plan.json',run/'analysis_plan.json')
    shutil.copy2(ROOT/'configs/legacy_spec.json',run/'data/legacy_spec.json')
    import pandas as pd
    dictionary=json.loads((ROOT/'configs/feature_dictionary.json').read_text(encoding='utf-8'))
    pd.DataFrame(dictionary).to_csv(run/'data/feature_dictionary.csv',index=False)
    manifest=json.loads((source/'input_manifest.json').read_text(encoding='utf-8'))
    expected={row['file']:row['sha256'] for row in manifest['files']}
    required=['cohort.csv','cohort_multiple_episode_sensitivity.csv','cohort_icu_sensitivity.csv','legacy_history.csv','clinical_spec.json']
    optional=['multicentre_collection.json']
    selected=required+[name for name in optional if (source/name).is_file()]
    for name in selected:
        path=source/name
        if not path.is_file():raise FileNotFoundError(f'Required authorised input: {name}')
        if name not in expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[name]:
            raise ValueError(f'Input manifest mismatch: {name}')
    # Synchronise optional inputs when reusing an output directory.
    for name in optional:
        if name not in selected:
            (run/'data'/name).unlink(missing_ok=True)
    for name in selected:
        shutil.copy2(source/name,run/'data'/name)
    clinical_spec=json.loads((run/'data/clinical_spec.json').read_text(encoding='utf-8'))
    if 'antibody_codes' not in clinical_spec:
        clinical_spec['antibody_codes']=json.loads((ROOT/'configs/antibody_codes.json').read_text(encoding='utf-8'))
    (run/'data/clinical_spec.json').write_text(json.dumps(clinical_spec,ensure_ascii=False,indent=2),encoding='utf-8')


def clinical_stage(run):
    import pandas as pd,numpy as np,clinical
    import publication_tables
    f=pd.read_csv(run/'data/cohort.csv',dtype={'antibody_codes':str},float_precision='round_trip')
    dev=f.loc[f.cohort.eq('development')].copy()
    def st(s):return np.select([s.le(2),s.eq(3),s.between(4,5),s.eq(6)],['I','D3','D45','X'],default='M')
    states=[st(dev.worst_mrs),st(dev.mrs6),st(dev.mrs12),st(dev.mrs24)]
    # Known death is absorbing; missing living assessments stay unobserved.
    for i in range(1,len(states)):states[i][states[i-1]=='X']='X'
    order=['I','D3','D45','X','M'];rows=[]
    for interval,left,right in zip(['worst_to_six','six_to_twelve','twelve_to_twentyfour'],states[:-1],states[1:]):
        for a in order:
            for b in order:rows.append(dict(interval=interval,from_state=a,to_state=b,n=int(((left==a)&(right==b)).sum())))
    flow=pd.DataFrame(rows);flow.to_csv(run/'data/longitudinal_transition_counts.csv',index=False)
    clinical.run(f,json.loads((run/'data/clinical_spec.json').read_text(encoding='utf-8')),pd.read_csv(run/'data/feature_dictionary.csv'),run/'results',flow)
    script('clinical_recovery.py',run)
    publication_tables.main_tables(f,run)
    (run/'results/completed_clinical.json').write_text(json.dumps({'completed':True,'source':'patient-row recomputation'}))

def posthoc_stage(run,prediction_dir,jobs):
    import pandas as pd,posthoc_clinical,posthoc_prediction,publication_tables
    f=pd.read_csv(run/'data/cohort.csv',dtype={'antibody_codes':str},float_precision='round_trip')
    out=run/'results/posthoc'
    posthoc_clinical.run(f,out,draws=2000,seed=20261115,jobs=jobs)
    source=Path(prediction_dir) if prediction_dir else run/'results/frozen_models'
    predictions=pd.read_csv(source/'overall_dynamic_predictions.csv',float_precision='round_trip')
    posthoc_prediction.diagnostics(f,predictions,out)
    plan=json.loads((run/'analysis_plan.json').read_text(encoding='utf-8'))
    posthoc_prediction.nested(f,plan,out,jobs=jobs)
    publication_tables.posthoc_tables(run)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['clinical','primary','precision','multiple','icu','legacy','posthoc','tables','figures','all','smoke','check-code'],required=True)
    p.add_argument('--inputs',type=Path,help='Authorised directory holding the four analysis CSVs')
    p.add_argument('--saved-predictions',type=Path,help='Directory of original frozen-model prediction CSVs for S20-S21')
    p.add_argument('--out',type=Path);p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    if a.stage=='check-code':
        subprocess.run([sys.executable,str(ROOT/'tests/test_public_code.py')],check=True)
        return
    if a.out is None or a.inputs is None:
        p.error('--inputs and --out are required for numerical stages')
    run=a.out.resolve()
    if run==ROOT or ROOT.is_relative_to(run) or run.is_relative_to(ROOT):
        p.error('Choose an output directory outside the software repository')
    prepare(run,a.inputs)
    started=time.time();stages=['clinical','primary','precision','multiple','icu','legacy','posthoc','tables','figures'] if a.stage=='all' else [a.stage]
    for stage in stages:
        print('Running',stage,flush=True)
        if stage=='clinical':clinical_stage(run)
        elif stage in ['primary','precision','multiple','icu']:script('run_experiments.py',run,'--stage',stage,'--jobs',a.jobs)
        elif stage=='legacy':
            script('check_legacy_execution.py',run)
            script('run_legacy.py',run,'--jobs',a.jobs,'--engine','accelerated')
        elif stage=='posthoc':posthoc_stage(run,a.saved_predictions,a.jobs)
        elif stage=='tables':
            script('make_supplement_tables.py',run)
            import pandas as pd,publication_tables
            publication_tables.main_tables(pd.read_csv(run/'data/cohort.csv'),run);publication_tables.posthoc_tables(run)
        elif stage=='figures':script('make_figures.py',run)
        elif stage=='smoke':
            clinical_stage(run)
            import pandas as pd,evaluate
            plan=json.loads((run/'analysis_plan.json').read_text(encoding='utf-8'))
            plan['primary']['development_refits']=2;plan['primary']['validation_patient_draws']=4
            plan['frozen_output']=str(run/'results/frozen_models')
            evaluate.run(pd.read_csv(run/'data/cohort.csv'),plan,run/'results',a.jobs)
            (run/'qa/smoke_plan.json').write_text(json.dumps(plan,indent=2))
            posthoc_stage(run,a.saved_predictions or run/'results/frozen_models',a.jobs)
            script('make_figures.py',run)
    summary={'stage':a.stage,'elapsed_seconds':time.time()-started,
             'smoke_is_full_primary_bootstrap':False if a.stage=='smoke' else None,'completed':True}
    (run/'qa'/f'completed_{a.stage}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
