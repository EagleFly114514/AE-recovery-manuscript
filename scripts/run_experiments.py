# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
import os
"""Run the fixed analysis specification on authorised runtime inputs."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
ROOT=Path(os.environ['AE_RUN_ROOT']).resolve()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLBACKEND']='Agg';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import pandas as pd
import evaluate, clinical

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['primary','precision','multiple','icu','clinical'],required=True);p.add_argument('--jobs',type=int,default=8);a=p.parse_args()
    plan=json.loads((ROOT/'analysis_plan.json').read_text(encoding='utf-8'))
    source={'multiple':'cohort_multiple_episode_sensitivity.csv','icu':'cohort_icu_sensitivity.csv'}.get(a.stage,'cohort.csv')
    frame=pd.read_csv(ROOT/'data'/source,dtype={'antibody_codes':str},float_precision='round_trip')
    where=ROOT/'results' if a.stage in ['primary','precision','clinical'] else ROOT/'sensitivity'/a.stage
    where.mkdir(parents=True,exist_ok=True);plan['frozen_output']=str(where/'frozen_models')
    start=time.time()
    if a.stage in ['primary','multiple','icu']:evaluate.run(frame,plan,where,a.jobs)
    elif a.stage=='precision':evaluate.run_precision(frame,plan,where,a.jobs)
    elif a.stage=='clinical':
        spec=json.loads((ROOT/'data/clinical_spec.json').read_text(encoding='utf-8'))
        clinical.run(frame,spec,pd.read_csv(ROOT/'data/feature_dictionary.csv'),where,longitudinal=pd.read_csv(ROOT/'data/longitudinal_transition_counts.csv'))
    record=dict(stage=a.stage,elapsed_seconds=time.time()-start,input_file=source,input_sha256=hashlib.sha256((ROOT/'data'/source).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((ROOT/'analysis_plan.json').read_bytes()).hexdigest(),completed=True)
    (where/f'completed_{a.stage}.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
