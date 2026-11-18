# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
import os
"""Nested historical comparisons with input-identity checks."""
from pathlib import Path
import argparse,os,sys,json,time,hashlib
ROOT=Path(os.environ['AE_RUN_ROOT']).resolve()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np,pandas as pd
from joblib import Parallel,delayed
import legacy_omissions as legacy
from functional_baseline import stable_seed

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=8);p.add_argument('--benchmark',action='store_true');p.add_argument('--point-only',action='store_true');p.add_argument('--engine',choices=['reference','accelerated'],default='reference');a=p.parse_args()
    f=pd.read_csv(ROOT/'data/cohort.csv');spec,frames=legacy.load_data(f,ROOT/'data');out=ROOT/'results'
    engine=legacy
    if a.engine=='accelerated':
        parity=json.loads((ROOT/'preparation/legacy_execution_parity.json').read_text())
        assert parity['passed'], 'Numerical equivalence must be checked before using acceleration'
        for name,h in parity['checked_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
        import legacy_accelerated as engine
    started=time.time()
    if a.benchmark:
        frame,y=frames['dynamic'];value=legacy.bootstrap_draw(frame,y,'dynamic',0,spec)
        (ROOT/'preparation/legacy_benchmark.json').write_text(json.dumps(dict(seconds=time.time()-started,gain=float(value)),indent=2));print('one full nested draw',time.time()-started,value,flush=True);return
    rows=[]
    cache=out/'legacy_refits';cache.mkdir(exist_ok=True)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['data/cohort.csv','data/legacy_history.csv','data/legacy_spec.json','code/legacy_omissions.py','code/legacy_validation.py']}
    identity=cache/'input_identity.json'
    if identity.exists():assert json.loads(identity.read_text())==hashes,'Saved draws belong to different inputs or estimator'
    else:identity.write_text(json.dumps(hashes,indent=2))
    for family,(frame,y) in frames.items():
        with Parallel(n_jobs=a.jobs,verbose=5) as pool:
            seed=stable_seed(spec['validation']['seed'],f'targeted_v5_{family}_point')
            point=legacy.scores(frame,y,family,seed,list(legacy.OMISSIONS),spec,pool)
            pd.DataFrame([dict(family=family,index=i,brier=v) for i,v in enumerate(point)]).to_csv(cache/f'{family}_points.csv',index=False)
            draws=[]
            for start in range(0,0 if a.point_only else spec['bootstrap_draws'],100):
                path=cache/f'{family}_{start:04d}.csv'
                if path.exists():
                    loaded=pd.read_csv(path,float_precision='round_trip');assert loaded['index'].tolist()==list(range(start,min(start+100,spec['bootstrap_draws'])))
                    part=loaded['gain'].tolist();assert np.isfinite(part).all()
                else:
                    part=pool(delayed(engine.bootstrap_draw)(frame,y,family,i,spec) for i in range(start,min(start+100,spec['bootstrap_draws'])))
                    pd.DataFrame(dict(index=range(start,start+len(part)),gain=part)).to_csv(path,index=False)
                draws.extend(part);print(f'Legacy {family}: {len(draws)}/{spec["bootstrap_draws"]}',flush=True)
                (out/'legacy_progress.json').write_text(json.dumps(dict(family=family,completed_draws=len(draws),target_draws=spec['bootstrap_draws'],engine=a.engine,elapsed_seconds=time.time()-started),indent=2))
        lo,hi=np.quantile(draws,[.025,.975]) if draws else (None,None)
        for i,omission in enumerate(legacy.OMISSIONS):
            h,hg=point[2*i:2*i+2]
            rows.append(dict(omission=omission,outcome='Dependence or death' if family=='dynamic' else 'Recovery',H_brier=h,HG_brier=hg,gain=h-hg,lower=lo if i==0 else None,upper=hi if i==0 else None))
        pd.DataFrame(rows).to_csv(out/'legacy_omissions.csv',index=False)
    (out/('completed_legacy_points.json' if a.point_only else 'completed_legacy.json')).write_text(json.dumps(dict(completed=True,elapsed_seconds=time.time()-started,draws_per_endpoint=0 if a.point_only else spec['bootstrap_draws'],intervals_recomputed=not a.point_only,engine=a.engine,input_hashes=hashes),indent=2));print('Legacy point estimates complete' if a.point_only else 'Legacy complete',time.time()-started,flush=True)

if __name__=='__main__':main()
