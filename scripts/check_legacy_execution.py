# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
import os
"""Validate execution-only acceleration against the unchanged estimator."""
from pathlib import Path
import os,sys,json,time,hashlib
ROOT=Path(os.environ['AE_RUN_ROOT']).resolve()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np,pandas as pd
from joblib import Parallel,delayed
import legacy_omissions as ref
import legacy_accelerated as fast
from functional_baseline import stable_seed

def comparison(f,y,family,index,spec):
    seed=stable_seed(spec['validation']['seed'],'targeted_v5_bootstrap_outcome',family,index)
    choice=np.random.default_rng(stable_seed(spec['validation']['seed'],'targeted_v5_bootstrap_sample',family,index)).choice(len(f),len(f),replace=True)
    f=f.iloc[choice].reset_index(drop=True);y=y[choice];g=f.study_id.to_numpy(int)
    a,ha=ref.make_splits(y,g,folds=5,repeats=20,seed=seed)
    b,hb=fast.make_splits(y,g,folds=5,repeats=20,seed=seed)
    assert ha==hb
    for x,z in zip(a,b):
        for xx,zz in zip(x,z):np.testing.assert_array_equal(xx,zz)
    h=[c for c in spec['history_columns'] if c!=ref.OMISSIONS['SE timing proxy']]
    names=('HexpandedNoSE12','HexpandedNoSEG12') if family=='dynamic' else ('R_HexpandedNoSE','R_HexpandedNoSEG')
    schemas={names[0]:h,names[1]:[*h,*spec['state_columns'][family]]}
    maximum=0.;checked_inner=0
    for i in [0,19,43,72,99]:
        tr,te=a[i];innerseed=stable_seed(seed,'outer_inner',i)
        s1,dh1=ref.make_splits(y[tr],g[tr],folds=4,repeats=1,seed=innerseed)
        s2,dh2=fast.make_splits(y[tr],g[tr],folds=4,repeats=1,seed=innerseed)
        assert dh1==dh2;checked_inner+=len(s1)
        u,v=ref.feature_pair(f.iloc[tr],f.iloc[te],spec)
        x,z=fast.feature_pair(f.iloc[tr],f.iloc[te],spec)
        np.testing.assert_allclose(u,x,rtol=0,atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(v,z,rtol=0,atol=1e-12,equal_nan=True)
        for name,columns in schemas.items():
            prep=ref.make_analysis_ridge_pipeline(columns,seed=1,no_indicator=set(spec['no_indicator'])).named_steps['preprocess']
            p,q=prep.fit_transform(u),prep.transform(v)
            pp,qq=fast.preprocess(x,z,columns,set(spec['no_indicator']))
            np.testing.assert_allclose(p,pp,rtol=0,atol=1e-12)
            np.testing.assert_allclose(q,qq,rtol=0,atol=1e-12)
        _,p=ref.outer_fit(f,y,g,tr,te,i,seed,schemas,spec)
        _,q=fast.outer_fit(f,y,g,tr,te,i,seed,schemas,spec)
        diff=float(np.max(np.abs(p-q)));maximum=max(maximum,diff)
        np.testing.assert_allclose(p,q,rtol=0,atol=1e-10)
    start=time.time();p=ref.scores(f,y,family,seed,['SE timing proxy'],spec);oldtime=time.time()-start
    start=time.time();q=fast.scores(f,y,family,seed,['SE timing proxy'],spec);newtime=time.time()-start
    np.testing.assert_allclose(p,q,rtol=0,atol=1e-10)
    return dict(family=family,index=index,outer_folds_identical=len(a),inner_folds_identical=checked_inner,
        maximum_prediction_difference=maximum,reference_scores=p.tolist(),accelerated_scores=q.tolist(),
        reference_seconds=oldtime,accelerated_seconds=newtime)

if __name__=='__main__':
    spec,frames=ref.load_data(pd.read_csv(ROOT/'data/cohort.csv'),ROOT/'data')
    results=Parallel(n_jobs=2,verbose=5)(delayed(comparison)(f,y,family,index,spec) for family,(f,y) in frames.items() for index in [0,137])
    path=ROOT/'preparation/legacy_execution_parity.json'
    checked=['code/legacy_accelerated.py','code/legacy_omissions.py','code/legacy_validation.py','data/legacy_history.csv','data/cohort.csv','data/legacy_spec.json']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in checked}
    path.write_text(json.dumps(dict(passed=True,tolerance=1e-10,comparisons=results,checked_hashes=hashes),indent=2),encoding='utf-8')
    print(path.read_text(),flush=True)
