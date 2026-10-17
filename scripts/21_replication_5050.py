import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
"""Closer replication of Ashwood et al. 2022: first 90 trials of each session (50/50 block), inputs = stimulus, bias,
previous choice, previous stimulus side. 5-fold CV by session for K=1..4; full-data K=3 fit per mouse."""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time, pickle
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from glmhmm import GLMHMM
D=pd.read_parquet('data/features.parquet').sort_values(['subject','date','session','trial']).reset_index(drop=True)
D['side']=np.where(D.contrastRight.notna(),1.0,-1.0)
D['prev_side']=D.groupby('session').side.shift(1).fillna(0)
D['bias']=1.0
D=D[(D.trial<90)&(D.probabilityLeft==0.5)].reset_index(drop=True)
cols=['stim','bias','prev','prev_side']; Ks=(1,2,3,4); NR=5
def cv_job(a):
    s,f,tr,te=a; d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    ses=np.unique(g); trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te]); out=[]
    for K in Ks:
        best=None
        for sd in range(NR if K>1 else 1):
            m=GLMHMM(K,seed=sd).fit(X[trm],y[trm],g[trm])
            if best is None or m.train_ll>best.train_ll: best=m
        out.append(dict(subject=s,fold=f,K=K,test_ll=best.loglik(X[tem],y[tem],g[tem]),n_test=int(tem.sum())))
    return out
def full_job(s):
    d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    best=None
    for sd in range(NR*2):
        m=GLMHMM(3,seed=sd).fit(X,y,g)
        if best is None or m.train_ll>best.train_ll: best=m
    post=best.posteriors(X,y,g); return dict(subject=s,W=best.W,A=best.A,occ=post.mean(0),n=len(y),sessions=len(np.unique(g)))
if __name__=='__main__':
    mice=sorted(D.subject.unique()); t=time.time()
    tasks=[]
    for s in mice:
        ses=np.unique(D[D.subject==s].session)
        for f,(tr,te) in enumerate(GroupKFold(5).split(ses,groups=ses)): tasks.append((s,f,tr,te))
    with Pool(20) as p:
        res=[r for o in p.imap_unordered(cv_job,tasks) for r in o]; full=p.map(full_job,mice)
    pd.DataFrame(res).to_csv('results/replication_5050_cv.csv',index=False); pickle.dump(full,open('data/replication_5050_full.pkl','wb')); print('done',round(time.time()-t),'s')
    print('trials total',len(D),'| mice',len(mice),'| sessions',D.session.nunique())
