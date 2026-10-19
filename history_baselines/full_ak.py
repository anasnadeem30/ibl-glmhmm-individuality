import sys, pathlib; _h = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(_h), str(_h.parent)]  # find glmhmm.py (repo root) and sibling modules
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from glmhmm import GLMHMM
from feats_ak import build
NR=3
D=pd.read_parquet('data/features.parquet')
def glm_ll(Xtr,ytr,Xte,yte):
    m=LogisticRegression(C=100.0,max_iter=500,fit_intercept=False).fit(Xtr,ytr)
    p=np.clip(m.predict_proba(Xte)[:,1],1e-9,1-1e-9)
    return (yte*np.log(p)+(1-yte)*np.log(1-p)).sum()
def hmm_ll(K,Xtr,ytr,gtr,Xte,yte,gte):
    best=None
    for sd in range(NR):
        m=GLMHMM(K,seed=sd).fit(Xtr,ytr,gtr)
        if best is None or m.train_ll>best.train_ll: best=m
    return best.loglik(Xte,yte,gte)
def job(a):
    s,f,tr,te=a
    d=D[D.subject==s]; F=build(d); y,g=F['y'],F['g']
    ses=np.unique(g); trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te])
    out=dict(subject=s,fold=f,n_test=int(tem.sum()))
    for nm in ('X0','S1','S2'):
        X=F[nm]; out['glm_'+nm]=glm_ll(X[trm],y[trm],X[tem],y[tem])
    X=np.c_[F['X0'],F['prior']]; out['glm_X0_block']=glm_ll(X[trm],y[trm],X[tem],y[tem])
    for K in (3,):
        for nm in ('X0','S1','S2'):
            X=F[nm]; out[f'hmm{K}_{nm}']=hmm_ll(K,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
    return out
if __name__=='__main__':
    mice=sorted(D.subject.unique()) if sys.argv[1]=='all' else sys.argv[1].split(',')
    tasks=[]
    for s in mice:
        ses=np.unique(D[D.subject==s].session)
        for f,(tr,te) in enumerate(GroupKFold(5).split(ses,groups=ses)): tasks.append((s,f,tr,te))
    t=time.time(); res=[]
    with Pool(24) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            res.append(o); print(i,len(tasks),round(time.time()-t),'s',flush=True)
            if i%20==0: pd.DataFrame(res).to_csv(sys.argv[2]+'.partial',index=False)
    pd.DataFrame(res).to_csv(sys.argv[2],index=False); print('done',round(time.time()-t),'s')
