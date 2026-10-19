import sys, pathlib; _h = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(_h), str(_h.parent)]  # find glmhmm.py (repo root) and sibling modules
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time, sys, zlib
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from glmhmm import GLMHMM
from feats_ak import build
from full_ak import glm_ll, hmm_ll
from pers_ctrl import MixGLM, mix_ll
from sim_mimic import simulate, D
def fit_best(cls,F):
    best=None
    for sd in range(3):
        h=cls(3,seed=sd).fit(F['X0'],F['y'],F['g'])
        if best is None or h.train_ll>best.train_ll: best=h
    return best
def job(a):
    s,gn,rep=a
    rng=np.random.default_rng(5000+1000*rep+zlib.crc32(s.encode())%997)
    d=D[D.subject==s]; F=build(d)
    if gn=='stateless':
        m=LogisticRegression(C=100.0,max_iter=500,fit_intercept=False).fit(F['S2'],F['y']); gen=('S2',m.coef_[0])
    elif gn=='hmm':
        b=fit_best(GLMHMM,F); gen=('H0',b.W,b.A,b.pi)
    else:
        b=fit_best(MixGLM,F); gen=('H0',b.W,np.tile(b.pi,(3,1)),b.pi)
    sim=simulate(d,gen,rng); G=build(sim); y,g=G['y'],G['g']
    ses=np.unique(g); out=dict(subject=s,gen=gn,rep=rep,n=0,**{k:0.0 for k in ('glm_S2','hmm3_X0','mix3_X0','hmm3_S2','mix3_S2')})
    for tr,te in GroupKFold(5).split(ses,groups=ses):
        trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te]); out['n']+=int(tem.sum())
        for nm in ('X0','S2'):
            X=G[nm]
            if nm=='S2': out['glm_S2']+=glm_ll(X[trm],y[trm],X[tem],y[tem])
            out['hmm3_'+nm]+=hmm_ll(3,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
            out['mix3_'+nm]+=mix_ll(3,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
    return out
if __name__=='__main__':
    mice=sorted(D.subject.unique()); reps=int(sys.argv[2])
    tasks=[(s,gn,r) for r in range(reps) for s in mice for gn in ('stateless','hmm','mix')]
    t=time.time(); res=[]
    with Pool(24) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            res.append(o); print(i,len(tasks),round(time.time()-t),'s',flush=True)
    pd.DataFrame(res).to_csv(sys.argv[1],index=False); print('done',round(time.time()-t),'s')
