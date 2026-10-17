import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from glmhmm import GLMHMM
cols=['stim','prev','wsls','bias']; Ks=(5,6); NR=3
def job(a):
    s,f,tr,te=a
    d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    ses=np.unique(g); trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te]); out=[]
    for K in Ks:
        best=None
        for seed in range(NR if K>1 else 1):
            m=GLMHMM(K,seed=seed).fit(X[trm],y[trm],g[trm])
            if best is None or m.train_ll>best.train_ll: best=m
        out.append(dict(subject=s,fold=f,K=K,test_ll=best.loglik(X[tem],y[tem],g[tem]),n_test=int(tem.sum()),
                        train_ll=best.train_ll,n_train=int(trm.sum())))
    return out
D=pd.read_parquet('data/features.parquet'); D['bias']=1.0
if __name__=='__main__':
    tasks=[]
    for s,d in D.groupby('subject'):
        ses=np.unique(d.session)
        for f,(tr,te) in enumerate(GroupKFold(5).split(ses,groups=ses)): tasks.append((s,f,tr,te))
    t=time.time(); res=[]
    with Pool(16) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            res+=o
            if i%10==0: print(i,len(tasks),round(time.time()-t),'s',flush=True)
    pd.DataFrame(res).to_csv('results/sweep_k56.csv',index=False); print('done',round(time.time()-t),'s')
