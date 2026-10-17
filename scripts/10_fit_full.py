import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
from glmhmm import GLMHMM
cols=['stim','prev','wsls','bias']; NR=5
D=pd.read_parquet('data/features.parquet'); D['bias']=1.0
def job(a):
    s,K=a; d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    fits=[GLMHMM(K,seed=sd).fit(X,y,g) for sd in range(NR)]
    fits.sort(key=lambda m:-m.train_ll); b=fits[0]
    post=b.posteriors(X,y,g)
    return dict(subject=s,K=K,W=b.W,A=b.A,pi=b.pi,ll=b.train_ll,n=len(y),post=post.astype(np.float32),
                all_ll=[m.train_ll for m in fits],all_W=[m.W for m in fits],all_A=[m.A for m in fits])
if __name__=='__main__':
    tasks=[(s,K) for s in sorted(D.subject.unique()) for K in (3,4)]
    t=time.time(); out=[]
    with Pool(6) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            out.append(o)
            if i%10==0: print(i,len(tasks),round(time.time()-t),'s',flush=True)
    pickle.dump(out,open('data/full_fits.pkl','wb')); print('done',round(time.time()-t),'s')
