import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
from glmhmm import GLMHMM
cols=['stim','prev','wsls','bias']; Ks=(3,4); NR=5
D=pd.read_parquet('data/features.parquet'); D['bias']=1.0
mice=sorted(D.subject.unique())
def halves(s,kind):
    d=D[D.subject==s]; order=d.groupby('session').date.first().sort_values(kind='stable').index.values
    if kind=='interleaved': return order[0::2],order[1::2]
    h=len(order)//2; return order[:h],order[h:]
def job(a):
    s,kind,half=a; d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    sess=halves(s,kind)[half]; m_=np.isin(g,sess); out={}
    for K in Ks:
        fits=sorted([GLMHMM(K,seed=sd).fit(X[m_],y[m_],g[m_]) for sd in range(NR)],key=lambda m:-m.train_ll)
        out[K]=dict(W=fits[0].W,A=fits[0].A,pi=fits[0].pi,ll=fits[0].train_ll/m_.sum(),n=int(m_.sum()),all_W=[m.W for m in fits],all_ll=[m.train_ll/m_.sum() for m in fits])
    return (s,kind,half),out
if __name__=='__main__':
    t=time.time(); tasks=[(s,k,h) for s in mice for k in ('interleaved','temporal') for h in (0,1)]
    with Pool(20) as p: res=dict(p.map(job,tasks,chunksize=1))
    pickle.dump(res,open('data/split_half_fits.pkl','wb')); print('done',round(time.time()-t),'s')
