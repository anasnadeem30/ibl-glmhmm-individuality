import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from glmhmm import GLMHMM, emission_logp, _viterbi_free_loglik
cols=['stim','prev','wsls','bias']; Ks=(1,3,4); NR=5; FR=0.6
D=pd.read_parquet('data/features.parquet'); D['bias']=1.0
mice=sorted(D.subject.unique())
def split(s):
    d=D[D.subject==s]; order=d.groupby('session').date.first().sort_values(kind='stable').index.values
    n=int(round(FR*len(order))); return order[:n],order[n:]
def job(s):
    d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    early,_=split(s); trm=np.isin(g,early); out={}
    for K in Ks:
        best=None
        for sd in range(NR if K>1 else 1):
            m=GLMHMM(K,seed=sd).fit(X[trm],y[trm],g[trm])
            if best is None or m.train_ll>best.train_ll: best=m
        out[K]=(best.W,best.A,best.pi,best.train_ll/trm.sum())
    return s,out
def score(par,j,sess):
    W,A,pi=par[:3]; d=D[D.subject==j]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    tot=0.0;n=0
    for sid in sess:
        ix=np.where(g==sid)[0]; tot+=_viterbi_free_loglik(emission_logp(X[ix],y[ix],W),A,pi); n+=len(ix)
    return tot/n/np.log(2)
if __name__=='__main__':
    t=time.time()
    with Pool(20) as p: fits=dict(p.map(job,mice,chunksize=1))
    rows=[]
    for K in Ks:
        for j in mice:
            early,late=split(j)
            for i in mice:
                rows.append(dict(K=K,model_mouse=i,test_mouse=j,bits=score(fits[i][K],j,late)))
            rows.append(dict(K=K,model_mouse=j,test_mouse=j+'__EARLYTRAIN',bits=fits[j][K][3]/np.log(2)))
    pd.DataFrame(rows).to_csv('results/individuality_temporal_matrix.csv',index=False); print('done',round(time.time()-t),'s')
