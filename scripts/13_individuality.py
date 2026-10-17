import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from glmhmm import GLMHMM, emission_logp, _viterbi_free_loglik
cols=['stim','prev','wsls','bias']; Ks=(1,3,4); NR=3
D=pd.read_parquet('data/features.parquet'); D['bias']=1.0
mice=sorted(D.subject.unique())
def folds(s):
    ses=np.unique(D[D.subject==s].session); return ses,list(GroupKFold(5).split(ses,groups=ses))
def job(a):
    s,f=a; d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    ses,fo=folds(s); trm=np.isin(g,ses[fo[f][0]]); out={}
    for K in Ks:
        best=None
        for sd in range(NR if K>1 else 1):
            m=GLMHMM(K,seed=sd).fit(X[trm],y[trm],g[trm])
            if best is None or m.train_ll>best.train_ll: best=m
        out[K]=(best.W,best.A,best.pi)
    return (s,f),out
def score(params,s,f):
    W,A,pi=params; d=D[D.subject==s]; X=d[cols].values; y=d.right.values.astype(float); g=d.session.values
    ses,fo=folds(s); tem=np.isin(g,ses[fo[f][1]]); tot=0.0
    for sid in np.unique(g[tem]):
        ix=np.where(g==sid)[0]; tot+=_viterbi_free_loglik(emission_logp(X[ix],y[ix],W),A,pi)
    return tot, int(tem.sum())
if __name__=='__main__':
    t=time.time(); fits={}
    with Pool(20) as p:
        for i,(k,o) in enumerate(p.imap_unordered(job,[(s,f) for s in mice for f in range(5)])):
            fits[k]=o
            if i%20==0: print('fit',i,165,round(time.time()-t),'s',flush=True)
    pickle.dump(fits,open('data/ind_fits.pkl','wb'))
    rows=[]
    for K in Ks:
        for j in mice:
            for i in mice:
                ll=n=0
                for f in range(5):
                    a,b=score(fits[(i,f)][K],j,f); ll+=a; n+=b
                rows.append(dict(K=K,model_mouse=i,test_mouse=j,bits=ll/n/np.log(2)))
    pd.DataFrame(rows).to_csv('results/individuality_matrix.csv',index=False); print('done',round(time.time()-t),'s')
