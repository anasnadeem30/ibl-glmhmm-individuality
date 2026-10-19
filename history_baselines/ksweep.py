import sys, pathlib; _h = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(_h), str(_h.parent)]  # find glmhmm.py (repo root) and sibling modules
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from feats_ak import build
from full_ak import hmm_ll
from pers_ctrl import mix_ll, D
KS=(2,4,5)
def job(a):
    for _ in range(3):
        try: return job0(a)
        except MemoryError: time.sleep(30)
    raise MemoryError
def job0(a):
    s,f,tr,te=a
    d=D[D.subject==s]; F=build(d); y,g=F['y'],F['g']
    ses=np.unique(g); trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te])
    out=dict(subject=s,fold=f,n_test=int(tem.sum()))
    X=F['S2']
    for K in KS:
        out[f'hmm{K}_S2']=hmm_ll(K,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
        out[f'mix{K}_S2']=mix_ll(K,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
    return out
if __name__=='__main__':
    mice=sorted(D.subject.unique()); tasks=[]
    for s in mice:
        ses=np.unique(D[D.subject==s].session)
        for f,(tr,te) in enumerate(GroupKFold(5).split(ses,groups=ses)): tasks.append((s,f,tr,te))
    t=time.time(); res=[]
    with Pool(16) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            res.append(o); pd.DataFrame(res).to_csv(sys.argv[1]+'.partial',index=False); print(i,len(tasks),round(time.time()-t),'s',flush=True)
    pd.DataFrame(res).to_csv(sys.argv[1],index=False); print('done',round(time.time()-t),'s')
