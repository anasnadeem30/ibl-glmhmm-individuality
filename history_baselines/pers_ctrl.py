import sys, pathlib; _h = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(_h), str(_h.parent)]  # find glmhmm.py (repo root) and sibling modules
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from glmhmm import GLMHMM, _fb, emission_logp
from feats_ak import build
class MixGLM(GLMHMM):
    """K-component mixture of GLMs with i.i.d. state assignment each trial (no Markov persistence):
    every row of A equals the marginal state distribution. Same EM otherwise."""
    def fit(self, X, y, groups):
        idx = [np.where(groups == g)[0] for g in np.unique(groups)]
        self._init(X, y); K = self.K; prev = -np.inf
        for it in range(self.n_iter):
            gam = np.zeros((len(y), K)); xi = np.zeros((K, K)); first = np.zeros(K); tot = 0.0
            for ix in idx:
                g, x, ll = _fb(emission_logp(X[ix], y[ix], self.W), self.A, self.pi)
                gam[ix] = g; xi += x; first += g[0]; tot += ll
            if tot - prev < self.tol * len(y) and it > 5: break
            prev = tot
            pi = gam.sum(0) / gam.sum(); self.pi = pi; self.A = np.tile(pi, (K, 1))
            for k in range(K):
                Xr = np.vstack([X, X]); yr = np.r_[np.ones(len(y)), np.zeros(len(y))]
                wr = np.r_[gam[:, k] * y, gam[:, k] * (1 - y)]
                m = LogisticRegression(C=self.C, max_iter=200, fit_intercept=False)
                m.fit(Xr, yr, sample_weight=wr + 1e-12); self.W[k] = m.coef_[0]
        self.train_ll = tot; self.gamma = gam; self.idx = idx
        return self
NR=3
D=pd.read_parquet('data/features.parquet')
def mix_ll(K,Xtr,ytr,gtr,Xte,yte,gte):
    best=None
    for sd in range(NR):
        m=MixGLM(K,seed=sd).fit(Xtr,ytr,gtr)
        if best is None or m.train_ll>best.train_ll: best=m
    return best.loglik(Xte,yte,gte)
def job(a):
    s,f,tr,te=a
    d=D[D.subject==s]; F=build(d); y,g=F['y'],F['g']
    ses=np.unique(g); trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te])
    out=dict(subject=s,fold=f,n_test=int(tem.sum()))
    for nm in ('X0','S2'):
        X=F[nm]; out['mix3_'+nm]=mix_ll(3,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
    return out
if __name__=='__main__':
    mice=sorted(D.subject.unique())
    tasks=[]
    for s in mice:
        ses=np.unique(D[D.subject==s].session)
        for f,(tr,te) in enumerate(GroupKFold(5).split(ses,groups=ses)): tasks.append((s,f,tr,te))
    t=time.time(); res=[]
    with Pool(24) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            res.append(o); print(i,len(tasks),round(time.time()-t),'s',flush=True)
    pd.DataFrame(res).to_csv(sys.argv[1],index=False); print('done',round(time.time()-t),'s')
