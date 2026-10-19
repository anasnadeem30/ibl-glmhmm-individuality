import sys, pathlib; _h = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(_h), str(_h.parent)]  # find glmhmm.py (repo root) and sibling modules
import numpy as np
def ewma(a,g,tau,init=0.0):
    lam=np.exp(-1.0/tau); out=np.zeros(len(a))
    for s in np.unique(g):
        ix=np.where(g==s)[0]; m=init
        for i in ix:
            out[i]=m; m=lam*m+(1-lam)*a[i]
    return out
def build(d):
    """d: one mouse's rows (ordered by session, trial). Returns dict of design matrices."""
    g=d.session.values; y=d.right.values.astype(float); act=2*y-1
    stim=d.stim.values; side=np.sign(stim); corr=(d.feedbackType.values==1).astype(float)
    X0=np.c_[d[['stim','prev','wsls']].values,np.ones(len(d))]
    AK=np.c_[[ewma(act,g,t) for t in (2,5,10,20,40)]].T
    SK=np.c_[[ewma(side,g,t) for t in (5,20)]].T
    ACC=np.c_[[ewma(corr,g,t,0.75) for t in (5,20)]].T-0.75
    GAIN=np.c_[stim*ACC[:,0],stim*ACC[:,1]]
    tf=np.zeros(len(d))
    for s in np.unique(g):
        ix=np.where(g==s)[0]; tf[ix]=np.arange(len(ix))/max(len(ix)-1,1)
    TF=tf[:,None]
    return dict(X0=X0,S1=np.c_[X0,AK],S2=np.c_[X0,AK,SK,ACC,GAIN,TF],y=y,g=g,prior=d.prior.values)
