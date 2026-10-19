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
D=pd.read_parquet('data/features.parquet')
TAU=(2,5,10,20,40); SKT=(5,20); ACT=(5,20)
def sig(a): return 1/(1+np.exp(-a))
def simulate(d, gen, rng):
    """closed-loop replay of real stimuli; gen = ('S2',w) stateless or ('H0',W,A,pi) hidden-state."""
    g=d.session.values; stim=d.stim.values; fb=d.feedbackType.values; right=d.right.values
    corr_right=np.where(fb==1,right,1-right)          # true correct side from the real data
    y=np.zeros(len(d)); prev=np.zeros(len(d)); wsls=np.zeros(len(d)); fbs=np.zeros(len(d))
    for s in np.unique(g):
        ix=np.where(g==s)[0]; n=len(ix)
        ak=np.zeros(len(TAU)); sk=np.zeros(len(SKT)); ac=np.zeros(len(ACT))   # ACC stored as (ewma-0.75)
        pc=0.0; pr=0.0
        if gen[0]=='H0': z=rng.choice(len(gen[3]),p=gen[3])
        for t,i in enumerate(ix):
            x0=np.array([stim[i],pc,pc*pr,1.0])
            if gen[0]=='S2':
                x=np.r_[x0,ak,sk,ac,stim[i]*ac,t/max(n-1,1)]; p=sig(x@gen[1])
            else:
                if t>0: z=rng.choice(len(gen[3]),p=gen[2][z])
                p=sig(x0@gen[1][z])
            c=float(rng.random()<p); act=2*c-1; ok=(c==corr_right[i]); r=1.0 if ok else -1.0
            y[i]=c; prev[i]=pc; wsls[i]=pc*pr; fbs[i]=r
            for j,tau in enumerate(TAU): l=np.exp(-1/tau); ak[j]=l*ak[j]+(1-l)*act
            for j,tau in enumerate(SKT): l=np.exp(-1/tau); sk[j]=l*sk[j]+(1-l)*np.sign(stim[i])
            for j,tau in enumerate(ACT): l=np.exp(-1/tau); ac[j]=l*ac[j]+(1-l)*(float(ok)-0.75)
            pc=act; pr=r
    o=d.copy(); o['right']=y.astype(int); o['feedbackType']=fbs; o['prev']=prev; o['wsls']=wsls
    return o
def job(a):
    s,gen_name,rep=a
    rng=np.random.default_rng(1000*rep+zlib.crc32(s.encode())%997)
    d=D[D.subject==s]; F=build(d)
    if gen_name=='stateless':
        m=LogisticRegression(C=100.0,max_iter=500,fit_intercept=False).fit(F['S2'],F['y'])
        gen=('S2',m.coef_[0])
    else:
        best=None
        for sd in range(3):
            h=GLMHMM(3,seed=sd).fit(F['X0'],F['y'],F['g'])
            if best is None or h.train_ll>best.train_ll: best=h
        gen=('H0',best.W,best.A,best.pi)
    sim=simulate(d,gen,rng); G=build(sim); y,g=G['y'],G['g']
    ses=np.unique(g); out=dict(subject=s,gen=gen_name,rep=rep,n=0,glm_X0=0.0,glm_S2=0.0,hmm3_X0=0.0,hmm3_S2=0.0)
    for tr,te in GroupKFold(5).split(ses,groups=ses):
        trm=np.isin(g,ses[tr]); tem=np.isin(g,ses[te]); out['n']+=int(tem.sum())
        for nm in ('X0','S2'):
            X=G[nm]; out['glm_'+nm]+=glm_ll(X[trm],y[trm],X[tem],y[tem])
            out['hmm3_'+nm]+=hmm_ll(3,X[trm],y[trm],g[trm],X[tem],y[tem],g[tem])
    return out
if __name__=='__main__':
    mice=sys.argv[1].split(','); reps=int(sys.argv[3])
    tasks=[(s,gn,r) for s in mice for gn in ('stateless','hmm') for r in range(reps)]
    t=time.time(); res=[]
    with Pool(min(24,len(tasks))) as p:
        for i,o in enumerate(p.imap_unordered(job,tasks)):
            res.append(o); print(i,len(tasks),round(time.time()-t),'s',flush=True)
    pd.DataFrame(res).to_csv(sys.argv[2],index=False); print('done',round(time.time()-t),'s')
