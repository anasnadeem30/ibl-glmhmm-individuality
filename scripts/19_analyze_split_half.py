import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pickle, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
R=pickle.load(open('data/split_half_fits.pkl','rb')); mice=sorted({k[0] for k in R})
def sd_of(K,kind):
    allW=np.vstack([R[(s,kind,h)][K]['W'] for s in mice for h in (0,1)]); return allW.std(0)
def dist(Wa,Wb,sd):
    C=np.linalg.norm((Wa[:,None,:]-Wb[None,:,:])/sd,axis=2); r,c=linear_sum_assignment(C); return C[r,c].mean()
def dwell(A): return 1/(1-np.diag(A)+1e-9)
out=[]
for kind in ('interleaved','temporal'):
    for K in (3,4):
        sd=sd_of(K,kind); n=len(mice)
        M=np.array([[dist(R[(a,kind,0)][K]['W'],R[(b,kind,1)][K]['W'],sd) for b in mice] for a in mice])   # half0 of a vs half1 of b
        same=np.diag(M); diff=(M.sum(1)-same)/(n-1)
        rank=np.array([(M[i]<M[i,i]).sum()+1 for i in range(n)]); rank_c=np.array([(M[:,j]<M[j,j]).sum()+1 for j in range(n)])
        ident=(rank==1).mean(); ident_c=(rank_c==1).mean()
        # also raw-weight distance for transparency
        sd1=np.ones(4); M1=np.array([[dist(R[(a,kind,0)][K]['W'],R[(b,kind,1)][K]['W'],sd1) for b in mice] for a in mice]); rk1=np.array([(M1[i]<M1[i,i]).sum()+1 for i in range(n)])
        # dwell-time agreement of same mouse: correlation across mice of sorted-by-stim dwell
        def prof(h,s):
            f=R[(s,kind,h)][K]; o=np.argsort(-f['W'][:,0]); return dwell(f['A'])[o]
        dw=np.array([[prof(0,s),prof(1,s)] for s in mice]); rho=np.corrcoef(np.log(dw[:,0,:]).ravel(),np.log(dw[:,1,:]).ravel())[0,1]
        print(f'--- {kind}, K={K} ---')
        print(f' standardized matched weight distance: same mouse (halves) median {np.median(same):.2f}; different mice median {np.median(diff):.2f}; same smaller than mean-of-others in {(same<diff).sum()}/{n}')
        print(f' identification (is the nearest other-half the same mouse?): {ident:.0%} (half0->half1), {ident_c:.0%} (reverse), chance {1/n:.0%}; mean rank of same mouse {rank.mean():.1f} (chance 17); raw-weight identification {(rk1==1).mean():.0%}')
        print(f' dwell times (log, states sorted by stim weight) correlation between halves: r={rho:.2f}')
        out.append(dict(kind=kind,K=K,same=np.median(same),diff=np.median(diff),ident=ident,ident_rev=ident_c,mean_rank=rank.mean(),dwell_r=rho))
pd.DataFrame(out).to_csv('results/split_half_summary.csv',index=False)
