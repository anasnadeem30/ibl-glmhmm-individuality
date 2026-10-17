import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pickle, numpy as np, pandas as pd, itertools
from scipy.optimize import linear_sum_assignment
F=pickle.load(open('data/full_fits.pkl','rb'))
def dist(Wa,Wb,Aa=None,Ab=None):
    C=np.linalg.norm(Wa[:,None,:]-Wb[None,:,:],axis=2); r,c=linear_sum_assignment(C); return C[r,c].mean(), (r,c)
rows=[]
for K in (3,4):
    fs=[f for f in F if f['K']==K]
    for f in fs:
        al=np.array(f['all_ll']); o=np.argsort(-al); best=f['all_W'][o[0]]
        for k in o[1:]:
            d,_=dist(best,f['all_W'][k]); rows.append(dict(K=K,subject=f['subject'],kind='restart_vs_best_same_mouse',dist=d,ll_gap=(al[o[0]]-al[k])/f['n']/np.log(2)))
    bests={f['subject']:f['W'] for f in fs}
    for a,b in itertools.combinations(sorted(bests),2):
        d,_=dist(bests[a],bests[b]); rows.append(dict(K=K,subject=a+'|'+b,kind='best_vs_best_different_mice',dist=d,ll_gap=np.nan))
R=pd.DataFrame(rows); R.to_csv('results/repro_restarts.csv',index=False)
for K in (3,4):
    s=R[(R.K==K)&(R.kind=='restart_vs_best_same_mouse')]; b=R[(R.K==K)&(R.kind=='best_vs_best_different_mice')]
    print(f'K={K}: matched weight distance (Euclid, 4 weights): same mouse, restart vs best: median {s.dist.median():.2f} (IQR {s.dist.quantile(.25):.2f}-{s.dist.quantile(.75):.2f}); different mice: median {b.dist.median():.2f} (IQR {b.dist.quantile(.25):.2f}-{b.dist.quantile(.75):.2f})')
    print(f'   restarts within 0.05 of best-weights distance: {(s.dist<0.05).mean():.0%}; restarts closer to own best than the median different-mouse distance: {(s.dist<b.dist.median()).mean():.0%}')
    # restarts with LL gap < 0.001 bits: do their weights agree?
    t=s[s.ll_gap<0.001]; print(f'   among restarts with near-identical LL (gap<0.001 bits, n={len(t)}): median distance {t.dist.median():.3f}; with larger LL gap (n={len(s)-len(t)}): median {s[s.ll_gap>=0.001].dist.median():.2f}')
