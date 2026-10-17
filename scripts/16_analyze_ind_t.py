import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from scipy import stats
M=pd.read_csv('results/individuality_temporal_matrix.csv'); lab=pd.read_csv('results/cohort.csv').set_index('subject').lab
tr=M[M.test_mouse.str.endswith('__EARLYTRAIN')]; M=M[~M.test_mouse.str.endswith('__EARLYTRAIN')]
R=pd.read_csv('results/individuality_summary.csv')
for K in (1,3,4):
    P=M[M.K==K].pivot(index='model_mouse',columns='test_mouse',values='bits'); mice=P.columns
    own=np.diag(P.values); off=(P.values.sum(0)-own)/(len(mice)-1); adv=own-off
    rank=np.array([P[j].rank(ascending=False)[j] for j in mice]); w=stats.wilcoxon(adv)
    sl=np.array([np.mean([P.loc[i,j] for i in mice if i!=j and lab[i]==lab[j]]) for j in mice]); dl=np.array([np.mean([P.loc[i,j] for i in mice if lab[i]!=lab[j]]) for j in mice])
    a0=R[R.K==K].set_index('mouse').adv
    t=tr[tr.K==K].set_index('model_mouse').bits; t.index=[i for i in t.index]
    trainll=np.array([t[m] for m in mice])
    print(f'--- K={K}: train early 60% of sessions, test last 40% ---')
    print(f' own beats mean-of-others in {(adv>0).sum()}/{len(mice)} mice; mean advantage {adv.mean():+.4f} bits (random-split test was {a0.mean():+.4f}); Wilcoxon p={w.pvalue:.2g}')
    print(f' own rank: mean {rank.mean():.1f} (chance 17), #rank1 {(rank==1).sum()}, top5 {(rank<=5).sum()}; same-lab minus other-lab {np.mean(sl-dl):+.4f}')
    print(f' own-model score: early-train {trainll.mean():.4f} -> late-test {own.mean():.4f} bits/trial (drop {trainll.mean()-own.mean():.4f}); others on late: {off.mean():.4f}')
    print(f' mice where own model is NOT best: {list(mice[rank>1])[:6]} ranks {rank[rank>1][:6]}')
