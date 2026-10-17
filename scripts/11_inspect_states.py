import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
F=pickle.load(open('data/full_fits.pkl','rb'))
rows=[]
for f in F:
    W,A,post=f['W'],f['A'],f['post']; o=np.argsort(-W[:,0]); occ=post.mean(0)
    for rank,k in enumerate(o):
        rows.append(dict(subject=f['subject'],K=f['K'],rank=rank,stim=W[k,0],prev=W[k,1],wsls=W[k,2],bias=W[k,3],occ=occ[k],stay=A[k,k],dwell=1/(1-A[k,k]+1e-9)))
S=pd.DataFrame(rows); S.to_csv('results/states_full.csv',index=False)
for K in (3,4):
    print(f'--- K={K}: median across 33 mice, states ranked by stim weight (0 = most stimulus-driven) ---')
    print(S[S.K==K].groupby('rank')[['stim','prev','wsls','bias','occ','dwell']].median().round(2))
    print('abs bias >1 (side-biased state) count per rank:', (S[S.K==K].assign(b=lambda x:x.bias.abs()>1).groupby('rank').b.sum()).to_dict())
# restart reproducibility: spread of top LLs
sp=[ (max(f['all_ll'])-np.sort(f['all_ll'])[-2])/f['n']/np.log(2) for f in F]
print('gap best vs 2nd-best restart (bits/trial) median %.4f max %.4f'%(np.median(sp),max(sp)))
fig,ax=plt.subplots(1,2,figsize=(11,4))
for a,K in zip(ax,(3,4)):
    s=S[S.K==K]; sc=a.scatter(s.stim,s.bias,c=s['rank'],cmap='viridis',s=20+300*s.occ,alpha=.7); a.axhline(0,ls=':',c='k'); a.set_xlabel('stimulus weight'); a.set_ylabel('bias weight'); a.set_title(f'K={K} states (size = occupancy, color = rank)')
plt.tight_layout(); plt.savefig('figures/states_K3_K4.png',dpi=130)
