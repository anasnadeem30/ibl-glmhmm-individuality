import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
# Figure A: block confound (state occupancy in block favouring the state's side vs the other side)
B=pd.read_csv('results/block_control_states.csv'); fig,ax=plt.subplots(1,2,figsize=(10,4))
for a,K in zip(ax,(3,4)):
    r=B[(B.K==K)&B.biased]; own=np.where(B[(B.K==K)&B.biased].bias>0,r.occ_right_block,r.occ_left_block); oth=np.where(r.bias>0,r.occ_left_block,r.occ_right_block)
    a.scatter(oth,own,s=22,color='tab:red',alpha=.7); m=max(own.max(),oth.max())*1.05; a.plot([0,m],[0,m],'k:')
    a.set_xlabel('occupancy when block favours the OTHER side'); a.set_ylabel("occupancy when block favours the state's side"); a.set_title(f'K={K}: low-stimulus side-biased states ({len(r)})')
plt.tight_layout(); plt.savefig('figures/block_control.png',dpi=130); plt.close()
# Figure B: individuality summary (random-session split vs early/late split), K=3
S=pd.read_csv('results/individuality_summary.csv'); T=pd.read_csv('results/individuality_temporal_matrix.csv'); T=T[~T.test_mouse.str.endswith('__EARLYTRAIN')]
P=T[T.K==3].pivot(index='model_mouse',columns='test_mouse',values='bits'); mice=P.columns
own=np.diag(P.values); oth=(P.values.sum(0)-own)/(len(mice)-1); rank_t=np.array([P[j].rank(ascending=False)[j] for j in mice])
R3=S[S.K==3].set_index('mouse'); fig,ax=plt.subplots(1,3,figsize=(14,4))
ax[0].scatter(R3.others,R3.own,s=22,color='k'); lo=min(R3.others.min(),R3.own.min())-.02; hi=max(R3.others.max(),R3.own.max())+.02; ax[0].plot([lo,hi],[lo,hi],'k:'); ax[0].set_xlabel('mean score of the 32 other mice\'s models'); ax[0].set_ylabel("mouse's own model"); ax[0].set_title('Random-session split: own model vs others')
ax[1].scatter(oth,own,s=22,color='tab:blue'); lo=min(oth.min(),own.min())-.02; hi=max(oth.max(),own.max())+.02; ax[1].plot([lo,hi],[lo,hi],'k:'); ax[1].set_xlabel('mean score of the 32 other mice\'s models'); ax[1].set_title('Early-train / late-test split')
bins=np.arange(.5,34.5,1); ax[2].hist(R3['rank'],bins=bins,alpha=.6,label='random split',color='gray'); ax[2].hist(rank_t,bins=bins,alpha=.6,label='early/late split',color='tab:blue'); ax[2].set_xlim(0,34); ax[2].set_xlabel("rank of own model among 33 (1 = best)"); ax[2].set_ylabel('mice'); ax[2].legend(); ax[2].set_title('Rank of own model (chance mean = 17)')
plt.tight_layout(); plt.savefig('figures/individuality_summary.png',dpi=130); plt.close()
# Figure C: replication on 50/50 trials vs all trials: held-out score by K
A=pd.read_csv('results/sweep_k_summary.csv',index_col=0); A.columns=A.columns.astype(int); Rr=pd.read_csv('results/replication_5050_summary.csv',index_col=0); Rr.columns=Rr.columns.astype(int)
fig,ax=plt.subplots(figsize=(5.8,4)); 
for D_,lab,c in ((A,'all trials (incl. 20/80 blocks)','tab:orange'),(Rr,'first 90 trials only (50/50), Ashwood-style','tab:green')):
    d=D_.loc[:, [k for k in D_.columns if k<=4]]; g=d.sub(d[1],axis=0).mean(); ax.plot(g.index,g.values,'o-',color=c,label=lab,lw=2)
ax.plot([3],[.13],'k*',ms=13,label='Ashwood et al. 2022 (+0.13 at K=3)'); ax.set_xlabel('number of states K'); ax.set_ylabel('held-out gain over 1 state (bits/trial)'); ax.set_xticks([1,2,3,4]); ax.legend(fontsize=8,loc='lower right')
plt.tight_layout(); plt.savefig('figures/replication_vs_all_trials.png',dpi=130)
print('ok')
