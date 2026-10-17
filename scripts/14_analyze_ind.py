import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
M=pd.read_csv('results/individuality_matrix.csv'); lab=pd.read_csv('results/cohort.csv').set_index('subject').lab
out=[]
for K in (1,3,4):
    P=M[M.K==K].pivot(index='model_mouse',columns='test_mouse',values='bits'); mice=P.columns
    own=np.diag(P.values); off=(P.values.sum(0)-own)/(len(mice)-1)
    adv=own-off; rank=np.array([(P[j].rank(ascending=False))[j] for j in mice])
    # same-lab vs other-lab others
    sl=[];dl=[]
    for j in mice:
        o=[i for i in mice if i!=j]; s=[P.loc[i,j] for i in o if lab[i]==lab[j]]; d=[P.loc[i,j] for i in o if lab[i]!=lab[j]]
        sl.append(np.mean(s) if s else np.nan); dl.append(np.mean(d))
    sl=np.array(sl); dl=np.array(dl)
    w=stats.wilcoxon(adv); 
    print(f'--- K={K} ---')
    print(f' own model beats mean-of-others in {(adv>0).sum()}/{len(mice)} mice; mean advantage {adv.mean():+.4f} bits/trial (median {np.median(adv):+.4f}); Wilcoxon p={w.pvalue:.2g}')
    print(f' own-model rank (1=best of 33): mean {rank.mean():.1f} (chance 17), median {np.median(rank):.0f}, #rank1: {(rank==1).sum()}, #in top 5: {(rank<=5).sum()}')
    ok=~np.isnan(sl); print(f' same-lab others vs other-lab others (on mice with lab-mates, n={ok.sum()}): same-lab minus other-lab = {np.mean(sl[ok]-dl[ok]):+.4f} bits; own minus same-lab = {np.mean(own[ok]-sl[ok]):+.4f}')
    out.append(pd.DataFrame(dict(K=K,mouse=mice,own=own,others=off,adv=adv,rank=rank,same_lab=sl,diff_lab=dl)))
    if K==3:
        # difficulty-normalised matrix: subtract the column mean (how hard each test mouse is)
        Z=P.values-P.values.mean(0,keepdims=True); fig,ax=plt.subplots(1,2,figsize=(12,5))
        im=ax[0].imshow(Z,cmap='RdBu',vmin=-.1,vmax=.1); ax[0].set_xlabel('test mouse'); ax[0].set_ylabel('model mouse'); ax[0].set_title('K=3: score minus column mean (bits/trial)'); plt.colorbar(im,ax=ax[0])
        ax[1].hist(adv,bins=15,color='gray'); ax[1].axvline(0,c='k',ls=':'); ax[1].set_xlabel('own-model advantage (bits/trial)'); ax[1].set_title('K=3: own minus mean of others, per mouse')
        plt.tight_layout(); plt.savefig('figures/individuality_K3.png',dpi=130)
pd.concat(out).to_csv('results/individuality_summary.csv',index=False)
