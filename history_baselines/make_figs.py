import sys, pathlib; _h = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(_h), str(_h.parent)]  # find glmhmm.py (repo root) and sibling modules
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=pd.read_csv('results/history_baselines/full_ak_bits.csv').set_index('subject')
p=pd.read_csv('results/history_baselines/pers_ctrl.csv'); n=p.groupby('subject').n_test.sum()
mx=p.groupby('subject')[['mix3_X0','mix3_S2']].sum().div(n,axis=0)/np.log(2)
print(set(mx.index)==set(b.index))
b=b.join(mx)
g0=b.glm_X0
cols={'GLM + action kernels (S1)':'glm_S1','GLM + all covariates (S2)':'glm_S2','iid mixture (X0)':'mix3_X0','GLM-HMM (X0)':'hmm3_X0','iid mixture (S2)':'mix3_S2','GLM-HMM (S2)':'hmm3_S2'}
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
rng=np.random.default_rng(0)
for i,(lab,c) in enumerate(cols.items()):
    v=(b[c]-g0).values; ax[0].bar(i,v.mean(),color='0.75',width=.7); ax[0].scatter(i+rng.uniform(-.2,.2,len(v)),v,s=8,color='k',alpha=.6)
ax[0].set_xticks(range(len(cols))); ax[0].set_xticklabels(cols.keys(),rotation=40,ha='right',fontsize=8)
ax[0].set_ylabel('held-out gain over 4-input GLM (bits/trial)'); ax[0].set_title('A  Model comparison, 33 mice')
s2share=(b.glm_S2-b.glm_X0)/(b.hmm3_X0-b.glm_X0)
mixshare=(b.mix3_S2-b.glm_S2)/(b.hmm3_S2-b.glm_S2)
ax[1].hist(s2share.clip(0,1.2),bins=np.linspace(0,1.2,25),alpha=.6,label=f'stateless S2 share of HMM(X0) gain\nmedian {s2share.median():.2f}')
ax[1].hist(mixshare.clip(0,1.2),bins=np.linspace(0,1.2,25),alpha=.6,label=f'iid-mixture share of HMM(S2) gain\nmedian {mixshare.median():.2f}')
ax[1].set_xlabel('fraction'); ax[1].set_ylabel('mice'); ax[1].legend(fontsize=8); ax[1].set_title('B  Fractions of gain explained')
plt.tight_layout(); plt.savefig('history_baselines/paper/fig1_decomposition.png',dpi=200)
s=pd.read_csv('results/history_baselines/sim_mix.csv')
for c in ['glm_S2','hmm3_X0','mix3_X0','hmm3_S2','mix3_S2']: s[c]=s[c]/s.n/np.log(2)
s['S2']=s.hmm3_S2-s.mix3_S2; s['X0']=s.hmm3_X0-s.mix3_X0
real=pd.DataFrame({'S2':b.hmm3_S2-b.mix3_S2,'X0':b.hmm3_X0-b.mix3_X0})
order=[('stateless','data from stateless GLM'),('mix','data from iid mixture'),('hmm','data from persistent HMM'),('real','real mice')]
fig,ax=plt.subplots(1,2,figsize=(10,4))
for a,k,t in zip(ax,['X0','S2'],['A  4-input models (X0)','B  full-covariate models (S2)']):
    for i,(gn,lab) in enumerate(order):
        v=real[k].values if gn=='real' else s[s.gen==gn][k].values
        a.boxplot(v,positions=[i],widths=.5,showfliers=False); a.scatter(i+rng.uniform(-.15,.15,len(v)),v,s=8,color='k',alpha=.5)
    a.axhline(0,color='r',lw=.7); a.set_xticks(range(4)); a.set_xticklabels([o[1] for o in order],rotation=25,ha='right',fontsize=8)
    a.set_ylabel('HMM3 - iid mixture3 (bits/trial)'); a.set_title(t)
plt.tight_layout(); plt.savefig('history_baselines/paper/fig2_calibration.png',dpi=200)
print(real.agg(['mean','median']).round(4)); print(s2share.describe().round(3)); print(mixshare.describe().round(3))
