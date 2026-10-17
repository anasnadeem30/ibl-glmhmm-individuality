import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = pd.read_parquet('data/trials_cohort.parquet')
d['sc'] = d.contrastRight.fillna(0) - d.contrastLeft.fillna(0)   # signed contrast, + = stimulus on right
print('choice counts', d.choice.value_counts().to_dict(), '| no-go share', round((d.choice==0).mean(),4))
# which choice sign is rewarded for right-side stimuli at high contrast?
r = d[(d.sc>=0.5)&(d.feedbackType==1)]
print('rewarded choice when stim right (high contrast):', r.choice.value_counts().to_dict())
sign = r.choice.mode()[0]            # choice value that means "toward right stimulus"
d = d[d.choice!=0].copy()
d['right'] = (d.choice==sign).astype(int)
d.to_parquet('data/trials_clean.parquet')
print('trials after dropping no-go:', len(d), '| contrast levels:', sorted(d.sc.unique()))
levels = sorted(d.sc.unique())
fig, ax = plt.subplots(1,3,figsize=(15,4.2),sharey=True)
for a,(p,title) in zip(ax[:2],[(0.5,'Unbiased block (50/50)'),(None,'Biased blocks')]):
    if p: sub=d[d.probabilityLeft==p]; a.set_title(title)
    else: sub=d[d.probabilityLeft!=0.5]; a.set_title(title)
    pm = sub.groupby(['subject','sc']).right.mean().unstack(0)
    a.plot(pm.index, pm.values, color='gray', alpha=.3, lw=.8)
    a.plot(pm.index, pm.mean(1), color='k', lw=2.5)
    a.axhline(.5,ls=':',c='k'); a.axvline(0,ls=':',c='k'); a.set_xlabel('signed contrast (+ = right)')
ax[0].set_ylabel('P(choose right)')
a=ax[2]; a.set_title('Pooled by block type')
for p,col in [(0.2,'tab:red'),(0.5,'k'),(0.8,'tab:blue')]:
    pm = d[d.probabilityLeft==p].groupby(['subject','sc']).right.mean().unstack(0)
    m, se = pm.mean(1), pm.sem(1)
    a.errorbar(m.index, m.values, se.values, color=col, lw=2, label=f'P(left)={p}')
a.legend(); a.axhline(.5,ls=':',c='k'); a.axvline(0,ls=':',c='k'); a.set_xlabel('signed contrast (+ = right)')
plt.tight_layout(); plt.savefig('figures/psychometric_cohort.png',dpi=130)
# per-mouse summary
acc = d[d.sc!=0].assign(ok=lambda x:(x.feedbackType==1)).groupby('subject').ok.mean()
print('per-mouse accuracy (non-zero contrast): min %.2f median %.2f max %.2f'%(acc.min(),acc.median(),acc.max()))
print('trials/session median', d.groupby('session').size().median())
