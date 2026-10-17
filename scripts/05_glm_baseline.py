import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
d = pd.read_parquet('data/trials_clean.parquet').sort_values(['subject','date','session','trial']).reset_index(drop=True)
# features (Ashwood-style): stimulus, previous choice, win-stay-lose-switch, bias (intercept)
y = d.right.values
pc = np.where(d.right==1,1,-1)                      # choice as +-1
g = d.groupby('session')
prev_c = g.right.shift(1).map({1:1,0:-1}).fillna(0).values
prev_r = g.feedbackType.shift(1).fillna(0).values   # +1 reward, -1 no reward
d['stim']=d.contrastRight.fillna(0)-d.contrastLeft.fillna(0)
d['prev']=prev_c; d['wsls']=prev_c*prev_r
d['prior']=(0.5-d.probabilityLeft)*2*(-1)          # +1 when right more likely (block prior); NOT given to the mouse directly
d.to_parquet('data/features.parquet')
def fit_eval(sub, cols):
    X=sub[cols].values; Y=sub.right.values; gr=sub.session.values
    ll=[];acc=[]
    for tr,te in GroupKFold(5).split(X,Y,gr):
        m=LogisticRegression(C=1e4,max_iter=500).fit(X[tr],Y[tr])
        p=np.clip(m.predict_proba(X[te])[:,1],1e-6,1-1e-6)
        ll.append(np.sum(Y[te]*np.log2(p)+(1-Y[te])*np.log2(1-p))); acc.append((m.predict(X[te])==Y[te]).sum())
    n=len(Y); return sum(ll)/n, sum(acc)/n
sets={'stim only':['stim'],'stim+prev':['stim','prev'],'stim+prev+wsls':['stim','prev','wsls'],'+block prior':['stim','prev','wsls','prior']}
rows=[]
for s,sub in d.groupby('subject'):
    # null model: bias only
    p0=sub.right.mean(); null=(p0*np.log2(p0)+(1-p0)*np.log2(1-p0))
    r={'subject':s,'null_bits':null}
    for k,c in sets.items(): r[k],r[k+'_acc']=fit_eval(sub,c)
    rows.append(r)
R=pd.DataFrame(rows).set_index('subject'); R.to_csv('results/glm_baseline_cv.csv')
print('mean held-out log-lik (bits/trial; higher=better, 0=perfect, -1=coin flip):')
print(R[['null_bits']+list(sets)].mean().round(4))
print('accuracy:', R[[k+'_acc' for k in sets]].mean().round(4).to_dict())
# full-data weights per mouse for the 3-history model
W=[]
for s,sub in d.groupby('subject'):
    m=LogisticRegression(C=1e4,max_iter=500).fit(sub[['stim','prev','wsls']].values,sub.right.values)
    W.append(dict(subject=s,stim=m.coef_[0][0],prev=m.coef_[0][1],wsls=m.coef_[0][2],bias=m.intercept_[0]))
W=pd.DataFrame(W).set_index('subject'); W.to_csv('results/glm_weights.csv'); print(W.describe().round(2).loc[['mean','std','min','max']])
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(R[list(sets)].T.values,color='gray',alpha=.4); ax[0].plot(R[list(sets)].mean().values,'k',lw=3)
ax[0].set_xticks(range(len(sets))); ax[0].set_xticklabels(sets,rotation=15); ax[0].set_ylabel('held-out log-lik (bits/trial)'); ax[0].set_title('Each mouse: adding history helps')
W.plot.box(ax=ax[1]); ax[1].set_title('Per-mouse GLM weights'); ax[1].axhline(0,ls=':',c='k')
plt.tight_layout(); plt.savefig('figures/glm_baseline.png',dpi=130)
