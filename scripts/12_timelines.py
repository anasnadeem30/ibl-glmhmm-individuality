import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
F={(f['subject'],f['K']):f for f in pickle.load(open('data/full_fits.pkl','rb'))}
D=pd.read_parquet('data/features.parquet')
cols=['#2a9d8f','#e9c46a','#e76f51']
fig,axes=plt.subplots(2,2,figsize=(14,6.5))
for j,s in enumerate(['KS019','CSHL_014']):
    f=F[(s,3)]; d=D[D.subject==s].reset_index(drop=True); post=f['post']; o=np.argsort(-f['W'][:,0]); P=post[:,o]
    sess=d.session.unique(); sid=sess[len(sess)//2]; ix=np.where(d.session.values==sid)[0]
    ax=axes[0,j]; ax.stackplot(np.arange(len(ix)),P[ix].T,colors=cols,alpha=.85,labels=[f'state {r+1}  (stim w={f["W"][o[r],0]:.1f}, bias={f["W"][o[r],3]:+.1f})' for r in range(3)])
    ch=d.right.values[ix]; ax.plot(np.arange(len(ix)),1.04+0.05*ch,'|',color='k',ms=4); ax.set_ylim(0,1.12)
    ax.set_title(f'{s}: one session, posterior P(state) per trial (ticks on top: choices)'); ax.set_xlabel('trial'); ax.legend(loc='lower left',fontsize=7)
    ax=axes[1,j]; d['sess_i']=pd.factorize(d.session)[0]
    occ=pd.DataFrame(P).groupby(d.sess_i.values).mean(); ax.stackplot(occ.index,occ.values.T,colors=cols,alpha=.85)
    ax.set_title(f'{s}: fraction of trials in each state, session by session'); ax.set_xlabel('session number'); ax.set_ylim(0,1)
plt.tight_layout(); plt.savefig('figures/state_timelines.png',dpi=130)
