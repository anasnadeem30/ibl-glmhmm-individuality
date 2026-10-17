import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, pickle
r=pd.read_csv('results/replication_5050_cv.csv'); m=r.groupby(['subject','K']).agg(ll=('test_ll','sum'),n=('n_test','sum')).reset_index(); m['bits']=m.ll/m.n/np.log(2)
P=m.pivot(index='subject',columns='K',values='bits'); P.to_csv('results/replication_5050_summary.csv')
print('mean held-out bits/trial by K:',P.mean().round(4).to_dict()); print('gain vs K=1:',(P.sub(P[1],axis=0)).mean().round(4).to_dict())
print('gain vs previous K:',P.diff(axis=1).mean().round(4).dropna().to_dict())
print('K=3 beats K=1 in',int((P[3]>P[1]).sum()),'/',len(P),'mice; K=2 beats K=1:',int((P[2]>P[1]).sum()),'; K=3 beats K=2:',int((P[3]>P[2]).sum()),'; K=4 beats K=3:',int((P[4]>P[3]).sum()))
print('best K per mouse:',P.idxmax(axis=1).value_counts().sort_index().to_dict())
F=pickle.load(open('data/replication_5050_full.pkl','rb')); rows=[]
for f in F:
    o=np.argsort(-f['W'][:,0]); W=f['W'][o]; occ=f['occ'][o]; dw=1/(1-np.diag(f['A'])[o]+1e-9)
    rows.append(dict(subject=f['subject'],eng_stim=W[0,0],eng_occ=occ[0],dwell_eng=dw[0],dwell_b1=dw[1],dwell_b2=dw[2],b1_bias=W[1,1],b2_bias=W[2,1],stim1=W[1,0],stim2=W[2,0],n=f['n']))
S=pd.DataFrame(rows); S.to_csv('results/replication_5050_states.csv',index=False)
print('K=3 full-data (50/50 trials only), medians over mice: engaged occupancy %.2f; dwell engaged %.1f, other two %.1f / %.1f; engaged stim weight %.1f; other states stim weight %.1f / %.1f'%(S.eng_occ.median(),S.dwell_eng.median(),S.dwell_b1.median(),S.dwell_b2.median(),S.eng_stim.median(),S.stim1.median(),S.stim2.median()))
print('mice whose two non-engaged states have opposite-sign bias (left vs right):',int((np.sign(S.b1_bias)!=np.sign(S.b2_bias)).sum()),'/',len(S))
print('trials per mouse median',S.n.median())
