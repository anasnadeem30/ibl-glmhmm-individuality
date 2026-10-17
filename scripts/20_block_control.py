import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pickle, numpy as np, pandas as pd
F=pickle.load(open('data/full_fits.pkl','rb')); D=pd.read_parquet('data/features.parquet')
rows=[]
for f in F:
    d=D[D.subject==f['subject']]; pl=d.probabilityLeft.values; P=f['post']; W=f['W']
    for k in range(f['K']):
        b=W[k,3]; occ_r=P[pl==0.2,k].mean(); occ_l=P[pl==0.8,k].mean()     # pl=0.2: stimulus mostly on RIGHT
        rows.append(dict(subject=f['subject'],K=f['K'],state=k,stim=W[k,0],bias=b,occ=P[:,k].mean(),occ_right_block=occ_r,occ_left_block=occ_l,
                         aligned=(occ_r-occ_l)*np.sign(b)))
R=pd.DataFrame(rows); R['biased']=(R.bias.abs()>1)&(R.stim<10)
R.to_csv('results/block_control_states.csv',index=False)
for K in (3,4):
    r=R[(R.K==K)&R.biased]
    print(f'K={K}: low-stimulus side-biased states: {len(r)} (in {r.subject.nunique()} mice); '
          f'occupancy when block favors the bias side minus when it favors the other side: mean {r.aligned.mean():+.3f}, median {r.aligned.median():+.3f}, '
          f'positive in {(r.aligned>0).sum()}/{len(r)}; mean occupancy overall {r.occ.mean():.3f}')
    e=R[(R.K==K)&(R.stim>=10)]; print(f'   engaged-like states (stim>=10): {len(e)}; aligned effect mean {e.aligned.mean():+.3f} (bias-signed)')
# block modulation of the whole state mix per mouse: how much of the posterior varies with block?
for K in (3,4):
    ps=[]
    for f in F:
        if f['K']!=K: continue
        pl=D[D.subject==f['subject']].probabilityLeft.values; P=f['post']
        ps.append(np.abs(P[pl==0.2].mean(0)-P[pl==0.8].mean(0)).sum()/2)   # total variation distance between state mixes
    print(f'K={K}: total-variation distance between state occupancy in right-favoring vs left-favoring blocks: median {np.median(ps):.3f}')
