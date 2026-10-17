import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, time
from glmhmm import GLMHMM
rng=np.random.default_rng(1)
Wt=np.array([[6.0,0.0,0.0],[0.5,0.0,2.0]])       # state0 = engaged (follows stim), state1 = right-biased, ignores stim (cols: stim, prev, bias)
A=np.array([[0.97,0.03],[0.04,0.96]])
X=[];Y=[];G=[];Z=[]
for s in range(30):
    z=rng.integers(2)
    for t in range(600):
        x=np.array([rng.choice([-1,-.25,-.125,-.0625,0,.0625,.125,.25,1]),0,1.0])
        p=1/(1+np.exp(-Wt[z]@x)); y=rng.random()<p
        X.append(x);Y.append(y);G.append(s);Z.append(z); z=rng.choice(2,p=A[z])
X=np.array(X);Y=np.array(Y,float);G=np.array(G);Z=np.array(Z)
for K in (1,2,3):
    t=time.time(); m=GLMHMM(K,seed=0).fit(X,Y,G)
    print(f'K={K} train LL/trial (bits) {m.train_ll/len(Y)/np.log(2):.4f}  iters {len(m.ll_trace)}  {time.time()-t:.1f}s')
    if K==2:
        print('weights\n',m.W.round(2)); print('A\n',m.A.round(3))
        post=m.posteriors(X,Y,G).argmax(1); acc=max((post==Z).mean(),(post!=Z).mean()); print('state decoding accuracy',round(acc,3))
        print('EM monotone:', bool(np.all(np.diff(m.ll_trace)>-1e-6)))
