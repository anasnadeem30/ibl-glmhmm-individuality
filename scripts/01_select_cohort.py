import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org', username='intbrainlab', password='international', silent=True)
one.load_cache()
s = one._cache.sessions.reset_index()
s['date'] = pd.to_datetime(s['date'])
b = s[s.task_protocol.str.contains('biasedChoiceWorld|ephysChoiceWorld')]
g = b.groupby('subject').agg(n=('id','size'), first=('date','min'), last=('date','max'), lab=('lab','first'))
g['span_days'] = (g['last']-g['first']).dt.days
print(len(g), 'mice with biased/ephys sessions')
print(g.n.describe())
for k in (10,15,20,30): print('>=',k,'sessions:', (g.n>=k).sum())
print(g[g.n>=15].sort_values('n',ascending=False).head(15))
print(g[g.n>=15].lab.value_counts())
g.to_csv('results/mice_summary.csv')
