import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org', username='intbrainlab', password='international', silent=True)
one.load_cache()
g = pd.read_csv('results/mice_summary.csv')
c = g[g.n>=30].sort_values('n',ascending=False).groupby('lab').head(5)
print(len(c), 'mice;', c.n.sum(), 'sessions'); print(c.lab.value_counts().to_dict())
c.to_csv('results/cohort.csv', index=False)
s = one._cache.sessions.reset_index()
s = s[s.subject.isin(c.subject) & s.task_protocol.str.contains('biasedChoiceWorld|ephysChoiceWorld')]
from concurrent.futures import ThreadPoolExecutor
def get(a):
    eid,sub,date=a
    try:
        t = one.load_dataset(eid,'_ibl_trials.table.pqt').reset_index(drop=True)
        t = t[['choice','contrastLeft','contrastRight','probabilityLeft','feedbackType']].copy()
        t['subject']=sub; t['session']=eid; t['date']=str(date); t['trial']=np.arange(len(t))
        return t
    except Exception as e:
        return None
with ThreadPoolExecutor(8) as ex: res = list(ex.map(get, zip(s.id,s.subject,s.date)))
rows=[r for r in res if r is not None]; fail=len(res)-len(rows)
d = pd.concat(rows); d.to_parquet('data/trials_cohort.parquet')
print('trials:', len(d), 'sessions ok:', d.session.nunique(), 'failed:', fail)
