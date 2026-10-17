import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org', username='intbrainlab', password='international', silent=True)
one.load_cache()
d = pd.read_parquet('data/trials_cohort.parquet')
c = pd.read_csv('results/cohort.csv')
s = one._cache.sessions.reset_index()
s = s[s.subject.isin(c.subject) & s.task_protocol.str.contains('biasedChoiceWorld|ephysChoiceWorld')]
miss = s[~s.id.isin(d.session.unique())]
def get(a):
    eid,sub,date=a
    try:
        o = one.load_object(eid,'trials',collection='alf')
        t = pd.DataFrame({k:np.asarray(o[k]) for k in ['choice','contrastLeft','contrastRight','probabilityLeft','feedbackType']})
        t['subject']=sub; t['session']=eid; t['date']=str(date); t['trial']=np.arange(len(t))
        return t
    except Exception as e:
        return str(e)[:80]
with ThreadPoolExecutor(8) as ex: res = list(ex.map(get, zip(miss.id,miss.subject,miss.date)))
ok=[r for r in res if isinstance(r,pd.DataFrame)]; bad=[r for r in res if not isinstance(r,pd.DataFrame)]
print('recovered', len(ok), 'still failed', len(bad), set(bad))
d = pd.concat([d]+ok); d.to_parquet('data/trials_cohort.parquet')
print('total trials', len(d), 'sessions', d.session.nunique())
