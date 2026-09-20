from pathlib import Path
import json
import time
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[2]
key=dict(l.split('=',1) for l in (ROOT/'.env').read_text().splitlines() if '=' in l)['DG_BERT_API_KEY']
cases=[
 ('sun-1',1,'星導は太陽ですか。',True),
 ('not-sun-2',2,'星導は太陽ですか。',False),
 ('mercury-5',5,'星導は水星ですか。',True),
 ('warning-40',40,'この資料は慢心に注意するよう述べていますか。',True),
 ('not-recommend-40',40,'この資料は慢心することを推奨していますか。',False),
 ('unknown-weather',40,'2026年9月19日の東京の気温は20度ですか。',None),
]
out=[]
for name,number,question,expected in cases:
 body={'state':{'number':number},'rag':{'number':number},'sources_only':True,'samples':'auto',
       'questions':{'answer':{'type':'noul','instructions':question}}}
 start=time.monotonic()
 req=Request('http://127.0.0.1:8011/v1/systemone',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urlopen(req,timeout=180) as r: value=json.load(r)
 answer=value['answers']['answer']
 actual=None if answer is None else answer['noul']>=.5
 out.append({'case':name,'expected':expected,'actual':actual,'pass':actual==expected,
             'seconds':round(time.monotonic()-start,3),'result':value})
(ROOT/'records/rag-acceptance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'passed':sum(r['pass'] for r in out),'total':len(out),'cases':[{k:r[k] for k in ['case','actual','pass','seconds']} for r in out]}))

raise SystemExit(0 if all(r["pass"] for r in out) else 1)
