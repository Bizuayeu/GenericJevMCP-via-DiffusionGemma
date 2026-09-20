"""Local real-model acceptance checks. Credentials are read only from model-local env files."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[2]
def keys(root):
    return dict(line.split('=',1) for line in (root/'.env').read_text().splitlines() if '=' in line)
gemma_key=keys(ROOT.parent/'gemma-nvfp4')['VLLM_API_KEY']
dg_keys=keys(ROOT)
records=[]
def post(port,route,body,key):
    started=time.monotonic()
    req=Request(f'http://127.0.0.1:{port}'+route,data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=180) as response:
            data=json.load(response)
    except HTTPError as error:
        raise RuntimeError(str(error)+': '+error.read(2000).decode(errors='replace')) from error
    return data,round(time.monotonic()-started,3)
def generation(port,model,key,label):
    body={'model':model,'messages':[{'role':'user','content':'雨上がりの庭について、日本語で三文を書いてください。'}],
          'max_tokens':256,'temperature':.4,'seed':42,'chat_template_kwargs':{'enable_thinking':False}}
    if port==8010:
        body.pop('seed')
        body.pop('temperature')
    data,seconds=post(port,'/v1/chat/completions',body,key)
    text=data['choices'][0]['message']['content']
    assert text and any('\u3040'<=c<='\u30ff' for c in text)
    item={'case':label,'seconds':seconds,'text':text,'usage':data.get('usage'),'finish_reason':data['choices'][0]['finish_reason']}
    records.append(item)
    return item

def decision(number,label,samples=1):
    body={'state':f'数霊{number}の説明を参照して判断してください。','rag':{'number':number},'sources_only':True,'samples':samples,
          'questions':{'supported':{'type':'noul','instructions':'この資料は指定された数霊の説明ですか。'},
                       'topic':{'type':'choice','instructions':'この資料の主題は何ですか。',
                                'criteria':{'数霊の象意':None,'料理の手順':None,'交通案内':None}},
                       'explicit':{'type':'score','instructions':'指定された数霊の説明は、どの程度直接書かれていますか。',
                                   'criteria':['記載なし','間接的','直接明記']}}}
    data,seconds=post(8011,'/v1/systemone',body,dg_keys['DG_BERT_API_KEY'])
    assert data['answers']['supported']['noul'] >= .5
    assert data['answers']['topic']['choice'] == '数霊の象意'
    assert max(data['answers']['explicit']['probabilities'], key=data['answers']['explicit']['probabilities'].get) == '2'
    assert any(source['title'] == f'数霊 {number}' for source in data['sources'])
    item={'case':label,'seconds':seconds,'result':data}
    records.append(item)
    return item

try:
    generation(8010,'dgemma',dg_keys['VLLM_API_KEY'],'diffusion-cold')
    generation(8010,'dgemma',dg_keys['VLLM_API_KEY'],'diffusion-warm')
    decision(40,'rag-40-cold')
    decision(40,'rag-40-warm')
    decision(5,'rag-5-auto','auto')
    empty={'state':'xyzqv nonexistent symbol','rag':{'query':'xyzqv'},'questions':{'q':{'type':'noul'}}}
    data,seconds=post(8011,'/v1/systemone',empty,dg_keys['DG_BERT_API_KEY'])
    assert data['abstention']=='no_retrieval_match' and data['diagnostics']['reads']==0
    records.append({'case':'rag-no-match','seconds':seconds,'result':data})
    tool={'type':'function','function':{'name':'add','description':'Add two integers.',
          'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}
    body={'model':'gemma-nvfp4','messages':[{'role':'user','content':'必ずaddツールを使って2と3を足してください。'}],
          'tools':[tool],'tool_choice':'auto','max_tokens':256,'temperature':0,
          'chat_template_kwargs':{'enable_thinking':False}}
    data,seconds=post(8888,'/v1/chat/completions',body,gemma_key)
    message=data['choices'][0]['message']
    call=message['tool_calls'][0]
    assert call['function']['name']=='add'
    args=json.loads(call['function']['arguments'])
    assert args=={'a':2,'b':3}
    body['messages'] += [message,{'role':'tool','tool_call_id':call['id'],'content':'5'}]
    final,end_seconds=post(8888,'/v1/chat/completions',body,gemma_key)
    assert '5' in final['choices'][0]['message']['content']
    records.append({'case':'gemma-tool-roundtrip','seconds':seconds+end_seconds,'result':final['choices'][0]['message']})
    for batch in range(3):
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(generation,8888,'gemma-nvfp4',gemma_key,f'mixed-{batch}-gemma'),
                  pool.submit(generation,8010,'dgemma',dg_keys['VLLM_API_KEY'],f'mixed-{batch}-diffusion'),
                  pool.submit(decision,31,f'mixed-{batch}-rag31'),
                  pool.submit(decision,40,f'mixed-{batch}-rag40')]
            for job in jobs:
                job.result()
    status='complete'
except Exception as error:
    status='failed'
    records.append({'case':'failure','type':type(error).__name__,'message':str(error)})
finally:
    out={'status':status,'time_utc':datetime.now(timezone.utc).isoformat(),'cases':records}
    path=ROOT/'records'/'acceptance.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':status,'cases':len(records),'record':str(path)},ensure_ascii=False))

sys.exit(0 if status=="complete" else 1)
