"""Authenticated local DG-BERT API: bounded decisions with optional archive retrieval."""
import argparse
from collections import defaultdict, deque
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

from .corpus import Corpus
from .decision import Engine, validate
from .media import BODY_LIMIT

RATE_PER_MINUTE=600  # Initial local ceiling; revisit against measured production traffic.

def execute(body, engine, corpus):
    schema=validate(body)
    seed=body.get('seed',42)
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    rag=body.get('rag',False)
    sources_only=body.get('sources_only',False)
    if type(sources_only) is not bool:
        raise ValueError('sources_only must be boolean')
    if rag is not False and not isinstance(rag,dict):
        raise ValueError('rag must be false or an object')
    evidence=[]
    if rag is not False:
        if set(rag)-{'query','number'}:
            raise ValueError('Unknown rag field')
        query=rag.get('query',json.dumps(schema['state'],ensure_ascii=False)+' '+' '.join(q['instructions'] for q in schema['questions']))
        if not isinstance(query,str) or len(query)>32768:
            raise ValueError('Invalid retrieval query')
        number=rag.get('number')
        if number is not None and (type(number) is not int or not 1<=number<=91):
            raise ValueError('number must be 1..91')
        if number is None:
            match=re.search(r'数霊\s*([0-9０-９]{1,2})(?![0-9０-９])',query)
            if match:
                number=int(match[1])
        evidence=corpus.search(query,number=number)
        if not evidence:
            return {'model':'dg-bert','answers':{q['id']:None for q in schema['questions']},
                    'sources':[],'abstention':'no_retrieval_match','diagnostics':{'reads':0}}
        schema['state']={'request':schema['state'],
                         'sources':[{'id':d['id'],'title':d['title'],'text':d['text']} for d in evidence]}
    if sources_only:
        for q in schema['questions']:
            q['instructions']='Use only the supplied text or image, without outside knowledge. '+q['instructions']
        schema['questions']=schema['questions']+[{
            'id':'_dg_supported','type':'noul',
            'instructions':'Can ALL the other questions be answered from the supplied text or image without outside knowledge or guessing? Both a supported YES and a supported NO count as answerable. An explicit fact contradicting a proposition is sufficient evidence to answer NO to that proposition. Say no here only when needed information is missing, not because another answer is negative.',
            'choices':[('yes',None),('no',None)],'labels':['yes','no']}]
        if schema['mode'] == 'separate':
            # The grounding gate sees the questions explicitly when its peers are absent.
            schema['questions'][-1]['instructions'] += '\nQuestions to assess: ' + json.dumps(
                [{'instructions': q['instructions'], 'alternatives': q['choices']}
                 for q in schema['questions'][:-1]], ensure_ascii=False)
    try:
        result=engine.decide(schema,seed)
    except (ValueError,KeyError,IndexError,TypeError) as error:
        raise RuntimeError('Invalid backend decision response') from error
    if sources_only:
        support=result['answers'].pop('_dg_supported')
        if support is None or support['noul']<.5:
            result['answers']={key:None for key in result['answers']}
            result['abstention']='insufficient_source_evidence'
    result['sources']=[{k:d[k] for k in ('id','title','source_url','archive_path','source_sha256','archived_at_source_timezone','images') if k in d} for d in evidence]
    result['model']='dg-bert'
    result['diagnostics']['input_flags']=['instruction_like_text'] if re.search(
        r'ignore.{0,30}(instruction|system)|system.prompt|API[_ ]?KEY|以前の指示|システムプロンプト',
        json.dumps(body,ensure_ascii=False),re.I) else []
    return result

def strict_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key]=value
    return result

def make_handler(engine,corpus,key,upstream_key):
    active=threading.BoundedSemaphore(2)  # Two reads + ordinary generation fit MAX_SEQS=4.
    requests=defaultdict(deque)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def send_json(self,status,value):
            data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            if any(secret.encode() in data for secret in (key,upstream_key)):
                status=500
                data=b'{"error":"output validation failed"}'
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path!='/health':
                return self.send_json(404,{'error':'unknown route'})
            try:
                with urlopen(engine.upstream+'/health',timeout=2) as response:
                    ready=response.status==200
            except Exception:
                ready=False
            self.send_json(200 if ready else 503,{'status':'ok' if ready else 'backend_unavailable'})
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+key):
                return self.send_json(401,{'error':'authentication required'})
            if self.path not in ('/v1/systemone','/v1/chat/completions'):
                return self.send_json(404,{'error':'unknown route'})
            now=time.monotonic()
            with lock:
                queue=requests[self.client_address[0]]
                while queue and queue[0]<=now-60:
                    queue.popleft()
                if len(queue)>=RATE_PER_MINUTE:
                    return self.send_json(429,{'error':'rate limit'})
                queue.append(now)
            if not active.acquire(blocking=False):
                return self.send_json(429,{'error':'server busy'})
            try:
                self.connection.settimeout(15)  # Local request-body deadline, not inference.
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=BODY_LIMIT:
                    return self.send_json(413,{'error':f'body must be 1..{BODY_LIMIT} bytes'})
                def invalid_constant(value):
                    raise ValueError('Non-finite JSON number')
                body=json.loads(self.rfile.read(length),object_pairs_hook=strict_pairs,parse_constant=invalid_constant)
                if self.path=='/v1/chat/completions':
                    msgs=body.get('messages')
                    if not isinstance(msgs,list) or len(msgs)!=2 or msgs[0].get('role')!='system' or msgs[1].get('role')!='user':
                        raise ValueError('Expected schema system message and state user message')
                    schema=json.loads(msgs[0]['content'],object_pairs_hook=strict_pairs,parse_constant=invalid_constant)
                    state=json.loads(msgs[1]['content'],object_pairs_hook=strict_pairs,parse_constant=invalid_constant)
                    body={**schema,'state':state}
                result=execute(body,engine,corpus)
                if self.path=='/v1/chat/completions':
                    result={'object':'chat.completion','model':'dg-bert','choices':[
                        {'index':0,'message':{'role':'assistant','content':json.dumps(result,ensure_ascii=False,allow_nan=False)},'finish_reason':'stop'}]}
                self.send_json(200,result)
            except (ValueError,TypeError,KeyError,AttributeError):
                self.send_json(422,{'error':'invalid request'})
            except (RuntimeError,URLError,TimeoutError):
                self.send_json(502,{'error':'backend decision failed'})
            except Exception:
                self.send_json(500,{'error':'internal error'})
            finally:
                active.release()
    return Handler

def main():
    from tokenizers import Tokenizer
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',default='http://127.0.0.1:8010')
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--corpus',help='Optional local corpus JSONL; omitted for knowledge/text/image decisions')
    p.add_argument('--port',type=int,default=8011)
    a=p.parse_args()
    key=os.environ['DG_BERT_API_KEY']
    upstream_key=os.environ['VLLM_API_KEY']
    if not key or not upstream_key:
        raise ValueError('Both API keys are required')
    engine=Engine(Tokenizer.from_file(a.tokenizer),a.upstream,upstream_key)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(engine,Corpus.load(a.corpus) if a.corpus else Corpus([]),key,upstream_key))
    print('DG-BERT ready on loopback:'+str(a.port),flush=True)
    server.serve_forever()

if __name__=='__main__':
    main()
