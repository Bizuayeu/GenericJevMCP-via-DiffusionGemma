"""Call the stack locally, or through authenticated SSH without exporting API keys."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]

def receive(payload):
    env=dict(line.split('=',1) for line in (ROOT/'.env').read_text().splitlines() if '=' in line and not line.startswith('#'))
    if payload['service']=='dg-bert':
        port,route,key=8011,'/v1/systemone',env['DG_BERT_API_KEY']
    elif payload['service']=='gemma':
        port,route,key=8888,'/v1/chat/completions',env['GEMMA_API_KEY']
    elif payload['service']=='diffusion':
        port,route,key=8010,'/v1/chat/completions',env['VLLM_API_KEY']
    else:
        raise ValueError('Unknown service')
    req=Request(f'http://127.0.0.1:{port}'+route,data=json.dumps(payload['body']).encode(),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urlopen(req,timeout=180) as response:
        return json.load(response)

def render_decision(value):
    """Present API values without another model call or probabilistic rewriting."""
    lines=[]
    for qid, answer in value['answers'].items():
        if answer is None:
            lines.append(f'{qid}: 保留')
            continue
        kind=answer['type']
        if kind=='choice':
            selected=answer['choice']
        elif kind=='noul':
            selected='yes' if answer['noul']>=.5 else 'no'
        else:
            selected=f"{answer['score']:.3f}（0始まりの期待値）"
        lines.append(f'{qid}: {selected}')
        legend=answer.get('legend',{})
        lines.append(' / '.join(f'{legend.get(name,name)}: {prob:.2%}'
                               for name,prob in answer['probabilities'].items()))
    if value.get('abstention'):
        lines.append('保留理由: '+value['abstention'])
    for source in value.get('sources',[]):
        lines.append('参照: '+source.get('source_url',source.get('title',source.get('id',''))))
    for source in value.get('input_sources',[]):
        lines.append('参照: '+source)
    lines.append(f"所要時間（全体）: {value['elapsed_seconds']:.3f} 秒")
    inference=value.get('diagnostics',{}).get('elapsed_seconds')
    if inference is not None:
        lines.append(f'判定時間: {inference:.3f} 秒')
    else:
        lines.append('判定時間: 未計測')
    return '\n'.join(lines)

def main():
    start=time.monotonic()
    config_path=ROOT/'client-config.json'
    config=json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.exists() else {}
    p=argparse.ArgumentParser()
    p.add_argument('--ssh-config',type=Path,
                   default=os.environ.get('JEV_SSH_CONFIG',config.get('ssh_config')))
    p.add_argument('--ssh-host',default=os.environ.get('JEV_SSH_HOST',config.get('ssh_host','spark')))
    p.add_argument('--remote-root',default=os.environ.get('JEV_REMOTE_ROOT',config.get('remote_root','/opt/GenericJevMCP')))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('receive')
    gen=sub.add_parser('generate')
    gen.add_argument('--model',choices=['gemma','diffusion'],default='gemma')
    gen.add_argument('--prompt',required=True)
    gen.add_argument('--max-tokens',type=int,default=512)
    decide=sub.add_parser('decide')
    decide.add_argument('--number',type=int)
    decide.add_argument('--query')
    decide.add_argument('--choices',nargs='+',help='Ordered choice names for a single question')
    state=decide.add_mutually_exclusive_group()
    state.add_argument('--state',help='Optional additional text; model knowledge remains enabled')
    state.add_argument('--state-file',type=Path,action='append',help='Read a UTF-8 text file with provenance; repeat for multiple files')
    decide.add_argument('--format',choices=['json','text'],default='json',help='JSON or ready-to-display decisions with timing')
    decide.add_argument('--image',type=Path,help='Additional PNG/JPEG image (one, at most 4 MiB)')
    source=decide.add_mutually_exclusive_group(required=True)
    source.add_argument('--question')
    source.add_argument('--request',type=Path,help='UTF-8 JSON body for /v1/systemone')
    source.add_argument('--request-json',help='Inline JSON body, or - to read stdin (recommended for Windows shells)')
    a=p.parse_args()
    if a.command=='receive':
        payload=json.load(sys.stdin)
    elif a.command=='generate':
        payload={'service':a.model,'body':{'model':'gemma-nvfp4' if a.model=='gemma' else 'dgemma',
                 'messages':[{'role':'user','content':a.prompt}],'max_tokens':a.max_tokens,
                 'chat_template_kwargs':{'enable_thinking':False}}}
    elif a.request is not None or a.request_json is not None:
        if any(value is not None for value in (a.number,a.query,a.choices,a.state)):
            p.error('--request cannot be combined with --number, --query, --choices or --state')
        from .decision import validate
        from .server import strict_pairs
        def invalid_constant(value):
            raise ValueError('Non-finite JSON number')
        try:
            text=a.request.read_text(encoding='utf-8-sig') if a.request is not None else (sys.stdin.read() if a.request_json=='-' else a.request_json)
            body=json.loads(text.lstrip('\ufeff'),
                            object_pairs_hook=strict_pairs,parse_constant=invalid_constant)
            validate(body)
        except (OSError,ValueError,TypeError) as error:
            p.error('Invalid request file: '+str(error))
        payload={'service':'dg-bert','body':body}
    else:
        rag={}
        if a.number is not None: rag['number']=a.number
        if a.query is not None: rag['query']=a.query
        payload={'service':'dg-bert','body':{'state':a.state if a.state is not None else a.query or '', 'rag':rag if rag else False,
                 'questions':{'answer':{'type':'noul','instructions':a.question}}}}
        if a.choices is not None:
            if len(set(a.choices)) != len(a.choices):
                p.error('Duplicate choices')
            payload['body']['questions']['answer']={
                'type':'choice','instructions':a.question,
                'criteria':{name:None for name in a.choices}}
        from .decision import validate
        try:
            validate(payload['body'])
        except (ValueError,TypeError) as error:
            p.error(str(error))
    if a.command=='decide' and a.state_file:
        if 'state' in payload['body'] and (a.request is not None or a.request_json is not None):
            p.error('state specified both in request and --state-file')
        try:
            sources=[]
            for path in a.state_file:
                # The API state ceiling also bounds local reads; never silently truncate.
                with path.open(encoding='utf-8-sig') as stream:
                    text=stream.read(32769)
                sources.append({'source':str(path.resolve()),'text':text})
                validate({**payload['body'],'state':{'sources':sources}})
            payload['body']['state']={'sources':sources}
        except (OSError,ValueError,TypeError) as error:
            p.error('Invalid state file: '+str(error))
    if a.command=='decide' and a.image is not None:
        from .media import encode_image, MAX_IMAGE_BYTES
        if 'image' in payload['body']:
            p.error('image specified both in request and --image')
        try:
            with a.image.open('rb') as source:
                payload['body']['image']=encode_image(source.read(MAX_IMAGE_BYTES+1))
        except (OSError,ValueError) as error:
            p.error('Invalid image: '+str(error))
    if a.ssh_config and a.command!='receive':
        import re
        if not re.fullmatch(r'[\w.-]+',a.ssh_host) or not re.fullmatch(r'/[\w/.-]+',a.remote_root):
            p.error('Invalid SSH host or remote root')
        args=['ssh','-F',str(a.ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10',
              a.ssh_host,'cd '+a.remote_root+' && python3 -m jev.client receive']
        result=subprocess.run(args,input=json.dumps(payload,ensure_ascii=False).encode(),
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=200)
        if result.returncode:
            raise RuntimeError('SSH request failed: '+result.stderr.decode(errors='replace')[:500])
        value=json.loads(result.stdout)
    else:
        value=receive(payload)
    if a.command=='decide':
        value['elapsed_seconds']=time.monotonic()-start
        if a.state_file:
            value['input_sources']=[str(path.resolve()) for path in a.state_file]
        value['timing']={'total_seconds':value['elapsed_seconds'],
                         'decision_seconds':value.get('diagnostics',{}).get('elapsed_seconds')}
        value['display']=render_decision(value)
        if a.format=='text':
            print(value['display'])
            return
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
