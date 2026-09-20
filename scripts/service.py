"""Operate the fixed DiffusionGemma backend and the CPU-only DG-BERT adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'state'
IMAGE=os.environ.get('JEV_IMAGE','sha256:b3a86ba00fb26aac2807a49ad8d60377373e2725d76f52dc1d6b0bf30c47a5a0')
BACKEND='vllm-diffusiongemma'
ADAPTER='dg-bert'

def run(*args):
    p=subprocess.run(args,text=True,capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    return p.stdout.strip()

def init(download_status):
    value=json.loads(Path(download_status).read_text())
    entry=value['models']['diffusion']
    lock=json.loads((ROOT/'models.lock.json').read_text())['diffusion']
    if value['status']!='complete' or entry['status']!='complete' or entry['revision']!=lock['revision']:
        raise RuntimeError('Download must match the fixed model revision')
    STATE.mkdir(exist_ok=True)
    (STATE/'download-status.json').write_text(json.dumps({'status':'complete','models':{'diffusion':entry}},indent=2))
    if not (ROOT/'.env').exists():
        fd=os.open(ROOT/'.env',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w') as out:
            out.write('VLLM_API_KEY='+secrets.token_urlsafe(32)+'\n')
            out.write('DG_BERT_API_KEY='+secrets.token_urlsafe(32)+'\n')

def backend_args(model):
    return [model,'--served-model-name','dgemma','--host','127.0.0.1','--port','8010',
            '--dtype','bfloat16','--quantization','modelopt','--kv-cache-dtype','bfloat16',
            '--gpu-memory-utilization','0.32','--kv-cache-memory',str(6*1024**3),
            '--max-model-len','131072','--max-num-seqs','4','--max-num-batched-tokens','2048',
            '--limit-mm-per-prompt','{"image":1,"video":0}','--attention-backend','TRITON_ATTN',
            '--enable-prefix-caching','--enable-chunked-prefill','--async-scheduling',
            '--max-logprobs','128','--diffusion-config','{"canvas_length":256,"max_denoising_steps":48}',
            '--override-generation-config','{"max_new_tokens":null}',
            '--enable-auto-tool-choice','--tool-call-parser','gemma4','--reasoning-parser','gemma4']

def start(kind):
    data=json.loads((STATE/'download-status.json').read_text())['models']['diffusion']
    cache=Path.home()/'.cache/huggingface'
    model='/hf/'+Path(data['snapshot']).relative_to(cache).as_posix()
    name=BACKEND if kind=='backend' else ADAPTER
    corpus=ROOT/'corpus/surei.jsonl'
    source_hash=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in
        (['scripts/service.py','models.lock.json'] if kind=='backend' else ['jev/server.py','jev/decision.py','jev/media.py','jev/corpus.py']+(['corpus/surei.jsonl'] if corpus.exists() else [])))).hexdigest()
    identity=hashlib.sha256(json.dumps([IMAGE,model,kind,source_hash]).encode()).hexdigest()
    exists=name in run('docker','ps','-a','--format','{{.Names}}').splitlines()
    if exists:
        info=json.loads(run('docker','inspect',name))[0]
        if (info['Config'].get('Labels') or {}).get('llllm.identity')!=identity:
            raise RuntimeError('Existing settings differ; stop and rename the saved container first.')
        if not info['State']['Running']:
            print(run('docker','start',name))
        else:
            print(name+' already running')
        return
    if kind=='backend':
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        available=int(mem['MemAvailable'].split()[0])/1024**2
        # Upstream bound: 19 GiB weights + 6 KV + 4*256 rows*262144*4*10 transient + 12 host.
        required=19+6+4*256*262144*4*10/1024**3+12
        if available<required:
            raise RuntimeError(f'Need {required:.1f} GiB available, have {available:.1f}')
        jit=STATE/'runtime-cache'
        jit.mkdir(exist_ok=True)
        command=['docker','run','-d','--name',name,'--label','llllm.identity='+identity,
                 '--gpus','all','--network','host','--memory','60g','--memory-swap','60g','--shm-size','2g',
                 '--env-file',str(ROOT/'.env'),'-e','DG_GPU_MEMORY_CAP=0.38',
                 '-e','HF_HUB_OFFLINE=1','-e','TRITON_CACHE_DIR=/cache/triton',
                 '-e','TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor','-e','XDG_CACHE_HOME=/cache/xdg',
                 '-e','VLLM_CACHE_ROOT=/cache/vllm','-v',str(jit)+':/cache',
                 '-v',str(cache)+':/hf:ro',IMAGE,*backend_args(model)]
    else:
        command=['docker','run','-d','--name',name,'--label','llllm.identity='+identity,
                 '--network','host','--memory','1g','--memory-swap','1g',
                 '--env-file',str(ROOT/'.env'),'-e','NVIDIA_VISIBLE_DEVICES=void',
                 '-v',str(ROOT)+':/app:ro','-v',str(cache)+':/hf:ro',
                 '--workdir','/app','--entrypoint','python3',IMAGE,'-m','jev.server','--tokenizer',model+'/tokenizer.json',
                 *(['--corpus','/app/corpus/surei.jsonl'] if corpus.exists() else [])]
    print(run(*command))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['init','start-backend','start-adapter','stop-backend','stop-adapter'])
    p.add_argument('--download-status')
    a=p.parse_args()
    if a.action=='init':
        if not a.download_status: p.error('--download-status is required')
        init(a.download_status)
    elif a.action.startswith('start-'):
        start(a.action.split('-')[1])
    else:
        print(run('docker','stop',BACKEND if a.action.endswith('backend') else ADAPTER))
