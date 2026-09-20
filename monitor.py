"""Observe only this stack; stop its GPU containers below the upstream 12-GiB reserve."""
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'state'
STATE.mkdir(exist_ok=True)
names=['vllm-gemma26-nvfp4','vllm-diffusiongemma']
expected=os.environ.get('JEV_IMAGE','sha256:b3a86ba00fb26aac2807a49ad8d60377373e2725d76f52dc1d6b0bf30c47a5a0')
history=deque(maxlen=1800)  # Last hour at a two-second interval.
while True:
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(mem['MemAvailable'].split()[0])/1024**2
    row={'time_utc':datetime.now(timezone.utc).isoformat(),'available_gib':round(available,3)}
    history.append(row)
    if available<12:
        stopped=[]
        for name in names:
            p=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
            if p.returncode:
                continue
            obj=json.loads(p.stdout)[0]
            if obj['Config']['Image']==expected and (obj['Config'].get('Labels') or {}).get('llllm.identity'):
                subprocess.run(['docker','stop','--time','5',obj['Id']],check=True,capture_output=True)
                stopped.append(name)
        row['stopped']=stopped
        (STATE/'memory-stop.json').write_text(json.dumps(row,indent=2))
        print('Memory reserve crossed; stopped own GPU containers.',flush=True)
    tmp=STATE/'memory.tmp'
    tmp.write_text(json.dumps({'latest':row,'minimum_gib':min(r['available_gib'] for r in history),'samples':list(history)}))
    tmp.replace(STATE/'memory.json')
    if available<12:
        break
    time.sleep(2)
