"""Download or verify the pinned snapshot and write service.py's local manifest."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def verify_snapshot(snapshot, lock):
    snapshot=Path(snapshot).absolute()
    cache=Path.home()/'.cache/huggingface'
    if not snapshot.is_relative_to(cache) or snapshot.name!=lock['revision']:
        raise ValueError('Snapshot must use the pinned revision under ~/.cache/huggingface')
    for entry in lock['files']:
        path=snapshot/entry['path']
        if not path.is_file() or path.stat().st_size!=entry['bytes']:
            raise ValueError('Missing or wrong-sized model file: '+entry['path'])
    return {'status':'complete','models':{'diffusion':{
        'status':'complete','revision':lock['revision'],'snapshot':str(snapshot)}}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--download',action='store_true')
    source.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    lock=json.loads((ROOT/'models.lock.json').read_text())['diffusion']
    snapshot=a.snapshot
    if a.download:
        from huggingface_hub import snapshot_download
        snapshot=snapshot_download(lock['model'],revision=lock['revision'],
            cache_dir=str(Path.home()/'.cache/huggingface/hub'))
    value=verify_snapshot(snapshot,lock)
    state=ROOT/'state'
    state.mkdir(exist_ok=True)
    (state/'download-status.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print('Verified pinned snapshot; run service.py init --download-status state/download-status.json')

if __name__=='__main__':
    main()
