"""Verify and install a pinned Python-only vLLM overlay."""
import compileall
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
version = importlib.metadata.version('vllm')
if '+g' + manifest['base_commit'][:9] not in version:
    raise RuntimeError('Unexpected base vLLM version: ' + version)
site = Path(importlib.util.find_spec('vllm').origin).parent
for name, hashes in manifest['files'].items():
    dst = site / Path(name).relative_to('vllm')
    src = root / 'overlay' / name
    if hashlib.sha256(src.read_bytes()).hexdigest() != hashes['fork_sha256']:
        raise RuntimeError('Overlay hash mismatch: ' + name)
    if hashes['base_sha256'] is None:
        if dst.exists():
            raise RuntimeError('New overlay path already exists: ' + name)
    elif hashlib.sha256(dst.read_bytes()).hexdigest() != hashes['base_sha256']:
        raise RuntimeError('Base file mismatch: ' + name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

model = site / 'model_executor/models/diffusion_gemma.py'
text = model.read_text()
# Adapted from razorback16/vllm commit 9bbf741 (Apache-2.0).
helper = '''
def _concat_logprob_stashes(parts, cu_num_generated_tokens):
    width = max(p.logprob_token_ids.shape[1] for p in parts)
    def pad(t, value):
        return F.pad(t, (0, width - t.shape[1]), value=value)
    return LogprobsTensors(
        logprob_token_ids=torch.cat([pad(p.logprob_token_ids, 0) for p in parts]),
        logprobs=torch.cat([pad(p.logprobs, float("-inf")) for p in parts]),
        selected_token_ranks=torch.cat([p.selected_token_ranks for p in parts]),
        cu_num_generated_tokens=cu_num_generated_tokens,
    )

'''
anchor = '@torch.compile('
idx = text.index(anchor, text.index('def _compute_num_rejected'))
text = text[:idx] + helper + '\n' + 'torch._dynamo.config.recompile_limit = 64\n\n' + text[idx:]
old = '''            if parts_ids:
                logprobs_tensors = LogprobsTensors(
                    logprob_token_ids=torch.cat(parts_ids),
                    logprobs=torch.cat(parts_lp),
                    selected_token_ranks=torch.cat(parts_ranks),
                    cu_num_generated_tokens=cu_gen,
                )'''
new = '''            if parts_ids:
                logprobs_tensors = _concat_logprob_stashes(
                    [LogprobsTensors(logprob_token_ids=i, logprobs=p,
                                     selected_token_ranks=r)
                     for i, p, r in zip(parts_ids, parts_lp, parts_ranks)], cu_gen
                )'''
if text.count(old) != 1:
    raise RuntimeError('Mixed-logprobs patch anchor changed')
text = text.replace(old, new)
if '(soft_embeds * sc_keep).to(sc_embeds.dtype)' not in text:
    raise RuntimeError('Eager dtype fix absent')
model.write_text(text)

worker = site / 'v1/worker/gpu_worker.py'
text = worker.read_text()
anchor = '            torch.accelerator.set_device_index(self.device)\n'
if text.count(anchor) != 1:
    raise RuntimeError('Worker cap anchor changed')
text = text.replace(anchor, anchor + '''
            import os as _cap_os
            _cap = float(_cap_os.environ["DG_GPU_MEMORY_CAP"])
            if not 0 < _cap < 1:
                raise ValueError("Invalid DG_GPU_MEMORY_CAP")
            torch.cuda.set_per_process_memory_fraction(_cap, self.device)
''')
worker.write_text(text)
assert compileall.compile_dir(site, quiet=1, force=True)
print('Verified overlay and runtime fixes installed:', version)
