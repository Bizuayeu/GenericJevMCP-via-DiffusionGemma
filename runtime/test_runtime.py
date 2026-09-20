"""CPU test: mixed logprob widths must concatenate without changing values."""
import ast
from collections import namedtuple
from pathlib import Path
import importlib.util
import torch
import torch.nn.functional as F

site = Path(importlib.util.find_spec('vllm').origin).parent
source = (site / 'model_executor/models/diffusion_gemma.py').read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_concat_logprob_stashes')
Row = namedtuple('Row', 'logprob_token_ids logprobs selected_token_ranks cu_num_generated_tokens', defaults=[None])
scope = {'torch': torch, 'F': F, 'LogprobsTensors': Row}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream-helper>', 'exec'), scope)
parts = [Row(torch.arange(9).reshape(3,3), -torch.arange(9).reshape(3,3).float(), torch.zeros(3)),
         Row(torch.arange(22).reshape(2,11), -torch.arange(22).reshape(2,11).float(), torch.zeros(2))]
out = scope['_concat_logprob_stashes'](parts, [0,3])
assert out.logprobs.shape == (5,11)
assert torch.equal(out.logprobs[:3,:3], parts[0].logprobs)
assert torch.isneginf(out.logprobs[:3,3:]).all()
assert not out.logprob_token_ids[:3,3:].any()
assert torch.equal(out.logprobs[3:], parts[1].logprobs)
assert out.cu_num_generated_tokens == [0,3]
print('mixed-width logprob CPU regression: PASS')
