# GenericJevMCP via DiffusionGemma

[日本語](README.ja.md) · [Validation](docs/validation.md) · [Provenance](NOTICE.md)

**Structured yes/no, choice, and score decisions through MCP, a terminal, or HTTP.** A CPU adapter reads bounded answer-token slots from a shared DiffusionGemma NVFP4 backend, returning candidate probabilities and measured elapsed time without generating an essay for each decision.

This is an experimental, self-hosted implementation of Jev-style decisions. The internal API name `dg-bert` is historical: **no BERT model is loaded**. The calling assistant is independent of the decision backend.

## Architecture

```text
MCP host / terminal → Python client → optional SSH → CPU adapter :8011
                                                        ↓
                                          DiffusionGemma / vLLM :8010
```

Text, optional evidence, or one image can be evaluated with multiple questions in a joint read or with isolated sequential questions. Ordinary generation on port 8010 and decisions share the same loaded weights. Decisions use one diffusion step per read; ordinary generation uses the denoising schedule.

## Requirements

| Component | Required environment / tested scope |
|---|---|
| GPU host | Linux ARM64, GB10, 128 GB unified memory; tested on MSI EdgeXpert (DGX Spark-class) |
| Runtime | NVIDIA-enabled Docker; pinned [base image and overlay](runtime/manifest.json), built with [Dockerfile](Dockerfile) |
| Weights | Exact NVIDIA DiffusionGemma 26B-A4B NVFP4 revision in [models.lock.json](models.lock.json), downloaded separately |
| Python | Python 3.10+ syntax; client/CPU tests verified on 3.13; install [requirements.txt](requirements.txt) |
| MCP host | Node.js 20+, Python on PATH or JEV_PYTHON, and npm ci |
| Remote connection | OpenSSH key authentication; APIs bind to loopback |

Provision storage for approximately 19 GB of model files, Docker layers, and runtime caches. Weights, credentials and private corpora are not distributed. Other GPU types, x86 hosts and upstream runtime versions are outside the tested deployment.

## GPU server setup

Run on the Linux GPU host. Review the [model card and model terms](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) before downloading. The code license does not replace model terms.

```sh
git clone https://github.com/Bizuayeu/GenericJevMCP-via-DiffusionGemma.git
cd GenericJevMCP-via-DiffusionGemma
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python prepare_model.py --download
# Or verify an existing snapshot:
# python prepare_model.py --snapshot /absolute/cache/path/snapshots/REVISION
python service.py init --download-status state/download-status.json

docker build -t generic-jev:local .
export JEV_IMAGE=$(docker image inspect generic-jev:local --format '{{.Id}}')
python service.py start-backend
# Wait until ready; first compilation can take longer than warm inference.
curl --fail http://127.0.0.1:8010/health
python service.py start-adapter
curl --fail http://127.0.0.1:8011/health
```

The preparation command downloads the pinned revision under ~/.cache/huggingface and verifies filenames/sizes. Initialization generates local credentials with mode 0600 without replacing an existing .env. No corpus is required.

Keep **JEV_IMAGE** set to your built image ID for service operations and monitoring. The source default is the original tested image ID, not an image already present on your machine. The build verifies the pinned base and overlay hashes.

Optionally run `python monitor.py` in another terminal with the same JEV_IMAGE. It stops matching managed GPU containers when host memory falls below the configured reserve; it is not an all-allocation OOM guarantee. Neither services nor monitor are registered for OS autostart. Stop with `python service.py stop-adapter` / `stop-backend`. When settings change, stop and retain/rename the old container before recreating it: the service refuses silent configuration replacement.

## Three decision types

```sh
python client.py decide --question 'Does water contain hydrogen?' --format text
python client.py decide --question 'Capital of Japan?' --choices Tokyo Osaka Kyoto --format text
python client.py decide --request examples/three-types.json
```

| Type | Request | Meaning |
|---|---|---|
| yes/no | type: noul | noul = P(yes); probabilities contains yes/no |
| choice | type: choice; criteria maps candidate names to descriptions or null | choice = highest-probability candidate; distribution over supplied candidates |
| score | type: score; criteria = ordered label array | score = **zero-based expected index**, not necessarily an integer |

For score probabilities [0.1, 0.2, 0.7], the result is 0×0.1 + 1×0.2 + 2×0.7 = **1.6**. The legend maps indices to labels. For every type, confidence is the largest candidate probability; when the answer is no, noul and confidence differ.

```json
{
  "state": "The package contains three red balls.",
  "questions": {
    "contains_red": {"type": "noul", "instructions": "Does it contain a red ball?"},
    "color": {"type": "choice", "instructions": "What color?", "criteria": {"red": null, "blue": null}},
    "explicitness": {"type": "score", "instructions": "How explicitly is the color stated?", "criteria": ["not stated", "implied", "explicit"]}
  }
}
```

State is optional: model knowledge is enabled by default. Add `--state 'text'`, repeatable `--state-file file.md`, or `--image photo.png`. Files are read by the client and images are encoded as data URLs. A path/URL inside state is not fetched. Existing state and state-file cannot be combined. Failed reads and oversized inputs are errors, never silent truncation.

For inline JSON use `--request-json '…'` or UTF-8 stdin with `--request-json -`. PowerShell stdin preserves quotes:

```powershell
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
@'
{"questions":{"answer":{"type":"noul","instructions":"Does water contain hydrogen?"}}}
'@ | python -X utf8 client.py decide --request-json - --format text
```

## Remote clients

Clone the source on the client too. Use your SSH alias and the **actual server checkout path**:

```sh
python client.py --ssh-config /path/to/ssh_config --ssh-host spark --remote-root /opt/GenericJevMCP decide --question 'Capital of Japan?' --choices Tokyo Osaka Kyoto
```

Alternatively set JEV_SSH_CONFIG, JEV_SSH_HOST, JEV_REMOTE_ROOT, or put ssh_config, ssh_host, remote_root in an ignored client-config.json beside client.py. CLI flags override environment variables, which override the file. Without an SSH config, calls are local. API keys remain on the server.

## MCP setup

Run `npm ci` in the checkout on the MCP host. Configure your host with absolute paths:

```json
{
  "mcpServers": {
    "jev": {
      "command": "node",
      "args": ["/absolute/path/GenericJevMCP-via-DiffusionGemma/mcp/server.mjs"],
      "env": {
        "JEV_PYTHON": "/absolute/path/to/python",
        "JEV_SSH_CONFIG": "/absolute/path/to/ssh_config",
        "JEV_SSH_HOST": "spark",
        "JEV_REMOTE_ROOT": "/opt/GenericJevMCP"
      },
      "timeoutSeconds": 240
    }
  }
}
```

Omit SSH variables on the GPU host. Host formats vary; timeoutSeconds is the tested Antigravity setting, allowing for the client's 200-second SSH timeout. The bridge starts the fixed Python client with shell:false. Tool callers cannot select an executable or change the SSH destination.

Call decide with:

```json
{"request":{"questions":{"answer":{"type":"noul","instructions":"Does water contain hydrogen?"}}}}
```

Optional top-level arguments are image (one local path) and state_files (UTF-8 paths). The MCP host reads them and transmits their content to the configured GPU server.

For Antigravity, register this server, copy [skills/jev](skills/jev) to ~/.gemini/antigravity-cli/skills/jev, and restart. Use `/jev [AdditionalInput] <OutputCategory>`. To auto-allow only this tool, add `mcp(jev/decide)` to permissions.allow in the host settings. No all-command permission is required. The tool description carries probability semantics instead of repeating a disclaimer in each result.

## Probability calibration and abstention

Probabilities are **uncalibrated and normalized within the supplied candidates**, not probabilities of factual correctness. A missing best answer can still yield a confident selection. Diagnostics expose label mass, label entropy, whether the vocabulary argmax is a permitted label, and read count. If every read's argmax falls outside the labels, the answer is null.

With sources_only:true, an extra evidence-sufficiency decision nulls unsupported answers. That gate is itself a model judgment. Explicit retrieval with no match abstains without inference.

[calibration.py](calibration.py) fits temperature on labeled examples and evaluates accuracy, NLL and ECE on **disjoint context groups**:

```json
[{"group":"document-001","probabilities":[0.9,0.1],"correct":0}]
```

```sh
python calibration.py --fit records/fit.json --evaluate records/evaluation.json
```

Use deployment-matched model, prompt, candidate order, mode and samples. Questions from one source share a group; overlapping fit/evaluation groups are rejected. Do not invent probabilities for abstentions. The upstream-derived grid is 0.2–4.0 in 0.05 steps; improvement on held-out data must be measured.

**This tool is an offline evaluator, not automatic serving calibration.** It does not modify the API, which retains calibrated:false. No real domain calibration dataset has been validated here; the numerical tests use synthetic distributions. This is distinct from NVFP4 quantization calibration.

## Latency breakdown

| Field / measurement | What it includes |
|---|---|
| Client elapsed_seconds | Client main entry through argument/file processing and API/SSH response receipt; excludes interpreter/import startup and final rendering |
| diagnostics.elapsed_seconds | Decision engine prompt/slot processing and backend HTTP reads, including repeats; excludes adapter validation, retrieval and source-only postprocessing |
| Difference | Input preparation **plus** SSH/network and adapter overhead; not a separate input-only timer |
| Full assistant turn | Also caller LLM planning, tool orchestration and final response; outside client timing |

Individual Windows→SSH→GB10 observations, 2026-09-20:

| Request | Client total | Decision | Difference |
|---|---:|---:|---:|
| Single choice | 1.387 s | 0.120 s | 1.267 s |
| Two short questions | 0.728 s | 0.127 s | 0.601 s |
| Text file + choice | 11.147 s | 0.382 s | 10.765 s |
| MCP yes/no | 1.698 s | 0.097 s | 1.601 s |

The MCP example's full assistant turn took 8.472 s. These are single observations, not percentiles or an optimization comparison. The slow file example does not isolate file-read cost. No-match retrieval has zero reads and can omit decision time.

Local HTTP probes: four short questions took 0.108 s joint/warm and 0.374 s separate/warm. Two-question image probes took 4.341 s on first shape use and 0.326–0.335 s warm. Compilation and concurrency matter. [Evidence and limits](docs/validation.md).

## One Spark-class machine: capacity and limits

Verified hardware: **one GB10 system with 128 GB unified memory**. These are deployed settings, not a maximum-capacity benchmark:

| Setting | Value |
|---|---:|
| Backend context limit / maximum sequences | 131,072 tokens / 4 |
| BF16 KV budget | 6 GiB |
| Generation canvas / denoising steps | 256 / 48 |
| Decision steps | 1 per read; auto uses 1 or 4 reads |
| Adapter simultaneous requests | 2 |
| Questions / candidates per question | 1–16 / 2–26 |
| Serialized state | 32,768 characters |
| Images / body | One PNG/JPEG, 4 MiB raw / 8 MiB HTTP body |
| PyTorch worker memory cap | 38% of device memory |
| Backend / adapter container limits | 60 GiB / 1 GiB |

Startup admission estimate: **19 GiB weights + 6 KV + 10 sampler transient + 12 host reserve = 47 GiB available**. This is not measured peak consumption. The PyTorch cap does not constrain every other library allocation.

Co-residency with a separate Gemma 26B NVFP4 backend and mixed generation/decision calls were tested. Host available memory was about 43 GiB after later vision validation, not a measurement of every startup peak. Maximum-context quality, saturation throughput and long-duration stability are unverified. The 16-question/26-candidate limits are also bounded by the answer canvas and 128 unique label-token IDs: not every maximal combination is valid.

## Retrieval, modes, and HTTP

The generic path needs no corpus. corpus.py is a **site-specific HTTrack importer**, not a universal document importer. No source archive is distributed. An optional corpus/surei.jsonl can contain your own entries with id, title and text plus optional provenance. Query retrieval uses Japanese character-bigram BM25; number lookup retains the original 1–91 convention. Restart the adapter after corpus changes.

POST /v1/systemone on loopback port 8011 requires Bearer authentication. The /v1/chat/completions wrapper accepts two messages: system = schema JSON, user = state JSON. Free-form generation uses port 8010. Use SSH forwarding for remote HTTP access.

Mode joint is the default; separate processes independent question inputs sequentially with a stable seed and per-question adaptive reads. Samples auto uses four reads if initial label entropy exceeds 0.1 nats or argmax lies outside labels. Fixed samples: 1–32. Isolation and seed do not guarantee bit-identical GPU results.

## Tests

```sh
pip install -r requirements.txt
# Download only the pinned tokenizer, not model shards.
python -c "import json, shutil; from huggingface_hub import hf_hub_download; m=json.load(open('models.lock.json'))['diffusion']; shutil.copyfile(hf_hub_download(m['model'],'tokenizer.json',revision=m['revision']), 'tests/tokenizer.json')"
python -m unittest discover -s tests -v
npm ci
npm test
```

Runtime overlay checks run during Docker build. Live scripts verify_stack.py and verify_rag.py retain companion-model/private-corpus assumptions; they are not fresh-install acceptance tests. CPU tests and generic examples need no private corpus. See [validation](docs/validation.md) for the actual published verification scope.


With a running configured GPU service, `npm run smoke` exercises MCP initialize, tool discovery and all three decision types. It does not start the GPU backend.

## License and provenance

Code: [Apache-2.0](LICENSE). [NOTICE.md](NOTICE.md) credits the pinned vLLM forks, [mmastrac/djev-spark](https://github.com/mmastrac/djev-spark), and [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev). Model weights have their own terms, including the Gemma terms referenced by the NVIDIA model card. This is not an official Google or NVIDIA product.
