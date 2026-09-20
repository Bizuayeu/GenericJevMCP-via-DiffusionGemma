# Validation / 検証範囲

[English README](../README.md) · [日本語README](../README.ja.md)

## Hardware and runtime / 実機とランタイム

Observations below were collected on 2026-09-19–20 on one MSI EdgeXpert, a GB10 system with 128 GB unified memory. DiffusionGemma NVFP4 shared the device with a separate Gemma 26B NVFP4 service. Model revision, base image and overlay hashes are pinned in models.lock.json and runtime/manifest.json. No benchmark below establishes the maximum supported workload.

以下はGB10・統合メモリ128 GBの1台での観測です。同居環境で機能確認しており、限界負荷試験ではありません。

## Functional checks / 機能確認

- Japanese generation, structured decisions, mixed generation/decision requests, and restart probes completed. The initial deployment suite contained 19 live cases.
- Six archive-grounding cases (affirmative, negative, and unsupported questions) matched expected outcomes. That private archive is not distributed.
- Joint/separate modes, question-order input isolation, adaptive read counts, and the grounding gate were checked. Separate mode is sequential; it does not guarantee identical GPU probabilities.
- Knowledge-only requests and irrelevant additional text succeeded; explicit sources_only with irrelevant evidence abstained.
- Synthetic red-triangle and blue-circle PNG images, plus a blue-circle JPEG, were passed as actual image bytes and classified by color and shape. These examples do not establish general vision accuracy.
- A real Antigravity headless session completed skill expansion, one jev/decide MCP call, and final response under request-review mode without an all-permissions flag. The tool permission was explicitly granted in host settings.
- The optional file/directory/URL workflow was tested with Markdown and a local static HTML server. A missing URL halted the request. This does not cover authenticated sites, JavaScript-only content, PDF, or OCR.

通常生成・判定・同居・再起動、正負の根拠と資料不足、joint/separate、実画像の転送、MCPの個別許可による完走を確認しています。非公開資料そのものは配布せず、汎用サンプルは資料なしで実行できます。

## Timing observations / 時間の観測

### Local HTTP and decision engine

| Probe | Observed time | Conditions |
|---|---:|---|
| Four text questions, joint, samples=1 | 1.180 s first shape use; 0.108 s warm | First request answered 3/4 correctly; warm request 4/4 |
| Four text questions, separate, samples=1 | 0.584 s first; 0.374 s warm | 4/4 on these examples |
| Separate, repeated same order | 0.381–0.406 s | Three repetitions |
| Separate, reversed order | 0.379 s | Additional probe |
| Joint / separate, samples=auto | 0.437 / 0.682 s | Four short questions |
| Two image questions, samples=auto | 4.341 s first PNG; 0.326 s warm PNG; 0.335 s JPEG | Adapter decision timing |

“First” means first use of the tested shape, not necessarily a freshly restarted process. Probabilities varied by up to 0.0414 across same-order repetitions; the reversed-order difference of 0.0135 was within that range. The cause of GPU variation was not isolated.

初回は「その形状の初回」であり、プロセス起動直後とは限りません。初回compileの影響をwarm性能の保証へ混ぜません。少数例の正答数を一般精度と解釈しません。

### Client and assistant

| Probe | Client | Decision engine | Client minus engine |
|---|---:|---:|---:|
| Single choice | 1.387 s | 0.120 s | 1.267 s |
| Two short questions | 0.728 s | 0.127 s | 0.601 s |
| UTF-8 text file + choice | 11.147 s | 0.382 s | 10.765 s |
| MCP yes/no | 1.698 s | 0.097 s | 1.601 s |

The last assistant turn took 8.472 s including orchestration and response generation. Client minus engine includes input preparation, transport and adapter work; it does not isolate input cost. These are individual measurements, not p50/p95 or before/after optimization results.

差分は入力準備・接続・通信・adapter等の合計で、入力のみの独立測定ではありません。呼出し元LLMの時間はクライアント計時の外側です。

## Memory and capacity / メモリと容量

The initial mixed-workload observation interval saw a minimum host available memory of 46.36 GiB; later vision validation observed about 43 GiB available after loading/testing. Neither value covers every startup allocation. The 47 GiB startup admission budget is an estimate. Backend context 128K, max sequences 4 and adapter concurrency 2 are configuration ceilings, not validated saturation workloads.

起動全ピーク・最大文脈品質・飽和スループット・長期耐久は未検証です。監視値と見積もりを区別します。

## Calibration / 校正

Temperature fitting and held-out evaluation have numerical tests, including disjoint context groups. No real domain calibration study has been completed. Serving remains uncalibrated, with calibrated:false. The evidence-sufficiency gate is also a model judgment.

校正コードの動作検証と、実データによる校正成功は別です。

## Publication checks / 公開版の確認

The publication preparation adds a standalone MCP package, explicit SSH configuration, optional corpus startup, and pinned-model preparation. Python CPU tests and MCP input-boundary tests cover these changes. A clean-checkout check and standalone MCP live probe are recorded below when completed. The existing fixed GPU runtime was previously exercised; a fresh GPU Docker rebuild is not claimed for this documentation/package release.

公開準備でGPU runtimeの固定revision・生成設定は変更していません。新規GPUホストへの一式再構築を今回再実施したとは扱いません。

Standalone MCP publication probe: initialize → tools/list → tools/call returned all three types successfully against the existing GPU API (client 1.240 s / engine 0.126 s). No GPU restart or rebuild was performed.

Clean exported checkout: Python CPU suite 46/46 passed using the declared dependency and pinned tokenizer fixture; npm ci from package-lock.json and both MCP boundary tests passed. No .env, client-config.json, private corpus or source-tree node_modules was present in that checkout. Local Markdown links and private machine-path checks passed.

## Package layout refactor / ディレクトリ整理

Python modules are under jev/, operational scripts under scripts/, and live probes under tests/live/. Python CPU checks: 48 passed; MCP boundary tests: 2 passed. The standalone MCP smoke probe was launched from outside the checkout and returned all three types through the new remote module command (client 1.235 s / decision 0.086 s). The existing GPU services were not restarted. Container command construction is covered by a CPU test; a fresh Docker deployment was not repeated.

設定・データのルート位置を保持し、MCP側が作業ディレクトリを指定する。クライアントとSSH受信側は同じ配置版を使う。稼働済みサービスの全体再配備とは区別する。
