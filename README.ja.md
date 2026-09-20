# GenericJevMCP via DiffusionGemma

[English](README.md) · [検証範囲](docs/validation.md) · [由来](NOTICE.md)

**文章・追加資料・画像から、yes/no・choice・scoreの構造化判定を返すMCP／CLI／HTTPツール。** CPUアダプターが、共有するDiffusionGemma NVFP4バックエンドの回答トークン枠を読み、候補確率と実測秒数を返します。判定ごとに説明文を生成する必要はありません。

Jev方式を使った実験的なセルフホスト実装です。内部API名の `dg-bert` は開発時の呼称で、**BERTモデルを別にロードするものではありません**。呼び出し側のアシスタントと判定バックエンドは独立しています。

## 構成

```text
MCPホスト／ターミナル → Pythonクライアント → 任意のSSH → CPUアダプター :8011
                                                              ↓
                                                DiffusionGemma／vLLM :8010
```

モデル知識を既定で使い、文章・UTF-8ファイル・画像1枚を追加できます。複数問をまとめて読むjointと、独立した入力で順次読むseparateを選べます。通常生成と判定は同じ重みを共有し、判定は1 readにつきdiffusion 1 step、通常生成は設定されたdenoising scheduleを使います。

## ディレクトリ構成

```text
.
├── jev/             Python本体・API・CLI・校正
├── mcp/             MCP stdioサーバー
├── scripts/         モデル準備・起動停止・監視
├── tests/           CPU・MCPテスト
│   └── live/        稼働中GPUに対する任意の実機検証
├── runtime/         固定vLLM差分・ビルド時検査
├── examples/        要求サンプル
├── skills/          アシスタント用スキル
└── docs/            検証範囲
```

Pythonの入口はリポジトリルートから `python -m jev.client`、`python -m jev.calibration`、`python scripts/service.py` で実行します。MCPはPythonの作業ディレクトリを自動でルートへ設定します。非追跡の `.env`・`client-config.json`・`state/`・`corpus/` の場所はルートのままです。

## 必要環境

| 要素 | 必要環境・検証範囲 |
|---|---|
| GPUホスト | Linux ARM64、GB10、統合メモリ128 GB。MSI EdgeXpert（DGX Spark相当機）で検証 |
| ランタイム | NVIDIA対応Docker。[固定base／overlay](runtime/manifest.json)を[Dockerfile](Dockerfile)でビルド |
| モデル | [models.lock.json](models.lock.json)で固定したNVIDIA DiffusionGemma 26B-A4B NVFP4。別途取得 |
| Python | コードは3.10以降の構文。クライアント・CPUテストは3.13で検証。[requirements.txt](requirements.txt)を導入 |
| MCPホスト | Node.js 20以降、PATH上のPythonまたはJEV_PYTHON、npm ci |
| リモート接続 | OpenSSHの鍵認証。APIはloopback待受 |

モデルファイル約19 GBに加え、Dockerのイメージ・ビルド層・実行キャッシュのディスク容量が必要です。重み・認証情報・非公開コーパスは同梱しません。他GPU、x86ホスト、別のvLLM版は今回の動作検証範囲外です。

## GPUサーバーのセットアップ

LinuxのGPUホスト上で実行します。取得前に[モデルカードと利用条件](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)を確認してください。本コードのライセンスはモデルの利用条件を置き換えません。

```sh
git clone https://github.com/Bizuayeu/GenericJevMCP-via-DiffusionGemma.git
cd GenericJevMCP-via-DiffusionGemma
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_model.py --download
# 取得済みの場合：
# python scripts/prepare_model.py --snapshot /absolute/cache/path/snapshots/REVISION
python scripts/service.py init --download-status state/download-status.json

docker build -t generic-jev:local .
export JEV_IMAGE=$(docker image inspect generic-jev:local --format '{{.Id}}')
python scripts/service.py start-backend
# readyになるまで待つ。初回コンパイルはwarm時より時間がかかる。
curl --fail http://127.0.0.1:8010/health
python scripts/service.py start-adapter
curl --fail http://127.0.0.1:8011/health
```

scripts/prepare_model.pyは固定revisionを ~/.cache/huggingface に取得し、ファイル名・サイズを照合します。initは0600の権限でローカル認証情報を生成し、既存.envを上書きしません。コーパスなしで起動できます。

**JEV_IMAGEには自分でビルドしたイメージIDを設定**し、サービス操作・監視で同じ値を使います。ソース内の既定IDは開発時に検証したイメージであり、導入先に存在するとは限りません。ビルド時にbaseとoverlayのhashを照合します。

任意の監視は、同じJEV_IMAGEを設定した別ターミナルで `python scripts/monitor.py`。hostの空きメモリが予約値を下回ると、条件の一致する管理対象GPUコンテナを停止します。あらゆるOOMを防ぐ保証ではありません。サービスと監視のOS自動起動は設定しません。停止は `python scripts/service.py stop-adapter`／`stop-backend`。構成を変える場合は旧コンテナを停止・保存用にrenameしてから再作成します。設定が異なる既存コンテナの黙った再利用は拒否します。

## yes/no・choice・scoreの3型

```sh
python -m jev.client decide --question '水には水素が含まれますか' --format text
python -m jev.client decide --question '日本の首都はどれですか' --choices 東京 大阪 京都 --format text
python -m jev.client decide --request examples/three-types.json
```

| 型 | 要求 | 戻り値の意味 |
|---|---|---|
| yes/no | type: noul | noulはP(yes)。probabilitiesはyes／noの分布 |
| choice | type: choice、criteriaは候補名→説明またはnull | choiceは最大確率の候補、probabilitiesは指定候補間の分布 |
| score | type: score、criteriaは順序付き尺度名の配列 | scoreは**0始まりの期待値**。整数の評点とは限らない |

scoreの確率が[0.1, 0.2, 0.7]なら、0×0.1 + 1×0.2 + 2×0.7 = **1.6**です。legendが番号と尺度名を対応付けます。全型のconfidenceは最大候補確率なので、no判定ではnoulとconfidenceが異なります。

```json
{
  "state": "袋に赤いボールが3個入っています。",
  "questions": {
    "contains_red": {"type": "noul", "instructions": "赤いボールが入っていますか"},
    "color": {"type": "choice", "instructions": "何色ですか", "criteria": {"赤": null, "青": null}},
    "explicitness": {"type": "score", "instructions": "色はどの程度明記されていますか", "criteria": ["記載なし", "間接的", "直接明記"]}
  }
}
```

stateは任意。追加文章は `--state '本文'`、UTF-8ファイルは `--state-file file.md`（複数回可）、画像は `--image photo.png`。クライアントがファイルを読み、画像をdata URLへ符号化します。stateにパスやURL文字列を入れるだけでは取得しません。既存stateとstate-fileは併用不可。取得失敗・上限超過はエラーとし、黙って切り捨てません。

JSONは `--request-json '…'` またはUTF-8標準入力の `--request-json -` でも渡せます。PowerShellでは引用符を保つため標準入力を使います。

```powershell
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
@'
{"questions":{"answer":{"type":"noul","instructions":"水には水素が含まれますか"}}}
'@ | python -X utf8 -m jev.client decide --request-json - --format text
```

## 別のPCから呼ぶ

クライアント側にもcloneし、自分のSSH設定・ホスト名・**サーバー上の実際のcheckoutパス**を指定します。

```sh
python -m jev.client --ssh-config /path/to/ssh_config --ssh-host spark --remote-root /opt/GenericJevMCP decide --question '日本の首都は？' --choices 東京 大阪 京都
```

JEV_SSH_CONFIG、JEV_SSH_HOST、JEV_REMOTE_ROOTの環境変数、またはリポジトリルートの非追跡client-config.jsonにssh_config／ssh_host／remote_rootを置く方法もあります。優先順位はCLI引数→環境変数→設定ファイル。SSH設定がなければローカル呼び出しです。APIキーはサーバー側に残ります。

## MCPの設定

MCPホスト側のcheckoutで `npm ci` を実行し、ホストの設定に絶対パスで登録します。

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

GPUホスト上で動かすならSSH変数を省略します。設定形式はMCPホストに依存し、timeoutSecondsはAntigravityで検証した項目です。240秒はクライアントのSSHタイムアウト200秒を含む設定。bridgeは固定Pythonクライアントをshell:falseで起動し、ツール引数から実行プログラムやSSH接続先を変更できません。

decideの引数：

```json
{"request":{"questions":{"answer":{"type":"noul","instructions":"水には水素が含まれますか"}}}}
```

任意の追加引数はimage（画像1枚のローカルパス）とstate_files（UTF-8ファイルパス配列）。MCPホストが読み、設定済みGPUサーバーへ内容を送ります。

AntigravityではMCPを登録し、[skills/jev](skills/jev)を ~/.gemini/antigravity-cli/skills/jev へコピーして再起動します。呼び方は `/jev [AdditionalInput] <OutputCategory>`。自動承認するならホスト設定のpermissions.allowに **mcp(jev/decide)だけ**を追加します。全シェルコマンドの許可は不要です。確率の意味はMCPツールの説明に置き、毎回の結果に注意書きを繰り返しません。

## 確率・保留・校正

確率は**指定された候補内で正規化した未校正の確率**であり、事実としての正答率ではありません。適切な答えが候補にない場合も、高確率の選択が出ることがあります。diagnosticsにはlabel mass、label entropy、語彙全体のargmaxが候補labelかどうか、read数を返します。全readでargmaxがlabel外なら、その回答はnullです。

sources_only:trueでは根拠十分性の追加判定を行い、不足時は回答をnullにします。このゲート自体もモデル判断です。検索を明示して一致がなかった場合は推論せず保留します。

[jev/calibration.py](jev/calibration.py)は正解付きデータで温度をfitし、**文脈グループを分離した評価データ**でaccuracy・NLL・ECEを比較します。

```json
[{"group":"document-001","probabilities":[0.9,0.1],"correct":0}]
```

```sh
python -m jev.calibration --fit records/fit.json --evaluate records/evaluation.json
```

本番とモデル・質問文・候補順・mode・samplesを揃え、同じ資料からの質問は同じgroupにします。fit／評価のgroup重複は拒否します。保留を架空の確率へ置き換えません。温度探索は上流由来の0.2〜4.0、0.05刻み。別データでの改善は実測して判断します。

**校正ツールはオフラインの評価用です。APIの確率を自動変更しません。** 配信側はcalibrated:falseのままです。実ドメインの校正データセットは未検証で、数値テストは合成分布を使います。NVFP4重みを作る量子化校正とは別の処理です。

## レイテンシの内訳

クライアントとMCPは `timing.total_seconds`（全体）と `timing.decision_seconds`（判定）を別の数値項目として返し、未計測の判定時間はnullにします。既存のelapsed項目は互換性のため保持します。固定表示用の `display` も返し、MCPはoutputSchemaとstructuredContentを公開します。text内容はdisplayをtextコードブロックで包んだもので、スキルはtextコードブロックで逐語提示します。任意の呼出し元モデルによる最終回答の書き換えまでは強制できませんが、ツール結果の形式はコードで固定しています。

| 項目 | 測定範囲 |
|---|---|
| クライアントJSONのelapsed_seconds | main開始から引数・ファイル処理、API／SSH応答受信まで。Python起動・import・最終表示整形は含まない |
| diagnostics.elapsed_seconds | 判定エンジンのprompt／slot処理とbackend HTTP read。追加readを含み、adapterの検証・検索・資料限定の後処理は含まない |
| 両者の差 | 入力準備＋SSH／通信＋adapter等の周辺処理。入力処理だけの独立計測ではない |
| アシスタントのターン全体 | 呼出し元LLMの解釈、ツール制御、最終回答生成も含む。クライアント計時の外側 |

2026-09-20のWindows→SSH→GB10による単発実測：

| 要求 | 全体 | 判定 | 差分 |
|---|---:|---:|---:|
| 単一choice | 1.387秒 | 0.120秒 | 1.267秒 |
| 短文2問 | 0.728秒 | 0.127秒 | 0.601秒 |
| 本文ファイル＋choice | 11.147秒 | 0.382秒 | 10.765秒 |
| MCP yes/no | 1.698秒 | 0.097秒 | 1.601秒 |

MCP例のアシスタントターン全体は8.472秒でした。いずれも単発値で、percentileや改善率のベンチマークではありません。ファイル例の遅さをファイル読込時間と断定できません。検索不一致の保留はread数0で、判定時間を持たない場合があります。

ローカルHTTPでは短文4問のjoint warmが0.108秒、separate warmが0.374秒。画像2問は初回形状4.341秒、warmで0.326〜0.335秒でした。初回compileや同時実行の影響があります。[検証範囲](docs/validation.md)。

## Spark 1枚でどこまで持つか

検証機は**GB10・統合メモリ128 GBの1台**。以下は配信設定であり、限界まで負荷をかけた結果ではありません。

| 設定 | 値 |
|---|---:|
| backend文脈上限／最大sequence | 131,072 token／4 |
| BF16 KV予算 | 6 GiB |
| 通常生成canvas／denoising step | 256／48 |
| 判定step | 1 readにつき1、autoは1または4 read |
| adapter同時要求 | 2 |
| 質問数／各候補数 | 1〜16／2〜26 |
| JSON化されたstate | 32,768文字 |
| 画像／body | PNG・JPEG 1枚、raw 4 MiB／HTTP 8 MiB |
| PyTorch workerメモリ上限 | deviceメモリの38% |
| backend／adapterコンテナ上限 | 60 GiB／1 GiB |

起動時の必要空き容量見積もりは **重み19 GiB＋KV 6＋sampler一時領域10＋host予約12＝47 GiB**。実測ピークではありません。PyTorchの上限は他ライブラリの全確保を制約しません。

別のGemma 26B NVFP4との同居、生成と判定の混在実行を検証しています。後続の画像検証後のhost空きは約43 GiBでしたが、起動中の全ピークを測った数字ではありません。最大文脈品質・飽和スループット・長時間耐久は未検証。16問／26択の範囲内でも、回答canvasとunique label-token ID最大128の制約により、最大値同士の全組み合わせが成立するわけではありません。

## 検索・mode・HTTP

汎用判定にはコーパス不要です。jev/corpus.pyは**特定サイト向けHTTrack importer**であり、汎用文書取込器ではありません。原本資料は配布しません。任意のcorpus/surei.jsonlにはid・title・textと任意の出典情報を持つ自分のレコードを置けます。query検索は日本語文字bigramのBM25、number検索は開発時の1〜91直接参照の規約です。変更時はadapterを再起動します。

loopback 8011のPOST /v1/systemoneはBearer認証が必要です。互換の/v1/chat/completionsはsystemにschema JSON、userにstate JSONの2メッセージ形式。自由生成は8010を使います。遠隔HTTP接続はSSH転送を使います。

modeはjointが既定。separateは質問ごとに独立した入力・同じseedで逐次実行し、追加readも個別に判断します。samples:autoは初回label entropyが0.1 nats超またはargmaxがlabel外なら4 read。固定samplesは1〜32。分離やseed固定はGPUのbit再現を保証しません。

## テスト

```sh
pip install -r requirements.txt
# 重みshardを取得せず、固定tokenizerだけを取得する。
python -c "import json, shutil; from huggingface_hub import hf_hub_download; m=json.load(open('models.lock.json'))['diffusion']; shutil.copyfile(hf_hub_download(m['model'],'tokenizer.json',revision=m['revision']), 'tests/tokenizer.json')"
python -m unittest discover -s tests -v
npm ci
npm test
```

Docker build中にもruntime overlay検査を行います。実機用tests/live/verify_stack.py／tests/live/verify_rag.pyには開発時の同居モデル・非公開コーパスの前提が残るため、新規導入の受入試験ではありません。CPUテストと汎用サンプルに非公開コーパスは不要です。公開時に実際に確認した範囲は[検証文書](docs/validation.md)を参照してください。


設定済みGPUサービスが稼働中なら `npm run smoke` でMCP接続・ツール一覧・3型の判定を実走できます。GPUの起動は行いません。

## ライセンスと謝辞

コードは[Apache-2.0](LICENSE)。固定vLLM fork、[mmastrac/djev-spark](https://github.com/mmastrac/djev-spark)、[open-alternative-jev](https://github.com/ikermoel/open-alternative-jev)の由来を[NOTICE.md](NOTICE.md)に記載しています。モデル重みにはNVIDIAモデルカードから参照されるGemma利用条件など、別の条件があります。Google・NVIDIA公式の製品ではありません。
