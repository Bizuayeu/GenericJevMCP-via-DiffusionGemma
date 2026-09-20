---
name: jev
description: /jev の真偽・選択・段階評価を専用MCPツールで実行し、判定・候補確率・所要秒数を返す。
---

# Jev を実行する

ユーザーは /jev を呼び出し済み。判定は **jev MCP の decide ツール**で実行する。シェル経由にせず、JSONファイルも作らない。使い方だけを聞かれた場合は説明でよい。

「魏延は五虎将軍？」の呼び出し引数：

```json
{"request":{"questions":{"answer":{"type":"noul","instructions":"魏延は五虎将軍？"}}}}
```

選択式は type:choice、criteria:{"東京":null,"大阪":null,"京都":null}。段階評価は type:score、criteriaを順序付き尺度名配列にする。質問を複数にすれば一回で判定できる。

追加文章はrequest.state、UTF-8本文ファイルはstate_files（絶対パスの配列）、画像はimage（PNG/JPEGの絶対パス1枚）へ。画像を説明文で代用しない。追加入力は任意、既定でモデル知識を使う。資料だけに限定する場合のみrequest.sources_only:true。詳細が必要なら[追加入力](references/requests.md)を読む。

ツール結果内のtextコードブロック（structuredContent.displayを包んだもの）だけを逐語提示する。CLIが外側に付けるCreated At・Completed At等のメタデータは含めない。要約・項目名の翻訳・並べ替えはしない。「所要時間（全体）」と「判定時間」は別行のまま残し、complexity等の質問IDと混ぜない。所要時間は入力処理・SSH通信を含み、呼出し元LLMの解釈・最終表示時間は含まない。確率の意味はMCPツールの説明に従い、通常の結果に注意書きを繰り返さない。保留を自己判定で埋めない。

jev/decideの実行許可はMCPホストの設定に従う。ツールが見えなければCLIの再起動が必要と報告する。実際の拒否・失敗はそのまま報告し、自己判定や別APIへ切り替えず、GPUを再起動しない。
