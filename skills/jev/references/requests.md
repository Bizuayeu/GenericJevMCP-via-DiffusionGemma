# 追加入力

- ファイル全文はMCP引数state_filesに絶対パスの配列で渡す。本文のLLM転記は不要。request.stateとの併用は不可。UTF-8（BOM可）、APIのstate上限32,768文字を超えた場合はエラー。
- ディレクトリは関連ファイルを選ぶ。URLは読取ツールで本文を取得し、request.stateへ {"sources":[{"source":"URL","text":"原文・抜粋"}]} として入れる。URL文字列だけでは取得されない。原文にない内容を補わず参照箇所を保持する。
- 画像はimageへPNG/JPEGの絶対パスを渡す（1枚・4 MiBまで）。画像URLなら取得してから渡す。自動縮小・OCR代用はしない。
- 指定された資料の取得失敗と、追加入力の省略は区別する。失敗時は止める。
- 通常はrequest.ragを省略する。数霊の登録資料検索だけ {"number":40} または {"query":"検索語"}。資料限定は明示時のみsources_only:true。
- request.modeはjoint（既定）またはseparate、samplesはauto（既定）または1〜32。最大16問・各2〜26択。scoreは0始まりの期待値。

例：
```json
{"request":{"questions":{"color":{"type":"choice","instructions":"画像の図形は何色ですか","criteria":{"赤":null,"青":null,"緑":null}},"shape":{"type":"choice","instructions":"画像の図形は何ですか","criteria":{"三角":null,"円":null,"四角":null}}}},"image":"C:/path/to/image.png"}
```

ターミナルから直接使う場合は[本体README](../../../README.md)の `python -m jev.client` を使える。/jevの通常実行はMCP一回で行う。
