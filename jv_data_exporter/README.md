# JV-DataLab JVD Exporter

JRA-VAN DataLab から保存した `.jvd` ファイルを、まずは **行単位の raw CSV** に落とすための最小ツールです。

## できること
- 指定フォルダ配下の `*.jvd` を再帰的に探索
- `cp932` / `utf-8-sig` / `utf-8` の順でデコードを試行
- 1行ごとに CSV 出力
- 以下の列を出力
  - `source_file`
  - `line_no`
  - `record_prefix_2`
  - `record_prefix_8`
  - `char_len`
  - `raw_text`

## 想定用途
現時点では **完全なJV仕様パース** ではなく、
まず `jvd` の中身を Python で扱える CSV に変換するところまでを目的にしています。

その後、`record_prefix_2` や `record_prefix_8` を見ながら、必要なレコードだけ個別パーサーを追加していく構成です。

## フォルダ構成
- `export_jvd_folder_to_csv.py` : メイン実行スクリプト
- `jv_parser.py` : 行単位の共通変換ロジック
- `requirements.txt` : 必須ライブラリ（標準ライブラリのみなので実質空）

## 使い方
### 1. Python を用意
Windows で Python 3.10 以上を想定します。

### 2. 実行
```bash
python export_jvd_folder_to_csv.py --input "F:\\TFJV" --output "F:\\TFJV\\exports\\jvd_raw.csv"
```

### 3. 出力
`jvd_raw.csv` が作られます。

## よくある注意点
- `.jvd` は保存直後のファイルを使ってください
- DataLab 検証ツールで `データを保存する` を ON にしてから `ファイル取得` を行ってください
- 文字化けする場合は `cp932` で開いているか確認してください

## 次の拡張候補
- レコード種別ごとの列展開
- `RACE` / `SE` / `JG` ごとの専用パーサー
- pandas を使った整形スクリプト追加
