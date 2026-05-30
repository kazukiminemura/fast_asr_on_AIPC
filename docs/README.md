# Docs

このディレクトリには、`fast_asr_on_AIPC` を理解、運用、再実装するためのドキュメントを置きます。

## ドキュメント一覧

- [implementation.md](implementation.md): 現行実装を再現するための実装ガイド。構成、処理フロー、API、デバイス選択、キャッシュ方針を説明します。

## アプリの起動

プロジェクトルートで実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python web_app.py
```

ブラウザで次を開きます。

```text
http://127.0.0.1:8000
```
