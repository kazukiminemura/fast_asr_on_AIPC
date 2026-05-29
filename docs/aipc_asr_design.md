# AIPC ASR Design

## Purpose

Raptor mini (Preview) などの AIPC 上で、OpenVINO GenAI Whisper を使った音声認識を高速に実行する。

最初の利用体験はブラウザGUIにする。入力はリアルタイムマイクを主経路にし、CLI はベンチマーク、検証、自動化のために残す。

## Target

- Target device: Raptor mini (Preview)
- Primary inference device: GPU
- Optional inference devices: NPU, CPU
- Runtime: OpenVINO / OpenVINO GenAI
- ASR model family: Whisper
- Main input: Browser realtime microphone
- Main UI: Browser GUI
- Secondary interface: CLI

## User Flow

ブラウザGUIの基本フロー:

1. `python web_app.py` でローカルサーバを起動する
2. ブラウザで `http://127.0.0.1:8000` を開く
3. 変換済み Whisper モデルディレクトリを指定する
4. 推論デバイスを選ぶ
5. `Start Live` でマイク入力を開始する
6. 初回はウォームアップ表示を出し、モデルロードと OpenVINO キャッシュ作成が終わるまで待つ
7. 音声を短いチャンクに分けて継続的に文字起こしする
8. テキスト結果とベンチマーク指標を確認する
9. 終了時に `Stop` を押す

## Architecture

```text
Browser GUI realtime mic
  |
  | WAV chunks over multipart/form-data
  v
Python HTTP server
  |
  | temporary chunk file
  v
Shared ASR runner
  |
  | OpenVINO GenAI WhisperPipeline
  v
OpenVINO device: GPU / NPU / CPU
```

主要ファイル:

- `web_app.py`: GUI 用 HTTP サーバと API
- `static/index.html`: GUI の画面構造
- `static/app.js`: リアルタイムマイク処理、WAV チャンク変換、API 呼び出し
- `static/styles.css`: GUI スタイル
- `main.py`: CLI と共有 ASR 実行処理
- `requirements.txt`: Python 依存関係
- `cache/openvino`: OpenVINO のデバイスコンパイルキャッシュ

## Device Selection

推論デバイスは以下から選ぶ。

```text
auto | gpu | npu | cpu
```

`auto` の優先順位:

```text
GPU -> NPU -> CPU
```

方針:

- GPU を最速候補として最優先にする
- NPU と CPU は比較、検証、フォールバック用に使えるようにする
- 明示指定されたデバイスが存在しない場合はエラーにする
- GUI 起動時に `/api/devices` で OpenVINO の利用可能デバイスを表示する

## Model Handling

OpenVINO GenAI の `WhisperPipeline` は、OpenVINO 形式に変換済みの Whisper モデルディレクトリを読み込む。

初期推奨モデル:

- `openai/whisper-tiny`
- `openai/whisper-base`

理由:

- AIPC 上でデバイスごとの速度差を確認しやすい
- 初回ロードと推論が比較的軽い
- 精度と速度のバランスを後から上位モデルで確認できる

モデルは `models/` 配下に置く想定だが、リポジトリには含めない。

## Audio Input

主入力はブラウザのリアルタイムマイク。

ブラウザマイク:

- Web Audio API でマイク入力を取得する
- `Chunk seconds` ごとに音声を切り出す
- ブラウザ側で 16 kHz mono WAV に変換する
- `multipart/form-data` でサーバへ送る
- サーバから返った文字起こしをチャンク単位で画面へ追記する

CLI マイク:

- `sounddevice` で指定秒数録音する
- 16 kHz mono float32 として ASR 実行経路に渡す

## Runtime Flow

ASR 実行時の処理:

1. モデルパスと入力音声を検証する
2. OpenVINO の利用可能デバイスを取得する
3. `auto` または明示指定に従ってデバイスを決定する
4. `WhisperPipeline(model_path, selected_device)` を取得する
5. 音声を 16 kHz mono として準備する
6. `pipe.generate(...)` を実行する
7. 文字起こし結果を抽出する
8. ベンチマーク指標を返す

リアルタイムマイク処理では、録音開始前に `/api/warmup` を呼び出してモデルとデバイスを先にウォームアップする。
同じモデルとデバイスの `WhisperPipeline` はサーバプロセス内でキャッシュする。
初回ウォームアップではモデルロード時間が発生し、以降のチャンクではキャッシュ済みパイプラインを再利用する。

OpenVINO のデバイスコンパイルキャッシュは `cache/openvino` に保存する。
これによりアプリ再起動後も OpenVINO のコンパイル時間短縮を期待できる。

## API

### `GET /api/devices`

OpenVINO で利用可能なデバイス一覧を返す。

```json
{
  "devices": ["CPU", "GPU"]
}
```

### `POST /api/transcribe`

`multipart/form-data` で音声チャンクと設定を受け取り、文字起こし結果を返す。

主なフォーム項目:

- `audio`: 音声ファイル
- `model`: モデルディレクトリ
- `device`: `auto`, `gpu`, `npu`, `cpu`
- `language`: Whisper 言語トークン。例: `<|ja|>`
- `task`: `transcribe` または `translate`
- `max_new_tokens`: 生成トークン上限
- `timestamps`: `true` または `false`

レスポンスには以下を含める。

- `text`
- `chunks`
- `requested_device`
- `selected_device`
- `available_devices`
- `input_source`
- `benchmark`

### `POST /api/warmup`

モデルロードと OpenVINO デバイスキャッシュ作成を先に実行する。

リクエスト:

```json
{
  "model": "models\\whisper-base",
  "device": "auto"
}
```

レスポンス:

```json
{
  "requested_device": "auto",
  "selected_device": "GPU",
  "available_devices": ["CPU", "GPU"],
  "model_load_seconds": 12.3,
  "cache_hit": false,
  "cache_dir": "cache\\openvino"
}
```

## Benchmark Metrics

計測する項目:

- `model_load_seconds`
- `audio_preprocess_seconds`
- `inference_seconds`
- `postprocess_seconds`
- `total_processing_seconds`
- `audio_duration_seconds`
- `rtf`

RTF:

```text
RTF = total_processing_time_seconds / audio_duration_seconds
```

`RTF < 1.0` ならリアルタイムより高速。

現状の `total_processing_seconds` には、デバイス選択、モデルロード、音声前処理、推論、後処理を含める。

## CLI

CLI は GUI と同じ ASR 実行処理を使う。

マイク録音:

```text
python main.py --device auto --model models\whisper-base --mic --duration 5 --benchmark
```

## Current Scope

現在のスコープ:

- ローカルブラウザGUI
- リアルタイムマイク入力
- チャンク単位の継続文字起こし
- 初回ウォームアップ表示
- サーバプロセス内の `WhisperPipeline` キャッシュ
- `cache/openvino` による次回起動向けキャッシュ
- OpenVINO デバイス切り替え
- WhisperPipeline による文字起こし
- ベンチマーク表示

まだ含めないもの:

- トークン単位の真のストリーミング ASR
- 話者分離
- VAD
- 字幕ファイル出力
- モデルの自動ダウンロードと自動変換
- 複数音声のバッチ処理
- GUI からの音声ファイル入力

## Next Decisions

次に決めること:

1. 初期推奨モデルを `whisper-tiny` にするか `whisper-base` にするか
2. Raptor mini 上で GPU/NPU/CPU の実測値を比較する
3. GUI でモデル変換まで扱うか、手順として README に残すか
4. チャンク秒数の初期値を 3 秒にするか 4 秒にするか
5. 精度評価用の音声セットをどう用意するか
