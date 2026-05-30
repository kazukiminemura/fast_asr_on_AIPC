# fast_asr_on_AIPC 実装ガイド

このドキュメントは、`fast_asr_on_AIPC` を同じ仕様で再実装できるように、現行実装の構成、責務、処理フロー、API、キャッシュ、デバイス選択をまとめたものです。

## 目的

AIPC 上で日本語音声認識を低遅延に試せる、ブラウザ GUI 付きのローカル ASR アプリを作る。

- ブラウザでマイク音声を録音する
- 音声を短いチャンクに分割してサーバーへ送る
- サーバー側で Qwen3-ASR を実行する
- Intel NPU/GPU がある場合は OpenVINO を優先する
- 認識結果と RTF をチャンクごとに画面へ追記する
- 初回ロード後はモデルと OpenVINO IR を `cache/` に保存する

## ファイル構成

```text
fast_asr_on_AIPC/
|-- main.py                  # ASR 実行コアと CLI
|-- web_app.py               # HTTP サーバー、静的配信、API
|-- requirements.txt         # Python 依存関係
|-- static/
|   |-- index.html           # GUI の HTML
|   |-- app.js               # 録音、チャンク化、API 呼び出し、表示更新
|   `-- styles.css           # GUI の見た目
|-- third_party/
|   `-- qwen_3_asr_helper.py # Qwen3-ASR の OpenVINO 変換/実行ヘルパー
|-- cache/
|   |-- huggingface/         # Hugging Face モデルキャッシュ
|   `-- openvino/            # OpenVINO IR と compiled cache
`-- docs/
    `-- implementation.md
```

`cache/` は生成物なのでリポジトリに含めない。

## 依存関係

`requirements.txt` の主要依存は以下。

```text
torch
transformers
accelerate
openvino
optimum-intel[openvino]
librosa
soundfile
sounddevice
qwen-asr
```

役割は次の通り。

- `qwen-asr`: Qwen3-ASR の通常実行
- `torch`, `transformers`: 非 OpenVINO モデル実行、デバイス検出
- `openvino`, `optimum-intel`: Intel NPU/GPU 向け推論と変換
- `librosa`: 音声ファイルを 16 kHz mono float32 に読み込み
- `soundfile`: Qwen3-ASR に渡す一時 WAV を書き出し
- `sounddevice`: CLI の固定秒数マイク録音

## 起動方法

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python web_app.py
```

ブラウザで開く。

```text
http://127.0.0.1:8000
```

ポートを変える場合。

```powershell
python web_app.py --host 127.0.0.1 --port 8080
```

CLI で直接実行する場合。

```powershell
python main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --audio sample.wav --benchmark
python main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --mic --duration 5 --benchmark --json
```

## モデル

既定モデルは次。

```text
neosophie/Qwen3-ASR-1.7B-JA
```

GUI では以下を選べる。

- `neosophie/Qwen3-ASR-1.7B-JA`: 日本語向け Qwen3-ASR
- `Qwen/Qwen3-ASR-1.7B`: 多言語 Qwen3-ASR
- `Custom`: Hugging Face model id またはローカルモデルディレクトリ

`main.py` では `DEFAULT_MODEL`, `QWEN3_ASR_MODEL`, `MODEL_CHOICES` で定義する。

## 全体アーキテクチャ

```text
Browser
  static/index.html
  static/app.js
    |
    | GET  /api/models
    | GET  /api/devices?model=...
    | POST /api/warmup
    | POST /api/transcribe multipart/form-data
    v
web_app.py
  WebAsrHandler
    |
    v
main.py
  warmup_asr_model()
  run_asr()
  runner.transcribe()
    |
    v
Qwen3-ASR / OpenVINO / PyTorch
```

ブラウザは Web Audio API でマイク音声を取り、16 kHz WAV チャンクへ変換して `POST /api/transcribe` に送る。サーバーは受け取った WAV を一時ファイルに保存し、`main.run_asr()` に渡す。

## バックエンド設計

`main.py` は ASR 実行コアで、CLI と Web API の両方から使う。

主要な公開関数。

- `available_devices()`: 使用可能な推論デバイスを返す
- `device_choices(model_ref)`: GUI 用のデバイス選択肢を返す
- `model_choices()`: GUI 用のモデル選択肢を返す
- `warmup_asr_model(model_ref, requested_device)`: モデルをロードし、必要なら OpenVINO 変換する
- `run_asr(args)`: 音声読み込み、推論、ベンチマーク生成を行う

### デバイス選択

GUI/CLI の入力値は `DEVICE_ALIASES` で内部表現へ変換する。

```text
auto      -> AUTO
intel_npu -> OPENVINO_NPU
intel_gpu -> OPENVINO_GPU
gpu       -> CUDA
cpu       -> CPU
```

`auto` の優先順は次。

```text
OPENVINO_NPU -> OPENVINO_GPU -> CUDA -> CPU
```

OpenVINO の実デバイス名へ渡すときは、`OPENVINO_NPU` を `NPU`、`OPENVINO_GPU` を `GPU` に変換する。

### Runner の種類

`get_moonshine_runner(model_ref, device)` がモデルとデバイスに応じて Runner を選ぶ。関数名に Moonshine が残っているが、現行の既定モデルは Qwen3-ASR。

- `Qwen3AsrRunner`: Qwen3-ASR を `qwen-asr` で実行する
- `OpenVINOQwen3AsrRunner`: Qwen3-ASR を `third_party.qwen_3_asr_helper` で OpenVINO IR に変換して実行する
- `MoonshineRunner`: Transformers の `AutoModelForSpeechSeq2Seq` で実行する汎用 Runner
- `OpenVINOMoonshineRunner`: Optimum Intel の `OVModelForSpeechSeq2Seq` で実行する汎用 OpenVINO Runner

選択ルール。

```text
Qwen3-ASR + OPENVINO_* -> OpenVINOQwen3AsrRunner
Qwen3-ASR + その他    -> Qwen3AsrRunner
その他 + OPENVINO_*   -> OpenVINOMoonshineRunner
その他 + その他       -> MoonshineRunner
```

Runner は `(model_ref, device)` をキーに `RUNNER_CACHE` へ保存する。Web GUI の連続チャンク処理では、同じモデルとデバイスの Runner を再利用する。

## キャッシュ

キャッシュ先は `main.py` の定数で管理する。

```text
HF_CACHE_DIR       = cache/huggingface
OPENVINO_CACHE_DIR = cache/openvino
```

Hugging Face のモデルファイルは `cache/huggingface` に保存する。OpenVINO 変換済みモデルは `cache/openvino/<model_refを安全な名前にした文字列>/` に保存する。OpenVINO compiled cache は `cache/openvino/compiled` に置く。

モデル参照をディレクトリ名に変換するときは `sanitize_model_ref()` を使い、英数字、`-`, `_`, `.` 以外を `_` に置き換える。

## Web サーバー

`web_app.py` は標準ライブラリの `ThreadingHTTPServer` と `SimpleHTTPRequestHandler` だけで実装する。

### 静的ファイル

`translate_path()` で次のように配信する。

- `/` -> `static/index.html`
- `/static/<path>` -> `static/<path>`
- その他 -> `static/<path>`

### API

#### `GET /api/models`

モデル選択肢を返す。

レスポンス例。

```json
{
  "models": [
    {
      "value": "neosophie/Qwen3-ASR-1.7B-JA",
      "label": "Qwen3-ASR JA",
      "description": "Japanese ASR via qwen-asr / OpenVINO"
    }
  ]
}
```

#### `GET /api/devices?model=<model>`

使用可能なデバイスと GUI 用選択肢を返す。

レスポンス例。

```json
{
  "devices": ["OPENVINO_NPU", "CPU"],
  "choices": [
    {
      "value": "auto",
      "label": "auto (OPENVINO_NPU)",
      "available": true,
      "target": "OPENVINO_NPU"
    }
  ]
}
```

#### `POST /api/warmup`

モデルを事前ロードする。初回はモデルダウンロードや OpenVINO 変換が走る。

リクエスト。

```json
{
  "model": "neosophie/Qwen3-ASR-1.7B-JA",
  "device": "auto"
}
```

レスポンス。

```json
{
  "model": "neosophie/Qwen3-ASR-1.7B-JA",
  "requested_device": "auto",
  "selected_device": "OPENVINO_NPU",
  "available_devices": ["OPENVINO_NPU", "CPU"],
  "model_load_seconds": 12.3456,
  "cache_hit": false,
  "cache_dir": "cache\\openvino",
  "fallback_device": null
}
```

#### `POST /api/transcribe`

`multipart/form-data` で音声チャンクを受け取り、認識結果とベンチマークを返す。

フォーム項目。

```text
model: Hugging Face model id またはローカルモデルディレクトリ
device: auto | intel_npu | intel_gpu | cpu | gpu
max_new_tokens: 生成トークン上限
duration: GUI 上のチャンク秒数
audio: WAV などの音声ファイル
```

レスポンス。

```json
{
  "text": "認識結果",
  "chunks": [],
  "model": "neosophie/Qwen3-ASR-1.7B-JA",
  "requested_device": "auto",
  "selected_device": "OPENVINO_NPU",
  "fallback_device": null,
  "available_devices": ["OPENVINO_NPU", "CPU"],
  "input_source": "C:\\Users\\...\\tmp.wav",
  "benchmark": {
    "model_load_seconds": 0.0,
    "audio_preprocess_seconds": 0.0123,
    "inference_seconds": 0.4567,
    "postprocess_seconds": 0.0,
    "total_processing_seconds": 0.469,
    "audio_duration_seconds": 1.0,
    "rtf": 0.469
  }
}
```

エラー時は HTTP 400 と `{ "error": "message" }` を返す。

## フロントエンド

`static/index.html` は単一画面の操作 UI を持つ。

- モデル選択
- カスタムモデル入力
- デバイス選択
- Max tokens
- Chunk seconds
- Start Live / Stop / Clear
- 音量メーター
- チャンクごとのログ
- Transcript
- メトリクス表示

`static/app.js` は状態を `state` オブジェクトにまとめる。

重要な状態。

```js
{
  liveRunning: false,
  warmed: false,
  stream: null,
  audioContext: null,
  source: null,
  processor: null,
  analyser: null,
  meterTimer: null,
  chunkTimer: null,
  chunks: [],
  submitChain: Promise.resolve(),
  chunkIndex: 0
}
```

### ライブ録音フロー

1. `Start Live` を押す
2. `warmupModel()` が `POST /api/warmup` を呼ぶ
3. `navigator.mediaDevices.getUserMedia({ audio: true })` でマイク許可を取る
4. `AudioContext` を作る
5. `MediaStreamSource -> Analyser -> ScriptProcessor -> destination` を接続する
6. `onaudioprocess` で Float32 PCM を `state.chunks` に積む
7. `setInterval(flushLiveChunk, Chunk seconds)` でチャンク送信する
8. `Stop` 時に最後のチャンクを送信し、録音リソースを解放する

### チャンク処理

`flushLiveChunk()` の処理。

1. `state.chunks` を取り出して空にする
2. `mergeFloat32()` で連結する
3. `resampleLinear()` でブラウザの sample rate から 16 kHz へ変換する
4. `audioRms()` が `0.003` 未満なら無音として捨てる
5. `encodeWav()` で 16-bit PCM WAV にする
6. `submitAudio()` で `POST /api/transcribe` へ送る
7. `appendLiveResult()` で `[chunk番号] テキスト` を追記する
8. `renderMetrics()` で RTF や推論時間を更新する

`submitChain` で送信を直列化する。チャンクの推論が重なって Runner を同時に叩かないようにするため。

## ASR 実行フロー

`run_asr(args)` の処理。

1. `validate_inputs()` でモデル名と音声パスを確認する
2. `available_devices()` で使用可能なデバイスを調べる
3. `select_model_device()` で要求デバイスを実デバイスに決める
4. `get_moonshine_runner()` で Runner を取得または作成する
5. `--mic` なら `read_microphone_16khz()`、`--audio` なら `read_audio_16khz()` で 16 kHz mono にする
6. `runner.transcribe(raw_speech, max_new_tokens)` を呼ぶ
7. 必要なら fallback を行う
8. `BenchmarkResult` を作る
9. Web/CLI 共通の payload を返す

`BenchmarkResult` の項目。

```text
model_load_seconds
audio_preprocess_seconds
inference_seconds
postprocess_seconds
total_processing_seconds
audio_duration_seconds
rtf
```

RTF は次で計算する。

```text
rtf = total_processing_seconds / audio_duration_seconds
```

`rtf < 1.0` なら、音声長より短い時間で処理できている。

## Qwen3-ASR 実行

### 通常実行

`Qwen3AsrRunner` は `qwen_asr.Qwen3ASRModel.from_pretrained()` を使う。

- CUDA の場合は `device_map="cuda:0"`、`dtype=torch.bfloat16`
- その他は `device_map="cpu"`、`dtype=torch.float32`
- `max_inference_batch_size=1`
- `max_new_tokens=64`

`qwen-asr` はファイルパスを入力に取るため、`transcribe()` では一時 WAV を作る。

```text
raw_speech list[float] -> temp .wav at 16 kHz -> model.transcribe(audio=...)
```

### OpenVINO 実行

`OpenVINOQwen3AsrRunner` は `third_party.qwen_3_asr_helper` を使う。

1. `convert_qwen3_asr_model(model_id, output_dir, quantization_config=None)` で変換する
2. `OVQwen3ASRModel.from_pretrained(model_dir, device, ...)` でロードする
3. 通常実行と同じく、一時 WAV を作って `model.transcribe(audio=...)` に渡す

変換先は `cache/openvino/<sanitized_model_ref>/`。

## 汎用 SpeechSeq2Seq 実行

Qwen3-ASR 以外のモデルを Custom で指定した場合に使う。

### PyTorch

`MoonshineRunner` は `AutoProcessor` と `AutoModelForSpeechSeq2Seq` をロードする。

- CUDA の場合は `cuda:0` と `float16`
- その他は `cpu` と `float32`
- `processor(..., sampling_rate=...)` で入力を作る
- `model.generate(**inputs, max_length=max_length)` で生成する
- `processor.decode(..., skip_special_tokens=True)` で文字列化する

### OpenVINO

`OpenVINOMoonshineRunner` は Optimum Intel を使う。

- 変換済み IR があれば `OVModelForSpeechSeq2Seq.from_pretrained(ov_model_dir, device=...)`
- なければ `optimum.exporters.openvino.main_export()` で変換する
- `ov_config={"CACHE_DIR": "cache/openvino/compiled"}` を指定する

現行実装では、Qwen3-ASR 以外を OpenVINO で実行して空文字になり、かつ入力音量が十分ある場合、CPU Runner で再認識する。

## 再実装手順

最小構成で同じアプリを作るなら、次の順で実装する。

1. `requirements.txt` を作る
2. `main.py` に定数、デバイス検出、モデル選択、Runner キャッシュを作る
3. `Qwen3AsrRunner` と `OpenVINOQwen3AsrRunner` を実装する
4. `read_audio_16khz()` と `read_microphone_16khz()` を実装する
5. `run_asr()` を実装し、JSON payload を返せるようにする
6. CLI の `argparse` と `main()` を追加する
7. `web_app.py` で静的配信、`/api/models`, `/api/devices`, `/api/warmup`, `/api/transcribe` を実装する
8. `static/index.html` で操作 UI を作る
9. `static/app.js` でモデル/デバイス取得、ウォームアップ、録音、チャンク化、送信、表示更新を実装する
10. `static/styles.css` で 1 画面に収まる作業 UI を整える
11. `cache/` と `.venv/` を `.gitignore` に入れる

## 動作確認

Python の構文確認。

```powershell
python -m py_compile main.py web_app.py
```

Web サーバー起動。

```powershell
python web_app.py --host 127.0.0.1 --port 8000
```

API 確認。

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/models | Select-Object -ExpandProperty Content
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/devices | Select-Object -ExpandProperty Content
```

音声ファイルで CLI 確認。

```powershell
python main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --audio sample.wav --benchmark --json
```

ブラウザ確認。

1. `http://127.0.0.1:8000` を開く
2. `Start Live` を押す
3. マイク許可を与える
4. 初回ウォームアップが終わるまで待つ
5. 話した内容が `[1] ...`, `[2] ...` の形で追記されることを確認する
6. `rtf`, `infer sec`, `total sec` が表示されることを確認する

## 注意点

- 初回はモデルダウンロードと OpenVINO 変換で時間がかかる
- OpenVINO NPU/GPU がない環境では `auto` は CUDA または CPU へ落ちる
- ブラウザのマイク利用には `http://127.0.0.1` または HTTPS が必要
- GUI のマイク録音はブラウザ側で行う。CLI の `--mic` は `sounddevice` で固定秒数だけ録音する別経路
- `ScriptProcessorNode` は古い API だが、現行実装では簡単さを優先して使っている
- `multipart/form-data` のパーサーは `web_app.py` 内の簡易実装。複雑なフォームに拡張するなら標準/外部の堅牢なパーサーを検討する
- 長時間運用では `RUNNER_CACHE` がモデル/デバイスの組み合わせごとにメモリを保持する
