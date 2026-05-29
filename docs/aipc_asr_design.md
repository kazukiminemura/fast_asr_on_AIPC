# AIPC ASR Design

## Purpose

Raptor mini (Preview) などの AIPC 上で、`UsefulSensors/moonshine-tiny-ja` を使った日本語音声認識をリアルタイム寄りに実行する。

最初の利用体験はブラウザGUIにする。入力はリアルタイムマイクを主経路にし、CLI はベンチマーク、検証、自動化のために残す。

## Target

- Target device: Raptor mini (Preview)
- ASR model: `UsefulSensors/moonshine-tiny-ja`
- Runtime: OpenVINO / Optimum Intel for Intel GPU, Transformers / PyTorch for CPU or CUDA
- Main input: Browser realtime microphone
- Main UI: Browser GUI
- Secondary interface: CLI

## User Flow

ブラウザGUIの基本フロー:

1. `python web_app.py` でローカルサーバを起動する
2. ブラウザで `http://127.0.0.1:8000` を開く
3. モデルが `UsefulSensors/moonshine-tiny-ja` になっていることを確認する
4. 推論デバイスを選ぶ
5. `Start Live` でマイク入力を開始する
6. 初回はウォームアップ表示を出し、モデル取得とロードが終わるまで待つ
7. 音声を短いチャンクに分けて継続的に文字起こしする
8. テキスト結果とベンチマーク指標を確認する
9. 終了時に `Stop` を押す

## Architecture

```text
Browser GUI realtime mic
  |
  | 16 kHz WAV chunks over multipart/form-data
  v
Python HTTP server
  |
  | temporary chunk file
  v
Shared Moonshine runner
  |
  | Optimum Intel OVModelForSpeechSeq2Seq or Transformers AutoModelForSpeechSeq2Seq
  v
Device: OpenVINO GPU / CUDA / CPU
```

主要ファイル:

- `web_app.py`: GUI 用 HTTP サーバと API
- `static/index.html`: GUI の画面構造
- `static/app.js`: リアルタイムマイク処理、WAV チャンク変換、API 呼び出し
- `static/styles.css`: GUI スタイル
- `main.py`: CLI と共有 Moonshine ASR 実行処理
- `requirements.txt`: Python 依存関係
- `cache/huggingface`: Hugging Face モデルキャッシュ

## Device Selection

推論デバイスは以下から選ぶ。

```text
auto | intel_gpu | cpu | gpu
```

方針:

- `auto`: OpenVINO GPU が利用できる場合は Intel GPU、なければ CUDA、なければ CPU
- `intel_gpu`: OpenVINO の `GPU` を明示的に使う
- `cpu`: PyTorch CPU を明示的に使う
- `gpu`: CUDA GPU を明示的に使う
- GUI は `/api/devices` のレスポンスから選択肢を作る
- 利用できないデバイスは `unavailable` として無効化する

Intel GPU 経路では Optimum Intel の `OVModelForSpeechSeq2Seq` を使う。初回ウォームアップで OpenVINO IR へ変換し、`cache/openvino` に保存する。
ただし Optimum Intel は Moonshine の OpenVINO export を公式サポートしていないため、OpenVINO GPU が空文字を返した有音チャンクは CPU runner でフォールバック認識する。

## Model Handling

既定モデル:

```text
UsefulSensors/moonshine-tiny-ja
```

モデルカードでは Hugging Face Transformers の `AutoProcessor` と `AutoModelForSpeechSeq2Seq` で利用する方法が示されている。
CPU/CUDA では Moonshine 用の Transformers runner を使う。
Intel GPU では Optimum Intel の OpenVINO runner を使う。

初回ウォームアップ時にモデルを取得し、`cache/huggingface` に保存する。
Intel GPU では OpenVINO IR を `cache/openvino` に保存する。
サーバプロセス内では、同じモデルとデバイスの runner をメモリ上で再利用する。

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

ウォームアップ:

1. モデルIDまたはローカルモデルディレクトリを検証する
2. PyTorch の利用可能デバイスを取得する
3. `auto` または明示指定に従ってデバイスを決定する
4. `AutoProcessor.from_pretrained(...)` を実行する
5. Intel GPU なら `OVModelForSpeechSeq2Seq.from_pretrained(..., export=True, device="GPU")` を実行する
6. CPU/CUDA なら `AutoModelForSpeechSeq2Seq.from_pretrained(...)` を実行する
7. runner をメモリにキャッシュする

チャンク推論:

1. 音声チャンクを一時ファイルとして受け取る
2. `librosa` で 16 kHz mono に読み込む
3. processor で入力テンソルを作る
4. `model.generate(...)` を実行する
5. processor でテキストへ decode する
6. ベンチマーク指標と一緒に返す

短いチャンクでの繰り返し生成ループを避けるため、音声長から推定した `max_length` を `Max tokens` 上限内で設定する。

## API

### `GET /api/devices`

Transformers / PyTorch で利用可能なデバイス一覧を返す。

```json
{
  "devices": ["OPENVINO_GPU", "CPU"],
  "choices": [
    {
      "value": "auto",
      "label": "auto (OPENVINO_GPU)",
      "available": true,
      "target": "OPENVINO_GPU"
    },
    {
      "value": "intel_gpu",
      "label": "Intel GPU (OpenVINO)",
      "available": true,
      "target": "OPENVINO_GPU"
    },
    {
      "value": "cpu",
      "label": "CPU",
      "available": true,
      "target": "CPU"
    },
    {
      "value": "gpu",
      "label": "GPU (CUDA)",
      "available": false,
      "target": "CUDA"
    }
  ]
}
```

CUDA が利用可能な環境では `CUDA` も含める。

### `POST /api/warmup`

モデルロードを先に実行する。

リクエスト:

```json
{
  "model": "UsefulSensors/moonshine-tiny-ja",
  "device": "auto"
}
```

レスポンス:

```json
{
  "model": "UsefulSensors/moonshine-tiny-ja",
  "requested_device": "auto",
  "selected_device": "CPU",
  "available_devices": ["CPU"],
  "model_load_seconds": 12.3,
  "cache_hit": false,
  "cache_dir": "cache\\huggingface"
}
```

### `POST /api/transcribe`

`multipart/form-data` で音声チャンクと設定を受け取り、文字起こし結果を返す。

主なフォーム項目:

- `audio`: 音声チャンク
- `model`: Hugging Face model id またはローカルモデルディレクトリ
- `device`: `auto`, `intel_gpu`, `cpu`, `gpu`
- `max_new_tokens`: 生成トークン上限

レスポンスには以下を含める。

- `text`
- `model`
- `requested_device`
- `selected_device`
- `available_devices`
- `input_source`
- `benchmark`

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

## CLI

CLI は GUI と同じ Moonshine ASR 実行処理を使う。

```text
python main.py --device auto --model UsefulSensors/moonshine-tiny-ja --mic --duration 5 --benchmark
```

## Current Scope

現在のスコープ:

- ローカルブラウザGUI
- `UsefulSensors/moonshine-tiny-ja`
- リアルタイムマイク入力
- チャンク単位の継続文字起こし
- 初回ウォームアップ表示
- サーバプロセス内の Moonshine runner キャッシュ
- `cache/huggingface` による次回起動向けモデルキャッシュ
- OpenVINO GPU 空出力時の CPU フォールバック
- ベンチマーク表示

まだ含めないもの:

- Intel NPU 向け Moonshine 実行
- トークン単位の真のストリーミング ASR
- VAD
- 話者分離
- 字幕ファイル出力
- GUI からの音声ファイル入力

## Next Decisions

次に決めること:

1. Raptor mini 上で CPU の RTF を実測する
2. Intel GPU 上の RTF を実測し、CPU と比較する
3. チャンク秒数の初期値を 3 秒にするか 4 秒にするか
4. VAD を入れて無音チャンクを送らないようにするか
5. 精度評価用の日本語音声セットをどう用意するか
