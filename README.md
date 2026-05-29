# fast_asr_on_AIPC

Raptor mini (Preview) などの AIPC 上で、OpenVINO GenAI Whisper を使って音声認識を行うブラウザGUIアプリです。

GPU を主デバイスとして使い、CPU と NPU も選択できるようにしています。ブラウザのマイク入力を短い音声チャンクに分けて継続処理し、文字起こし結果と RTF を画面に追記します。

最初の開始時は Whisper モデルと OpenVINO デバイスをウォームアップするため時間がかかります。ウォームアップ後はメモリ上のパイプラインを再利用し、OpenVINO のキャッシュも `cache\openvino` に残して次回起動を速くします。

設計メモは [docs/aipc_asr_design.md](docs/aipc_asr_design.md) にあります。

## Features

- ブラウザGUIで操作
- OpenVINO デバイス選択: `auto`, `gpu`, `npu`, `cpu`
- `auto` では `GPU -> NPU -> CPU` の順で利用可能デバイスを選択
- ブラウザのリアルタイムマイク入力
- Whisper の言語指定: auto, Japanese, English
- `transcribe` / `translate` の切り替え
- 推論時間、全体処理時間、RTF の表示
- CLI からの実行にも対応

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare Model

このアプリは OpenVINO GenAI の `WhisperPipeline` を使います。事前に Whisper モデルを OpenVINO 形式へ変換してください。

最初は速度確認しやすい `openai/whisper-tiny` または `openai/whisper-base` を推奨します。

```powershell
pip install --upgrade-strategy eager "optimum-intel[openvino]" transformers
optimum-cli export openvino --trust-remote-code --model openai/whisper-base models\whisper-base
```

変換後、GUI の `Model directory` に以下のようなパスを指定します。

```text
models\whisper-base
```

## Run GUI

```powershell
python web_app.py
```

ブラウザで開きます。

```text
http://127.0.0.1:8000
```

別ポートで起動する場合:

```powershell
python web_app.py --port 8080
```

## GUI Usage

1. `Model directory` に変換済みモデルのディレクトリを入力する
2. `Device` で `auto`, `gpu`, `npu`, `cpu` を選ぶ
3. 必要に応じて `Language`, `Task`, `Max tokens` を調整する
4. `Chunk seconds` で1回の推論に渡す音声長を決める
5. `Start Live` を押す
6. 初回はウォームアップ表示が出るので、モデルロードと OpenVINO キャッシュ作成が終わるまで待つ
7. 話した内容がチャンクごとに `Transcript` へ追記される
8. 終了するときは `Stop` を押す

ブラウザでマイクを使う場合、ブラウザのマイク許可が必要です。

`Chunk seconds` を短くすると表示までの待ち時間は短くなりますが、推論回数が増えます。まずは `4` 秒から始める想定です。

初回ウォームアップが長いのは正常です。2回目以降の `Start Live` はメモリ上のパイプラインを使うため速くなります。アプリを再起動した後も、`cache\openvino` に残ったキャッシュにより OpenVINO のコンパイル時間短縮が期待できます。

## CLI Usage

GUI と同じ推論処理を CLI からも実行できます。ベンチマークや自動テストでは CLI が便利です。

CLI のマイク録音:

```powershell
python main.py --device auto --model models\whisper-base --mic --duration 5 --benchmark
```

JSON 出力:

```powershell
python main.py --device auto --model models\whisper-base --mic --duration 5 --benchmark --json
```

## Device Policy

`--device auto` または GUI の `auto` は、OpenVINO の利用可能デバイスを確認し、以下の順で選択します。

```text
GPU -> NPU -> CPU
```

明示的に `gpu`, `npu`, `cpu` を選んだ場合、そのデバイスが利用できなければエラーになります。

## Benchmark Metrics

アプリは以下を計測します。

- モデルロード時間
- 音声前処理時間
- 推論時間
- 後処理時間
- 全体処理時間
- 音声長
- RTF

RTF は以下です。

```text
RTF = total_processing_time_seconds / audio_duration_seconds
```

`RTF < 1.0` ならリアルタイムより高速です。

## Notes

- `models/` は `.gitignore` 対象です。変換済みモデルはリポジトリに含めません。
- ブラウザGUIのマイク入力は、ブラウザ側で録音した音声を 16 kHz WAV のチャンクにしてバックエンドへ送ります。
- サーバ側では同じモデルとデバイスの `WhisperPipeline` をキャッシュし、リアルタイム処理中の再ロードを避けます。
- OpenVINO のデバイスコンパイルキャッシュは `cache\openvino` に保存します。
- CLI の `--mic` は Python の `sounddevice` を使った固定秒数録音です。リアルタイム操作はブラウザGUIを使います。
