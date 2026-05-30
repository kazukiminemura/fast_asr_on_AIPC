# fast_asr_on_AIPC

Raptor mini (Preview) などの AIPC 上で、`neosophie/Qwen3-ASR-1.7B-JA` を使って日本語音声認識を行うブラウザ GUI アプリです。

ブラウザのマイク入力を短い音声チャンクに分けて継続処理し、文字起こし結果と RTF を画面に追記します。最初の開始時は Hugging Face からモデルを取得し、Intel NPU/GPU を使う場合は OpenVINO IR へ変換するため時間がかかります。ウォームアップ後はメモリ上のモデルを再利用し、モデルファイルも `cache\huggingface` と `cache\openvino` に残して次回起動を速くします。

## Features

- ブラウザ GUI で操作
- `neosophie/Qwen3-ASR-1.7B-JA` による日本語ASR
- Intel NPU/GPU 実行: OpenVINO
- ブラウザのリアルタイムマイク入力
- 発話と無音に合わせた動的チャンクの継続文字起こし
- 初回ウォームアップ表示
- モデルキャッシュ: `cache\huggingface`
- OpenVINO IR キャッシュ: `cache\openvino`
- 推論時間、全体処理時間、RTF の表示
- CLI からの固定秒数録音にも対応

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Model

既定モデルは Hugging Face の以下です。

```text
neosophie/Qwen3-ASR-1.7B-JA
```

初回実行時に自動でダウンロードされ、`cache\huggingface` に保存されます。Intel NPU/GPU を選んだ場合は OpenVINO 変換済みファイルも `cache\openvino` に保存されます。GUI は日本語認識に特化し、このモデルを固定で使います。

## Run GUI

```powershell
.\.venv\Scripts\python.exe web_app.py
```

ブラウザで開きます。

```text
http://127.0.0.1:8000
```

別ポートで起動する場合:

```powershell
.\.venv\Scripts\python.exe web_app.py --port 8080
```

## GUI Usage

1. `Model` が `neosophie/Qwen3-ASR-1.7B-JA` になっていることを確認する
2. `Device` で `auto`, `intel_npu`, `intel_gpu`, `cpu` を選ぶ
3. `Start Live` を押す
4. 初回はウォームアップ表示が出るので、モデル取得とロードが終わるまで待つ
5. `asr_text` の本文だけが `Transcript` へ追記される
6. 終了するときは `Stop` を押す

ブラウザでマイクを使う場合、ブラウザのマイク許可が必要です。

ライブ認識はブラウザ側で音量を見ながら、発話後の短い無音または最大長に達した時点で自動的に区切ります。無音だけのチャンクや空の認識結果は `Transcript` に表示しません。

## CLI Usage

GUI と同じ ASR バックエンドを CLI からも実行できます。

マイク録音:

```powershell
.\.venv\Scripts\python.exe main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --mic --duration 5 --benchmark
```

JSON 出力:

```powershell
.\.venv\Scripts\python.exe main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --mic --duration 5 --benchmark --json
```

## Device Policy

Qwen3-ASR JA は、Intel NPU/GPU では OpenVINO 変換済みモデル、それ以外では `qwen-asr` を使います。認識時は `qwen-asr` の正規言語名である `Japanese` を明示して実行します。

```text
auto | intel_npu | intel_gpu | cpu
```

- `auto`: Intel NPU が使える場合は OpenVINO NPU、なければ OpenVINO GPU、なければ CPU
- `intel_npu`: OpenVINO の `NPU`
- `intel_gpu`: OpenVINO の `GPU`
- `cpu`: PyTorch CPU

GUI の `Device` は起動時に利用可能なデバイスを確認して更新します。使えないデバイスは `unavailable` として無効表示されます。

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

- `cache/` は `.gitignore` 対象です。モデルキャッシュはリポジトリに含めません。
- ブラウザ GUI のマイク入力は、ブラウザ側で録音した音声を発話と無音に応じた 16 kHz WAV チャンクにしてバックエンドへ送ります。
- サーバ側では同じモデルとデバイスの ASR runner をキャッシュし、リアルタイム処理中の再ロードを避けます。
- Intel NPU/GPU 初回実行ではモデルを OpenVINO IR に変換し、`cache\openvino` に保存します。
- 認識対象は日本語音声です。日本語以外の音声は精度保証の対象外です。
- CLI の `--mic` は Python の `sounddevice` を使った固定秒数録音です。リアルタイム操作はブラウザ GUI を使います。

## Implementation Guide

再実装に必要な設計、API、処理フローは [docs/implementation.md](docs/implementation.md) にまとめています。
