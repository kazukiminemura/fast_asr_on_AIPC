# fast_asr_on_AIPC

Raptor mini (Preview) などの AIPC 上で、`UsefulSensors/moonshine-tiny-ja` を使って日本語音声認識を行うブラウザGUIアプリです。

ブラウザのマイク入力を短い音声チャンクに分けて継続処理し、文字起こし結果と RTF を画面に追記します。最初の開始時は Hugging Face からモデルを取得してウォームアップするため時間がかかります。ウォームアップ後はメモリ上のモデルを再利用し、モデルファイルも `cache\huggingface` に残して次回起動を速くします。

設計メモは [docs/aipc_asr_design.md](docs/aipc_asr_design.md) にあります。

## Features

- ブラウザGUIで操作
- `UsefulSensors/moonshine-tiny-ja` による日本語ASR
- Intel GPU 実行: OpenVINO / Optimum Intel
- ブラウザのリアルタイムマイク入力
- チャンク単位の継続文字起こし
- 初回ウォームアップ表示
- モデルキャッシュ: `cache\huggingface`
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
UsefulSensors/moonshine-tiny-ja
```

初回実行時に自動でダウンロードされ、`cache\huggingface` に保存されます。GUI の `Model` 欄には Hugging Face model id またはローカルモデルディレクトリを指定できます。

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

1. `Model` が `UsefulSensors/moonshine-tiny-ja` になっていることを確認する
2. `Device` で `auto`, `intel_gpu`, `cpu`, `gpu` を選ぶ
3. `Max tokens` と `Chunk seconds` を調整する
4. `Start Live` を押す
5. 初回はウォームアップ表示が出るので、モデル取得とロードが終わるまで待つ
6. 話した内容がチャンクごとに `Transcript` へ追記される
7. 終了するときは `Stop` を押す

ブラウザでマイクを使う場合、ブラウザのマイク許可が必要です。

`Chunk seconds` を短くすると表示までの待ち時間は短くなりますが、推論回数が増えます。まずは `4` 秒から始める想定です。

## CLI Usage

GUI と同じ Moonshine 処理を CLI からも実行できます。

マイク録音:

```powershell
python main.py --device auto --model UsefulSensors/moonshine-tiny-ja --mic --duration 5 --benchmark
```

JSON 出力:

```powershell
python main.py --device auto --model UsefulSensors/moonshine-tiny-ja --mic --duration 5 --benchmark --json
```

## Device Policy

この Moonshine バックエンドは、Intel GPU では OpenVINO / Optimum Intel、それ以外では Hugging Face Transformers / PyTorch を使います。

```text
auto | intel_gpu | cpu | gpu
```

- `auto`: Intel GPU が使える場合は OpenVINO GPU、なければ CUDA、なければ CPU
- `intel_gpu`: OpenVINO の `GPU`
- `cpu`: PyTorch CPU
- `gpu`: CUDA GPU

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
- ブラウザGUIのマイク入力は、ブラウザ側で録音した音声を 16 kHz WAV のチャンクにしてバックエンドへ送ります。
- サーバ側では同じモデルとデバイスの Moonshine runner をキャッシュし、リアルタイム処理中の再ロードを避けます。
- Intel GPU 初回実行では Moonshine を OpenVINO IR に変換し、`cache\openvino` に保存します。
- 現在の Optimum Intel は Moonshine の OpenVINO export を公式サポートしていないため、Intel GPU が空文字を返した有音チャンクは CPU runner でフォールバック認識します。
- CLI の `--mic` は Python の `sounddevice` を使った固定秒数録音です。リアルタイム操作はブラウザGUIを使います。
