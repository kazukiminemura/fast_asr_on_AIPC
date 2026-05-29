# AIPC ASR Design

## Goal

Raptor mini (Preview) 上で ASR をできるだけ高速に動かす。

主デバイスは GPU とし、CPU と NPU も実行時に選択できる構成にする。

## Target Hardware

- Device: Raptor mini (Preview)
- Primary accelerator: GPU
- Optional devices: CPU, NPU
- Runtime: OpenVINO

## Device Selection Policy

起動時にコマンドラインオプションで推論デバイスを選択する。

```text
--device auto | gpu | cpu | npu
```

初期方針:

- `auto`: 利用可能なデバイスを検出し、GPU -> NPU -> CPU の順で選択する
- `gpu`: GPU を明示的に使う
- `npu`: NPU を明示的に使う
- `cpu`: CPU を明示的に使う

GPU を最速候補として扱い、CPU と NPU は比較・検証・フォールバック用に切り替え可能にする。

## Runtime Flow

1. 起動時に CLI オプションを読む
2. OpenVINO で利用可能デバイスを取得する
3. 指定されたデバイス、または `auto` の優先順位に従って実行デバイスを決める
4. 同じ ASR モデルを選択デバイス向けにロードする
5. 音声入力を前処理する
6. ASR 推論を実行する
7. 結果と処理時間を出力する

## Initial CLI Shape

```text
python main.py --device auto --model <model_path> --audio <audio_path>
```

想定オプション:

- `--device`: `auto`, `gpu`, `cpu`, `npu`
- `--model`: OpenVINO IR または変換元モデルのパス
- `--audio`: 入力音声ファイル
- `--benchmark`: 推論時間を詳細出力する

## Benchmark Metrics

まずは以下を記録する。

- モデルロード時間
- 音声前処理時間
- 推論時間
- 後処理時間
- 全体処理時間
- Real Time Factor (RTF)

RTF は以下で計算する。

```text
RTF = total_processing_time_seconds / audio_duration_seconds
```

`RTF < 1.0` ならリアルタイムより高速。

## Open Questions

次に決めるべきこと:

1. ASR モデルは何を使うか
2. 入力はファイル処理から始めるか、リアルタイムマイク入力まで最初から含めるか
3. 最適化対象はレイテンシ重視か、スループット重視か
4. 精度評価用の音声データをどう用意するか
5. GPU/NPU の利用可否をどのログ粒度で見せるか

## Recommendation

最初の実装は、音声ファイルを入力にした CLI ベンチマークツールから始める。

理由:

- GPU, NPU, CPU の速度比較がしやすい
- リアルタイム入力の複雑さを後回しにできる
- OpenVINO のデバイス切り替え設計を先に固められる
- RTF を使って最速構成を定量的に比較できる
