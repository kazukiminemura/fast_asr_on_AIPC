# fast_asr_on_AIPC Implementation Guide

This document describes the current implementation so the app can be maintained or rebuilt from the same design.

## Goal

Provide a local browser GUI for low-latency Japanese ASR on an AIPC.

The current app is not a true streaming decoder. It records browser microphone audio, cuts it into short chunks, sends each chunk to the local server, and runs Qwen3-ASR on each chunk in order.

## Repository Layout

```text
fast_asr_on_AIPC/
|-- main.py                  # ASR runtime core and CLI
|-- web_app.py               # Static file server and HTTP API
|-- requirements.txt         # Python dependencies
|-- static/
|   |-- index.html           # Browser UI
|   |-- app.js               # Recording, chunking, API calls, UI updates
|   `-- styles.css           # Layout and Transcript scrolling
|-- third_party/
|   `-- qwen_3_asr_helper.py # Qwen3-ASR OpenVINO conversion/runtime helper
|-- cache/
|   |-- huggingface/         # Hugging Face cache
|   `-- openvino/            # OpenVINO IR cache
`-- docs/
    `-- implementation.md
```

`cache/` and `.venv/` are generated and are not tracked by Git.

## Dependencies

Main packages:

```text
torch
transformers
accelerate
openvino>=2026.2
optimum-intel[openvino]
librosa
soundfile
sounddevice
qwen-asr
```

Roles:

- `qwen-asr`: Qwen3-ASR model API
- `openvino`, `optimum-intel`: OpenVINO conversion and inference
- `torch`, `transformers`: model execution and generation utilities
- `soundfile`: in-memory upload decoding and temporary WAV output inside runners
- `librosa`: resampling to 16 kHz when needed
- `sounddevice`: CLI microphone recording

## Startup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe web_app.py
```

Open:

```text
http://127.0.0.1:8000
```

CLI examples:

```powershell
.\.venv\Scripts\python.exe main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --audio sample.wav --benchmark
.\.venv\Scripts\python.exe main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --mic --duration 5 --benchmark --json
```

Always start the app with the same Python environment that installed `requirements.txt`.
On Windows, prefer `.\.venv\Scripts\python.exe ...` in commands even after activation.
If the app is launched with the Microsoft Store/global `python`, `third_party/qwen_3_asr_helper.py`
can exist but still fail to import because `qwen_asr` is missing from that interpreter. That failure
may surface as:

```text
OpenVINO Qwen3-ASR helper is missing. Expected third_party/qwen_3_asr_helper.py.
```

When this appears, first confirm which interpreter is running:

```powershell
python -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -c "import qwen_asr; print(qwen_asr.__file__)"
```

Then restart with:

```powershell
.\.venv\Scripts\python.exe web_app.py
```

## Architecture

```text
Browser
  static/index.html
  static/app.js
    |
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
  run_asr_samples()
  runner.transcribe()
    |
    v
Qwen3-ASR / OpenVINO / PyTorch
```

The browser records microphone audio with Web Audio APIs, converts chunks to 16 kHz WAV, and posts them to `/api/transcribe`. The server decodes uploaded audio in memory and calls `main.run_asr_samples()`.

## Device Policy

User-facing device aliases:

```text
auto       -> AUTO
intel_npu  -> OPENVINO_NPU
npu        -> OPENVINO_NPU
intel_gpu  -> OPENVINO_GPU
cpu        -> CPU
```

For generic OpenVINO-capable models, `auto_device_target()` prefers:

```text
OPENVINO_NPU -> OPENVINO_GPU -> CPU
```

For Qwen3-ASR, NPU is not used. The current Intel NPU OpenVINO compiler cannot lower this Qwen3-ASR graph reliably, even after dynamic input bounds are added. Qwen3-ASR uses:

```text
OPENVINO_GPU -> CPU
```

For Qwen3-ASR, requests for `auto`, `intel_npu`, `npu`, `OPENVINO_NPU`, or `NPU` are routed to GPU or CPU. The GUI marks `intel_npu` unavailable for the default model.

## Cache

Constants:

```text
HF_CACHE_DIR              = cache/huggingface
OPENVINO_CACHE_DIR        = cache/openvino
RUNNER_CACHE              = {(model_ref, selected_device): runner}
AVAILABLE_DEVICES_CACHE   = list[str] | None
```

`available_devices()` caches OpenVINO device discovery. This avoids repeated device probing for every live audio chunk. Use `available_devices(refresh=True)` to force a fresh probe.

Runners are cached by `(model_ref, selected_device)` to avoid reloading the model for every chunk.

## main.py

Important functions:

- `available_devices(refresh=False)`: returns available inference devices
- `device_choices(model_ref)`: returns GUI device options
- `model_choices()`: returns GUI model options
- `warmup_asr_model(model_ref, requested_device)`: loads the selected runner before live transcription
- `run_asr(args)`: CLI entry path; reads audio then delegates to `run_asr_samples()`
- `run_asr_samples(...)`: shared runtime path for already-decoded float32 PCM

`run_asr_samples()` flow:

1. Validate model reference.
2. Get available devices.
3. Select the actual runtime device.
4. Get or create a runner.
5. Call `runner.transcribe(raw_speech, max_new_tokens)`.
6. Strip ASR tags from the result.
7. Return a JSON payload and `BenchmarkResult`.

## Runner Selection

`get_moonshine_runner(model_ref, device)` selects a runner. The function name still contains `moonshine`, but the default GUI model is Qwen3-ASR.

```text
Qwen3-ASR + OPENVINO_* -> OpenVINOQwen3AsrRunner
Qwen3-ASR + other      -> Qwen3AsrRunner
other + OPENVINO_*     -> OpenVINOMoonshineRunner
other + other          -> MoonshineRunner
```

### Qwen3AsrRunner

Uses `qwen_asr.Qwen3ASRModel.from_pretrained()`.

- CPU execution
- `max_inference_batch_size=1`
- `max_new_tokens=64`
- `language="Japanese"`

The `qwen-asr` API expects an audio path, so the runner writes a temporary 16 kHz WAV and passes the path to `model.transcribe()`.

### OpenVINOQwen3AsrRunner

Uses `third_party.qwen_3_asr_helper`.

1. `convert_qwen3_asr_model()` creates or reuses OpenVINO IR.
2. `OVQwen3ASRModel.from_pretrained()` loads the IR.
3. Inference writes a temporary 16 kHz WAV and passes the path to `model.transcribe()`.

If `device == "NPU"` is passed directly, the runner redirects to GPU or CPU. This protects against code paths that bypass normal device selection.

### third_party/qwen_3_asr_helper.py

This helper converts and runs Qwen3-ASR with OpenVINO.

It defines upper bounds for dynamic dimensions to avoid the NPU compiler error "Missing upper bound for one or more nodes":

```text
NPU_MAX_BATCH_SIZE
NPU_MAX_AUDIO_CHUNKS
NPU_MAX_AUDIO_SEQUENCE_LENGTH
NPU_MAX_TEXT_SEQUENCE_LENGTH
```

Those bounds are useful, but the full Qwen3-ASR graph still does not compile reliably for Intel NPU. The main app therefore does not route Qwen3-ASR to NPU.

## web_app.py

The server uses only standard-library HTTP classes:

- `ThreadingHTTPServer`
- `SimpleHTTPRequestHandler`

Static routing:

```text
/               -> static/index.html
/static/<path>  -> static/<path>
other           -> static/<path>
```

### GET /api/devices

Returns physical devices and GUI choices.

Example:

```json
{
  "devices": ["OPENVINO_NPU", "OPENVINO_GPU", "CPU"],
  "choices": [
    {
      "value": "auto",
      "label": "auto (OPENVINO_GPU)",
      "available": true,
      "target": "OPENVINO_GPU"
    },
    {
      "value": "intel_npu",
      "label": "Intel NPU (OpenVINO)",
      "available": false,
      "target": "OPENVINO_GPU",
      "reason": "Qwen3-ASR currently uses Intel GPU or CPU because the OpenVINO NPU compiler cannot lower this graph."
    }
  ]
}
```

### POST /api/warmup

Request:

```json
{
  "model": "neosophie/Qwen3-ASR-1.7B-JA",
  "device": "auto"
}
```

Response:

```json
{
  "model": "neosophie/Qwen3-ASR-1.7B-JA",
  "requested_device": "auto",
  "selected_device": "OPENVINO_GPU",
  "available_devices": ["OPENVINO_NPU", "OPENVINO_GPU", "CPU"],
  "model_load_seconds": 12.3456,
  "cache_hit": false,
  "cache_dir": "cache\\openvino",
  "fallback_device": null
}
```

### POST /api/transcribe

Accepts `multipart/form-data`.

Fields:

```text
model: Hugging Face model id
device: auto | intel_npu | intel_gpu | cpu
audio: uploaded audio chunk
```

The server does not write the uploaded chunk to a temporary file. It decodes audio bytes in memory:

```text
soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
```

If the sample rate is not 16 kHz, it resamples with `librosa.resample()`.

Response:

```json
{
  "text": "recognized text",
  "chunks": [],
  "model": "neosophie/Qwen3-ASR-1.7B-JA",
  "requested_device": "auto",
  "selected_device": "OPENVINO_GPU",
  "fallback_device": null,
  "available_devices": ["OPENVINO_NPU", "OPENVINO_GPU", "CPU"],
  "input_source": "live-1.wav",
  "benchmark": {
    "model_load_seconds": 0.0,
    "audio_preprocess_seconds": 0.0,
    "inference_seconds": 0.4567,
    "postprocess_seconds": 0.0,
    "total_processing_seconds": 0.469,
    "audio_duration_seconds": 1.0,
    "rtf": 0.469
  }
}
```

Errors return HTTP 400 with:

```json
{ "error": "message" }
```

## Frontend

`static/index.html` is a single-screen UI:

- Device selector
- Start / Stop / Clear buttons
- audio meter
- status
- warmup notice
- Transcript
- latest metrics

`static/app.js` handles:

- loading device choices
- model warmup
- microphone capture
- frame collection through `ScriptProcessorNode`
- RMS-based speech/silence detection
- chunk encoding to 16 kHz WAV
- serialized `/api/transcribe` calls
- Transcript and metrics updates

### Chunk Settings

```js
const chunking = {
  speechRmsThreshold: 0.01,
  submitRmsThreshold: 0.003,
  minChunkMs: 350,
  trailingSilenceMs: 280,
  maxChunkMs: 2400,
  idleDropMs: 900,
};
```

`submitChain` serializes requests so multiple chunks do not access the same runner concurrently.

### Transcript Layout

`static/styles.css` keeps the app inside the viewport:

- `body` and `.app-shell` use `100dvh`
- `.workspace` receives the remaining height
- `.transcript-pane` does not grow with content
- `#transcript` scrolls internally
- narrow layouts reserve at least 280 px for Transcript
- metrics are constrained to a small scroll area on narrow layouts

`appendLiveResult()` only auto-scrolls when the user is already near the bottom. If the user scrolls up to read older text, new chunks do not force the panel back to the bottom.

## BenchmarkResult

```text
model_load_seconds
audio_preprocess_seconds
inference_seconds
postprocess_seconds
total_processing_seconds
audio_duration_seconds
rtf
```

```text
rtf = total_processing_seconds / audio_duration_seconds
```

## Checks

Python syntax:

```powershell
.\.venv\Scripts\python.exe -m py_compile main.py web_app.py third_party/qwen_3_asr_helper.py
```

Device API:

```powershell
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/devices?model=neosophie%2FQwen3-ASR-1.7B-JA" | Select-Object -ExpandProperty Content
```

CLI:

```powershell
.\.venv\Scripts\python.exe main.py --device auto --model neosophie/Qwen3-ASR-1.7B-JA --audio sample.wav --benchmark --json
```

Browser:

1. Open `http://127.0.0.1:8000`.
2. Hard reload with `Ctrl + F5`.
3. Press `Start`.
4. Allow microphone access.
5. Confirm Transcript stays visible and scrolls internally.
6. Confirm metrics do not collapse the Transcript area.

## Notes

- Qwen3-ASR is run as short chunked inference, not true token streaming.
- First run can take a long time because of model download and OpenVINO IR conversion.
- Qwen3-ASR Intel NPU execution is disabled; use GPU or CPU.
- Browser microphone capture requires `http://127.0.0.1` or HTTPS.
- `ScriptProcessorNode` is an older API, kept here for simplicity.
- Long-running servers keep loaded model runners in memory through `RUNNER_CACHE`.
