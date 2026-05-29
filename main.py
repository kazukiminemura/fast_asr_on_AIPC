from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEVICE_PRIORITY = ("GPU", "NPU", "CPU")
DEVICE_ALIASES = {
    "auto": "AUTO",
    "gpu": "GPU",
    "npu": "NPU",
    "cpu": "CPU",
}
PIPELINE_CACHE: dict[tuple[str, str], Any] = {}
OPENVINO_CACHE_DIR = Path("cache") / "openvino"


@dataclass
class BenchmarkResult:
    model_load_seconds: float
    audio_preprocess_seconds: float
    inference_seconds: float
    postprocess_seconds: float
    total_processing_seconds: float
    audio_duration_seconds: float
    rtf: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Whisper ASR with OpenVINO GenAI on GPU, NPU, or CPU.",
    )
    parser.add_argument(
        "--device",
        choices=tuple(DEVICE_ALIASES),
        default="auto",
        help="Inference device. auto prefers GPU, then NPU, then CPU.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to an OpenVINO GenAI Whisper model directory.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio",
        type=Path,
        help="Path to an input audio file. It is resampled to 16 kHz.",
    )
    input_group.add_argument(
        "--mic",
        action="store_true",
        help="Record audio from the microphone instead of reading a file.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing metrics and Real Time Factor.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Microphone recording duration in seconds.",
    )
    parser.add_argument(
        "--input-device",
        help="Optional microphone device name or index used by sounddevice.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    parser.add_argument(
        "--language",
        help='Optional Whisper language token, for example "<|en|>" or "<|ja|>".',
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper task.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Return sentence-level timestamps when supported by the model.",
    )
    return parser.parse_args()


def require_dependency(module_name: str, package_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {package_name}. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc


def available_devices() -> list[str]:
    openvino = require_dependency("openvino", "openvino")
    core = openvino.Core()
    return list(core.available_devices)


def select_device(requested: str, devices: list[str]) -> str:
    requested_device = DEVICE_ALIASES[requested]
    if requested_device != "AUTO":
        if requested_device not in devices:
            available = ", ".join(devices) or "none"
            raise SystemExit(
                f"Requested device {requested_device} is not available. "
                f"Available devices: {available}"
            )
        return requested_device

    for device in DEVICE_PRIORITY:
        if device in devices:
            return device

    available = ", ".join(devices) or "none"
    raise SystemExit(f"No supported OpenVINO device found. Available devices: {available}")


def validate_paths(model_path: Path, audio_path: Path | None) -> None:
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise SystemExit(f"Model path must be an OpenVINO GenAI model directory: {model_path}")
    if audio_path is None:
        return
    if not audio_path.exists():
        raise SystemExit(f"Audio path does not exist: {audio_path}")
    if not audio_path.is_file():
        raise SystemExit(f"Audio path must be a file: {audio_path}")


def read_audio_16khz(audio_path: Path) -> tuple[list[float], float]:
    librosa = require_dependency("librosa", "librosa")
    raw_speech, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    duration_seconds = float(len(raw_speech) / sample_rate) if sample_rate else 0.0
    return raw_speech.astype("float32").tolist(), duration_seconds


def read_microphone_16khz(duration_seconds: float, input_device: str | None) -> tuple[list[float], float]:
    if duration_seconds <= 0:
        raise SystemExit("--duration must be greater than 0.")

    sounddevice = require_dependency("sounddevice", "sounddevice")
    sample_rate = 16000
    frame_count = int(duration_seconds * sample_rate)
    device: int | str | None = input_device
    if input_device is not None and input_device.isdigit():
        device = int(input_device)

    print(f"Recording microphone for {duration_seconds:.1f} seconds...", file=sys.stderr)
    recording = sounddevice.rec(
        frame_count,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sounddevice.wait()
    return recording.reshape(-1).tolist(), duration_seconds


def build_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "task": args.task,
    }
    if args.language:
        kwargs["language"] = args.language
    if args.timestamps:
        kwargs["return_timestamps"] = True
    return kwargs


def extract_text(result: Any) -> str:
    if hasattr(result, "texts") and result.texts:
        return str(result.texts[0]).strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()


def extract_chunks(result: Any) -> list[dict[str, Any]]:
    chunks = []
    for chunk in getattr(result, "chunks", []) or []:
        chunks.append(
            {
                "start": getattr(chunk, "start_ts", None),
                "end": getattr(chunk, "end_ts", None),
                "text": getattr(chunk, "text", ""),
            }
        )
    return chunks


def get_whisper_pipeline(model_path: Path, device: str) -> tuple[Any, float]:
    cache_key = (str(model_path.resolve()), device)
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key], 0.0

    ov_genai = require_dependency("openvino_genai", "openvino-genai")
    OPENVINO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    try:
        pipe = ov_genai.WhisperPipeline(
            str(model_path),
            device,
            CACHE_DIR=str(OPENVINO_CACHE_DIR),
        )
    except TypeError:
        pipe = ov_genai.WhisperPipeline(str(model_path), device)
    load_seconds = time.perf_counter() - load_start
    PIPELINE_CACHE[cache_key] = pipe
    return pipe, load_seconds


def warmup_asr_model(model_path: Path, requested_device: str) -> dict[str, Any]:
    validate_paths(model_path, None)
    devices = available_devices()
    selected_device = select_device(requested_device, devices)
    _pipe, model_load_seconds = get_whisper_pipeline(model_path, selected_device)
    return {
        "requested_device": requested_device,
        "selected_device": selected_device,
        "available_devices": devices,
        "model_load_seconds": model_load_seconds,
        "cache_hit": model_load_seconds == 0.0,
        "cache_dir": str(OPENVINO_CACHE_DIR),
    }


def run_asr(args: argparse.Namespace) -> tuple[dict[str, Any], BenchmarkResult]:
    validate_paths(args.model, args.audio)

    device_start = time.perf_counter()
    devices = available_devices()
    selected_device = select_device(args.device, devices)
    device_selection_seconds = time.perf_counter() - device_start

    pipe, model_load_seconds = get_whisper_pipeline(args.model, selected_device)

    preprocess_start = time.perf_counter()
    if args.mic:
        raw_speech, duration_seconds = read_microphone_16khz(args.duration, args.input_device)
        input_source = "microphone"
    else:
        raw_speech, duration_seconds = read_audio_16khz(args.audio)
        input_source = str(args.audio)
    audio_preprocess_seconds = time.perf_counter() - preprocess_start

    inference_start = time.perf_counter()
    result = pipe.generate(raw_speech, **build_generation_kwargs(args))
    inference_seconds = time.perf_counter() - inference_start

    postprocess_start = time.perf_counter()
    text = extract_text(result)
    chunks = extract_chunks(result)
    postprocess_seconds = time.perf_counter() - postprocess_start

    total_seconds = (
        device_selection_seconds
        + model_load_seconds
        + audio_preprocess_seconds
        + inference_seconds
        + postprocess_seconds
    )
    rtf = total_seconds / duration_seconds if duration_seconds > 0 else None

    benchmark = BenchmarkResult(
        model_load_seconds=model_load_seconds,
        audio_preprocess_seconds=audio_preprocess_seconds,
        inference_seconds=inference_seconds,
        postprocess_seconds=postprocess_seconds,
        total_processing_seconds=total_seconds,
        audio_duration_seconds=duration_seconds,
        rtf=rtf,
    )

    payload = {
        "text": text,
        "chunks": chunks,
        "requested_device": args.device,
        "selected_device": selected_device,
        "available_devices": devices,
        "input_source": input_source,
        "benchmark": asdict(benchmark),
    }
    return payload, benchmark


def print_human_output(payload: dict[str, Any], show_benchmark: bool) -> None:
    print(payload["text"])

    if payload["chunks"]:
        print()
        for chunk in payload["chunks"]:
            start = chunk["start"]
            end = chunk["end"]
            if start is None or end is None:
                print(chunk["text"])
            else:
                print(f"[{start:.2f} - {end:.2f}] {chunk['text']}")

    if show_benchmark:
        benchmark = payload["benchmark"]
        print()
        print(f"selected_device: {payload['selected_device']}")
        print(f"available_devices: {', '.join(payload['available_devices'])}")
        print(f"model_load_seconds: {benchmark['model_load_seconds']:.4f}")
        print(f"audio_preprocess_seconds: {benchmark['audio_preprocess_seconds']:.4f}")
        print(f"inference_seconds: {benchmark['inference_seconds']:.4f}")
        print(f"postprocess_seconds: {benchmark['postprocess_seconds']:.4f}")
        print(f"total_processing_seconds: {benchmark['total_processing_seconds']:.4f}")
        print(f"audio_duration_seconds: {benchmark['audio_duration_seconds']:.4f}")
        rtf = benchmark["rtf"]
        print(f"rtf: {rtf:.4f}" if rtf is not None else "rtf: n/a")


def main() -> int:
    args = parse_args()
    payload, _benchmark = run_asr(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human_output(payload, args.benchmark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
