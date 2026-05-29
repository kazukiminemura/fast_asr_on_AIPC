from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "UsefulSensors/moonshine-tiny-ja"
HF_CACHE_DIR = Path("cache") / "huggingface"
OPENVINO_CACHE_DIR = Path("cache") / "openvino"
DEVICE_ALIASES = {
    "auto": "AUTO",
    "intel_gpu": "OPENVINO_GPU",
    "gpu": "CUDA",
    "cpu": "CPU",
}
RUNNER_CACHE: dict[tuple[str, str], Any] = {}


@dataclass
class BenchmarkResult:
    model_load_seconds: float
    audio_preprocess_seconds: float
    inference_seconds: float
    postprocess_seconds: float
    total_processing_seconds: float
    audio_duration_seconds: float
    rtf: float | None


class MoonshineRunner:
    def __init__(self, model_ref: str, device: str) -> None:
        torch = require_dependency("torch", "torch")
        transformers = require_dependency("transformers", "transformers")

        self.torch = torch
        self.device_name = device
        self.torch_device = "cuda:0" if device == "CUDA" else "cpu"
        self.torch_dtype = torch.float16 if device == "CUDA" else torch.float32
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_ref,
            cache_dir=str(HF_CACHE_DIR),
        )
        self.model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_ref,
            cache_dir=str(HF_CACHE_DIR),
        )
        self.model.to(self.torch_device)
        self.model.to(self.torch_dtype)
        self.model.eval()

    def transcribe(self, raw_speech: list[float], max_tokens: int) -> str:
        sampling_rate = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(
            raw_speech,
            return_tensors="pt",
            sampling_rate=sampling_rate,
        )
        inputs = inputs.to(self.torch_device)
        if "input_values" in inputs:
            inputs["input_values"] = inputs["input_values"].to(self.torch_dtype)

        max_length = max(1, max_tokens)
        if "attention_mask" in inputs:
            token_limit_factor = 13 / sampling_rate
            seq_lens = inputs["attention_mask"].sum(dim=-1)
            estimated_limit = int((seq_lens * token_limit_factor).max().item())
            max_length = max(1, min(max_tokens, estimated_limit))

        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_length=max_length)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()


class OpenVINOMoonshineRunner:
    def __init__(self, model_ref: str, device: str) -> None:
        optimum_intel = require_dependency("optimum.intel", "optimum-intel[openvino]")
        openvino_export = require_dependency("optimum.exporters.openvino", "optimum-intel[openvino]")
        transformers = require_dependency("transformers", "transformers")

        ov_model_dir = OPENVINO_CACHE_DIR / sanitize_model_ref(model_ref)
        ov_config = {"CACHE_DIR": str(OPENVINO_CACHE_DIR / "compiled")}
        self.device_name = device
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_ref,
            cache_dir=str(HF_CACHE_DIR),
        )

        has_openvino_model = ov_model_dir.exists() and any(ov_model_dir.glob("*.xml"))
        if has_openvino_model:
            self.model = optimum_intel.OVModelForSpeechSeq2Seq.from_pretrained(
                ov_model_dir,
                device=device,
                ov_config=ov_config,
            )
        else:
            ov_model_dir.mkdir(parents=True, exist_ok=True)
            openvino_export.main_export(
                model_name_or_path=model_ref,
                output=ov_model_dir,
                task="automatic-speech-recognition",
                cache_dir=str(HF_CACHE_DIR),
                library_name="transformers",
                stateful=True,
            )
            self.model = optimum_intel.OVModelForSpeechSeq2Seq.from_pretrained(
                ov_model_dir,
                device=device,
                ov_config=ov_config,
            )
            self.processor.save_pretrained(ov_model_dir)
        self.model.main_input_name = "input_values"
        self.model.encoder.main_input_name = "input_values"

    def transcribe(self, raw_speech: list[float], max_tokens: int) -> str:
        sampling_rate = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(
            raw_speech,
            return_tensors="pt",
            sampling_rate=sampling_rate,
        )

        max_length = max(1, max_tokens)
        if "attention_mask" in inputs:
            token_limit_factor = 13 / sampling_rate
            seq_lens = inputs["attention_mask"].sum(dim=-1)
            estimated_limit = int((seq_lens * token_limit_factor).max().item())
            max_length = max(1, min(max_tokens, estimated_limit))

        generated_ids = self.model.generate(
            inputs=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
        )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Moonshine ASR with UsefulSensors/moonshine-tiny-ja.",
    )
    parser.add_argument(
        "--device",
        choices=tuple(DEVICE_ALIASES),
        default="auto",
        help="Inference device. auto prefers Intel GPU, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or local model directory.",
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
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    return parser.parse_args()


def require_dependency(module_name: str, package_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {package_name}. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc


def available_devices() -> list[str]:
    torch = require_dependency("torch", "torch")
    devices = ["CPU"]
    try:
        openvino = require_dependency("openvino", "openvino")
        core = openvino.Core()
        if "GPU" in core.available_devices:
            devices.insert(0, "OPENVINO_GPU")
    except SystemExit:
        pass
    if torch.cuda.is_available():
        insert_at = 1 if "OPENVINO_GPU" in devices else 0
        devices.insert(insert_at, "CUDA")
    return devices


def device_choices() -> list[dict[str, Any]]:
    devices = available_devices()
    auto_target = "OPENVINO_GPU" if "OPENVINO_GPU" in devices else "CUDA" if "CUDA" in devices else "CPU"
    return [
        {
            "value": "auto",
            "label": f"auto ({auto_target})",
            "available": True,
            "target": auto_target,
        },
        {
            "value": "intel_gpu",
            "label": "Intel GPU (OpenVINO)",
            "available": "OPENVINO_GPU" in devices,
            "target": "OPENVINO_GPU",
        },
        {
            "value": "cpu",
            "label": "CPU",
            "available": "CPU" in devices,
            "target": "CPU",
        },
        {
            "value": "gpu",
            "label": "GPU (CUDA)",
            "available": "CUDA" in devices,
            "target": "CUDA",
        },
    ]


def select_device(requested: str, devices: list[str]) -> str:
    requested_device = DEVICE_ALIASES[requested]
    if requested_device == "AUTO":
        if "OPENVINO_GPU" in devices:
            return "OPENVINO_GPU"
        return "CUDA" if "CUDA" in devices else "CPU"
    if requested_device not in devices:
        available = ", ".join(devices) or "none"
        raise SystemExit(
            f"Requested device {requested_device} is not available for the "
            f"Transformers Moonshine backend. Available devices: {available}"
        )
    return requested_device


def validate_inputs(model_ref: str, audio_path: Path | None) -> None:
    if not model_ref.strip():
        raise SystemExit("Model id or path is required.")
    model_path = Path(model_ref)
    if model_path.exists() and not model_path.is_dir():
        raise SystemExit(f"Local model path must be a directory: {model_path}")
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


def sanitize_model_ref(model_ref: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in model_ref)


def get_moonshine_runner(model_ref: str, device: str) -> tuple[Any, float]:
    cache_key = (model_ref, device)
    if cache_key in RUNNER_CACHE:
        return RUNNER_CACHE[cache_key], 0.0

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OPENVINO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    if device == "OPENVINO_GPU":
        runner = OpenVINOMoonshineRunner(model_ref, "GPU")
    else:
        runner = MoonshineRunner(model_ref, device)
    load_seconds = time.perf_counter() - load_start
    RUNNER_CACHE[cache_key] = runner
    return runner, load_seconds


def warmup_asr_model(model_ref: str, requested_device: str) -> dict[str, Any]:
    validate_inputs(model_ref, None)
    devices = available_devices()
    selected_device = select_device(requested_device, devices)
    _runner, model_load_seconds = get_moonshine_runner(model_ref, selected_device)
    fallback_load_seconds = 0.0
    if selected_device == "OPENVINO_GPU":
        _fallback_runner, fallback_load_seconds = get_moonshine_runner(model_ref, "CPU")
    return {
        "model": model_ref,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "available_devices": devices,
        "model_load_seconds": model_load_seconds + fallback_load_seconds,
        "cache_hit": model_load_seconds == 0.0 and fallback_load_seconds == 0.0,
        "cache_dir": str(OPENVINO_CACHE_DIR if selected_device == "OPENVINO_GPU" else HF_CACHE_DIR),
        "fallback_device": "CPU" if selected_device == "OPENVINO_GPU" else None,
    }


def run_asr(args: argparse.Namespace) -> tuple[dict[str, Any], BenchmarkResult]:
    validate_inputs(args.model, args.audio)

    device_start = time.perf_counter()
    devices = available_devices()
    selected_device = select_device(args.device, devices)
    device_selection_seconds = time.perf_counter() - device_start

    runner, model_load_seconds = get_moonshine_runner(args.model, selected_device)

    preprocess_start = time.perf_counter()
    if args.mic:
        raw_speech, duration_seconds = read_microphone_16khz(args.duration, args.input_device)
        input_source = "microphone"
    else:
        raw_speech, duration_seconds = read_audio_16khz(args.audio)
        input_source = str(args.audio)
    audio_preprocess_seconds = time.perf_counter() - preprocess_start

    inference_start = time.perf_counter()
    text = runner.transcribe(raw_speech, args.max_new_tokens)
    fallback_device = None
    if selected_device == "OPENVINO_GPU" and not text and audio_rms(raw_speech) > 0.001:
        fallback_runner, fallback_load_seconds = get_moonshine_runner(args.model, "CPU")
        model_load_seconds += fallback_load_seconds
        text = fallback_runner.transcribe(raw_speech, args.max_new_tokens)
        fallback_device = "CPU"
    inference_seconds = time.perf_counter() - inference_start

    postprocess_start = time.perf_counter()
    text = text.strip()
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
        "chunks": [],
        "model": args.model,
        "requested_device": args.device,
        "selected_device": selected_device,
        "fallback_device": fallback_device,
        "available_devices": devices,
        "input_source": input_source,
        "benchmark": asdict(benchmark),
    }
    return payload, benchmark


def audio_rms(raw_speech: list[float]) -> float:
    if not raw_speech:
        return 0.0
    square_sum = sum(sample * sample for sample in raw_speech)
    return (square_sum / len(raw_speech)) ** 0.5


def print_human_output(payload: dict[str, Any], show_benchmark: bool) -> None:
    print(payload["text"])

    if show_benchmark:
        benchmark = payload["benchmark"]
        print()
        print(f"model: {payload['model']}")
        print(f"selected_device: {payload['selected_device']}")
        if payload.get("fallback_device"):
            print(f"fallback_device: {payload['fallback_device']}")
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
