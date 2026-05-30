from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "neosophie/Qwen3-ASR-1.7B-JA"
JAPANESE_LANGUAGE = "Japanese"
QWEN3_ASR_MODELS = {DEFAULT_MODEL}
MODEL_CHOICES = [
    {
        "value": DEFAULT_MODEL,
        "label": "Japanese",
        "description": "Japanese ASR with Qwen3-ASR JA via qwen-asr / OpenVINO",
    },
]
HF_CACHE_DIR = Path("cache") / "huggingface"
OPENVINO_CACHE_DIR = Path("cache") / "openvino"
DEVICE_ALIASES = {
    "auto": "AUTO",
    "intel_npu": "OPENVINO_NPU",
    "npu": "OPENVINO_NPU",
    "intel_gpu": "OPENVINO_GPU",
    "cpu": "CPU",
}
RUNNER_CACHE: dict[tuple[str, str], Any] = {}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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


class Qwen3AsrRunner:
    def __init__(self, model_ref: str, device: str) -> None:
        try:
            qwen_asr = importlib.import_module("qwen_asr")
        except ImportError as exc:
            raise SystemExit(
                f"{model_ref} requires the qwen-asr package. Install it with "
                "`python -m pip install -U qwen-asr` and restart the app."
            ) from exc

        torch = require_dependency("torch", "torch")
        self.model_ref = model_ref
        self.device_name = device
        self.soundfile = require_dependency("soundfile", "soundfile")
        device_map = "cuda:0" if device == "CUDA" else "cpu"
        dtype = torch.bfloat16 if device == "CUDA" else torch.float32
        self.model = qwen_asr.Qwen3ASRModel.from_pretrained(
            model_ref,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=64,
        )

    def transcribe(self, raw_speech: list[float], max_tokens: int) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = Path(temp_audio.name)

        try:
            self.soundfile.write(str(temp_audio_path), raw_speech, 16000)
            results = self.model.transcribe(
                audio=str(temp_audio_path),
                language=JAPANESE_LANGUAGE,
            )
            if not results:
                return ""
            return getattr(results[0], "text", "").strip()
        finally:
            temp_audio_path.unlink(missing_ok=True)


class OpenVINOQwen3AsrRunner:
    def __init__(self, model_ref: str, device: str) -> None:
        try:
            qwen3_asr_helper = importlib.import_module("third_party.qwen_3_asr_helper")
        except ImportError as exc:
            raise SystemExit(
                "OpenVINO Qwen3-ASR helper is missing. Expected "
                "third_party/qwen_3_asr_helper.py."
            ) from exc

        self.model_ref = model_ref
        if device == "NPU":
            fallback_device = openvino_target_name(qwen3_asr_device_target(available_devices()))
            print(
                "Qwen3-ASR is not routed to Intel NPU because the OpenVINO NPU "
                f"compiler cannot lower this graph. Using {fallback_device} instead.",
                file=sys.stderr,
            )
            device = fallback_device
        self.device_name = f"OPENVINO_{device}"
        self.soundfile = require_dependency("soundfile", "soundfile")
        ov_model_dir = OPENVINO_CACHE_DIR / sanitize_model_ref(model_ref)
        with contextlib.redirect_stdout(sys.stderr):
            qwen3_asr_helper.convert_qwen3_asr_model(
                model_id=model_ref,
                output_dir=ov_model_dir,
                quantization_config=None,
            )
            self.model = qwen3_asr_helper.OVQwen3ASRModel.from_pretrained(
                model_dir=str(ov_model_dir),
                device=device,
                max_inference_batch_size=1,
                max_new_tokens=64,
            )

    def transcribe(self, raw_speech: list[float], max_tokens: int) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = Path(temp_audio.name)

        try:
            self.soundfile.write(str(temp_audio_path), raw_speech, 16000)
            results = self.model.transcribe(
                audio=str(temp_audio_path),
                language=JAPANESE_LANGUAGE,
            )
            if not results:
                return ""
            return getattr(results[0], "text", "").strip()
        finally:
            temp_audio_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run ASR with {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--device",
        choices=tuple(DEVICE_ALIASES),
        default="auto",
        help="Inference device. Qwen3-ASR auto prefers Intel GPU, then CPU.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Japanese ASR model id or local model directory.",
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
        default=64,
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
    devices = ["CPU"]
    try:
        openvino = require_dependency("openvino", "openvino")
        core = openvino.Core()
        if "NPU" in core.available_devices:
            devices.insert(0, "OPENVINO_NPU")
        if "GPU" in core.available_devices:
            insert_at = 1 if "OPENVINO_NPU" in devices else 0
            devices.insert(insert_at, "OPENVINO_GPU")
    except SystemExit:
        pass
    return devices


def auto_device_target(devices: list[str]) -> str:
    for candidate in ("OPENVINO_NPU", "OPENVINO_GPU", "CPU"):
        if candidate in devices:
            return candidate
    return "CPU"


def qwen3_asr_device_target(devices: list[str]) -> str:
    for candidate in ("OPENVINO_GPU", "CPU"):
        if candidate in devices:
            return candidate
    return "CPU"


def is_openvino_device(device: str) -> bool:
    return device.startswith("OPENVINO_")


def openvino_target_name(device: str) -> str:
    return device.removeprefix("OPENVINO_")


def device_choices(model_ref: str = DEFAULT_MODEL) -> list[dict[str, Any]]:
    devices = available_devices()
    if is_qwen3_asr_model(model_ref):
        auto_target = qwen3_asr_device_target(devices)
        return [
            {
                "value": "auto",
                "label": f"auto ({auto_target})",
                "available": True,
                "target": auto_target,
            },
            {
                "value": "intel_npu",
                "label": "Intel NPU (OpenVINO)",
                "available": False,
                "target": auto_target,
                "reason": "Qwen3-ASR currently uses Intel GPU or CPU because the OpenVINO NPU compiler cannot lower this graph.",
            },
            {
                "value": "intel_gpu",
                "label": "Intel GPU (OpenVINO)",
                "available": "OPENVINO_GPU" in devices,
                "target": "OPENVINO_GPU",
                "reason": "Uses converted OpenVINO IR under cache/openvino.",
            },
            {
                "value": "cpu",
                "label": "CPU",
                "available": "CPU" in devices,
                "target": "CPU",
            },
        ]

    auto_target = auto_device_target(devices)
    return [
        {
            "value": "auto",
            "label": f"auto ({auto_target})",
            "available": True,
            "target": auto_target,
        },
        {
            "value": "intel_npu",
            "label": "Intel NPU (OpenVINO)",
            "available": "OPENVINO_NPU" in devices,
            "target": "OPENVINO_NPU",
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
    ]


def model_choices() -> list[dict[str, str]]:
    return MODEL_CHOICES


def is_qwen3_asr_model(model_ref: str) -> bool:
    model_ref = model_ref.strip()
    return model_ref in QWEN3_ASR_MODELS or "Qwen3-ASR" in model_ref


def select_device(requested: str, devices: list[str]) -> str:
    requested_device = DEVICE_ALIASES[requested]
    if requested_device == "AUTO":
        return auto_device_target(devices)
    if requested_device not in devices:
        available = ", ".join(devices) or "none"
        raise SystemExit(
            f"Requested device {requested_device} is not available for the "
            f"ASR backend. Available devices: {available}"
        )
    return requested_device


def select_model_device(model_ref: str, requested: str, devices: list[str]) -> str:
    if is_qwen3_asr_model(model_ref):
        if requested in {"auto", "intel_npu", "npu", "OPENVINO_NPU", "NPU"}:
            return qwen3_asr_device_target(devices)
    selected_device = select_device(requested, devices)
    return selected_device


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
    if is_qwen3_asr_model(model_ref) and is_openvino_device(device):
        runner = OpenVINOQwen3AsrRunner(model_ref, openvino_target_name(device))
    elif is_qwen3_asr_model(model_ref):
        runner = Qwen3AsrRunner(model_ref, device)
    elif is_openvino_device(device):
        runner = OpenVINOMoonshineRunner(model_ref, openvino_target_name(device))
    else:
        runner = MoonshineRunner(model_ref, device)
    load_seconds = time.perf_counter() - load_start
    RUNNER_CACHE[cache_key] = runner
    return runner, load_seconds


def warmup_asr_model(model_ref: str, requested_device: str) -> dict[str, Any]:
    validate_inputs(model_ref, None)
    devices = available_devices()
    selected_device = select_model_device(model_ref, requested_device, devices)
    _runner, model_load_seconds = get_moonshine_runner(model_ref, selected_device)
    fallback_load_seconds = 0.0
    needs_cpu_fallback = not is_qwen3_asr_model(model_ref) and is_openvino_device(selected_device)
    if needs_cpu_fallback:
        _fallback_runner, fallback_load_seconds = get_moonshine_runner(model_ref, "CPU")
    return {
        "model": model_ref,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "available_devices": devices,
        "model_load_seconds": model_load_seconds + fallback_load_seconds,
        "cache_hit": model_load_seconds == 0.0 and fallback_load_seconds == 0.0,
        "cache_dir": str(OPENVINO_CACHE_DIR if is_openvino_device(selected_device) else HF_CACHE_DIR),
        "fallback_device": "CPU" if needs_cpu_fallback else None,
    }


def run_asr(args: argparse.Namespace) -> tuple[dict[str, Any], BenchmarkResult]:
    validate_inputs(args.model, args.audio)

    device_start = time.perf_counter()
    devices = available_devices()
    selected_device = select_model_device(args.model, args.device, devices)
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
    if not is_qwen3_asr_model(args.model) and is_openvino_device(selected_device) and not text and audio_rms(raw_speech) > 0.001:
        fallback_runner, fallback_load_seconds = get_moonshine_runner(args.model, "CPU")
        model_load_seconds += fallback_load_seconds
        text = fallback_runner.transcribe(raw_speech, args.max_new_tokens)
        fallback_device = "CPU"
    inference_seconds = time.perf_counter() - inference_start

    postprocess_start = time.perf_counter()
    text = extract_asr_text(text)
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


def extract_asr_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    asr_tag = "<asr_text>"
    if asr_tag in text:
        text = text.split(asr_tag, 1)[1]

    text = re.sub(r"</?asr_text>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^language\s+\S+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    return text.strip()


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
