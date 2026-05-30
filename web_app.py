from __future__ import annotations

import argparse
import io
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from main import (
    DEFAULT_MODEL,
    available_devices,
    device_choices,
    model_choices,
    require_dependency,
    run_asr_samples,
    warmup_asr_model,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"


class WebAsrHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed_path = urlparse(path).path
        if parsed_path == "/":
            return str(STATIC_DIR / "index.html")
        if parsed_path.startswith("/static/"):
            relative = parsed_path.removeprefix("/static/")
            return str(STATIC_DIR / relative)
        return str(STATIC_DIR / parsed_path.lstrip("/"))

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/devices":
            self.write_json(safe_device_payload(parsed_path.query))
            return
        if parsed_path.path == "/api/models":
            self.write_json({"models": model_choices()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/warmup":
            self.handle_warmup()
            return
        if parsed_path.path == "/api/transcribe":
            self.handle_transcribe()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_warmup(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            fields = json.loads(body.decode("utf-8") or "{}")
            model = fields.get("model", "").strip()
            if not model:
                raise ValueError("Model path is required.")
            result = warmup_asr_model(model, fields.get("device", "auto"))
            self.write_json(result)
        except SystemExit as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def handle_transcribe(self) -> None:
        try:
            fields, files = self.parse_multipart_form()
            result = transcribe_form(fields, files)
            self.write_json(result)
        except SystemExit as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def parse_multipart_form(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart/form-data.")

        boundary_token = "boundary="
        if boundary_token not in content_type:
            raise ValueError("Missing multipart boundary.")
        boundary = content_type.split(boundary_token, 1)[1].strip().strip('"')

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        delimiter = b"--" + boundary.encode("utf-8")
        fields: dict[str, str] = {}
        files: dict[str, tuple[str, bytes]] = {}

        for raw_part in body.split(delimiter):
            part = raw_part.strip()
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].strip()

            header_block, separator, value = part.partition(b"\r\n\r\n")
            if not separator:
                continue

            headers = parse_part_headers(header_block.decode("utf-8", errors="replace"))
            disposition = headers.get("content-disposition", "")
            disposition_items = parse_content_disposition(disposition)
            name = disposition_items.get("name")
            if not name:
                continue

            value = value.rstrip(b"\r\n")
            filename = disposition_items.get("filename")
            if filename:
                files[name] = (filename, value)
            else:
                fields[name] = value.decode("utf-8", errors="replace")

        return fields, files

    def write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_part_headers(header_block: str) -> dict[str, str]:
    headers = {}
    for line in header_block.split("\r\n"):
        key, separator, value = line.partition(":")
        if separator:
            headers[key.strip().lower()] = value.strip()
    return headers


def parse_content_disposition(value: str) -> dict[str, str]:
    items = {}
    for part in value.split(";"):
        key, separator, raw_value = part.strip().partition("=")
        if separator:
            items[key] = raw_value.strip().strip('"')
    return items


def safe_device_payload(query: str = "") -> dict:
    try:
        model = ""
        for item in query.split("&"):
            key, separator, value = item.partition("=")
            if separator and key == "model":
                from urllib.parse import unquote_plus

                model = unquote_plus(value).strip()
                break
        return {
            "devices": available_devices(),
            "choices": device_choices(model or DEFAULT_MODEL),
        }
    except SystemExit:
        return {"devices": [], "choices": []}


def transcribe_form(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes]],
) -> dict:
    model = fields.get("model", "").strip()
    if not model:
        raise ValueError("Model path is required.")

    audio_file = files.get("audio")
    if not audio_file or not audio_file[1]:
        raise ValueError("Audio input is required.")

    raw_speech, duration_seconds = read_audio_upload_16khz(audio_file[1])
    payload, _benchmark = run_asr_samples(
        model_ref=model,
        requested_device=fields.get("device", "auto"),
        raw_speech=raw_speech,
        duration_seconds=duration_seconds,
        input_source=audio_file[0],
        max_new_tokens=64,
    )
    return payload


def read_audio_upload_16khz(audio_bytes: bytes) -> tuple[list[float], float]:
    soundfile = require_dependency("soundfile", "soundfile")
    numpy = require_dependency("numpy", "numpy")
    audio, sample_rate = soundfile.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        librosa = require_dependency("librosa", "librosa")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    audio = numpy.asarray(audio, dtype="float32")
    duration_seconds = float(len(audio) / sample_rate) if sample_rate else 0.0
    return audio.tolist(), duration_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser GUI for AIPC ASR.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), WebAsrHandler)
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
