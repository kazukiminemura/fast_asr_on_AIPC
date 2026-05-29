from __future__ import annotations

import argparse
import json
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from main import available_devices, run_asr, warmup_asr_model


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
            self.write_json({"devices": safe_available_devices()})
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
            result = warmup_asr_model(Path(model), fields.get("device", "auto"))
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


def safe_available_devices() -> list[str]:
    try:
        return available_devices()
    except SystemExit:
        return []


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

    suffix = Path(audio_file[0]).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_file[1])
        temp_audio_path = Path(temp_audio.name)

    try:
        args = argparse.Namespace(
            device=fields.get("device", "auto"),
            model=Path(model),
            audio=temp_audio_path,
            mic=False,
            duration=float(fields.get("duration", "0") or "0"),
            input_device=None,
            benchmark=True,
            json=True,
            language=fields.get("language") or None,
            task=fields.get("task", "transcribe"),
            max_new_tokens=int(fields.get("max_new_tokens", "128") or "128"),
            timestamps=fields.get("timestamps") == "true",
        )
        payload, _benchmark = run_asr(args)
        return payload
    finally:
        temp_audio_path.unlink(missing_ok=True)


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
