# SPDX-License-Identifier: MIT
"""HTTP bridge for the OwnedLM UI and its local PyTorch checkpoint."""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

from generate import generate
from model import TinyLM, config_from_preset
from tokenizer import CharTokenizer
from dataset import load_text_files
import torch

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "dist"
CHECKPOINT = ROOT / "checkpoints" / "tiny.pt"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
MAX_BODY_BYTES = 16_384
MAX_PROMPT_CHARS = 2_000
MAX_NEW_TOKENS = 40
BOOTSTRAP_ERROR = ""


class OwnedLMHandler(BaseHTTPRequestHandler):
    server_version = "OwnedLM/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[OwnedLM] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_web(self, request_path: str) -> None:
        if not WEB_ROOT.is_dir():
            self._send_json(503, {"error": "واجهة الويب غير مبنية داخل الحاوية."})
            return
        relative = unquote(request_path).lstrip("/") or "index.html"
        candidate = (WEB_ROOT / relative).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._send_json(404, {"error": "المسار غير موجود."})
            return
        if not candidate.is_file():
            # Client-side routes should load the single-page app entry point.
            candidate = WEB_ROOT / "index.html"
        if not candidate.is_file():
            self._send_json(404, {"error": "ملف الواجهة غير موجود."})
            return
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            ready = CHECKPOINT.is_file()
            self._send_json(
                200,
                {
                    "ready": ready,
                    "engine": "local-pytorch",
                    "checkpoint": "checkpoints/tiny.pt",
                    "message": (
                        "النموذج التجريبي جاهز للتوليد."
                        if ready
                        else "لا توجد نقطة حفظ للنموذج داخل النشر بعد؛ الواجهة تعمل لكن الدردشة غير جاهزة."
                    ),
                    "bootstrap_error": BOOTSTRAP_ERROR,
                },
            )
            return
        if path.startswith("/api/"):
            self._send_json(404, {"error": "المسار غير موجود."})
            return
        self._serve_web(path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"error": "المسار غير موجود."})
            return

        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json(400, {"error": "طول الطلب غير صالح."})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "حجم الطلب غير مسموح."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "أرسل طلب JSON صالحاً."})
            return
        prompt = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(400, {"error": "اكتب رسالة أولاً."})
            return
        prompt = prompt.strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            self._send_json(413, {"error": f"الرسالة طويلة؛ الحد {MAX_PROMPT_CHARS} حرف."})
            return
        if not CHECKPOINT.is_file():
            self._send_json(
                503,
                {"error": "النموذج غير موجود على الخادم بعد؛ يجب توفير checkpoints/tiny.pt قبل تفعيل الدردشة."},
            )
            return
        try:
            max_new_tokens = int(payload.get("max_new_tokens", 32))
        except (TypeError, ValueError):
            max_new_tokens = 32
        max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS))
        try:
            # Limit CPU thread fan-out on small shared hosting instances.
            # This model is tiny; excessive PyTorch threads can make it slower.
            torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
            generated = generate(
                SimpleNamespace(
                    checkpoint=str(CHECKPOINT),
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=20,
                    seed=None,
                    device="cpu",
                    stop_at_eos=True,
                ),
                include_prompt=False,
            )
        except Exception as exc:
            print(f"[OwnedLM] generation error: {type(exc).__name__}: {exc}")
            self._send_json(500, {"error": "تعذّر توليد النص. راجع سجل خادم Python."})
            return
        self._send_json(
            200,
            {
                "reply": generated.strip() or "النموذج لم يولّد تتمة واضحة لهذه الرسالة.",
                "engine": "local-pytorch",
                "note": "تتمة نصية من نموذج تجريبي صغير، وليست إجابة موثوقة مثل المساعدات الكبيرة.",
            },
        )


def _ensure_checkpoint() -> None:
    """Create a tiny local checkpoint when deployment did not include one."""
    if CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 0:
        return

    print("[OwnedLM] No usable checkpoint found; training a tiny bootstrap model.")
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--data", str(ROOT / "data" / "corpus"),
        "--preset", "tiny",
        "--steps", os.environ.get("OWNEDLM_BOOTSTRAP_STEPS", "20"),
        "--batch-size", "2",
        "--seq-len", "64",
        "--log-every", "20",
        "--save-every", "20",
        "--out", str(CHECKPOINT),
        "--device", "cpu",
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=240)
        if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
            raise RuntimeError("training finished without a non-empty checkpoint")
        print(f"[OwnedLM] Bootstrap checkpoint created ({CHECKPOINT.stat().st_size} bytes).")
    except Exception as exc:
        global BOOTSTRAP_ERROR
        BOOTSTRAP_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[OwnedLM] Bootstrap training failed: {BOOTSTRAP_ERROR}")
        # Last-resort checkpoint: keep the API usable even if short bootstrap
        # training fails in a restricted hosting environment. This fallback
        # is randomly initialized and must not be presented as a trained model.
        try:
            documents = load_text_files(ROOT / "data" / "corpus")
            texts = [text for _name, text in documents]
            if not texts:
                raise RuntimeError("no training text files found")
            tokenizer = CharTokenizer.from_texts(texts)
            cfg = config_from_preset("tiny", tokenizer.vocab_size, pad_id=tokenizer.pad_id)
            model = TinyLM(cfg)
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": 1,
                    "step": 0,
                    "config": cfg.to_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "model": model.state_dict(),
                },
                CHECKPOINT,
            )
            if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
                raise RuntimeError("fallback checkpoint was not saved")
            print(
                f"[OwnedLM] Created fallback checkpoint ({CHECKPOINT.stat().st_size} bytes); "
                "weights are untrained."
            )
        except Exception as fallback_exc:
            BOOTSTRAP_ERROR += (
                f"; fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
            )
            print(f"[OwnedLM] Fallback checkpoint failed: {fallback_exc}")


def main() -> None:
    _ensure_checkpoint()
    server = ThreadingHTTPServer((HOST, PORT), OwnedLMHandler)
    print(f"OwnedLM server listening on http://{HOST}:{PORT}")
    print(f"Checkpoint: {CHECKPOINT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nإيقاف خادم OwnedLM...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
