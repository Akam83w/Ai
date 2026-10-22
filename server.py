# SPDX-License-Identifier: MIT
"""Local-only HTTP bridge between the OwnedLM web UI and the trained checkpoint.

Run with: python server.py
The server binds to loopback only and never calls an external model API.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from generate import generate

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "tiny.pt"
HOST = "127.0.0.1"
PORT = 8090
MAX_BODY_BYTES = 16_384
MAX_PROMPT_CHARS = 2_000
MAX_NEW_TOKENS = 120


class OwnedLMHandler(BaseHTTPRequestHandler):
    server_version = "OwnedLMLocal/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep requests visible in the terminal without exposing prompt text.
        print(f"[OwnedLM] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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
                        "النموذج المدرّب جاهز للتوليد."
                        if ready
                        else "لا توجد نقطة حفظ بعد. درّب النموذج أولاً حتى checkpoints/tiny.pt."
                    ),
                },
            )
            return
        self._send_json(404, {"error": "المسار غير موجود."})

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
                {
                    "error": "النموذج غير مدرّب بعد. شغّل أمر التدريب الظاهر في الواجهة أولاً."
                },
            )
            return

        try:
            max_new_tokens = int(payload.get("max_new_tokens", 80))
        except (TypeError, ValueError):
            max_new_tokens = 80
        max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS))

        try:
            generated = generate(
                SimpleNamespace(
                    checkpoint=str(CHECKPOINT),
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=30,
                    seed=None,
                    device="cpu",
                    stop_at_eos=True,
                )
            )
        except Exception as exc:
            # Details remain in the local terminal; do not leak tracebacks to the browser.
            print(f"[OwnedLM] generation error: {type(exc).__name__}: {exc}")
            self._send_json(500, {"error": "تعذّر توليد النص. راجع سجل خادم Python."})
            return

        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        self._send_json(
            200,
            {
                "reply": continuation.strip() or "النموذج لم يولّد تتمة واضحة لهذه الرسالة.",
                "engine": "local-pytorch",
                "note": "تتمة نصية من نموذج تجريبي صغير، وليست إجابة موثوقة مثل المساعدات الكبيرة.",
            },
        )


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), OwnedLMHandler)
    print(f"OwnedLM local API: http://{HOST}:{PORT}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("لا توجد اتصالات بنماذج خارجية. أوقف الخادم باستخدام Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nإيقاف خادم OwnedLM...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
