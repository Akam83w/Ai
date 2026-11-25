# SPDX-License-Identifier: MIT
"""OwnedLM web server backed by the local pretrained Arabic Jais-3 checkpoint."""
from __future__ import annotations

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "dist"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
MODEL_ID = os.environ.get("JAIS_MODEL_ID", "Dhunun/OwnedLM-Jais-3-0.9B")
HF_HOME = os.environ.get("HF_HOME", "/data/huggingface")
MAX_BODY_BYTES = 48_000
MAX_PROMPT_CHARS = 4_000
MAX_HISTORY_MESSAGES = 12
MAX_NEW_TOKENS = 128
MIN_MODEL_RAM_GB = float(os.environ.get("OWNEDLM_MIN_RAM_GB", "4"))
MODEL = None
TOKENIZER = None
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
MODEL_LOADING = False
MODEL_ERROR = ""
MODEL_LOCK = threading.Lock()
GENERATION_LOCK = threading.Lock()
DEVICE = "cpu"
DTYPE = None

def container_memory_limit_bytes() -> int | None:
    """Return the container RAM limit when exposed by cgroups."""
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),  # cgroup v2
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),  # cgroup v1
    )
    limits = []
    for path in candidates:
        try:
            raw = path.read_text(encoding="ascii").strip()
            if raw.isdigit():
                value = int(raw)
                # cgroup v1 may report a huge sentinel when unlimited.
                if 0 < value < (1 << 60):
                    limits.append(value)
        except (OSError, ValueError):
            continue
    return min(limits) if limits else None


SYSTEM_PROMPT = (
    "أنت OwnedLM، مساعد عربي مستقل. أجب باللغة العربية الفصحى المعاصرة فقط. "
    "افهم سياق المحادثة وتذكّر الرسائل السابقة ضمن هذه الجلسة. "
    "قدّم إجابات واضحة ومفيدة، وإذا لم تعرف معلومة فقل إنك لا تعرفها ولا تختلق حقائق."
)


def load_model() -> None:
    global MODEL, TOKENIZER, MODEL_LOADING, MODEL_ERROR
    global torch, AutoModelForCausalLM, AutoTokenizer, DEVICE, DTYPE
    with MODEL_LOCK:
        if MODEL is not None or MODEL_LOADING:
            return
        MODEL_LOADING = True
    try:
        os.environ.setdefault("HF_HOME", HF_HOME)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        memory_limit = container_memory_limit_bytes()
        required_bytes = int(MIN_MODEL_RAM_GB * (1024 ** 3))
        if DEVICE == "cpu" and memory_limit is not None and memory_limit < required_bytes:
            actual_gb = memory_limit / (1024 ** 3)
            MODEL_ERROR = (
                f"ذاكرة الخادم غير كافية لتشغيل النموذج: المتاح للحاوية {actual_gb:.2f} GB، "
                f"والمطلوب مبدئياً {MIN_MODEL_RAM_GB:.1f} GB على الأقل. "
                "ارفع حد RAM في منصة الاستضافة أو استخدم نسخة مكمّمة أصغر."
            )
            print(f"[OwnedLM] {MODEL_ERROR}", flush=True)
            return
        # Import the heavyweight ML stack only after the RAM guard. This lets
        # the lightweight web/API process start on small hosting plans.
        import torch as torch_module
        from transformers import AutoModelForCausalLM as model_class, AutoTokenizer as tokenizer_class
        torch = torch_module
        AutoModelForCausalLM = model_class
        AutoTokenizer = tokenizer_class
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DTYPE = torch.bfloat16 if DEVICE == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        print(f"[OwnedLM] Loading {MODEL_ID} on {DEVICE} ({DTYPE})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model.to(DEVICE)
        model.eval()
        with MODEL_LOCK:
            TOKENIZER = tokenizer
            MODEL = model
            MODEL_ERROR = ""
        print("[OwnedLM] Pretrained model loaded successfully.", flush=True)
    except Exception as exc:
        with MODEL_LOCK:
            MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[OwnedLM] Model load failed: {MODEL_ERROR}", flush=True)
    finally:
        with MODEL_LOCK:
            MODEL_LOADING = False


def begin_model_load() -> None:
    threading.Thread(target=load_model, name="ownedlm-model-loader", daemon=True).start()


def _clean_output(text: str) -> str:
    for marker in ("<|begin_of_thought|>", "<ifm|think>", "<|begin_of_solution|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    for marker in ("<|end_of_thought|>", "<|end_of_solution|>", "</ifm|think>", "<|ifm|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]
    return text.strip()


def generate_reply(payload: dict) -> str:
    model, tokenizer = MODEL, TOKENIZER
    if model is None or tokenizer is None:
        raise RuntimeError("النموذج ما زال قيد التحميل أو تعذّر تحميله.")
    raw_messages = payload.get("messages")
    messages = []
    if isinstance(raw_messages, list):
        for item in raw_messages[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(item, dict):
                continue
            role, content = item.get("role"), item.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            content = content.strip()
            if content and len(content) <= MAX_PROMPT_CHARS:
                messages.append({"role": role, "content": content})
    if not messages:
        prompt = payload.get("message", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("اكتب رسالة أولاً.")
        messages = [{"role": "user", "content": prompt.strip()}]
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("رسالة المستخدم غير موجودة.")
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    prompt_tokens = inputs["input_ids"].shape[-1]
    try:
        max_new_tokens = int(payload.get("max_new_tokens", 256))
    except (TypeError, ValueError):
        max_new_tokens = 256
    max_new_tokens = max(32, min(max_new_tokens, MAX_NEW_TOKENS))
    with GENERATION_LOCK, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.08,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
    answer = _clean_output(answer)
    return answer or "لم أتمكن من صياغة إجابة واضحة. أعد صياغة سؤالك من فضلك."


class OwnedLMHandler(BaseHTTPRequestHandler):
    server_version = "OwnedLM/2.0"

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
            with MODEL_LOCK:
                ready, loading, error = MODEL is not None, MODEL_LOADING, MODEL_ERROR
            self._send_json(200, {
                "ready": ready,
                "loading": loading,
                "engine": "jais-3-local" if ready else "jais-3-loading" if loading else "jais-3-error",
                "model": MODEL_ID,
                "device": DEVICE,
                "message": "محرك OwnedLM الجديد يعمل." if ready else "يجري تجهيز النموذج لأول مرة..." if loading else "تعذّر تحميل النموذج.",
                "error": error or None,
            })
            return
        if path.startswith("/api/"):
            self._send_json(404, {"error": "المسار غير موجود."})
            return
        self._serve_web(path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"error": "المسار غير موجود."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
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
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "صيغة الطلب غير صالحة."})
            return
        if MODEL is None:
            with MODEL_LOCK:
                loading, error = MODEL_LOADING, MODEL_ERROR
            self._send_json(503, {
                "error": "النموذج قيد التحميل، انتظر قليلاً ثم أعد الإرسال." if loading else "تعذّر تحميل النموذج. راجع سجل النشر.",
                "loading": loading,
                "detail": error or None,
            })
            return
        try:
            answer = generate_reply(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            print(f"[OwnedLM] generation error: {type(exc).__name__}: {exc}", flush=True)
            self._send_json(500, {"error": "حدث خطأ أثناء التوليد. راجع سجل الخادم."})
            return
        self._send_json(200, {
            "reply": answer,
            "engine": "jais-3-local",
            "model": MODEL_ID,
            "note": "توليد محلي من الأوزان المحمّلة داخل الخادم.",
        })


def main() -> None:
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)
    begin_model_load()
    server = ThreadingHTTPServer((HOST, PORT), OwnedLMHandler)
    print(f"OwnedLM server listening on http://{HOST}:{PORT}", flush=True)
    print(f"Model: {MODEL_ID}; cache: {HF_HOME}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nإيقاف خادم OwnedLM...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
