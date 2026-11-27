import os, re, sqlite3
from pathlib import Path
from typing import Literal
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from knowledge import retrieve_answer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("OWNEDLM_CHECKPOINT", str(ROOT / "checkpoints" / "model.pt")))
KNOWLEDGE_DIR = Path(os.getenv("OWNEDLM_KNOWLEDGE_DIR", str(ROOT / "data" / "corpus")))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_PATH = Path(os.getenv("OWNEDLM_DB_PATH", "/data/ownedlm.sqlite3"))
ADMIN_TOKEN = os.getenv("OWNEDLM_ADMIN_TOKEN", "")
JAIS_MODEL_ID = os.getenv("JAIS_MODEL_ID", "Dhunun/OwnedLM-Jais-3-0.9B")
JAIS_ADAPTER_ID = os.getenv("JAIS_ADAPTER_ID", "").strip()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEVICE.type == "cuda" else torch.float32
app = FastAPI(title="OwnedLM Arabic Companion API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET","POST","DELETE"], allow_headers=["Content-Type","X-Admin-Token"])

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=24)
    session_id: str = Field(min_length=1, max_length=100)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

_model = None
_tokenizer = None
_load_error = None
_db_error = None

@contextmanager
def db_connection():
    if DATABASE_URL.startswith(("postgres://", "postgresql://")):
        import psycopg
        conn = psycopg.connect(DATABASE_URL)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=15)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

def init_database():
    global _db_error
    try:
        with db_connection() as conn:
            cur = conn.cursor()
            if DATABASE_URL.startswith(("postgres://", "postgresql://")):
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id BIGSERIAL PRIMARY KEY,
                        session_id VARCHAR(100) NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        used_for_training BOOLEAN NOT NULL DEFAULT FALSE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_messages(session_id, id)")
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        used_for_training INTEGER NOT NULL DEFAULT 0
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_messages(session_id, id)")
        _db_error = None
        return True
    except Exception as exc:
        _db_error = f"{type(exc).__name__}: {exc}"
        return False

def save_exchange(session_id: str, user_text: str, assistant_text: str):
    with db_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO conversation_messages (session_id, role, content) VALUES (%s, %s, %s)"
            if DATABASE_URL.startswith(("postgres://", "postgresql://"))
            else "INSERT INTO conversation_messages (session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "user", user_text), (session_id, "assistant", assistant_text)]
        )

def require_admin(token: str | None):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Set OWNEDLM_ADMIN_TOKEN on the server first")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

@app.on_event("startup")
def startup():
    init_database()
    # Load the pretrained base during startup so the first chat request is not a cold load.
    load_checkpoint()

@app.get("/")
def home():
    return FileResponse(ROOT / "jais-chat.html")

def load_checkpoint():
    global _model, _tokenizer, _load_error
    if _model is not None:
        return _model, _tokenizer
    try:
        _load_error = None
        tok = AutoTokenizer.from_pretrained(JAIS_MODEL_ID, trust_remote_code=True)
        net = AutoModelForCausalLM.from_pretrained(
            JAIS_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        if JAIS_ADAPTER_ID:
            from peft import PeftModel
            net = PeftModel.from_pretrained(net, JAIS_ADAPTER_ID)
        net.to(DEVICE)
        net.eval()
        _model, _tokenizer = net, tok
        return _model, _tokenizer
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        return None, None

def extract_answer(text: str) -> str:
    for marker in ("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            break
    for marker in ("<|ifm|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()

def starter_reply(text: str) -> str:
    s = " ".join(text.strip().lower().split())
    if re.search(r"(هلو|هلا|مرحبا|السلام عليكم|شلونك|كيف حالك)", s):
        return "هلا بيك! أني صديقك العربي قيد التطوير. بعدني أتعلم شلون أسولف وياك بصورة طبيعية."
    if re.search(r"(اسمك|من أنت|منو انت|من تكون)", s):
        return "أني صديقك العربي OwnedLM، ودا نبني شخصيتي وذاكرتي خطوة بخطوة."
    return "وصلتني رسالتك، بس بعدني ما أملك أوزاناً لغوية مدرّبة حتى أقدر أجاوبك من فهمي. نحتاج نكمل تدريبي."

def build_dialogue(messages: list[Message], limit: int) -> str:
    lines = []
    for item in messages[-8:]:
        if item.role == "system":
            continue
        speaker = "المستخدم" if item.role == "user" else "الصديق"
        content = " ".join(item.content.split()).strip()
        lines.append(f"{speaker}: {content}")
    lines.append("الصديق:")
    return "\n".join(lines)[-limit:]

@app.get("/health")
def health():
    model, _ = load_checkpoint()
    db_ok = init_database()
    return {
        "status": "ok" if model is not None else "degraded",
        "service": "OwnedLM",
        "device": DEVICE.type,
        "model_id": JAIS_MODEL_ID,
        "adapter_id": JAIS_ADAPTER_ID or None,
        "model_loaded": model is not None,
        "mode": "jais_pretrained" if model is not None else "model_load_error",
        "database_connected": db_ok,
        "database_type": "postgresql" if DATABASE_URL.startswith(("postgres://", "postgresql://")) else "sqlite",
        "database_error": _db_error,
        "knowledge_directory": str(KNOWLEDGE_DIR),
        "knowledge_files": len(list(KNOWLEDGE_DIR.rglob("*.txt"))) if KNOWLEDGE_DIR.is_dir() else 0,
        "model_parameters": sum(p.numel() for p in model.parameters()) if model is not None else None,
        "model_vocab_size": model.config.vocab_size if model is not None else None,
        "model_context_length": model.config.max_position_embeddings if model is not None else None,
        "load_error": _load_error
    }

@app.get("/admin/training-stats")
def training_stats(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    init_database()
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversation_messages")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE role = %s" if DATABASE_URL.startswith(("postgres://", "postgresql://")) else "SELECT COUNT(*) FROM conversation_messages WHERE role = ?", ("user",))
        user_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE used_for_training = TRUE" if DATABASE_URL.startswith(("postgres://", "postgresql://")) else "SELECT COUNT(*) FROM conversation_messages WHERE used_for_training = 1")
        trained_count = cur.fetchone()[0]
    return {"saved_messages": total, "user_messages": user_count, "messages_marked_trained": trained_count, "pending_user_messages": user_count - trained_count}

@app.get("/admin/training-data", response_class=PlainTextResponse)
def training_data(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    init_database()
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM conversation_messages WHERE used_for_training = FALSE ORDER BY id" if DATABASE_URL.startswith(("postgres://", "postgresql://")) else "SELECT role, content FROM conversation_messages WHERE used_for_training = 0 ORDER BY id")
        rows = cur.fetchall()
    lines = []
    for role, content in rows:
        speaker = "المستخدم" if role == "user" else "الصديق"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) + ("\n" if lines else "")

@app.delete("/v1/conversations/{session_id}")
def delete_conversation(session_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    try:
        with db_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM conversation_messages WHERE session_id = %s" if DATABASE_URL.startswith(("postgres://", "postgresql://")) else "DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
            deleted = cur.rowcount
        return {"deleted": True, "deleted_messages": deleted}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database delete failed: {type(exc).__name__}")

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    model, tok = load_checkpoint()

    # Prefer a high-confidence answer explicitly present in the project's own corpus.
    knowledge_match = retrieve_answer(user_text, KNOWLEDGE_DIR)
    if knowledge_match is not None:
        answer, mode = knowledge_match["answer"], "knowledge_retrieval"
    elif model is None:
        raise HTTPException(
            status_code=503,
            detail=f"OwnedLM pretrained model is not available: {_load_error or 'model loading failed'}",
        )
    else:
        messages = [{"role": item.role, "content": item.content} for item in req.messages]
        if not any(item["role"] == "system" for item in messages):
            messages.insert(0, {
                "role": "system",
                "content": "أنت OwnedLM، مساعد عربي مستقل. أجب باللغة العربية الفصحى المعاصرة فقط، بدقة ووضوح. لا تخترع معلومات عند عدم التأكد."
            })
        inputs = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="high",
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=20,
                repetition_penalty=1.0,
                pad_token_id=tok.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        generated_text = tok.decode(output[0, prompt_length:], skip_special_tokens=False)
        answer = extract_answer(generated_text)
        if not answer:
            answer = "لم أتمكن من إنشاء إجابة."
        mode = "jais_pretrained"

    saved = False
    try:
        if user_text:
            save_exchange(req.session_id, user_text, answer)
            saved = True
    except Exception as exc:
        global _db_error
        _db_error = f"{type(exc).__name__}: {exc}"
    response = {
        "model": "OwnedLM",
        "base_model": JAIS_MODEL_ID,
        "mode": mode,
        "saved_to_database": saved,
        "choices": [{"message": {"role": "assistant", "content": answer}}],
    }
    if knowledge_match is not None:
        response["knowledge_match"] = {
            "source": knowledge_match["source"],
            "matched_question": knowledge_match["matched_question"],
            "score": knowledge_match["score"],
        }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
