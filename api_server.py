import os, re, sqlite3
from pathlib import Path
from typing import Literal
from contextlib import contextmanager

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from model import ModelConfig, TinyLM
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("OWNEDLM_CHECKPOINT", str(ROOT / "checkpoints" / "model.pt")))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_PATH = Path(os.getenv("OWNEDLM_DB_PATH", "/data/ownedlm.sqlite3"))
ADMIN_TOKEN = os.getenv("OWNEDLM_ADMIN_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="OwnedLM Arabic Companion API", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET","POST"], allow_headers=["Content-Type","X-Admin-Token"])

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=24)
    session_id: str = Field(min_length=1, max_length=100)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)

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

@app.get("/")
def home():
    return FileResponse(ROOT / "jais-chat.html")

def load_checkpoint():
    global _model, _tokenizer, _load_error
    if _model is not None: return _model, _tokenizer
    if not CHECKPOINT.is_file(): return None, None
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        tok = CharTokenizer.from_dict(ckpt["tokenizer"])
        cfg = ModelConfig.from_dict(ckpt["config"])
        net = TinyLM(cfg).to(DEVICE)
        net.load_state_dict(ckpt["model"])
        net.eval()
        _model, _tokenizer = net, tok
        return _model, _tokenizer
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        return None, None

def starter_reply(text: str) -> str:
    s = re.sub(r"\\s+", " ", text.strip().lower())
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
        content = re.sub(r"\\s+", " ", item.content).strip()
        lines.append(f"{speaker}: {content}")
    lines.append("الصديق:")
    return "\n".join(lines)[-limit:]

@app.get("/health")
def health():
    model, _ = load_checkpoint()
    db_ok = init_database()
    return {
        "status": "ok",
        "service": "OwnedLM",
        "device": DEVICE.type,
        "model_loaded": model is not None,
        "mode": "trained_model" if model is not None else "starter",
        "database_connected": db_ok,
        "database_type": "postgresql" if DATABASE_URL.startswith(("postgres://", "postgresql://")) else "sqlite",
        "database_error": _db_error,
        "checkpoint": str(CHECKPOINT),
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

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt_text = build_dialogue(req.messages, 120)
    model, tok = load_checkpoint()
    if model is None:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        answer, mode = starter_reply(last_user), "starter"
    else:
        prompt_text = prompt_text[-(model.cfg.max_seq_len - 1):]
        ids = tok.encode(prompt_text, add_bos=True) or [tok.bos_id]
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            out = model.generate(x, max_new_tokens=min(req.max_new_tokens,128), temperature=req.temperature, top_k=20, eos_id=tok.eos_id)
        decoded = tok.decode(out[0].tolist()).strip()
        answer = decoded[len(prompt_text):].strip() if decoded.startswith(prompt_text) else decoded
        if not answer:
            answer = "ما كدرت أكمّل الرد من الأوزان الحالية."
        mode = "trained_model"
    user_text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    saved = False
    try:
        if user_text:
            save_exchange(req.session_id, user_text, answer)
            saved = True
    except Exception as exc:
        global _db_error
        _db_error = f"{type(exc).__name__}: {exc}"
    return {"model":"OwnedLM","mode":mode,"saved_to_database":saved,"choices":[{"message":{"role":"assistant","content":answer}}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
