import os, re
from pathlib import Path
from typing import Literal
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from model import ModelConfig, TinyLM
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("OWNEDLM_CHECKPOINT", str(ROOT / "checkpoints" / "model.pt")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="OwnedLM Arabic Companion API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET","POST"], allow_headers=["Content-Type"])

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)
class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=24)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)

_model = None
_tokenizer = None
_load_error = None

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
    return "\\n".join(lines)[-limit:]

@app.get("/health")
def health():
    model, _ = load_checkpoint()
    return {"status":"ok","service":"OwnedLM","device":DEVICE.type,"model_loaded":model is not None,"mode":"trained_model" if model is not None else "starter","checkpoint":str(CHECKPOINT),"load_error":_load_error}

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
    return {"model":"OwnedLM","mode":mode,"choices":[{"message":{"role":"assistant","content":answer}}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
