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
app = FastAPI(title="OwnedLM Arabic API", version="1.0.1")
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
        return "هلا بيك! أنا نموذجك العربي قيد التطوير. وصلتني رسالتك، شنو تحب نحچي؟"
    if re.search(r"(اسمك|من أنت|منو انت|من تكون)", s):
        return "أنا نموذجك العربي المستقل OwnedLM، وهذه نسخة تشغيل أولية."
    if re.search(r"(شكرا|شكرًا|ممنون|تسلم)", s): return "العفو، بالخدمة دائماً."
    if re.search(r"(ساعدني|تساعدني|مساعدة)", s):
        return "أكيد أساعدك. اكتب طلبك بشكل محدد، وسأحاول توجيهك ضمن قدرات النسخة الحالية."
    return "وصلتني رسالتك. الخادم يعمل، لكن أوزان النموذج المدرّبة غير موجودة على الخادم بعد؛ هذا ردّ تشغيل أولي وليس توليداً من النموذج المتعلّم."

@app.get("/health")
def health():
    model, _ = load_checkpoint()
    return {"status":"ok","service":"OwnedLM","device":DEVICE.type,"model_loaded":model is not None,"mode":"trained_model" if model is not None else "starter","checkpoint":str(CHECKPOINT),"load_error":_load_error}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompts = [m.content for m in req.messages if m.role == "user"]
    if not prompts:
        answer, mode = "اكتب رسالتك حتى أجاوبك.", "starter"
    else:
        prompt = prompts[-1]
        model, tok = load_checkpoint()
        if model is None:
            answer, mode = starter_reply(prompt), "starter"
        else:
            ids = tok.encode(prompt, add_bos=True) or [tok.bos_id]
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.inference_mode():
                out = model.generate(x, max_new_tokens=min(req.max_new_tokens,128), temperature=req.temperature, top_k=20, eos_id=tok.eos_id)
            answer = tok.decode(out[0].tolist()).strip()
            answer = answer[len(prompt):].strip() if answer.startswith(prompt) else answer
            if not answer: answer = "لم أتمكن من توليد رد واضح من الأوزان الحالية."
            mode = "trained_model"
    return {"model":"OwnedLM","mode":mode,"choices":[{"message":{"role":"assistant","content":answer}}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
