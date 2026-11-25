"""Optional local/Colab API for the Jais 3 model.

Run:
  pip install -r requirements-jais.txt
  python jais_api.py

The model weights are downloaded from Hugging Face on first launch and are
never committed to this Git repository.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_ID = os.getenv("JAIS_MODEL_ID", "inception42/Jais-3-0.9B")
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="OwnedLM Jais API", version="0.1.0")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=24)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_generator():
    return pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        trust_remote_code=True,
        device=DEVICE,
        torch_dtype=DTYPE,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": "cuda" if DEVICE >= 0 else "cpu"}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    try:
        conversation = [
            {"role": item.role, "content": item.content}
            for item in request.messages
        ]
        result = get_generator()(
            conversation,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            return_full_text=False,
        )
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            answer = next(
                (item.get("content", "") for item in reversed(generated)
                 if item.get("role") == "assistant"),
                "",
            )
        else:
            answer = str(generated)
        return {"model": MODEL_ID, "choices": [{"message": {"role": "assistant", "content": answer}}]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {type(exc).__name__}") from exc
