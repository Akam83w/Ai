# Jais 3 integration (optional)

This repository keeps its original from-scratch OwnedLM model. Jais 3 is an
optional pretrained inference engine; it does not replace or overwrite the
original model files.

- Project checkpoint: https://huggingface.co/Dhunun/Jais-3.0.9B
- Upstream model and license: https://huggingface.co/inception42/Jais-3-0.9B (Apache-2.0).
- Keep the upstream license and attribution when redistributing derived weights.
- Model weights are downloaded at runtime and must not be committed to GitHub.

## Start the API

Use Python 3.10+ and preferably a CUDA GPU with enough available memory.

```bash
pip install -r requirements-jais.txt
# If the project checkpoint is private, authenticate first using a fresh
# Hugging Face token stored locally as HF_TOKEN (never paste it into chat).
export HF_TOKEN=your_hugging_face_token
export JAIS_MODEL_ID=Dhunun/Jais-3.0.9B
python jais_api.py
```

The API listens on port 8000.

- Health: `GET http://localhost:8000/health`
- Chat: `POST http://localhost:8000/v1/chat/completions`

The server uses the model's native K2 Horizon chat template with
`reasoning_effort="high"` and removes the reasoning segment from the returned
user-facing answer.

Example request:

```json
{
  "messages": [
    {"role": "system", "content": "أنت مساعد مفيد. أجب بالعربية العراقية باختصار."},
    {"role": "user", "content": "هلو، شلونك؟"}
  ],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.95
}
```

Example response shape:

```json
{
  "model": "Dhunun/Jais-3.0.9B",
  "choices": [
    {"message": {"role": "assistant", "content": "نص الإجابة"}}
  ]
}
```

## Important

- This is a local inference service, not a public hosted endpoint.
- The machine running the API must remain on while it is being used.
- Do not expose the API publicly without authentication, rate limits, and HTTPS.
- The model can make mistakes. Test Iraqi dialect quality before relying on it.
- The upstream model uses custom model code; review upstream code and pin a
  reviewed revision before production deployment.
