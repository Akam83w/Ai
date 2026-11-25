# Jais 3 integration (optional)

This repository keeps its original from-scratch OwnedLM model. Jais 3 is an
optional pretrained inference engine; it does not replace or overwrite the
original model files.

- Hugging Face model: https://huggingface.co/inception42/Jais-3-0.9B
- Model license: Apache-2.0 (retain the model's license and notices when redistributing).
- Model weights are downloaded at runtime and must not be committed to GitHub.

## Start the API

Use Python 3.10+ and preferably a CUDA GPU with enough available memory.

```bash
pip install -r requirements-jais.txt
python jais_api.py
```

The first launch downloads the model. The API listens on port 8000.

- Health: `GET http://localhost:8000/health`
- Chat: `POST http://localhost:8000/v1/chat/completions`

Example request:

```json
{
  "messages": [
    {"role": "system", "content": "أنت مساعد مفيد. أجب بالعربية العراقية باختصار."},
    {"role": "user", "content": "هلو، شلونك؟"}
  ],
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95
}
```

Example response shape:

```json
{
  "model": "inception42/Jais-3-0.9B",
  "choices": [
    {"message": {"role": "assistant", "content": "نص الإجابة"}}
  ]
}
```

## Important

- This is an optional inference service, not a public hosted endpoint.
- For Colab, the notebook runtime must remain active while the API is used.
- Do not expose the API publicly without authentication, rate limits, and HTTPS.
- The model can make mistakes. Test Iraqi dialect quality before relying on it.
- The upstream model uses custom model code; review upstream code and pin a
  reviewed revision before production deployment.
