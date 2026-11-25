# Jais-3 Arabic teacher and base model

The project uses the locally hosted Jais-3 checkpoint as its Arabic-capable teacher and as the supported pretrained base for OwnedLM customization.

- Project checkpoint: https://huggingface.co/Dhunun/Jais-3.0.9B (private).
- Upstream model: https://huggingface.co/inception42/Jais-3-0.9B.
- Upstream license: Apache-2.0; retain attribution and license notices.
- Architecture: K2 Horizon, approximately 1.078B parameters.
- Upstream safetensors weights: approximately 2.16 GB. Runtime memory use is higher.
- Model weights are stored on Hugging Face and downloaded by the runtime; do not commit binary weights to GitHub.

## Run the local API

```bash
pip install -r requirements-jais.txt
# Configure HF_TOKEN as a secret/environment variable if the private project checkpoint requires authentication.
JAIS_MODEL_ID=Dhunun/Jais-3.0.9B python jais_api.py
```

The API listens on port 8000.

- Health: `GET http://localhost:8000/health`
- Chat: `POST http://localhost:8000/v1/chat/completions`

Use Modern Standard Arabic in system instructions. Example:

```json
{
  "messages": [
    {"role": "system", "content": "أنت مساعد عربي فصيح. أجب بالعربية الفصحى المعاصرة بوضوح ودقة."},
    {"role": "user", "content": "اشرح الفرق بين الفعل والاسم."}
  ],
  "max_new_tokens": 512,
  "temperature": 0.5,
  "top_p": 0.9
}
```

## Arabic training and customization

See [distillation/README.md](distillation/README.md) for the Arabic-only curriculum, teacher-data generation, dataset validation/splitting, and LoRA fine-tuning workflow.

The project checkpoint is private. Never paste tokens into chat or commit them. Configure secrets in the deployment or training environment.

## Deployment safety

- This is a local inference service, not a hosted external model API.
- The machine running the API must remain on while it is used.
- Before public deployment, restrict CORS to the actual website origin and add authentication, rate limits, and HTTPS.
- Keep the original base checkpoint immutable; save customized adapters/checkpoints under versioned project paths.
