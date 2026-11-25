# Arabic learning and distillation pipeline

This pipeline uses the project's locally hosted Jais-3 teacher to generate Arabic-only learning material for the OwnedLM project.

## Teacher model

- Project checkpoint: `Dhunun/Jais-3.0.9B` (private Hugging Face repository).
- Upstream base: `inception42/Jais-3-0.9B`.
- Architecture: K2 Horizon, approximately 1.078B parameters; the current safetensors file is about 2.16 GB.
- The upstream model card declares Apache-2.0. Keep the upstream license, copyright notices, and attribution with redistributed model code or weights.
- Runtime generation is performed by the project's own Jais API. Do not call a hosted external inference API.

## Curriculum scope

The curriculum is Modern Standard Arabic only. It covers phonetics, spelling, grammar, morphology, semantics, rhetoric, reading comprehension, writing, conversation, literature, translation, mathematics, natural sciences, computing, history, geography, economics, reasoning, and teaching/evaluation. Regional dialect tasks are intentionally excluded.

## 1. Generate teacher examples

Start the local teacher API:

```bash
pip install -r requirements-jais.txt
JAIS_MODEL_ID=Dhunun/Jais-3.0.9B python jais_api.py
```

Keep any Hugging Face token in the runtime's secret manager or environment; never commit it.

Generate a small smoke-test batch:

```bash
python distillation/generate_teacher_data.py --limit 4 --samples-per-task 1
```

Generate the complete curriculum with several variations:

```bash
python distillation/generate_teacher_data.py --samples-per-task 3 --pause 0.5
```

Output is appended under `data/private/` by default. It is not committed to Git.

## 2. Validate and split the dataset

```bash
python distillation/prepare_sft_dataset.py \
  --input data/private/jais_arabic_distilled.jsonl \
  --output-dir data/private/arabic_sft \
  --validation-ratio 0.05
```

This removes malformed, empty, too-short, and exact duplicate records, then makes a deterministic train/validation split. Review generated facts before training.

## 3. Fine-tune the Jais-3 base with LoRA

The current tiny randomly initialized OwnedLM Transformer cannot absorb Jais weights: its architecture and tokenizer are incompatible. The fast, technically valid path is to use the Jais-3 weights as the Arabic-capable base, then fine-tune that base for the project's behavior with LoRA. The adapter is the project's customization; it is not a full pretraining run.

Install optional training dependencies:

```bash
pip install -r requirements-jais-train.txt
```

Run on a CUDA GPU:

```bash
python distillation/train_jais_lora.py \
  --model-id Dhunun/Jais-3.0.9B \
  --train-file data/private/arabic_sft/train.jsonl \
  --validation-file data/private/arabic_sft/validation.jsonl \
  --output-dir adapters/ownedlm-arabic-v1
```

Use `--max-length`, `--batch-size`, and `--gradient-accumulation` to fit available VRAM. The script saves adapter weights separately and does not overwrite the original teacher checkpoint.

## What is and is not transferred

- The Jais checkpoint already contains its pretrained Arabic and general-language capabilities.
- Teacher generation creates new prompt/answer examples from the model's visible outputs; it does not extract hidden training data or copy the full internal knowledge into the old tiny model.
- LoRA fine-tuning adapts the Jais base into the project's own checkpoint lineage while preserving the original base weights.
- Keep training and validation sets separate. Never mark generated examples as verified facts without review.
- Do not commit model weights, private generated data, tokens, or user conversations to the public Git repository.

## References

- Upstream model: https://huggingface.co/inception42/Jais-3-0.9B
- Project checkpoint: https://huggingface.co/Dhunun/Jais-3.0.9B
- License: https://www.apache.org/licenses/LICENSE-2.0
