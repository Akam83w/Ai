# Arabic distillation from Jais

This directory defines a repeatable way to ask the project's Jais teacher for Arabic learning material.

## What this does

- Builds a broad Arabic-first curriculum: grammar, morphology, vocabulary, writing, comprehension, Iraqi conversation, basic sciences, history, logic, and practical tasks.
- Calls the project's existing Jais-compatible API and saves generated examples as JSONL plus plain UTF-8 text.
- Keeps generated data local under `data/private/` by default.

It does **not** extract hidden knowledge, copy model parameters, or make the tiny randomly initialized OwnedLM instantly fluent. It creates teacher-generated training material. The student must still be upgraded and trained on a suitable compute machine.

## Run

Start the Jais API using the instructions in [JAIS-SETUP.md](../JAIS-SETUP.md), then run:

```bash
python distillation/generate_teacher_data.py --base-url http://127.0.0.1:8000
```

For a small test:

```bash
python distillation/generate_teacher_data.py --limit 3 --samples-per-task 1
```

The default curriculum currently contains 40+ task prompts. Use `--samples-per-task 5` to request more variations. Generation requires a running Jais inference server; GitHub itself does not run the model.

## Output

- `data/private/jais_arabic_distilled.jsonl`: prompt/response records with topic labels.
- `data/private/jais_arabic_distilled.txt`: concatenated Arabic training text.

Do not commit private or very large generated corpora to Git. Keep source/license records for every dataset. Jais-derived outputs and any redistributed derived weights must follow the applicable model license and attribution terms. Review generated material for factual errors, repetition, dialect quality, and contamination before training.

## Next engineering stage

The current OwnedLM uses a character tokenizer and a very small randomly initialized Transformer. A serious Arabic student needs a subword tokenizer (for example, BPE trained on permitted Arabic text), a larger context/model configuration, a supervised fine-tuning or continued-pretraining pipeline, and held-out Arabic evaluation. Do not overwrite the original model until the new training path passes tests.
