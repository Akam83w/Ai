# نموذج لغوي صغير من الصفر (OwnedLM)

نموذج **Transformer** سببي صغير، مكتوب بـ Python وPyTorch، يُدرَّب من **أوزان عشوائية** على نصوص تضعها أنت. بعد التدريب يعمل التوليد من ملف الأوزان على جهازك، من غير اتصال بـ ChatGPT أو OpenAI أو Grok أو أي واجهة نموذج خارجية، ومن غير تنزيل أوزان جاهزة.

هذا ليس مساعداً بمستوى ChatGPT، ولن يصبح كذلك بتدريب البذرة المرفقة. هو نواة تملكها وتكبرها: بنية النموذج، حلقة التدريب، حفظ الأوزان، واستئناف التدريب.

عند إضافة هذا النموذج لم يكن في المستودع ترخيص ولا بيانات تدريب. ملفات واجهة الويب التي كانت على `main` (مثل `package.json` و`vite.config.ts`) بقيت ولم تُحذف. هذا الالتزام يضيف ترخيص MIT لكود النموذج وللنصين الأصليين في `data/corpus/`.

## ما الذي يفعله النموذج

- محوّل فكّ فقط (decoder-only): تضمين رموز، مواضع متعلَّمة، انتباه ذاتي سببي متعدد الرؤوس، طبقة تغذية أمامية، وربط أوزان التضمين برأس الإخراج.
- مفردات التدريب الجديدة تستخدم Byte-level BPE يتعلم أجزاء الكلمات من ملفاتك، مع الاحتفاظ بدعم نقاط الاستئناف القديمة ذات ترميز الحروف.
- التدريب: AdamW، قص المتدرجات، فقدان الانتروبيا المتقاطعة، حفظ نقطة استئناف.
- التوليد: اختيار عشوائي بدرجة حرارة و`top-k`، أو اختيار جشع إذا كانت الحرارة `0`.

## ما الذي لا يفعله

- لا يستدعي API ولا يحمّل نموذجاً مدرَّباً.
- لا يعرف العالم. يرى فقط الملفات التي تعطيه إياها، وقد يحفظها حرفياً إذا طال التدريب عليها.
- البذرة الحالية بضع كيلوبايتات من نص أصلي. بعدها المتوقع: ضياع في المحارف، لا محادثة.
- لا يوجد ضمان أمان أو صحة للإجابات. لا تستخدم المخرجات في قرار طبي أو قانوني أو مالي.

## الحجم والذاكرة والوقت

الأرقام تقريبية لأوزان fp32. زمن التدريب الفعلي يعتمد على حجم نصك وعدد الخطوات.

| الإعداد | المعاملات (مفردات نموذجية) | الأوزان | جهاز مناسب | ماذا تتوقع |
| --- | --- | --- | --- | --- |
| `tiny` | نحو 0.1–0.3 مليون (مفردات حروف صغيرة، عرض 64، طبقتان، سياق 128) | أقل من 2MB | معالج عادي وذاكرة أقل من 1GB | تجربة الحلقة. الخسارة قد تنخفض على نص مكرر. الكلام يبقى ضعيفاً |
| `small` | نحو 5–8 مليون (مفردات أكبر، عرض 256، 6 طبقات، سياق 256) | نحو 20–40MB | يفضَّل GPU بذاكرة 4GB. على CPU غالباً خطوات قليلة في الثانية وذاكرة نحو 2GB | تقليد أسلوب corpus متوسط، وليس مساعداً عاماً |

للمقارنة فقط، حتى تبقى الحدود واضحة: النماذج الحوارية الكبيرة تُقاس بمليارات المعاملات وبكميات نصوص هائلة (مئات المليارات من الرموز فأكثر) وبعناقيد GPU. هذا المستودع لا يوفّر تلك البيانات ولا تلك القدرة، ولا يدّعي ذلك المستوى.

إعداد `tiny` هو الافتراضي حتى يبقى التشغيل ممكناً على جهاز شخصي.

## البيانات

اقرأ [data/README.md](data/README.md) و[data/SOURCES.md](data/SOURCES.md) قبل التدريب.

- أضف ملفات `.txt` بترميز UTF-8 تملك حق استخدامها.
- النصوص القابلة للنشر توضع في `data/corpus/`.
- النصوص التي لا تريد رفعها توضع في `data/private/` (مستبعدة من Git) وتمرر مجلدها إلى `--data`.
- سجّل المصدر والرخصة في `data/SOURCES.md`.
- لا تنسخ كتباً أو مواقع محمية، ولا أوزان شركات أخرى.

البذرة `seed_ar.txt` و`seed_en.txt` نص أصلي كُتب لهذا المستودع تحت ترخيص MIT. احذفها أو استبدلها متى شئت.

## التشغيل

يتطلب Python 3.10 أو أحدث.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

تثبيت PyTorch على CPU فقط، إن لم تكن تريد حزمة CUDA الكبيرة:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

الاختبارات (إنشاء النموذج، قناع الانتباه السببي، انخفاض الخسارة على نص مكرر، التدريب، الاستئناف، والتوليد):

```bash
python -m unittest discover -s tests -t . -v
```

تدريب قصير على البذرة. الناتج يبقى محلياً داخل `checkpoints/` ولا يُرفع إلى Git:

```bash
python train.py --data data/corpus --preset tiny --tokenizer-type bpe --vocab-size 2048 --steps 200 --batch-size 16 --seq-len 64 --out checkpoints/tiny-bpe.pt
```

استئناف التدريب من نفس الملف:

```bash
python train.py --data data/corpus --steps 200 --resume checkpoints/tiny-bpe.pt --out checkpoints/tiny-bpe.pt
```

توليد محلي من غير شبكة:

```bash
python generate.py --checkpoint checkpoints/tiny-bpe.pt --prompt "في ورشة" --max-new-tokens 80 --temperature 0.9 --top-k 40
```

تقسيم تحقق اختياري عندما يكون عندك أكثر من ملف: `--val-ratio 0.1`. القيمة الافتراضية `0` حتى تُستخدم البذرة كلها في التدريب.

أهم الخيارات في `train.py`:

| الخيار | المعنى | الافتراضي |
| --- | --- | --- |
| `--data` | مجلد ملفات `.txt` | `data/corpus` |
| `--preset` | `tiny` أو `small` | `tiny` |
| `--tokenizer-type` | `bpe` أو `char` (الافتراضي BPE) | `bpe` |
| `--vocab-size` | حجم مفردات BPE | `2048` |
| `--steps` | عدد خطوات هذه الجولة | `200` |
| `--seq-len` | طول النافذة، ولا يتجاوز حد الإعداد | `64` |
| `--lr` | معدل تعلم AdamW | `0.0003` |
| `--resume` | نقطة استئناف | فارغ |
| `--device` | `auto` أو `cpu` أو `cuda` | `auto` |

نقطة الاستئناف تحفظ الأوزان والمحسّن ورقم الخطوة والإعداد والمفردات. لا تبنِ مفردات جديدة عند الاستئناف: الحرف غير الموجود يصبح `<?>`. درّب من الصفر إذا تغيّرت المفردات كثيراً.

ملفات `.pt` صيغة pickle. حمّل فقط نقاط استئناف درّبتها بنفسك.

## هيكل الملفات

```text
model.py          النموذج
tokenizer.py      مفردات الحروف من نصك
dataset.py        قراءة الملفات وتقطيع النوافذ
train.py          التدريب والاستئناف
generate.py       التوليد المحلي
tests/test_lm.py  الاختبارات
data/corpus/      البذرة الأصلية
data/private/     بيانات محلية غير متتبعة
```

## فحص أولي، وليس مستوى محادثة

الاختبارات في `tests/test_lm.py` تتحقق من شكل المخرجات، ومن أن الرموز اللاحقة لا تغيّر لوغيتات الماضي، ومن أن الخسارة تنخفض على جملة مكررة، ومن أن التدريب يحفظ نقطة استئناف ويكمل منها، ومن أن التوليد يعيد نصاً من غير استيراد أي عميل نموذج خارجي.

تشغيل فعلي على CPU، إعداد `tiny`، 100 خطوة، على بذرة المستودع فقط (83 حرفاً في المفردات، 113,600 معاملاً، نحو 0.45MB لأوزان fp32):

- متوسط خسارة التدريب نزل من حوالي 4.13 في أول 25 خطوة إلى حوالي 3.66 على المئة كلها، وآخر خطوة كانت 3.17.
- خط الأساس العشوائي لهذه المفردات هو `ln(83)` أي حوالي 4.42. الانخفاض حقيقي وصغير.
- عينة بعد هذه المئة، بالمقدمة `في ورشة`، كانت غير مفهومة. هذا هو المتوقع من نص بحجم بضع كيلوبايتات.

لا تقرأ هذا الجدول كجودة منتج. اقرأه كدليل على أن الحلقة تعمل وأن القدرة ما زالت محدودة جداً.

## الخطوة التالية

1. أضف نصاً تملكه، أكبر بكثير من البذرة، وسجّل مصدره.
2. درّب `tiny` حتى تنخفض خسارة التدريب بثبات على عيّنة مكررة ثم على نصك.
3. انتقل إلى `--preset small` على GPU، مع `--val-ratio` وملفات تحقق لم تُستخدم في التدريب.
4. بعدها: مجزئ BPE تدرّبه أنت على نصك، وقياس الحيرة (perplexity) على قسم محجوز، ثم تكبير العرض والطبقات فقط إذا بقيت الخسارة على التحقق تتحسن.
5. لا تقارن المخرجات بمساعد تجاري قبل أن تمتلك بيانات وحساباً بهذا الحجم، وقبل أن تقيس ذلك على نص لم يره النموذج.


## Iraqi dialogue data preparation

A separate optional importer can prepare an Iraqi-dialect dialogue corpus
locally. The selected source is the
[Iraqi Arabic Sales Dialogue Dataset](https://huggingface.co/datasets/ameer4wisam/iraqi-arabic-sales-dialogue-dataset),
published under CC BY 4.0. It is synthetic, template-generated, mainly
Baghdadi-based, and contains retail plus everyday topics; it is not natural
Moslawi speech. The importer skips the disclosed scraped-car subset, removes
phone-number patterns, ignores system prompts, deduplicates exact conversations,
and preserves the published train/validation split.

Install the optional preparation dependency and run:

```bash
pip install -r requirements-data.txt
python prepare_iraqi_dataset.py --config v4
```

The result is written under `data/private/iraqi_sales/`, which is excluded
from Git. Inspect the sample conversations and `import_report.json` before
training. Do not publish the downloaded data or generated checkpoints without
preserving the dataset attribution and checking its license obligations.

Train against the training split and evaluate against the held-out split:

```bash
python train.py --data data/private/iraqi_sales/train --validation-data data/private/iraqi_sales/validation --preset small --tokenizer-type bpe --vocab-size 2048 --steps 10000 --batch-size 8 --seq-len 256 --device cuda --out checkpoints/iraqi-small.pt
```

This is a starting experiment, not a promise of a capable assistant. The
dataset is template-generated and Baghdadi-based; add reviewed Moslawi
examples and broader factual text you have rights to use, then compare
validation loss and held-out conversations before deploying any checkpoint.

## Local knowledge retrieval and deployment diagnostics

The API can now retrieve a high-confidence answer from adjacent
`المستخدم:` / `الصديق:` pairs in `data/corpus/*.txt`. This is an
extractive lookup over text you supplied; it is **not** model training,
semantic understanding, or a general web search. A match is returned only
when the local lexical-similarity threshold is met. The response mode is
`knowledge_retrieval` and includes the matched question, source file, and
score. If no strong match exists, the API continues to the trained model or
the honest starter fallback.

The `/health` endpoint now reports whether the checkpoint exists, its byte
size, load errors, model parameter count, vocabulary size, context length,
and the number of local knowledge files. Interpret the mode as follows:

- `trained_model`: checkpoint loaded successfully.
- `checkpoint_missing`: no checkpoint exists at the reported path.
- `checkpoint_invalid`: a file exists but could not be loaded; inspect
  `load_error`.

The Docker image includes the local corpus and training utilities
(`train.py`, `dataset.py`, and `generate.py`) but deliberately does **not**
invent or train a checkpoint during deployment. Train a checkpoint separately
with your own data, then mount or provide it at `OWNEDLM_CHECKPOINT`. Keep
that file on persistent storage; a container rebuild can otherwise discard it.

Conversation storage is still separate from training. Saving a chat does not
change model weights. Review conversation candidates, remove private or
sensitive content, prepare a curated training corpus, run the training script,
evaluate the new checkpoint, and only then deploy it.

## Conversation storage and learning-data collection

The API stores each incoming user message and generated assistant reply in
`conversation_messages`. Set `DATABASE_URL` to a PostgreSQL connection string
(preferred for hosted deployments). If it is not set, the API falls back to
SQLite at `/data/ownedlm.sqlite3`; this file survives redeploys only when the
hosting platform provides persistent storage mounted at `/data`.

Set `OWNEDLM_ADMIN_TOKEN` to a long random secret in the hosting environment.
The following private endpoints require that value in the `X-Admin-Token`
header:

- `GET /admin/training-stats` — count stored and pending messages.
- `GET /admin/training-data` — export pending conversation text for review
  and later training.

Collected conversations are **training candidates**, not immediate weight
updates. Review and filter them before training; do not train on private,
sensitive, abusive, or low-quality content. A training run and deployment of
its checkpoint are still required before the model's weights change.

## Current pretrained Arabic runtime

The main FastAPI service now loads the copied Jais-3-0.9B checkpoint as the OwnedLM inference engine. It no longer generates chat replies with the randomly initialized TinyLM. The model files are hosted at https://huggingface.co/Dhunun/OwnedLM-Jais-3-0.9B and are downloaded and cached locally by the runtime; this is local inference, not a remote inference API. The Docker deployment uses requirements-jais.txt and needs persistent storage mounted at /data. On CPU, allow several gigabytes of RAM and expect slower generation; GPU acceleration is preferred when available.

## Arabic-first Jais-3 training path

For the project's Arabic-only pretrained base, teacher-generated learning curriculum, validated SFT dataset, and LoRA customization workflow, see [JAIS-SETUP.md](JAIS-SETUP.md) and [distillation/README.md](distillation/README.md). This path uses the project's private Jais-3 checkpoint as a locally hosted base; the older tiny from-scratch Transformer remains a separate experimental implementation and cannot directly load Jais weights. Generated Arabic training data and adapter/model files stay outside the public Git repository.


## تأسيس اللغة العربية من مصادر متعددة

أُضيف مسار مستقل لتوسيع العربية في OwnedLM بالاعتماد على منهج لغوي شامل ومصادر بيانات عامة يجب تدقيق تراخيصها قبل الاستخدام. ابدأ من [دليل التدريب العربي](training/README.md)، وراجع [منهج اللغة العربية](training/arabic_curriculum.json) و[سجل مصادر البيانات](training/DATA_SOURCES.md). هذا المسار لا يشغّل Jais ولا يربط النموذج به وقت التشغيل؛ إنه ينقل خريطة المهارات وأفضل ممارسات إعداد البيانات إلى مشروع مستقل. لا تُنسخ أوزان Jais أو مفرداته مباشرة إلى TinyLM لأنها غير متوافقة مع بنيته.
