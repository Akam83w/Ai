# نموذج لغوي صغير من الصفر (OwnedLM)

نموذج **Transformer** سببي صغير، مكتوب بـ Python وPyTorch، يُدرَّب من **أوزان عشوائية** على نصوص تضعها أنت. بعد التدريب يعمل التوليد من ملف الأوزان على جهازك، من غير اتصال بـ ChatGPT أو OpenAI أو Grok أو أي واجهة نموذج خارجية، ومن غير تنزيل أوزان جاهزة.

هذا ليس مساعداً بمستوى ChatGPT، ولن يصبح كذلك بتدريب البذرة المرفقة. هو نواة تملكها وتكبرها: بنية النموذج، حلقة التدريب، حفظ الأوزان، واستئناف التدريب.

عند إضافة هذا النموذج لم يكن في المستودع ترخيص ولا بيانات تدريب. ملفات واجهة الويب التي كانت على `main` (مثل `package.json` و`vite.config.ts`) بقيت ولم تُحذف. هذا الالتزام يضيف ترخيص MIT لكود النموذج وللنصين الأصليين في `data/corpus/`.

## ما الذي يفعله النموذج

- محوّل فكّ فقط (decoder-only): تضمين رموز، مواضع متعلَّمة، انتباه ذاتي سببي متعدد الرؤوس، طبقة تغذية أمامية، وربط أوزان التضمين برأس الإخراج.
- المفردات تُبنى من حروف ملفاتك (UTF-8)، بما فيها العربية، من غير مفردات منزلة.
- التدريب: AdamW، قص المتدرجات، فقدان الانتروبيا المتقاطعة، حفظ نقطة استئناف.
- التوليد: اختيار عشوائي بدرجة حرارة و`top-k`، أو اختيار جشع إذا كانت الحرارة `0`.

## ما الذي لا يفعله

- لا يستدعي API ولا يحمّل نموذجاً مدرَّباً.
- لا يعرف العالم. يرى فقط الملفات التي تعطيه إياها، وقد يحفظها حرفياً إذا طال التدريب عليها.
- البذرة الحالية بضع كيلوبايتات من نص أصلي. بعدها المتوقع: ضياع في المحارف، لا محادثة.
- لا يوجد ضمان أمان أو صحة للإجابات. لا تستخدم المخرجات في قرار طبي أو قانوني أو مالي.

## البيانات والتقسيم

اقرأ [data/README.md](data/README.md) و[data/SOURCES.md](data/SOURCES.md) و[data/ADDING_DATA.md](data/ADDING_DATA.md) قبل إضافة نص خارجي.

- التقسيم إلى تدريب/تحقق **على مستوى الملف كاملاً** (`--val-ratio`) حتى لا تتسرب نوافذ من نفس الملف.
- معاني `train_loss` و`val_loss` وحدودها: [METRICS.md](METRICS.md).
- عينات توليد ثابتة للمقارنة قبل/بعد:

```bash
python eval_samples.py --checkpoint checkpoints/tiny.pt --out samples/after.txt
```

## التشغيل

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
python train.py --data data/corpus --preset tiny --steps 200 --val-ratio 0.5 --out checkpoints/tiny.pt
python generate.py --checkpoint checkpoints/tiny.pt --prompt "في ورشة" --max-new-tokens 80
```

### تشغيل واجهة المحادثة المحلية

بعد تثبيت PyTorch وتدريب النموذج، افتح طرفيتين من مجلد المشروع:

1. شغّل خادم التوليد المحلي: `python server.py`
2. في الطرفية الثانية: `npm install && npm run dev`
3. افتح `http://127.0.0.1:8080`.

إذا لم يوجد `checkpoints/tiny.pt`، درّب أولاً بالأمر أعلاه. تفاصيل الربط وحدوده في [WEB.md](WEB.md). واجهة المحادثة تجربة أولية؛ النموذج لا يتبع التعليمات بالضرورة ولا يستخدم سجل المحادثة كسياق.


## هيكل الملفات

```text
model.py          النموذج
tokenizer.py      مفردات الحروف من نصك
dataset.py        قراءة الملفات وتقطيع النوافذ (تقسيم بالملف)
train.py          التدريب والاستئناف وval_loss
generate.py       التوليد المحلي
eval_samples.py   عينات مطالبات عربية ثابتة
METRICS.md        معاني train/val loss وحدودها
tests/test_lm.py  الاختبارات
data/ADDING_DATA.md  دليل إضافة بيانات مرخّصة
data/corpus/      البذرة الأصلية
data/private/     بيانات محلية غير متتبعة
```

## فحص أولي، وليس مستوى محادثة

الاختبارات تتحقق من الشكل، القناع السببي، انخفاض الخسارة على نص مكرر، التقسيم بلا تسرب، قياس train/val بشكل منفصل، التدريب والاستئناف، والتوليد من غير عملاء خارجيين.

تشغيل فعلي على بذرة المستودع (ملف عربي للتدريب وملف إنجليزي للتحقق، 50 خطوة، `tiny`):

- `train_loss` انخفض تقريباً من 4.09 إلى 3.88 (متوسط النافذة).
- `val_loss` على الملف الإنجليزي بقي حوالي 4.4–4.5 (لا تحسن واضح).
- عينات `eval_samples.py` بعد 50 خطوة بقيت غير مفهومة.

هذا يؤكد أن الحلقة تعمل، وأن البذرة أصغر من أن تدعم تعميماً أو توليداً مقروءاً. نجاح الاختبارات **لا** يعني أن النموذج صار مساعداً.

## الخطوة التالية

1. أضف نصاً عربياً مرخّصاً وموثّقاً في SOURCES.md (انظر ADDING_DATA.md).
2. درّب مع `--val-ratio` على ملفات مستقلة، وراقب val_loss لا train_loss وحده.
3. لا تكبّر النموذج قبل أن تتحسن val_loss على بيانات لم تدخل التدريب.


## Pretrained chat runtime (deploy/deplexo)

The deployment branch now serves the React UI with the Jais-3-0.9B checkpoint from `Dhunun/OwnedLM-Jais-3-0.9B`. It loads the weights locally in the container using Transformers 4.57.1 and trusted model code from the Apache-2.0 checkpoint. The chat API accepts the recent conversation messages as context. Configure a persistent volume at `/data` so the Hugging Face cache survives redeployments; first startup downloads about 2.16 GB. CPU inference requires several GB of available RAM and can be slow. Check `/api/status` for loading/ready/error state.
