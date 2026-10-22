# واجهة الويب — الحالة الفعلية

## ماذا كان مكسوراً؟

على الفرع `feature/train-val-metrics` (وقبلها على `main`) وُجدت ملفات:

- `package.json` يستدعي `scripts/with-app-env.mjs`
- `vite.config.ts` يستورد `./scripts/app-env-plugin.mjs` ويُعدّ TanStack Start مع `src/router.tsx`

لكن **مجلد `scripts/` غير موجود في المستودع**، و**لا يوجد** `src/router.tsx` ولا مسارات (`routes/`) ولا معظم مكدس تطبيق Grok App Builder. ما تبقى تحت `src/lib/app-data/` شظايا فقط.

لذلك `npm run dev` كان يفشل فوراً بخطأ «الملف غير موجود»، وليس بسبب الشبكة أو GPU.

## ماذا أُصلح في هذه المرحلة؟

1. `scripts/with-app-env.mjs` — غلاف بسيط يشغّل الأمر التالي كما هو.
2. `scripts/app-env-plugin.mjs` — بقي كمرجع؛ الإعداد الحالي لـ Vite لم يعد يعتمد عليه.
3. `vite.config.ts` — Vite + React فقط (بدون TanStack Start / Nitro) لأن بقية السقالة غير مرفوعة.
4. `index.html` + `src/main.tsx` + `src/App.tsx` + `src/styles.css` — صفحة حالة للمشروع OwnedLM.
5. `package.json` — تبسيط سكربتات التشغيل لتطابق ما يوجد فعلاً.

هذا **ليس** استعادة كاملة لواجهة App Builder السابقة، ولا لوحة تحكم للتدريب. الهدف: تشغيل واجهة خفيفة تشرح المشروع دون ادعاء قدرات غير موجودة.

## كيف تشغّل الواجهة؟

```bash
npm install
npm run dev
```

ثم افتح `http://127.0.0.1:8080`.

## ما الذي لا تفعله الواجهة؟

- لا تستدعي نماذج خارجية.
- لا تدرّب النموذج من المتصفح.
- لا تعرض «مساعداً ذكياً».
