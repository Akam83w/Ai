export default function App() {
  return (
    <main className="page">
      <header>
        <h1>OwnedLM</h1>
        <p className="tag">نموذج لغوي صغير يُدرَّب من الصفر على جهازك</p>
      </header>

      <section>
        <h2>حالة المشروع</h2>
        <ul>
          <li>التدريب والتوليد محليان (Python + PyTorch) — بلا API خارجية.</li>
          <li>نجاح اختبارات الحلقة لا يعني أن النموذج صار مساعداً حوارياً.</li>
          <li>
            البذرة في <code>data/corpus/</code> صغيرة؛ التوليد المتوقع غير مفهوم حتى تُضاف بيانات
            مرخّصة أكبر.
          </li>
        </ul>
      </section>

      <section>
        <h2>أوامر مفيدة</h2>
        <pre>{`python -m unittest discover -s tests -t . -v
python train.py --data data/corpus --preset tiny --steps 200 --val-ratio 0.5 --out checkpoints/tiny.pt
python generate.py --checkpoint checkpoints/tiny.pt --prompt "في ورشة" --max-new-tokens 80
python eval_samples.py --checkpoint checkpoints/tiny.pt --out samples/after.txt`}</pre>
      </section>

      <section>
        <h2>وثائق</h2>
        <ul>
          <li>
            <code>METRICS.md</code> — معاني train_loss و val_loss
          </li>
          <li>
            <code>data/ADDING_DATA.md</code> — شروط إضافة بيانات عربية
          </li>
          <li>
            <code>WEB.md</code> — ماذا تعمل واجهة الويب وماذا لا تعمل
          </li>
        </ul>
      </section>
    </main>
  );
}
