import { useEffect, useRef, useState, type FormEvent } from "react";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

type ApiStatus = "checking" | "loading" | "ready" | "error" | "offline";

const TRAIN_COMMAND =
  "python train.py --data data/corpus --preset tiny --steps 200 --val-ratio 0.5 --out checkpoints/tiny.pt";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<ApiStatus>("checking");
  const [statusMessage, setStatusMessage] = useState("أتحقق من خادم النموذج...");
  const [nextId, setNextId] = useState(1);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const check = async () => {
      try {
        const response = await fetch("/api/status");
        if (!response.ok) throw new Error("status request failed");
        const data = (await response.json()) as {
          ready: boolean; loading: boolean; message: string; error?: string | null;
        };
        if (!active) return;
        setStatus(data.ready ? "ready" : data.loading ? "loading" : "error");
        setStatusMessage(data.ready ? data.message : data.loading ? data.message : (data.error || data.message));
        if (data.loading) timer = setTimeout(check, 10000);
      } catch {
        if (!active) return;
        setStatus("offline");
        setStatusMessage("تعذّر الاتصال بخادم النموذج.");
      }
    };
    void check();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = input.trim();
    if (!prompt || busy) return;

    const userId = nextId;
    const assistantId = nextId + 1;
    setNextId(nextId + 2);
    setInput("");
    setMessages((previous) => [...previous, { id: userId, role: "user", content: prompt }]);
    setBusy(true);

    try {
      const conversation = [...messages, { id: userId, role: "user" as const, content: prompt }]
        .slice(-16)
        .map(({ role, content }) => ({ role, content }));
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: prompt, messages: conversation, max_new_tokens: 256 }),
      });
      const data = (await response.json()) as { reply?: string; error?: string };
      if (!response.ok) throw new Error(data.error || "فشل طلب التوليد.");
      setMessages((previous) => [
        ...previous,
        {
          id: assistantId,
          role: "assistant",
          content: data.reply || "ما وصلني نص من النموذج.",
        },
      ]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "صار خطأ غير متوقع أثناء التوليد.";
      setMessages((previous) => [
        ...previous,
        { id: assistantId, role: "assistant", content: `تعذّر التوليد: ${message}` },
      ]);
      if (message.includes("خادم Python")) {
        setStatus("offline");
        setStatusMessage("خادم Python مو شغّال. شغّل python server.py بطرفية ثانية.");
      }
    } finally {
      setBusy(false);
    }
  }

  const statusLabel: Record<ApiStatus, string> = {
    checking: "جاري الفحص",
    loading: "تحميل النموذج",
    ready: "النموذج جاهز",
    error: "تعذّر تحميل النموذج",
    offline: "الخادم غير متصل",
  };

  return (
    <main className="app-shell" dir="rtl">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">O</div>
          <div>
            <h1>OwnedLM</h1>
            <p>مساعدك العربي المستقل</p>
          </div>
        </div>

        <div className={`status-card status-${status}`}>
          <span className="status-dot" />
          <div>
            <strong>{statusLabel[status]}</strong>
            <p>{statusMessage}</p>
          </div>
        </div>

        <section className="side-section">
          <span className="eyebrow">عن النموذج</span>
          <h2>محلي ومفتوح للتطوير</h2>
          <p>
            يعمل بمحرك Jais-3-0.9B محلياً داخل خادم OwnedLM. لا يستخدم واجهة استدلال خارجية.
          </p>
        </section>

        <section className="side-section setup-card">
          <span className="eyebrow">محرك اللغة</span>
          <h2>Jais-3 · 0.9B</h2>
          <p>أوزان مدرّبة مسبقاً مع Tokenizer خاص، وتُحمّل على الخادم عند بدء التشغيل.</p>
          <p className="small-note">أول تشغيل قد يستغرق وقتاً لتنزيل نحو 2.16 جيجابايت من الأوزان.</p>
        </section>

        <div className="sidebar-footer">نسخة تجريبية · لا تعتمد على المخرجات كمعلومات مؤكدة<br /><span>تطوير: ذنون</span></div>
      </aside>

      <section className="chat-panel" aria-label="محادثة النموذج">
        <header className="chat-header">
          <div>
            <span className="eyebrow">مساحة التجربة</span>
            <h2>جرّب OwnedLM</h2>
          </div>
          <span className="local-pill"><span /> محلي</span>
        </header>

        <div className="notice" role="note">
          المحادثة تستخدم النموذج المدرّب مسبقاً، وترسل آخر الرسائل كسياق حتى يفهم تسلسل الحوار.
        </div>

        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article key={message.id} className={`message message-${message.role}`}>
              <div className="message-avatar" aria-hidden="true">
                {message.role === "user" ? "أنت" : "O"}
              </div>
              <div className="message-body">
                <span className="message-name">
                  {message.role === "user" ? "إنت" : "OwnedLM"}
                </span>
                <p>{message.content}</p>
              </div>
            </article>
          ))}
          {busy && (
            <article className="message message-assistant">
              <div className="message-avatar" aria-hidden="true">O</div>
              <div className="message-body">
                <span className="message-name">OwnedLM</span>
                <p className="typing">النموذج دا يولّد<span>…</span></p>
              </div>
            </article>
          )}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <label className="sr-only" htmlFor="prompt">اكتب رسالة للنموذج</label>
          <textarea
            id="prompt"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="اكتب رسالتك إلى OwnedLM..."
            rows={2}
            maxLength={2000}
            disabled={busy}
          />
          <div className="composer-footer">
            <span>المحادثة بالعربية · التوليد محلي</span>
            <button type="submit" disabled={busy || !input.trim()}>
              {busy ? "لحظة..." : "إرسال"}
              <span aria-hidden="true"> ←</span>
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
