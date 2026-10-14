// TeachArm 상담 챗봇 위젯
// 페이지에 <link rel="stylesheet" href="chatbot/widget/chat-widget.css"> 와
// <script src="chatbot/widget/chat-widget.js" defer></script> 를 넣으면 오른쪽 아래에 버튼이 생긴다.
(function () {
  const ENDPOINT = "https://eeoponnsqoclkzkcfndi.supabase.co/functions/v1/chat";
  const MAX_HISTORY = 4; // 서버에 보낼 최근 대화 수 (질문+답변 각각 1개로 셈)

  // 입력창 위 빠른 질문 버튼: 누르면 question을 대신 보낸다
  const QUICK_QUESTIONS = [
    { label: "서비스 개요", question: "TeachArm은 어떤 서비스인가요?" },
    { label: "기술 문의", question: "TeachArm은 어떤 기술로 동작하나요?" },
    { label: "도입 문의", question: "TeachArm을 도입하려면 무엇이 필요하고 비용은 얼마나 드나요?" },
  ];

  const WELCOME = "안녕하세요! TeachArm 상담 챗봇이에요. 서비스, 기술, 도입에 대해 궁금한 점을 물어보세요.";

  const ICON_CHAT =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>';
  const ICON_SEND =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';

  const history = []; // 페이지 메모리에만 보관, 새로고침하면 초기화
  let busy = false;

  // ---------- DOM ----------
  const root = document.createElement("div");
  root.className = "tac-root";
  root.innerHTML = `
    <button class="tac-launcher" type="button" aria-label="상담 챗봇 열기">${ICON_CHAT}</button>
    <section class="tac-panel" role="dialog" aria-label="TeachArm 상담 챗봇">
      <header class="tac-header">
        <span class="tac-dot"></span>
        <div>
          <div class="tac-title">TeachArm 상담</div>
          <div class="tac-subtitle">프로젝트 문서 기반 AI 답변</div>
        </div>
        <button class="tac-close" type="button" aria-label="닫기">&times;</button>
      </header>
      <div class="tac-messages" aria-live="polite"></div>
      <footer class="tac-footer">
        <div class="tac-quick"></div>
        <form class="tac-form">
          <textarea class="tac-input" rows="1" maxlength="500" placeholder="질문을 입력하세요" aria-label="질문"></textarea>
          <button class="tac-send" type="submit" aria-label="보내기">${ICON_SEND}</button>
        </form>
      </footer>
    </section>`;
  document.body.appendChild(root);

  const $ = (sel) => root.querySelector(sel);
  const messages = $(".tac-messages");
  const form = $(".tac-form");
  const input = $(".tac-input");
  const quick = $(".tac-quick");

  QUICK_QUESTIONS.forEach(({ label, question }) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "tac-chip";
    chip.textContent = label;
    chip.addEventListener("click", () => ask(question));
    quick.appendChild(chip);
  });

  // ---------- 렌더링 ----------
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // 답변의 최소 마크다운(**굵게**, - 목록)과 [1], [1, 4] 근거 번호를 HTML로 바꾼다
  function renderAnswer(text, sources) {
    const inline = (line) =>
      escapeHtml(line)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) =>
          nums
            .split(",")
            .map((n) => {
              const s = sources[Number(n) - 1];
              return s
                ? `<a class="tac-cite" href="${escapeHtml(s.url)}" target="_blank" rel="noopener">[${n.trim()}]</a>`
                : "";
            })
            .join("")
        );
    let html = "";
    let inList = false;
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      const item = line.match(/^[-*•]\s+(.*)$/);
      if (item) {
        if (!inList) (html += "<ul>"), (inList = true);
        html += `<li>${inline(item[1])}</li>`;
        continue;
      }
      if (inList) (html += "</ul>"), (inList = false);
      if (line) html += `<p>${inline(line)}</p>`;
    }
    if (inList) html += "</ul>";
    return html;
  }

  // 답변에서 실제로 인용한 근거만 원래 번호 그대로 보여 준다
  function renderSources(text, sources) {
    const cited = new Set();
    for (const m of text.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
      m[1].split(",").forEach((n) => cited.add(Number(n)));
    }
    // 같은 섹션이 여러 청크로 인용되면 '[1][3] 제목 › 섹션'처럼 한 줄로 합친다
    const groups = new Map();
    [...cited]
      .sort((a, b) => a - b)
      .filter((n) => sources[n - 1])
      .forEach((n) => {
        const s = sources[n - 1];
        const section = (s.heading_path || "").split(" > ").pop();
        const label = section && section !== s.title ? `${s.title} › ${section}` : s.title;
        if (!groups.has(label)) groups.set(label, { url: s.url, nums: [] });
        groups.get(label).nums.push(n);
      });
    const items = [...groups].map(
      ([label, g]) =>
        `<a href="${escapeHtml(g.url)}" target="_blank" rel="noopener" title="${escapeHtml(label)}">${g.nums
          .map((n) => `[${n}]`)
          .join("")} ${escapeHtml(label)}</a>`
    );
    // 기본은 접어 두고, 펼칠 때만 목록을 보여 준다
    return items.length
      ? `<details class="tac-sources"><summary>출처 ${items.length}개</summary>${items.join("")}</details>`
      : "";
  }

  function addMessage(className, html) {
    const el = document.createElement("div");
    el.className = `tac-msg ${className}`;
    el.innerHTML = html;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function setBusy(value) {
    busy = value;
    root.querySelectorAll(".tac-chip, .tac-send").forEach((b) => (b.disabled = value));
    input.disabled = value;
    if (!value) input.focus();
  }

  // ---------- 질문 보내기 ----------
  async function ask(question) {
    question = question.trim();
    if (!question || busy) return;
    addMessage("tac-user", escapeHtml(question));
    input.value = "";
    autoresize();
    setBusy(true);
    const typing = addMessage("tac-bot tac-typing", "답변 작성 중");

    try {
      const res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: history.slice(-MAX_HISTORY) }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.answer) throw new Error(data.error || "잠시 후 다시 시도해 주세요.");

      typing.remove();
      addMessage("tac-bot", renderAnswer(data.answer, data.sources || []) + renderSources(data.answer, data.sources || []));
      history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
    } catch (err) {
      typing.remove();
      const msg = err instanceof TypeError ? "네트워크 오류가 발생했어요. 잠시 후 다시 시도해 주세요." : err.message;
      addMessage("tac-bot tac-error", escapeHtml(msg));
    } finally {
      setBusy(false);
    }
  }

  // ---------- 이벤트 ----------
  function autoresize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 96)}px`;
  }

  function open() {
    root.classList.add("tac-open");
    if (!messages.children.length) addMessage("tac-bot", `<p>${escapeHtml(WELCOME)}</p>`);
    input.focus();
  }

  function close() {
    root.classList.remove("tac-open");
    $(".tac-launcher").focus();
  }

  $(".tac-launcher").addEventListener("click", open);
  $(".tac-close").addEventListener("click", close);
  root.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    ask(input.value);
  });
  input.addEventListener("input", autoresize);
  input.addEventListener("keydown", (e) => {
    // Enter는 보내기, Shift+Enter는 줄바꿈 (한글 조합 중 Enter는 무시)
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      ask(input.value);
    }
  });
})();
