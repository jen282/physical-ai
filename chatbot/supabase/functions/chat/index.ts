// TeachArm 챗봇 질문 처리: 질문 재작성 → 임베딩 → match_chunks 검색 → Gemini 답변
// POST { question: string, history?: { role: "user" | "assistant", content: string }[] }
// → { answer: string, sources: { title, heading_path, url }[] }
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const GEMINI_KEY = Deno.env.get("GEMINI_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
// Supabase가 자동으로 넣어 주는 서버 키 (새 secret 키가 없으면 legacy service_role 키)
const SECRET_KEYS = JSON.parse(Deno.env.get("SUPABASE_SECRET_KEYS") ?? "{}");
const DB_KEY = SECRET_KEYS["default"] ?? Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

// gemini-3.8-flash는 혼잡해 답변 한 번에 90초 넘게 걸려서 lite로 바꿨다 (lite는 수 초)
const CHAT_MODEL = "gemini-3.5-flash-lite";
// 질문 재작성은 한 문장을 바꿔 쓰는 단순 작업이라 답변 모델과 상관없이 lite로 고정한다
const REWRITE_MODEL = "gemini-3.5-flash-lite";
const EMBED_MODEL = "gemini-embedding-2";
const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const REPO_BLOB = "https://github.com/jen282/physical-ai/blob/main";

const MATCH_COUNT = 5;
const MAX_QUESTION = 500;
const MAX_HISTORY = 4;

const ALLOWED_ORIGINS = [/^https:\/\/jen282\.github\.io$/, /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/];

// 서비스와 무관한 질문에 대한 고정 답변. 이 답변일 때는 출처를 보내지 않는다.
const OFF_TOPIC_ANSWER = "서비스 상담 전용 챗봇이에요. TeachArm 서비스 관련 내용만 알려드릴 수 있어요.";

const SYSTEM_PROMPT = `당신은 TeachArm 프로젝트(시연으로 가르치는 모방학습 로봇팔)의 문서를 안내하는 상담 챗봇입니다.
- TeachArm 서비스(로봇팔, 모방학습, 기술, 도입·비용 등)와 전혀 관련 없는 질문(날씨, 잡담, 코딩 부탁 등)에는 다른 말 없이 "${OFF_TOPIC_ANSWER}"라고만 답하세요.
- 아래 [근거] 안의 내용만으로 답하고, 문장 끝에 [1]처럼 근거 번호를 표시하세요.
- 서비스 관련 질문인데 근거에 직접적인 답이 없으면 "자료에 없는 내용입니다"라고 밝히고, 근거 중 관련된 내용이 있으면 함께 알려 주세요.
- 목표치와 실측치를 구분하세요. 예를 들어 성공률 60%는 실측 결과가 아니라 목표입니다.
- 한국어 존댓말로 간결하게 답하세요.`;

type Turn = { role: "user" | "assistant"; content: string };
type Chunk = { title: string; source_path: string; heading_path: string | null; content: string; similarity: number };

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function corsHeaders(origin: string | null): Record<string, string> {
  const allowed = origin && ALLOWED_ORIGINS.some((re) => re.test(origin));
  return {
    ...(allowed ? { "Access-Control-Allow-Origin": origin! } : {}),
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Vary": "Origin",
  };
}

async function gemini(path: string, body: unknown) {
  // 503(모델 혼잡)은 일시적인 경우가 많아 잠깐 기다렸다 두 번까지 다시 시도한다
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`${GEMINI_BASE}/${path}`, {
      method: "POST",
      headers: { "x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok) return res.json();
    if (res.status === 429) throw new HttpError(429, "무료 사용량을 초과했습니다. 잠시 후 다시 시도해 주세요.");
    if (res.status === 503 && attempt < 2) {
      await res.body?.cancel();
      await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
      continue;
    }
    if (res.status === 503) throw new HttpError(503, "지금 이용자가 많아 답변이 어렵습니다. 잠시 후 다시 시도해 주세요.");
    throw new Error(`Gemini ${res.status}: ${await res.text()}`);
  }
}

async function generate(model: string, system: string, contents: unknown[]): Promise<string> {
  const data = await gemini(`${model}:generateContent`, {
    systemInstruction: { parts: [{ text: system }] },
    contents,
    generationConfig: { thinkingConfig: { thinkingLevel: "minimal" } },
  });
  const parts = data.candidates?.[0]?.content?.parts ?? [];
  return parts.map((p: { text?: string }) => p.text ?? "").join("").trim();
}

// "그건 왜요?"처럼 앞 대화에 기대는 질문을 혼자서도 뜻이 통하는 질문으로 바꾼다
async function rewrite(question: string, history: Turn[]): Promise<string> {
  if (history.length === 0) return question;
  const log = history.map((t) => `${t.role === "user" ? "사용자" : "챗봇"}: ${t.content}`).join("\n");
  const out = await generate(
    REWRITE_MODEL,
    "대화 기록을 참고해 마지막 질문을 앞 대화 없이도 이해되는 하나의 질문으로 바꾸세요. 바꾼 질문만 출력하세요.",
    [{ role: "user", parts: [{ text: `[대화 기록]\n${log}\n\n[마지막 질문]\n${question}` }] }],
  );
  return out || question;
}

async function embed(question: string): Promise<number[]> {
  const data = await gemini(`${EMBED_MODEL}:embedContent`, {
    // gemini-embedding-2 권장 질문 입력 형식
    content: { parts: [{ text: `task: question answering | query: ${question}` }] },
    outputDimensionality: 768,
  });
  return data.embedding.values;
}

async function search(embedding: number[]): Promise<Chunk[]> {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/rpc/match_chunks`, {
    method: "POST",
    headers: { apikey: DB_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ query_embedding: `[${embedding.join(",")}]`, match_count: MATCH_COUNT }),
  });
  if (!res.ok) throw new Error(`match_chunks ${res.status}: ${await res.text()}`);
  return res.json();
}

async function answer(question: string, history: Turn[], chunks: Chunk[]): Promise<string> {
  const context = chunks
    .map((c, i) => `[${i + 1}] ${c.title}${c.heading_path ? ` > ${c.heading_path}` : ""}\n${c.content}`)
    .join("\n\n");
  const contents = [
    ...history.map((t) => ({ role: t.role === "user" ? "user" : "model", parts: [{ text: t.content }] })),
    { role: "user", parts: [{ text: `[근거]\n${context}\n\n[질문]\n${question}` }] },
  ];
  return await generate(CHAT_MODEL, SYSTEM_PROMPT, contents);
}

function parseBody(body: unknown): { question: string; history: Turn[] } {
  const b = body as { question?: unknown; history?: unknown };
  const question = typeof b?.question === "string" ? b.question.trim() : "";
  if (!question) throw new HttpError(400, "question이 비어 있습니다.");
  if (question.length > MAX_QUESTION) throw new HttpError(400, `질문은 ${MAX_QUESTION}자 이하로 입력해 주세요.`);
  const history = (Array.isArray(b.history) ? b.history : [])
    .filter((t): t is Turn =>
      (t?.role === "user" || t?.role === "assistant") && typeof t?.content === "string"
    )
    .slice(-MAX_HISTORY)
    .map((t) => ({ role: t.role, content: t.content.slice(0, 2000) }));
  // Gemini 대화는 사용자 차례로 시작해야 한다
  while (history[0]?.role === "assistant") history.shift();
  return { question, history };
}

Deno.serve(async (req) => {
  const cors = corsHeaders(req.headers.get("Origin"));
  const json = (data: unknown, status = 200) =>
    new Response(JSON.stringify(data), { status, headers: { ...cors, "Content-Type": "application/json" } });

  if (req.method === "OPTIONS") return new Response(null, { headers: cors });
  if (req.method !== "POST") return json({ error: "POST만 지원합니다." }, 405);

  try {
    const { question, history } = parseBody(await req.json().catch(() => null));
    const standalone = await rewrite(question, history);
    const chunks = await search(await embed(standalone));
    const text = await answer(question, history, chunks);

    if (text.includes(OFF_TOPIC_ANSWER)) return json({ answer: OFF_TOPIC_ANSWER, sources: [] });
    // 답변의 [n]과 맞도록 근거 순서 그대로 돌려준다
    const sources = chunks.map((c) => ({
      title: c.title,
      heading_path: c.heading_path,
      url: `${REPO_BLOB}/${c.source_path}`,
    }));
    return json({ answer: text, sources });
  } catch (e) {
    if (e instanceof HttpError) return json({ error: e.message }, e.status);
    console.error(e);
    return json({ error: "답변을 만드는 중 오류가 발생했습니다." }, 500);
  }
});
