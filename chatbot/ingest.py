"""TeachArm 챗봇용 문서 전처리: md 읽기 → 청킹 → 임베딩 → Supabase 적재

사용법 (프로젝트 루트에서):
    python chatbot/ingest.py --dry-run       # 청킹 결과만 확인 (Gemini 호출 없음)
    python chatbot/ingest.py                 # 청킹 → 임베딩 → chunks.json 저장 → Supabase 업로드
    python chatbot/ingest.py --upload-only   # 저장된 chunks.json을 다시 업로드 (임베딩 생략)

.env: GEMINI_KEY, SUPABASE_URL, SUPABASE_KEY_SECRET
"""

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "chatbot" / "data" / "chunks.json"

SOURCES = [
    "PRD/PRD_arm.md",
    "report/report_arm.md",
    "report/preprocess_arm.md",
    "report/checklist_arm.md",
    "report/object_detection.md",
    "report/model_selection.md",
]

MAX_CHARS = 1500  # 이보다 긴 섹션은 문단 기준으로 다시 나눔
MIN_CHARS = 200   # 이보다 짧은 섹션은 같은 상위 헤딩의 다음 섹션과 합침

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
EMBED_BATCH = 10  # 무료 등급 분당 한도 때문에 작게 나눔

# 장식용 이모지만 제거. ○△✕◐● 같은 표의 평가 기호는 의미가 있어서 남긴다.
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF⭐⚠️‍]")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
CHECKBOX_RE = re.compile(r"\[(v| )\]")
PLACEHOLDER_RE = re.compile(r"(?:\\_){2,}|_{3,}")


# ---------- 정제 ----------

def clean_checkbox_line(line: str) -> str:
    """'[v] A [ ] B [ ] C / 총 2대' → '선택: A / 총 2대'. 고르지 않은 보기는 버린다."""
    parts = CHECKBOX_RE.split(line)
    prefix, rest = parts[0].strip(), parts[1:]
    checked, extras = [], []
    for mark, text in zip(rest[0::2], rest[1::2]):
        text = text.strip()
        # 마지막 보기 뒤에 붙은 '/ 총 2대', '/ 모델명: ...' 같은 기재 내용은 선택 여부와 무관하게 남긴다
        if " / " in text:
            text, extra = text.split(" / ", 1)
            extras.append(extra.strip())
        # '4단계 이상 (단계 수: 3 — ...)'처럼 공백 뒤 괄호로 적은 기재 내용도 남긴다
        note = re.search(r"\s\((.+)\)$", text)
        if note and mark == " ":
            extras.append(f"({note.group(1)})")
        if mark == "v":
            checked.append(text)
    result = "선택: " + (", ".join(checked) if checked else "(미응답)")
    if extras:
        result += " / " + " / ".join(extras)
    return f"{prefix} {result}".strip()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = EMOJI_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*-{3,}\s*", line):  # 구분선
            continue
        if CHECKBOX_RE.search(line):
            line = clean_checkbox_line(line)
        line = PLACEHOLDER_RE.sub("(미기재)", line)
        if line.lstrip().startswith("|"):  # 표의 칸 맞춤용 공백·대시 줄이기
            line = re.sub(r" {2,}", " ", line)
            line = re.sub(r"-{4,}", "---", line)
        lines.append(line.rstrip())
    return "\n".join(lines)


# ---------- 청킹 ----------

def split_sections(text: str):
    """헤딩(#~###) 기준으로 나눈다. 반환: [(heading_path 리스트, 본문)]"""
    sections, path, buf = [], [], []
    in_code = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(path), body))
        buf.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
            buf.append(line)  # 헤딩 줄도 본문에 남겨 합쳐도 구조가 보이게 한다
        else:
            buf.append(line)
    flush()
    return sections


def split_blocks(body: str):
    """빈 줄 기준 문단 단위. 코드 블록은 쪼개지 않는다."""
    blocks, buf, in_code = [], [], False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def is_table(block: str) -> bool:
    lines = block.splitlines()
    return len(lines) >= 2 and all(l.lstrip().startswith("|") for l in lines)


def split_long_block(block: str):
    """MAX_CHARS보다 긴 문단 하나를 나눈다. 표는 조각마다 헤더 행(2줄)을 다시 붙인다."""
    lines = block.splitlines()
    header, rows = (lines[:2], lines[2:]) if is_table(block) else ([], lines)
    pieces, cur = [], list(header)
    for row in rows:
        if cur != header and len("\n".join(cur + [row])) > MAX_CHARS:
            pieces.append("\n".join(cur))
            cur = list(header)
        cur.append(row)
    pieces.append("\n".join(cur))
    return pieces


def pack(body: str):
    """섹션 본문을 MAX_CHARS 이하 조각으로 묶는다."""
    if len(body) <= MAX_CHARS:
        return [body]
    blocks = []
    for b in split_blocks(body):
        blocks.extend(split_long_block(b) if len(b) > MAX_CHARS else [b])
    pieces, cur = [], ""
    for b in blocks:
        # 헤딩·도입 문장처럼 짧은 앞부분은 혼자 남기지 않고 다음 문단에 붙인다 (MAX_CHARS를 조금 넘을 수 있음)
        if len(cur) >= MIN_CHARS and len(cur) + 2 + len(b) > MAX_CHARS:
            pieces.append(cur)
            cur = b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        pieces.append(cur)
    return pieces


def common_prefix(a, b):
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def is_heading_only(body: str) -> bool:
    return all(HEADING_RE.match(l) for l in body.splitlines() if l.strip())


def merge_short(sections):
    """MIN_CHARS 미만 섹션을 같은 상위 헤딩 아래의 앞뒤 섹션과 합친다."""
    # 1) 다음 섹션과 합치기. 길어지면 pack()이 다시 나눈다.
    merged = []
    for path, body in sections:
        if merged:
            prev_path, prev_body = merged[-1]
            parent = prev_path[:-1]
            # 헤딩만 있는 섹션은 앞에 붙이지 않는다 (다음 하위 섹션의 머리가 되어야 함)
            if len(prev_body) < MIN_CHARS and path[: len(parent)] == parent and not is_heading_only(body):
                # '## 4. 기술 스택'처럼 헤딩만 있는 섹션은 하위 섹션의 경로를 그대로 쓴다
                new_path = path if is_heading_only(prev_body) else common_prefix(prev_path, path)
                merged[-1] = (new_path, f"{prev_body}\n\n{body}")
                continue
        merged.append((path, body))
    # 2) 그룹의 마지막이라 남은 짧은 섹션은 앞 섹션 뒤에 붙인다.
    result = []
    for path, body in merged:
        if result and len(body) < MIN_CHARS and not is_heading_only(body):
            prev_path, prev_body = result[-1]
            parent = path[:-1]
            if prev_path[: len(parent)] == parent and len(prev_body) + 2 + len(body) <= MAX_CHARS:
                result[-1] = (common_prefix(prev_path, path), f"{prev_body}\n\n{body}")
                continue
        result.append((path, body))
    return result


def chunk_document(text: str):
    chunks = []
    for path, body in merge_short(split_sections(text)):
        for piece in pack(body):
            chunks.append({"heading_path": " > ".join(path) or None, "content": piece})
    return chunks


def doc_title(text: str) -> str:
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return next(l.strip() for l in text.splitlines() if l.strip())


# ---------- 임베딩 ----------

def embed_all(docs):
    from dotenv import load_dotenv
    from google import genai
    from google.genai import errors, types

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    items = [(d, c) for d in docs for c in d["chunks"]]
    for i in range(0, len(items), EMBED_BATCH):
        batch = items[i : i + EMBED_BATCH]
        # gemini-embedding-2 권장 문서 입력 형식: 'title: ... | text: ...'
        inputs = [
            f"title: {d['title']}" + (f" > {c['heading_path']}" if c["heading_path"] else "")
            + f" | text: {c['content']}"
            for d, c in batch
        ]
        for attempt in range(6):
            try:
                res = client.models.embed_content(
                    model=EMBED_MODEL,
                    # 문자열 리스트를 그대로 넘기면 embedding-2가 하나로 합쳐 임베딩 1개만 돌려준다
                    contents=[types.Content(parts=[types.Part(text=t)]) for t in inputs],
                    config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
                )
                break
            except errors.ClientError as e:
                if e.code != 429 or attempt == 5:
                    raise
                print("  분당 한도 초과, 30초 후 재시도")
                time.sleep(30)
        assert len(res.embeddings) == len(batch), f"임베딩 개수 불일치: {len(res.embeddings)} != {len(batch)}"
        for (_, c), e in zip(batch, res.embeddings):
            c["embedding"] = e.values
        print(f"  임베딩 {min(i + EMBED_BATCH, len(items))}/{len(items)}")


# ---------- 업로드 ----------

def upload(docs):
    """같은 source_path의 기존 문서를 지우고(청크는 cascade로 함께 삭제) 새로 넣는다."""
    import requests
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    base = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1"
    # RLS를 우회하는 secret 키. 이 스크립트에서만 쓰고 커밋·위젯에는 절대 넣지 않는다.
    headers = {"apikey": os.environ["SUPABASE_KEY_SECRET"], "Content-Type": "application/json"}

    def check(r):
        if not r.ok:
            raise RuntimeError(f"Supabase {r.status_code}: {r.text}")
        return r

    paths = ",".join(f'"{d["source_path"]}"' for d in docs)
    check(requests.delete(f"{base}/documents", params={"source_path": f"in.({paths})"}, headers=headers))

    rows = check(requests.post(
        f"{base}/documents",
        headers={**headers, "Prefer": "return=representation"},
        json=[{"title": d["title"], "source_path": d["source_path"]} for d in docs],
    )).json()
    doc_ids = {r["source_path"]: r["id"] for r in rows}

    chunks = [
        {
            "document_id": doc_ids[d["source_path"]],
            "chunk_index": c["chunk_index"],
            "heading_path": c["heading_path"],
            "content": c["content"],
            "embedding": "[" + ",".join(map(str, c["embedding"])) + "]",
        }
        for d in docs for c in d["chunks"]
    ]
    for i in range(0, len(chunks), 50):
        check(requests.post(f"{base}/chunks", headers=headers, json=chunks[i : i + 50]))
    print(f"업로드: 문서 {len(rows)}개, 청크 {len(chunks)}개")


# ---------- 실행 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="임베딩 없이 청킹 결과만 출력")
    parser.add_argument("--upload-only", action="store_true", help="저장된 chunks.json을 업로드만 함")
    args = parser.parse_args()

    if args.upload_only:
        upload(json.loads(OUT_PATH.read_text(encoding="utf-8")))
        return

    docs = []
    for src in SOURCES:
        text = clean_text((ROOT / src).read_text(encoding="utf-8"))
        chunks = chunk_document(text)
        for i, c in enumerate(chunks):
            c["chunk_index"] = i
        docs.append({"title": doc_title(text), "source_path": src, "chunks": chunks})
        sizes = [len(c["content"]) for c in chunks]
        print(f"{src}: {len(chunks)}개 청크 (최소 {min(sizes)} / 최대 {max(sizes)}자) — {docs[-1]['title']}")
    print(f"총 {sum(len(d['chunks']) for d in docs)}개 청크")

    if args.dry_run:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        preview = OUT_PATH.with_name("chunks_preview.md")
        with preview.open("w", encoding="utf-8") as f:
            for d in docs:
                for c in d["chunks"]:
                    f.write(f"\n\n========== {d['source_path']} #{c['chunk_index']} | {c['heading_path']} "
                            f"({len(c['content'])}자)\n{c['content']}")
        print(f"미리보기: {preview.relative_to(ROOT)}")
        return

    embed_all(docs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {OUT_PATH.relative_to(ROOT)}")
    upload(docs)


if __name__ == "__main__":
    main()
