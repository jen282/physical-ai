-- TeachArm 챗봇 RAG 스키마 (Supabase TeachARM 프로젝트)
create extension if not exists vector with schema extensions;

create table public.documents (
  id          bigint generated always as identity primary key,
  title       text not null,
  source_path text not null unique          -- 'report/model_selection.md'
);

create table public.chunks (
  id           bigint generated always as identity primary key,
  document_id  bigint not null references public.documents(id) on delete cascade,
  chunk_index  int  not null,               -- 문서 안에서의 순서
  heading_path text,                        -- '4. 결론 > 4.3 R2 큐브 탐지 보조 → YOLOE'
  content      text not null,
  embedding    extensions.vector(768) not null,
  unique (document_id, chunk_index)
);

create index chunks_document_id_idx on public.chunks (document_id);

-- RLS는 켜고 정책은 만들지 않는다: 공개 키로는 접근 불가, 서버 키(ingest.py, Edge Function)만 접근
alter table public.documents enable row level security;
alter table public.chunks    enable row level security;

-- 이 프로젝트는 새 테이블에 service_role 권한을 자동으로 주지 않아 명시적으로 부여
grant select, insert, delete on public.documents to service_role;
grant select, insert, delete on public.chunks    to service_role;

-- 질문 벡터와 가까운 청크 찾기
create function public.match_chunks(query_embedding extensions.vector(768), match_count int default 5)
returns table (title text, source_path text, heading_path text, content text, similarity float)
language sql stable
set search_path = ''
as $$
  select d.title, d.source_path, c.heading_path, c.content,
         1 - (c.embedding operator(extensions.<=>) query_embedding)
  from public.chunks c join public.documents d on d.id = c.document_id
  order by c.embedding operator(extensions.<=>) query_embedding
  limit match_count;
$$;

-- 검색 함수도 서버 키로만 호출
revoke execute on function public.match_chunks(extensions.vector, int) from public, anon, authenticated;
grant  execute on function public.match_chunks(extensions.vector, int) to service_role;
