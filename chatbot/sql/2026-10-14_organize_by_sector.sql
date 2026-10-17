-- TeachArm 챗봇 RAG: 문서를 섹터(기획 / 하드웨어 / 소프트웨어)별로 정리
-- Supabase 대시보드 > SQL Editor에 붙여 넣고 한 번 실행합니다. 전체가 한 트랜잭션이라 실패하면 아무것도 바뀌지 않습니다.
--
-- 바뀌는 것
--   - sectors 테이블 추가, documents.sector_id 추가 (기존 문서 7개는 경로로 분류)
--   - chunks.embedding에 HNSW 벡터 인덱스 추가
--   - match_chunks에 섹터 필터(filter_sector, 선택)와 결과 sector 컬럼 추가. 기존 호출은 그대로 동작
--   - 대시보드용 섹터별 뷰 6개 (데이터 복사 없음, 임베딩 컬럼 제외)

begin;

-- 1) 섹터
create table public.sectors (
  id         smallint generated always as identity primary key,
  code       text not null unique,   -- 'hardware' (뷰 이름·검색 필터에 쓰는 영문 코드)
  name       text not null,          -- '하드웨어'
  sort_order smallint not null default 0
);
alter table public.sectors enable row level security;
grant select on public.sectors to service_role;

insert into public.sectors (code, name, sort_order) values
  ('planning', '기획', 1),
  ('hardware', '하드웨어', 2),
  ('software', '소프트웨어', 3);

-- 2) 문서에 섹터 연결 (기존 문서는 경로로 분류)
alter table public.documents add column sector_id smallint references public.sectors(id);

update public.documents d set sector_id = s.id
from public.sectors s
where s.code = case d.source_path
  when 'PRD/PRD_arm.md'             then 'planning'
  when 'report/checklist_arm.md'    then 'planning'
  when 'report/hardware_arm.md'     then 'hardware'
  when 'report/report_arm.md'       then 'hardware'
  when 'report/preprocess_arm.md'   then 'software'
  when 'report/object_detection.md' then 'software'
  when 'report/model_selection.md'  then 'software'
end;

alter table public.documents alter column sector_id set not null;
create index documents_sector_id_idx on public.documents (sector_id);

-- 3) 벡터 검색 인덱스 (코사인 거리)
create index chunks_embedding_hnsw_idx on public.chunks
  using hnsw (embedding extensions.vector_cosine_ops);

-- 4) 검색 함수: 섹터 필터(선택)와 섹터 이름 추가. 기존 호출(query_embedding, match_count)은 그대로 동작
drop function public.match_chunks(extensions.vector, int);

create function public.match_chunks(
  query_embedding extensions.vector(768),
  match_count int default 5,
  filter_sector text default null          -- 'hardware' 등. null이면 전체 섹터 검색
)
returns table (sector text, title text, source_path text, heading_path text, content text, similarity float)
language sql stable
-- 섹터로 거른 뒤에도 match_count개를 채우도록 HNSW 반복 스캔을 켠다 (pgvector 0.8+)
set search_path = ''
set hnsw.iterative_scan = strict_order
as $$
  select s.name, d.title, d.source_path, c.heading_path, c.content,
         1 - (c.embedding operator(extensions.<=>) query_embedding)
  from public.chunks c
  join public.documents d on d.id = c.document_id
  join public.sectors s on s.id = d.sector_id
  where filter_sector is null or s.code = filter_sector
  order by c.embedding operator(extensions.<=>) query_embedding
  limit match_count;
$$;

revoke execute on function public.match_chunks(extensions.vector, int, text) from public, anon, authenticated;
grant  execute on function public.match_chunks(extensions.vector, int, text) to service_role;

-- 5) 대시보드용 섹터별 뷰 (데이터 복사 없음, 임베딩 컬럼은 빼서 읽기 쉽게)
create view public.planning_documents with (security_invoker = true) as
  select d.id, d.title, d.source_path, (select count(*) from public.chunks c where c.document_id = d.id) as chunk_count
  from public.documents d join public.sectors s on s.id = d.sector_id where s.code = 'planning';
create view public.planning_chunks with (security_invoker = true) as
  select c.id, d.title, c.chunk_index, c.heading_path, c.content
  from public.chunks c join public.documents d on d.id = c.document_id join public.sectors s on s.id = d.sector_id
  where s.code = 'planning';

create view public.hardware_documents with (security_invoker = true) as
  select d.id, d.title, d.source_path, (select count(*) from public.chunks c where c.document_id = d.id) as chunk_count
  from public.documents d join public.sectors s on s.id = d.sector_id where s.code = 'hardware';
create view public.hardware_chunks with (security_invoker = true) as
  select c.id, d.title, c.chunk_index, c.heading_path, c.content
  from public.chunks c join public.documents d on d.id = c.document_id join public.sectors s on s.id = d.sector_id
  where s.code = 'hardware';

create view public.software_documents with (security_invoker = true) as
  select d.id, d.title, d.source_path, (select count(*) from public.chunks c where c.document_id = d.id) as chunk_count
  from public.documents d join public.sectors s on s.id = d.sector_id where s.code = 'software';
create view public.software_chunks with (security_invoker = true) as
  select c.id, d.title, c.chunk_index, c.heading_path, c.content
  from public.chunks c join public.documents d on d.id = c.document_id join public.sectors s on s.id = d.sector_id
  where s.code = 'software';

-- 뷰도 서버 키로만 조회
revoke all on public.planning_documents, public.planning_chunks,
              public.hardware_documents, public.hardware_chunks,
              public.software_documents, public.software_chunks,
              public.sectors
  from anon, authenticated;
grant select on public.planning_documents, public.planning_chunks,
                public.hardware_documents, public.hardware_chunks,
                public.software_documents, public.software_chunks
  to service_role;

commit;
