# TeachArm — 나만의 피지컬 AI 제품 만들기

시연을 보고 배우는 티칭 로봇 팔 **TeachArm**을 기획하고, 제품 소개 홈페이지를 GitHub Pages로 배포하는 과제 저장소입니다.

## 폴더 구성

```
physical-ai/
├── index.html                  ← 제품 소개 홈페이지 (GitHub Pages 시작 파일)
├── hardware.html               ← 하드웨어 사양 페이지
├── technology.html             ← 소프트웨어(기술) 페이지
├── physical_ai_부품도감.html    ← 피지컬 AI 부품 선택 도감
├── images/                     ← 홈페이지에서 쓰는 로고·구조도
├── PRD/
│   └── PRD_arm.md              ← TeachArm 제품 요구사항 문서
├── report/                     ← 과제 보고서 (전처리, 모델 선택, 하드웨어, 추론 엔진, 컨테이너 등)
│   └── model_comparison/       ← 모델 비교 실험 결과 이미지·수치
├── chatbot/                    ← 문서 기반 RAG 상담 챗봇 (Supabase + Gemini)
│   ├── ingest.py               ← 문서 청킹 → 임베딩 → Supabase 적재
│   ├── sql/                    ← DB 스키마와 마이그레이션 (파일명 날짜 순서대로 실행)
│   ├── supabase/functions/chat ← 챗봇 응답 Edge Function
│   └── widget/                 ← 홈페이지에 붙는 채팅 위젯 (JS/CSS)
├── code/
│   ├── preprocess_code_arm.py          ← LeRobot 시연 데이터 전처리 모듈
│   ├── mediapipe_arm_hand_landmarks.py ← MediaPipe 팔·손 랜드마크 시각화
│   └── model_comparison/               ← 비전 모델 비교 실험 (데이터 준비 → 실행 → 지연시간 측정)
└── requirements.txt
```

## 개발 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

루트의 `.env`에 챗봇용 키를 넣습니다: `GEMINI_KEY`, `SUPABASE_URL`, `SUPABASE_KEY_SECRET`

## 실행

### 모델 비교 실험

```bash
python code/model_comparison/prepare_data.py   # 공식 샘플 데이터 내려받기 → code/model_comparison/data/
python code/model_comparison/run_models.py     # 예시 이미지·results.json → report/model_comparison/
python code/model_comparison/bench_latency.py  # CPU 지연시간 교차 측정 → report/model_comparison/latency.json
```

### MediaPipe 팔·손 랜드마크

```bash
python code/mediapipe_arm_hand_landmarks.py                  # 공식 샘플 3장
python code/mediapipe_arm_hand_landmarks.py my.jpg --no-show # 내 이미지, 창 없이 저장만
```

### 챗봇

1. Supabase 대시보드 > SQL Editor에서 `chatbot/sql/`의 파일을 날짜 순서대로 실행합니다.
2. 문서를 임베딩해 적재합니다.
   ```bash
   python chatbot/ingest.py --dry-run   # 청킹 결과만 확인 (Gemini 호출 없음)
   python chatbot/ingest.py             # 청킹 → 임베딩 → Supabase 업로드
   ```
3. `chatbot/` 폴더에서 Edge Function을 배포합니다. (`GEMINI_API_KEY` 시크릿 필요)
   ```bash
   supabase functions deploy chat
   ```

## GitHub Pages 배포

`main` 브랜치의 루트가 그대로 배포됩니다. 변경 사항을 커밋하고 push하면 됩니다.

```bash
git add .
git commit -m "수정 내용에 대한 설명"
git push origin main
```
