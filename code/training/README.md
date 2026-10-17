# TeachArm ACT 학습

SO-101 시연 데이터(LeRobot 형식)로 ACT 정책을 학습하는 코드를 모아 둔 폴더입니다.
전처리 기준은 `report/preprocess_arm.md`, 학습 목표와 인프라는 `PRD/PRD_arm.md`를 따릅니다.

## 현재 단계: 공개 데이터로 End-to-End PoC

자체 시연 데이터를 수집하기 전에, 공개 데이터셋으로 **전처리 → 학습 → 오프라인 검증 → ONNX 변환 → 지연 측정**을 끝까지 한 번 연결합니다. 목표는 성능이 아니라 각 단계가 막힘 없이 이어지는지 확인하는 것입니다. 자체 데이터가 생기면 데이터셋 ID만 바꿉니다.

| 용도 | 데이터셋 | 내용 |
| --- | --- | --- |
| PoC 기본 | `lerobot/svla_so100_pickplace` | SO-100, 50 에피소드, 19,631 프레임, 30fps, 카메라 top + wrist (640×480), 470MB, Apache-2.0. 작업: "Pick up the cube and place it in the box." |
| 시뮬 성공률 확인 (선택) | `lerobot/aloha_sim_transfer_cube_human` | ALOHA 14축, 50 에피소드, MIT. gym-aloha에서 성공률을 바로 측정할 수 있음 |

AI Hub의 손 동작 데이터(494, 71601)는 로봇 관절 상태·행동이 없어서 ACT 학습에 쓰지 않습니다.

## 폴더 구조

```
code/training/
├── README.md               ← 지금 보고 있는 문서
├── colab_setup.sh          ← Colab 런타임 공통 셋업. 모든 노트북에서 저장소를 받은 뒤 호출
├── requirements-train.txt  ← 학습 환경 고정 버전 (lerobot[training]==0.6.1, onnx, onnxruntime)
├── prepare_dataset.py      ← (작성 예정) 학습용 데이터셋 만들기
├── train_act.py            ← (작성 예정) LeRobot ACT 학습 실행
├── export_onnx.py          ← (작성 예정) ONNX 변환 + PyTorch 출력 비교
├── configs/                ← 학습 설정 파일 (실험별 YAML)
├── notebooks/              ← Colab 실행용 노트북 (00_setup.ipynb 작성됨)
├── data/                   ← 전처리된 데이터셋 (git 제외, 실행 시 생성)
└── outputs/                ← 체크포인트·학습 로그 (git 제외, 실행 시 생성)
```

## 파일·폴더별 역할

### `prepare_dataset.py`

원본 데이터셋을 학습에 쓸 데이터셋으로 바꿉니다. 처리 함수는 `code/preprocess_code_arm.py`의 것을 그대로 가져다 씁니다.

1. 무결성·동기 점검 (`check_sync`): 프레임 드랍과 영상/parquet 길이 불일치 탐지
2. 에피소드 필터링 (`filter_episodes`): 실패, 길이 이상, 드랍이 많은 에피소드 제외
3. 궤적 정리 (`clean_trajectory`, `idle_trim_range`): Hampel 스파이크 제거, 앞뒤 정지 구간 트리밍
4. 영상 품질 점검 (`image_quality`, `flag_bad_frames`): 블러·밝기 이상 프레임 표시
5. 영상 리사이즈: 320×240으로 재인코딩 (학습 중 영상 디코딩 부담 약 1/4)
6. 정규화 통계 재계산 (`compute_stats`): **필터링과 트리밍이 끝난 학습 세트로만** 계산
7. 데이터 분할: 시연 50개 기준 학습 45개 / 검증 5개 (에피소드 단위)

결과는 `data/`(Colab에서는 Drive)에 저장하고, 필요하면 Hugging Face Hub에 비공개로 버전을 올립니다.

### `train_act.py`

`configs/`의 설정 파일을 읽어 LeRobot ACT 학습(`lerobot-train`)을 실행합니다. 체크포인트와 로그는 `outputs/<실험 이름>/`에 남기고 W&B에 기록합니다.

### `export_onnx.py`

학습된 체크포인트를 `torch.onnx.export`로 변환하고, 같은 입력에서 PyTorch와 ONNX Runtime 출력이 일치하는지 확인합니다(최대 오차 < 1e-4). 지연 측정은 `code/model_comparison/bench_latency.py`의 방식(워밍업 + 반복 측정, p50/p95)을 따릅니다.

### `configs/`

실험마다 YAML 파일 하나를 둡니다. 코드를 고치지 않고 설정만 바꿔서 실험을 비교하기 위한 폴더입니다.

- PoC 설정: `act_so100_pickplace.yaml` (공개 데이터)
- 기본 설정: `act_so101.yaml` (입력 320×240, 액션 청크 크기, 학습 중 증강 Random shift + ColorJitter 등)
- 시연 수 비교 실험(PRD 기능 5): 10 / 25 / 50개 부분 데이터셋별 설정

### `notebooks/`

Colab에서 실행하는 노트북을 둡니다. 노트북은 저장소를 clone하고 `.py` 스크립트를 호출하는 역할만 합니다. 학습 로직은 노트북 셀이 아니라 `.py` 파일에 둡니다. 코드는 IDE에서 고쳐 push하고, Colab에서는 `git pull`만 합니다.

| 노트북 | 런타임 | 하는 일 |
| --- | --- | --- |
| `00_setup.ipynb` | CPU | 최초 1회 환경 점검 (아래 "셋업" 참고) |
| `01_data_prep.ipynb` | CPU | 데이터 받기 → `prepare_dataset.py` → Drive에 저장 |
| `02_train_act.ipynb` | **GPU (T4/L4)** | 스모크 테스트 → 짧은 학습 → 본 학습, 끊기면 이어서 학습 |
| `03_eval_export.ipynb` | CPU | 오프라인 검증 → `export_onnx.py` → 지연 측정 (참고용) |

GPU가 필요한 것은 02번뿐입니다. 01·03번은 CPU 런타임으로 돌려 GPU 크레딧을 아낍니다.

### `data/`, `outputs/`

용량이 커서 git에 올리지 않습니다(`.gitignore`에 등록). Colab에서는 런타임이 초기화되면 사라지므로 Google Drive에 둡니다. 데이터셋과 체크포인트 버전은 Hugging Face Hub와 W&B로 관리합니다.

## Colab 실행 방법

PoC는 모든 단계를 Colab에서 실행합니다.

### 사전 준비 (최초 1회, Colab 밖에서)

| 항목 | 할 일 |
| --- | --- |
| Hugging Face | 계정 생성, **write 권한 토큰** 발급 (체크포인트 업로드용) |
| Weights & Biases | 계정 생성, 프로젝트 `teacharm-act` 생성, API 키 확인 |
| Colab Secrets | 왼쪽 🔑 메뉴에 `HF_TOKEN`, `WANDB_API_KEY` 등록, 노트북 접근 허용 |
| Google Drive | `MyDrive/teacharm/` 폴더 생성 |

토큰은 노트북 셀이나 저장소에 직접 쓰지 않습니다.

### Drive 폴더 구조

```
MyDrive/teacharm/
├── data/       ← 전처리된 데이터셋 (01번 결과)
├── outputs/    ← 실험별 체크포인트·학습 설정 (02번 결과)
├── onnx/       ← ONNX 파일·출력 비교·지연 결과 (03번 결과)
└── env/        ← 실행마다 기록한 환경 정보 (env_<날짜>.json)
```

원본 데이터셋 다운로드 캐시는 Drive가 아니라 런타임 로컬 디스크(`/content`)에 둡니다. Drive는 작은 파일을 많이 읽고 쓰는 작업이 느리기 때문입니다.

### 셋업 (`colab_setup.sh`, `00_setup.ipynb`)

Colab 런타임은 끊길 때마다 초기화되므로, **모든 노트북의 앞 세 셀은 똑같이** 실행합니다.

1. **Drive 연결, 토큰 불러오기** (노트북 셀): `drive.mount('/content/drive')`, Colab Secrets의 `HF_TOKEN`, `WANDB_API_KEY`를 환경 변수로 설정. huggingface_hub와 wandb가 이 변수를 자동으로 읽습니다
2. **저장소 받기** (노트북 셀): `/content/physical-ai`에 `git clone` (이미 있으면 `git pull`)
3. **공통 셋업** (`colab_setup.sh`)
   - Python 3.12 이상인지 확인 (LeRobot 0.6.1 요구사항)
   - `nvidia-smi`로 GPU 확인, `ffmpeg` 확인 (없으면 설치)
   - `requirements-train.txt` 설치. LeRobot 0.6.1은 torch 2.12 미만을 요구하므로, Colab 기본 torch가 범위를 벗어나면 pip가 다시 설치하고 스크립트가 알려줍니다
   - `MyDrive/teacharm/` 하위 폴더 생성
   - Python·PyTorch·CUDA·LeRobot 버전, GPU 이름, 커밋 해시를 `MyDrive/teacharm/env/env_<날짜>.json`에 기록

`00_setup.ipynb`는 여기에 더해 아래 항목을 최초 1회 확인합니다.

4. 메모리의 torch와 설치된 torch 버전이 같은지, GPU 연산이 되는지
5. Hugging Face 로그인과 토큰 권한(write인지), W&B 로그인
6. `lerobot/svla_so100_pickplace` (커밋 `728583b`, `v3.0` 태그) 정보: 에피소드 50, 30fps, 카메라 `observation.images.top`·`observation.images.wrist`, 관절 상태·행동 6차원
7. 에피소드 0번 첫 프레임을 카메라별로 디코딩해 화면에 출력
8. `lerobot-train --help`에 02번에서 쓸 옵션이 있는지

### 학습 실행 (`02_train_act.ipynb`)

`train_act.py`는 YAML 설정을 아래와 같은 `lerobot-train` 명령으로 바꿔 실행합니다. 플래그 이름은 `requirements-train.txt`에 고정한 LeRobot 버전 기준으로 확인합니다.

```bash
lerobot-train \
  --policy.type=act --policy.device=cuda \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --batch_size=8 --steps=<N> --save_freq=<M> \
  --output_dir=/content/drive/MyDrive/teacharm/outputs/act_so100_50ep \
  --wandb.enable=true \
  --policy.repo_id=<hf계정>/teacharm_act_so100_50ep
```

- **부분 데이터셋 (실험 ①)**: `--dataset.episodes`로 에피소드 10 / 25 / 50개를 지정해 같은 명령으로 실행합니다.
- **이어서 학습**: 세션이 끊기면 Drive에 저장된 마지막 체크포인트의 `train_config.json`을 지정하고 `--resume=true`로 다시 실행합니다.
- **업로드**: 최종 체크포인트는 HF Hub에 실험 이름을 버전 태그로 붙여 올립니다.

step 수는 단계적으로 늘립니다. T4에서 ACT의 학습 속도는 아직 측정 전이므로, 본 학습 step 수는 짧은 학습 결과로 정합니다.

| 단계 | step 수 | 확인할 것 |
| --- | --- | --- |
| 스모크 테스트 | 수백 | 설치, 데이터 로딩, Drive 저장, W&B 기록이 동작함 |
| 짧은 학습 | 수천 | loss가 내려감. step당 속도를 재서 4시간(PRD) 안에 가능한 step 수 계산 |
| 본 학습 | 계산된 값 | 03번에 쓸 체크포인트 확보 |

### 검증·변환 (`03_eval_export.ipynb`)

1. 오프라인 검증: 검증 에피소드에서 예측 행동과 실제 행동 비교 (관절별 MSE, 궤적 겹쳐 그리기)
2. ONNX 변환 + PyTorch 출력 비교 (최대 오차 < 1e-4)
3. ONNX Runtime 지연 p50/p95 측정

## 주의

- ACT의 검증 손실은 실제 성공률과 상관관계가 약합니다. 최종 평가는 실물 로봇 테스트로 합니다.
- **Colab CPU에서 잰 지연 시간은 Pi 5 성능을 대표하지 않습니다**(x86 2코어 vs ARM 4코어). PoC에서는 측정 스크립트가 동작하는지만 확인하고, 실험 ④ 판정(p95 ≤ 300ms)은 Pi 5 실기기에서 측정합니다.
- 공개 데이터는 SO-100으로 수집되어, 여기서 학습한 정책을 우리 SO-101 팔에 그대로 쓰지 않습니다. 파이프라인 검증 용도입니다.
- 리더 팔 데이터와 손 추적 데이터는 분포가 달라서 섞지 않고 따로 학습합니다.
- MuJoCo 시뮬레이션 검증과 실물 평가 코드는 이 폴더가 아니라 `code/sim/`, `code/eval/`에 둘 예정입니다.
