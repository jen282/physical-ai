# TeachArm 추론 엔진 선택 보고서

| 항목      | 내용                                                                                       |
| --------- | ------------------------------------------------------------------------------------------ |
| 작성일    | 2026-10-15                                                                                 |
| 비교 대상 | ONNX Runtime, NCNN, OpenVINO, TFLite (LiteRT)                                              |
| 근거 문서 | `PRD/PRD_arm.md`, `report/hardware_arm.md`, `report/report_arm.md`, `report/model_selection.md` |
| 상태      | 학습된 모델이 아직 없어 **문서·공식 자료 기준으로 선택**합니다. 실측은 모델 학습 후 진행합니다(8장). |
| 표기      | ※ 는 추정치이거나 실측으로 확인해야 하는 내용입니다.                                       |

---

## 0. 결론

> **ONNX Runtime을 선택합니다.**
> TeachArm의 핵심 모델(ACT)은 PyTorch로 학습하는 Transformer 모델이고, 실행 장비는 **노트북(GPU)과 Raspberry Pi 5(ARM CPU)** 두 곳입니다.
> 네 엔진 중 **PyTorch Transformer 모델을 가장 쉽게 옮기고, 같은 모델 파일을 두 장비에서 그대로 돌릴 수 있는 엔진**이 ONNX Runtime입니다.

| 순위 | 엔진              | 점수 (5점 만점) | 한 줄 판정                                                          |
| ---- | ----------------- | --------------- | ------------------------------------------------------------------- |
| **1** | **ONNX Runtime** | **4.75**        | **채택.** 변환 쉬움, Transformer 지원 좋음, 노트북·Pi 5 모두 지원  |
| 2    | NCNN              | 3.50            | 예비안. ARM CPU에서 CNN은 가장 빠르지만 Transformer 변환이 까다로움 |
| 3    | OpenVINO          | 3.45            | 제외. Intel 칩 전용 최적화라 ARM(Pi 5)·NVIDIA GPU(노트북)에서 강점이 없음 |
| 4    | TFLite (LiteRT)   | 3.25            | 제외. 모바일에 강하지만 PyTorch Transformer 변환이 가장 불안정함    |

**이유 (세 줄)**

1. **변환이 쉽다:** PyTorch → ONNX는 `torch.onnx.export` 한 번이면 됩니다. 나머지 셋은 변환 도구를 한 단계 더 거쳐야 하고, Transformer에서 실패할 가능성이 큽니다.
2. **실험 ④를 공정하게 할 수 있다:** 같은 `.onnx` 파일을 노트북에서는 GPU로, Pi 5에서는 CPU로 실행합니다. 모델이 같으니 "장비 차이"만 비교할 수 있습니다.
3. **이미 PRD에 적힌 방향과 같다:** PRD의 Edge AI 항목이 "PyTorch CPU 또는 ONNX Runtime"이고, 기능 #7이 "Pi 5용 ONNX 변환"입니다. 계획을 바꾸지 않아도 됩니다.

---

## 1. 추론 엔진이란?

학습은 PyTorch로 하지만, PyTorch는 **학습용 도구**라서 무겁고 저가 장비에서는 느립니다.
**추론 엔진**은 학습이 끝난 모델을 받아 **"실행만" 빠르게 하도록 다듬은 프로그램**입니다.

```
[학습]  PyTorch로 ACT 학습 (Colab GPU)
           │  모델 파일 변환 (.pt → 엔진 전용 형식)
           ▼
[실행]  추론 엔진이 실행 (노트북 / Raspberry Pi 5)
           카메라 영상 + 관절 상태 → 관절 목표값 묶음(Action Chunk)
```

엔진마다 **잘 돌아가는 칩**과 **잘 받아들이는 모델 종류**가 다릅니다. 그래서 "우리 장비"와 "우리 모델"에 맞는 엔진을 골라야 합니다.

---

## 2. TeachArm 조건

### 2.1 하드웨어 — 어디서 돌리나

| 장비               | 연산 장치                             | 역할                         | 추론 목표 (PRD 실험 ④) |
| ------------------ | ------------------------------------- | ---------------------------- | ---------------------- |
| Raspberry Pi 5 (8GB) | **ARM CPU 4코어 (Cortex-A76)**, NPU 없음 | 실제 현장용 저가 엣지 장치   | **p95 ≤ 300ms**        |
| 노트북             | x86 CPU + **GPU (NVIDIA 가정 ※)**     | 기준(비교군), 학습·데이터 수집 | p95 ≤ 50ms             |

- Pi 5에는 AI 전용 칩(NPU)이 없습니다. 따라서 **ARM CPU에서 얼마나 빠른가**가 가장 중요합니다.
- 실험 ④는 "같은 정책을 노트북과 Pi 5에서 돌려 비교"합니다. 따라서 **두 장비를 모두 지원하는 엔진**이 유리합니다.
- 클라우드 추론은 PRD에서 제외했습니다(`report/report_arm.md`). 엔진은 **오프라인**으로 동작해야 합니다. 네 후보 모두 오프라인 실행이 가능합니다.

### 2.2 AI 모델 — 무엇을 돌리나

| 모델                    | 학습 도구 | 구조                                         | 실행 위치          | 엔진 선택 필요? |
| ----------------------- | --------- | -------------------------------------------- | ------------------ | --------------- |
| **ACT** (핵심 정책)     | PyTorch (LeRobot) | **ResNet18(CNN) + Transformer + CVAE**, 카메라 2대 + 관절 6개 입력 | 노트북, **Pi 5** | **필요 (이 보고서의 대상)** |
| Diffusion Policy (선택) | PyTorch (LeRobot) | CNN + 노이즈 제거 신경망을 여러 번 반복 실행 | 노트북, Pi 5       | ACT와 같은 엔진 사용 |
| MediaPipe (손 추적)     | -         | Google 배포 모델                              | 노트북             | 불필요 — 내부에 LiteRT가 이미 들어 있음 |
| YOLOE (큐브 탐지 보조)  | PyTorch   | CNN                                          | 노트북 (사후 처리) | 불필요 — `.pt` 그대로 사용 (`report/object_detection.md`) |

정리하면, 엔진이 필요한 모델은 **ACT 하나**이고, ACT는 다음 두 특징이 있습니다.

- **PyTorch로 학습한다** → PyTorch 모델을 잘 변환하는 엔진이 필요합니다.
- **CNN과 Transformer가 섞여 있다** → CNN만 빠른 엔진으로는 부족하고, Transformer(어텐션) 연산도 잘 지원해야 합니다.

---

## 3. 후보 4개 한눈에 보기

| 항목              | ONNX Runtime                 | NCNN                              | OpenVINO                        | TFLite (LiteRT)                     |
| ----------------- | ---------------------------- | --------------------------------- | ------------------------------- | ----------------------------------- |
| 개발              | Microsoft                    | Tencent                           | Intel                           | Google                              |
| 라이선스          | MIT (무료)                   | BSD-3 (무료)                      | Apache-2.0 (무료)               | Apache-2.0 (무료)                   |
| 원래 목표 장비    | **모든 장비** (서버~엣지)    | 모바일·ARM CPU                    | **Intel** CPU·GPU·NPU           | 모바일(안드로이드)·마이크로컨트롤러 |
| PyTorch → 변환    | **`torch.onnx.export` 1단계** | PyTorch → PNNX → NCNN             | PyTorch → OpenVINO 변환기 (ONNX 경유 가능) | PyTorch → AI Edge Torch → `.tflite` |
| Transformer 지원  | **좋음** (어텐션 최적화 내장) | 보통 (지원은 하나 변환 실패가 잦음 ※) | 좋음 (Intel 칩 기준)            | 보통 (변환 도구가 비교적 신생 ※)    |
| Pi 5 (ARM CPU)    | 좋음                         | **매우 좋음** (ARM 전용 최적화)   | 지원은 하나 최적화가 약함 ※     | 좋음 (XNNPACK)                      |
| 노트북 NVIDIA GPU | **지원** (CUDA, TensorRT)    | Vulkan으로 일부 지원              | 미지원                          | 사실상 미지원 (GPU 가속은 모바일 위주) |
| Python 사용       | **`pip install onnxruntime`** | 가능하나 C++ 중심                 | 가능                            | 가능                                |
| 대표 사용처       | 범용 배포 표준               | 모바일 앱, YOLO의 Pi 배포         | Intel PC·산업용 PC              | 안드로이드 앱, MediaPipe            |

---

## 4. 평가 기준과 점수

### 4.1 평가 기준 (가중치)

| 기준                     | 가중치 | 왜 중요한가                                                                 |
| ------------------------ | ------ | --------------------------------------------------------------------------- |
| ① PyTorch 모델 변환 쉬움 | 25%    | ACT는 LeRobot(PyTorch)으로 학습합니다. 변환이 막히면 Pi 5 실험 자체를 못 합니다. |
| ② Pi 5 ARM CPU 속도      | 25%    | 실험 ④ 목표 p95 300ms를 넘으면 안 됩니다.                                   |
| ③ Transformer 연산 지원  | 20%    | ACT의 절반은 Transformer입니다. CNN만 빠르면 의미가 없습니다.               |
| ④ 노트북·Pi 5 동시 지원  | 15%    | 같은 모델 파일로 두 장비를 비교해야 실험 ④가 공정합니다.                    |
| ⑤ 사용 난이도·자료 양    | 15%    | 소규모 팀(학부생 수준)이 짧은 기간에 적용해야 합니다.                       |

### 4.2 점수표 (5점 만점)

| 기준 (가중치)              | ONNX Runtime | NCNN     | OpenVINO | TFLite (LiteRT) |
| -------------------------- | ------------ | -------- | -------- | --------------- |
| ① 변환 쉬움 (25%)          | 5            | 3        | 4        | 3               |
| ② Pi 5 속도 (25%)          | 4            | 5        | 3        | 4               |
| ③ Transformer 지원 (20%)   | 5            | 3        | 4        | 3               |
| ④ 노트북·Pi 5 동시 (15%)   | 5            | 3        | 2        | 2               |
| ⑤ 사용 난이도 (15%)        | 5            | 3        | 4        | 4               |
| **가중 합계**              | **4.75**     | **3.50** | **3.45** | **3.25**        |

- 점수는 공식 문서와 일반적으로 알려진 특성 기준입니다. 실제 ACT 모델로 측정한 값은 아닙니다(※).
- ②에서 ONNX Runtime이 NCNN·LiteRT보다 1점 낮은 이유: 순수 CNN을 ARM CPU에서 돌리면 NCNN·LiteRT가 더 빠른 경우가 많습니다. 하지만 이 차이는 **변환이 성공해야** 의미가 있습니다.

---

## 5. 엔진별 판정

**ONNX Runtime — 채택**
- PyTorch에서 바로 내보낼 수 있고, ONNX는 업계 표준 형식이라 문제가 생겨도 자료를 찾기 쉽습니다.
- 같은 `.onnx` 파일을 노트북에서는 GPU(CUDA)로, Pi 5에서는 CPU로 실행합니다. 코드도 거의 같습니다.
- INT8 양자화(모델을 가볍게 만드는 기능)를 Python 몇 줄로 적용할 수 있습니다.
- 약점: ARM CPU에서 CNN만 비교하면 NCNN보다 다소 느릴 수 있습니다(※).

**NCNN — 예비안**
- ARM CPU 최적화가 가장 뛰어나고, Ultralytics도 Pi에서 YOLO를 돌릴 때 NCNN을 권장합니다.
- 하지만 PNNX라는 변환 단계를 한 번 더 거치고, ACT처럼 CNN·Transformer·여러 입력이 섞인 모델은 변환이 막힐 위험이 큽니다.
- 노트북 NVIDIA GPU(CUDA)를 쓰지 못해 실험 ④의 비교 조건이 달라집니다.
- **쓰는 경우:** ONNX Runtime으로 300ms를 못 맞추고, 병목이 ResNet18(CNN) 부분일 때 그 부분만 NCNN으로 옮기는 방안을 검토합니다(7장).

**OpenVINO — 제외**
- Intel CPU·GPU에서는 아주 빠르지만, TeachArm 장비는 **ARM(Pi 5)과 NVIDIA GPU(노트북 가정)**라서 이 강점을 쓸 수 없습니다.
- ARM CPU도 지원은 하지만 Intel 칩만큼 최적화되어 있지 않습니다(※).
- **다시 검토하는 경우:** 실제 서비스 단계에서 Intel 기반 산업용 PC로 장비를 바꿀 때.

**TFLite (LiteRT) — 제외**
- ARM CPU에서는 XNNPACK 덕분에 빠르고, MediaPipe도 내부에서 이 엔진을 씁니다.
- 하지만 TensorFlow 계열 형식이라, PyTorch 모델은 AI Edge Torch 같은 별도 도구로 변환해야 합니다. Transformer·CVAE가 섞인 ACT는 변환이 막힐 위험이 가장 큽니다(※).
- 노트북 GPU 가속이 사실상 없어 실험 ④ 비교에 불리합니다.
- **다시 검토하는 경우:** 안드로이드 앱이나 Google Coral 같은 TFLite 전용 가속기를 쓸 때.

---

## 6. 적용 계획 (ONNX Runtime)

### 6.1 진행 순서

```
① LeRobot으로 ACT 학습 (PyTorch, Colab)
② torch.onnx.export로 .onnx 변환
③ 정확도 확인: PyTorch 출력과 ONNX 출력 차이가 거의 없는지 비교
④ 노트북(GPU)에서 지연 측정 → 목표 p95 ≤ 50ms
⑤ Pi 5(CPU)에서 지연 측정 → 목표 p95 ≤ 300ms
⑥ 목표 미달이면 6.2의 최적화를 하나씩 적용
```

### 6.2 Pi 5 속도가 부족할 때 최적화 순서 (쉬운 것부터)

| 순서 | 방법                       | 내용                                                          | 주의할 점                                   |
| ---- | -------------------------- | ------------------------------------------------------------- | ------------------------------------------- |
| 1    | 그래프 최적화·스레드 설정  | ONNX Runtime 최적화 옵션 켜기, CPU 스레드 4개 지정            | 설정만 바꾸면 됨. 정확도 영향 없음          |
| 2    | 입력 해상도 축소           | 640×480 → 320×240 (PRD에 이미 계획된 방법)                    | **학습 때도 같은 해상도**로 다시 학습해야 함 |
| 3    | INT8 양자화                | 32비트 숫자를 8비트로 줄여 모델을 약 1/4로 줄이고 속도 향상   | 성공률이 떨어지는지 반드시 확인             |
| 4    | 추론 간격 조절             | ACT는 한 번에 수십 스텝을 출력하므로 매 스텝마다 추론하지 않아도 됨 | 간격을 너무 늘리면 반응이 늦어짐            |

- 최적화 후에는 **지연만 보지 말고 성공률도 다시 측정**합니다. PRD 기준은 "Pi 5 성공률이 노트북 대비 10%p 이내"입니다.
- 참고 추정: ACT는 약 5천만 개 파라미터 수준이고, 320×240 해상도·카메라 2대 기준이면 Pi 5 CPU에서 **수백 ms 이내**로 예상합니다(※ 실측 전 추정).

---

## 7. 위험과 대안

| 위험                                    | 가능성 | 대안                                                                                 |
| --------------------------------------- | ------ | ------------------------------------------------------------------------------------ |
| ACT의 일부 연산이 ONNX로 변환되지 않음  | 낮음   | ONNX 버전(opset) 변경, 학습 때만 쓰는 CVAE 인코더는 빼고 추론에 필요한 부분만 변환  |
| 최적화 후에도 Pi 5 p95 300ms 초과       | 중간   | ① 병목이 CNN이면 ResNet18만 NCNN으로 실행 ② 그래도 안 되면 Jetson 등 GPU 엣지 검토 (`report/hardware_arm.md` 4장) |
| INT8 양자화로 성공률 하락               | 중간   | INT8 대신 FP32 유지, 또는 해상도 축소만 적용                                         |
| Diffusion Policy가 Pi 5에서 너무 느림   | 높음 ※ | 반복 횟수가 많아 느릴 수 있음. 반복 횟수를 줄이거나 노트북 전용 비교군으로만 사용   |

---

## 8. 다음 할 일

1. MuJoCo 시뮬레이션용 ACT로 ONNX 변환이 되는지 먼저 확인 (실제 데이터 수집 전에 미리 검증)
2. PyTorch와 ONNX 출력값 비교 스크립트 작성 (오차 확인)
3. 노트북·Pi 5 지연 측정 스크립트 작성 (p50/p95, 기존 `code/model_comparison/bench_latency.py` 방식 재사용)
4. Pi 5 지연이 목표에 미달하면 6.2 최적화 적용 후 성공률 재측정
5. 결과를 바탕으로 이 보고서의 점수(※ 표시)를 실측값으로 갱신

---

### 참고 자료

- [ONNX Runtime 공식 문서](https://onnxruntime.ai/docs/) — 실행 장치(CPU, CUDA, TensorRT), 양자화
- [PyTorch ONNX 내보내기](https://pytorch.org/docs/stable/onnx.html)
- [NCNN GitHub](https://github.com/Tencent/ncnn), [PNNX](https://github.com/pnnx/pnnx)
- [OpenVINO 지원 장치](https://docs.openvino.ai/) — ARM CPU 지원 범위
- [LiteRT 공식 문서](https://ai.google.dev/edge/litert), [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch) — PyTorch 변환
- [Ultralytics Raspberry Pi 가이드](https://docs.ultralytics.com/guides/raspberry-pi/) — Pi에서 NCNN 권장 사례
- [LeRobot ACT](https://huggingface.co/docs/lerobot/) — ACT 구조와 학습 설정
