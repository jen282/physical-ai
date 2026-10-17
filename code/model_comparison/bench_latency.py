"""
6개 모델 CPU 지연시간 교차 측정 → report/model_comparison/latency.json

run_models.py 안의 단독 측정값은 실행 순서·발열 상태에 따라 최대 2배까지 흔들려서,
모든 모델을 한 프로세스에 올려 두고 라운드마다 번갈아 측정한다 (발열 영향이 모든 모델에 고르게 분산됨).
  - 라운드 3회 × 모델별 (워밍업 3회 + 측정 20회), 라운드별 중앙값을 구한 뒤
  - 보고값 = 라운드 중앙값들의 중앙값, 범위 = 라운드 중앙값의 최소~최대
입력: 모두 640×480 LeRobot 프레임(ep00_start.png). 단 OCR은 phototest.tif, MediaPipe는 공식 손 이미지.

실행: python code/model_comparison/bench_latency.py
"""
import contextlib
import json
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

import run_models as R

ROUNDS, WARMUP, RUNS = 3, 3, 20


def build() -> dict:
    from transformers import AutoModel, AutoProcessor
    from ultralytics import SAM, YOLOE

    import mediapipe_arm_hand_landmarks as mh

    img = R.frame(R.EXAMPLE)
    fns = {}

    yoloe = YOLOE(str(R.WEIGHTS / "yoloe-26s-seg.pt"))
    with contextlib.chdir(R.WEIGHTS):
        yoloe.set_classes(R.PROMPTS)
    fns["yoloe"] = lambda: yoloe.predict(img, conf=R.YOLOE_CONF, device=R.DEVICE, verbose=False)

    _, gd_detect = R.load_gdino()
    fns["grounding_dino"] = lambda: gd_detect(img, R.GDINO_BOX_T)

    sam = SAM(str(R.WEIGHTS / "sam2.1_b.pt"))
    boxes = [d["box"] for d in gd_detect(img, R.GDINO_BOX_T)]  # 박스는 미리 구해 두고 SAM 단계만 측정
    fns["sam2_1_b_box_prompt"] = lambda: sam.predict(img, bboxes=boxes, device=R.DEVICE, verbose=False)

    proc = AutoProcessor.from_pretrained(R.SIGLIP_ID, revision=R.SIGLIP_REV)
    sig = AutoModel.from_pretrained(R.SIGLIP_ID, revision=R.SIGLIP_REV).to(R.DEVICE).eval()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    texts = list(R.SIGLIP_LABELS.values())

    @torch.no_grad()
    def siglip():
        inputs = proc(text=texts, images=pil, padding="max_length", max_length=64, return_tensors="pt")
        return sig(**inputs).logits_per_image

    fns["siglip2"] = siglip

    en = R.DATA / "ocr/phototest.tif"
    fns["tesseract"] = lambda: R.tesseract(en, "eng")

    models, _ = mh.ensure_assets()
    hand_img = cv2.cvtColor(cv2.imread(str(R.DATA / "hand/male_full_height_hands.jpg")), cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_img)
    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(models["hand"])), num_hands=2))
    pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(models["pose"])), num_poses=1))
    fns["mediapipe_hand"] = lambda: hand_lm.detect(mp_img)
    fns["mediapipe_pose"] = lambda: pose_lm.detect(mp_img)
    return fns


def main() -> None:
    fns = build()
    per_round = {k: [] for k in fns}
    for rnd in range(ROUNDS):
        for name, fn in fns.items():
            for _ in range(WARMUP):
                fn()
            ts = []
            for _ in range(RUNS):
                t = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t) * 1e3)
            per_round[name].append(round(float(np.median(ts)), 1))
            print(f"round {rnd + 1} {name:<22} {per_round[name][-1]:>8.1f} ms", flush=True)
    out = {
        "method": f"{ROUNDS} rounds x (warmup {WARMUP} + {RUNS} runs), round-robin, CPU, batch 1",
        "torch_threads": torch.get_num_threads(),
        "latency_ms": {k: {"median": float(np.median(v)), "min_round": min(v), "max_round": max(v), "rounds": v}
                       for k, v in per_round.items()},
    }
    path = R.OUT / "latency.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
