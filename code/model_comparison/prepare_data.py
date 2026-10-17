"""
모델 비교용 공식 샘플 데이터 준비 (모두 공개 저장소의 고정 리비전에서 받음 → 누구나 같은 입력으로 재현 가능)

  1) LeRobot 공식 SO-101 데이터셋 `lerobot/svla_so101_pickplace` (Apache-2.0)
     - TeachArm과 같은 구성: SO-101 팔, 상단 카메라 640×480 30fps, 시연 50개, "블록을 집어 상자에 넣기"
     - 에피소드별 첫 프레임(블록이 책상 위) / 마지막 프레임(블록이 상자 안)을 PNG로 추출
     - 첫 프레임에서 분홍 블록의 정답 박스(GT)를 HSV 색 임계값으로 자동 생성 → gt_boxes.json
  2) Tesseract 공식 테스트 이미지 phototest.tif + 정답 텍스트 phototest.gold.txt
  3) EasyOCR 공식 예제 korean.png (한글 OCR 정성 비교용)
  4) MediaPipe 공식 테스트 이미지 male_full_height_hands.jpg

실행: python code/model_comparison/prepare_data.py   → code/model_comparison/data/
"""
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import requests

DATA = Path(__file__).resolve().parent / "data"

LEROBOT_REV = "f641879e22172be7e8161d5e6c1503c2d2feb657"
LEROBOT = f"https://huggingface.co/datasets/lerobot/svla_so101_pickplace/resolve/{LEROBOT_REV}"
FILES = {
    "lerobot/top.mp4": f"{LEROBOT}/videos/observation.images.up/chunk-000/file-000.mp4",
    "lerobot/episodes.parquet": f"{LEROBOT}/meta/episodes/chunk-000/file-000.parquet",
    "ocr/phototest.tif": "https://raw.githubusercontent.com/tesseract-ocr/test/232ff181c66516116ec0e84c4963f70de15050fd/testing/phototest.tif",
    "ocr/phototest.gold.txt": "https://raw.githubusercontent.com/tesseract-ocr/test/232ff181c66516116ec0e84c4963f70de15050fd/testing/phototest.gold.txt",
    "ocr/korean.png": "https://raw.githubusercontent.com/JaidedAI/EasyOCR/363afb184047ce452e436f4224f3098422df872e/examples/korean.png",
    "hand/male_full_height_hands.jpg": "https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/male_full_height_hands.jpg",
}
FPS = 30

# 분홍(마젠타 쪽 빨강) 블록. 상자 테두리의 주황 테이프(H≈10~20)와 겹치지 않게 H 170~180, 0~7 만 사용
HSV_RANGES = [((170, 120, 90), (180, 255, 255)), ((0, 120, 90), (7, 255, 255))]
MIN_BLOB_AREA = 80


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def brick_mask(bgr: np.ndarray) -> np.ndarray | None:
    """가장 큰 분홍 덩어리의 bool 마스크. 면적이 너무 작으면 None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in HSV_RANGES).clip(0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == i if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA else None


def brick_gt(bgr: np.ndarray) -> list[int] | None:
    """가장 큰 분홍 덩어리의 [x1, y1, x2, y2]. 없으면 None."""
    m = brick_mask(bgr)
    if m is None:
        return None
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def extract_frames() -> None:
    eps = pl.read_parquet(DATA / "lerobot/episodes.parquet")
    starts = (eps["videos/observation.images.up/from_timestamp"] * FPS).round().cast(int).to_list()
    lengths = eps["length"].to_list()
    out = DATA / "frames"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(DATA / "lerobot/top.mp4"))
    gt = {}
    for ep, (s, n) in enumerate(zip(starts, lengths)):
        for tag, idx in (("start", s), ("end", s + n - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            assert ok, f"frame {idx} decode 실패"
            name = f"ep{ep:02d}_{tag}.png"
            cv2.imwrite(str(out / name), frame)
            if tag == "start":
                gt[name] = brick_gt(frame)
    (DATA / "gt_boxes.json").write_text(json.dumps(gt, indent=1))
    print(f"frames: {len(starts) * 2}, GT boxes found: {sum(v is not None for v in gt.values())}/{len(gt)}")


if __name__ == "__main__":
    for rel, url in FILES.items():
        download(url, DATA / rel)
    extract_frames()
