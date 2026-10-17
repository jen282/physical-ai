"""
TeachArm 비전 모델 비교 실험: YOLOE / DINO-X / SAM 3 / SigLIP / OCR / MediaPipe

모든 모델을 같은 PC·같은 조건(CPU, 배치 1)으로 실행해서
  - 사용 예시 이미지 (입력 | 출력)      → report/model_comparison/*.jpg
  - 지연시간·모델 크기·과제 지표(JSON)  → report/model_comparison/results.json
을 만든다. 보고서(report/model_selection.md)의 모든 수치는 이 JSON에서 나온다.

실행 전: python code/model_comparison/prepare_data.py
실행:    python code/model_comparison/run_models.py

계정이 필요한 두 모델은 같은 계열의 공개 모델로 대신 실행한다 (보고서에 명시):
  - DINO-X  : 클라우드 API 전용(가중치 비공개, API 토큰 필요) → 같은 IDEA Research의 공개 전작 Grounding DINO
  - SAM 3   : HF 체크포인트가 승인제(gated) → SAM 3 이전 표준 파이프라인 Grounded-SAM 2 (Grounding DINO 박스 → SAM 2.1)

TeachArm 대리 과제 (LeRobot 공식 SO-101 데이터셋, 시연 50개):
  - 탐지  : 각 시연 첫 프레임에서 "pink lego brick"을 찾는가 (HSV 정답 박스와 IoU ≥ 0.5)
  - 판정  : 첫 프레임(책상 위) / 마지막 프레임(상자 안) 100장을 텍스트만으로 구분하는가
"""
import contextlib
import json
import platform
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
OUT = ROOT.parents[1] / "report" / "model_comparison"
sys.path.insert(0, str(ROOT.parent))  # code/ → mediapipe_arm_hand_landmarks 재사용
sys.path.insert(0, str(ROOT))  # prepare_data 의 HSV 정답 마스크 재사용

DEVICE = "cpu"  # Raspberry Pi 5 같은 엣지 장치와 조건을 맞추기 위해 GPU(MPS) 미사용
N_EPISODES = 50
EXAMPLE = "ep00_start.png"
PROMPTS = ["pink lego brick", "transparent box", "robot arm"]
BRICK = PROMPTS[0]
IOU_THRESH = 0.5
TESSDATA = Path("/opt/homebrew/share/tessdata")  # brew install tesseract tesseract-lang

GDINO_ID, GDINO_REV = "IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SIGLIP_ID, SIGLIP_REV = "google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
GDINO_BOX_T, GDINO_TEXT_T = 0.35, 0.25  # IDEA 공식 데모 기본값
YOLOE_CONF = 0.25                        # Ultralytics 기본값
# SigLIP 조건 A: 전체 화면 / 조건 B: 고정 카메라라서 상자 위치가 같음 → 상자 영역만 잘라서 판정
SIGLIP_LABELS = {
    "start": "a photo of a pink lego brick lying on the white table",
    "end": "a photo of a pink lego brick inside the transparent box",
}
SIGLIP_ROI = (60, 130, 260, 270)  # x1, y1, x2, y2 (50개 시연 모두 상자가 이 안에 들어옴)
SIGLIP_ROI_LABELS = {
    "start": "a photo of an empty transparent box",
    "end": "a photo of a pink lego brick inside a transparent box",
}
OCR_SCENE_GOLD = "서울 Seoul 56Km 평양 Pyeongyang 205Km"  # korean.png 표지판 글자를 사람이 옮겨 적은 정답

COLORS = {"pink lego brick": (180, 80, 255), "transparent box": (255, 200, 0), "robot arm": (80, 220, 80)}


# ---------- 공통 ----------
def frame(name: str) -> np.ndarray:
    return cv2.imread(str(DATA / "frames" / name))


def bench(fn, n: int = 10, warmup: int = 2) -> float:
    """중앙값 지연시간(ms)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return round(float(np.median(ts)), 1)


def n_params(module) -> float:
    return round(sum(p.numel() for p in module.parameters()) / 1e6, 1)


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def draw_boxes(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        c = COLORS.get(d["label"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        txt = f"{d['label']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def overlay_masks(img: np.ndarray, masks: list[np.ndarray], labels: list[str]) -> np.ndarray:
    out = img.copy()
    for m, lab in zip(masks, labels):
        c = np.array(COLORS.get(lab, (255, 255, 255)), np.uint8)
        out[m] = (0.45 * out[m] + 0.55 * c).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, tuple(int(v) for v in c), 2)
    return out


def save_pair(name: str, left: np.ndarray, right: np.ndarray, l_title: str, r_title: str) -> str:
    """[입력 | 출력] 나란히 + 상단 제목줄. 높이가 다르면 오른쪽을 왼쪽 높이에 맞춤."""
    h = left.shape[0]
    if right.shape[0] != h:
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    bar = 36
    canvas = np.full((h + bar, left.shape[1] + right.shape[1] + 10, 3), 255, np.uint8)
    canvas[bar:, : left.shape[1]] = left
    canvas[bar:, left.shape[1] + 10:] = right
    for x, t in ((8, l_title), (left.shape[1] + 18, r_title)):
        cv2.putText(canvas, t, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    path = OUT / name
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path.name


def detection_metrics(per_frame: dict[str, list[dict]], gt: dict, thresh: float) -> dict:
    """brick 탐지: 기본 임계값 재현율, 임계값 무관 top-1 정확도, 오탐 수 + 프레임별 근거."""
    hit, top1, fp, scores, rows = 0, 0, 0, [], []
    for name, dets in per_frame.items():
        bricks = sorted((d for d in dets if d["label"] == BRICK), key=lambda d: -d["score"])
        best = bricks[0] if bricks else None
        best_iou = iou(best["box"], gt[name]) if best else 0.0
        top1 += best_iou >= IOU_THRESH
        kept = [d for d in bricks if d["score"] >= thresh]
        matched = [d for d in kept if iou(d["box"], gt[name]) >= IOU_THRESH]
        hit += bool(matched)
        fp += len(kept) - min(1, len(matched))
        if matched:
            scores.append(max(d["score"] for d in matched))
        rows.append({"frame": name, "gt": gt[name], "top1_box": [round(v, 1) for v in best["box"]] if best else None,
                     "top1_score": round(best["score"], 3) if best else None, "top1_iou": round(best_iou, 3),
                     "kept": len(kept), "hit": bool(matched)})
    n = len(per_frame)
    return {
        "frames": n,
        "recall_at_default_thresh": round(hit / n, 3),
        "top1_accuracy": round(top1 / n, 3),
        "false_positives_at_default_thresh": fp,
        "mean_score_of_hits": round(float(np.mean(scores)), 3) if scores else None,
        "per_frame": rows,
    }


# ---------- 1) YOLOE ----------
def run_yoloe(gt: dict) -> dict:
    from ultralytics import YOLOE

    weights = WEIGHTS / "yoloe-26s-seg.pt"
    model = YOLOE(str(weights))
    with contextlib.chdir(WEIGHTS):  # 텍스트 인코더(MobileCLIP2)가 현재 폴더로 내려받아지므로 weights/ 에 모음
        model.set_classes(PROMPTS)

    def detect(img):
        r = model.predict(img, conf=0.01, device=DEVICE, retina_masks=True, verbose=False)[0]  # 낮게 받아서 지표 계산 때 임계값 적용
        return r, [{"label": r.names[int(c)], "score": float(s), "box": b.tolist()}
                   for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]

    img = frame(EXAMPLE)
    r, dets = detect(img)
    keep = [i for i, d in enumerate(dets) if d["score"] >= YOLOE_CONF]
    vis = img.copy()
    if r.masks is not None and keep:
        masks = [r.masks.data[i].numpy() > 0.5 for i in keep]
        vis = overlay_masks(vis, masks, [dets[i]["label"] for i in keep])
    vis = draw_boxes(vis, [dets[i] for i in keep])
    fig = save_pair("yoloe.jpg", img, vis, "Input (LeRobot SO-101 top camera)", f"YOLOE-26s text prompt, conf>={YOLOE_CONF}")

    per_frame = {f"ep{i:02d}_start.png": detect(frame(f"ep{i:02d}_start.png"))[1] for i in range(N_EPISODES)}
    return {
        "model": "YOLOE-26s-seg", "weights": weights.name, "run_as": "original",
        "size_mb": round(weights.stat().st_size / 2**20, 1), "params_m": n_params(model.model),
        "latency_ms": bench(lambda: model.predict(img, conf=YOLOE_CONF, device=DEVICE, verbose=False)),
        "threshold": YOLOE_CONF, "example": fig,
        "example_detections": [dets[i] for i in keep],
        "brick_detection": detection_metrics(per_frame, gt, YOLOE_CONF),
    }


# ---------- 2) DINO-X (대체: Grounding DINO) ----------
def load_gdino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(GDINO_ID, revision=GDINO_REV)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID, revision=GDINO_REV).to(DEVICE).eval()

    @torch.no_grad()
    def detect(img: np.ndarray, box_t: float = 0.05) -> list[dict]:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = proc(images=pil, text=[PROMPTS], return_tensors="pt").to(DEVICE)
        res = proc.post_process_grounded_object_detection(
            model(**inputs), inputs.input_ids, threshold=box_t, text_threshold=GDINO_TEXT_T,
            target_sizes=[img.shape[:2]])[0]
        return [{"label": lab, "score": float(s), "box": b.tolist()}
                for b, s, lab in zip(res["boxes"], res["scores"], res["text_labels"]) if lab in PROMPTS]

    return model, detect


def run_gdino(gt: dict, model, detect) -> dict:
    img = frame(EXAMPLE)
    dets = [d for d in detect(img) if d["score"] >= GDINO_BOX_T]
    fig = save_pair("dinox_substitute_grounding_dino.jpg", img, draw_boxes(img, dets),
                    "Input (LeRobot SO-101 top camera)", f"Grounding DINO-T (DINO-X substitute), box>={GDINO_BOX_T}")
    per_frame = {f"ep{i:02d}_start.png": detect(frame(f"ep{i:02d}_start.png")) for i in range(N_EPISODES)}
    return {
        "model": "Grounding DINO Tiny (DINO-X 대체)", "weights": f"{GDINO_ID}@{GDINO_REV[:7]}", "run_as": "substitute",
        "params_m": n_params(model), "size_mb": round(n_params(model) * 4 / 1.048576, 1),  # fp32 safetensors
        "latency_ms": bench(lambda: detect(img, GDINO_BOX_T)),
        "threshold": GDINO_BOX_T, "example": fig, "example_detections": dets,
        "brick_detection": detection_metrics(per_frame, gt, GDINO_BOX_T),
    }


# ---------- 3) SAM 3 (대체: Grounding DINO 박스 → SAM 2.1) ----------
def run_grounded_sam2(detect) -> dict:
    from ultralytics import SAM

    from prepare_data import brick_mask

    weights = WEIGHTS / "sam2.1_b.pt"
    sam = SAM(str(weights))
    img = frame(EXAMPLE)
    dets = [d for d in detect(img) if d["score"] >= GDINO_BOX_T]
    boxes = [d["box"] for d in dets]
    run = lambda: sam.predict(img, bboxes=boxes, device=DEVICE, verbose=False)[0]
    r = run()
    masks = [m.numpy().astype(bool) for m in r.masks.data]
    fig = save_pair("sam3_substitute_grounded_sam2.jpg", img, overlay_masks(img, masks, [d["label"] for d in dets]),
                    "Input (LeRobot SO-101 top camera)", "Grounding DINO boxes -> SAM 2.1-B masks (SAM 3 substitute)")

    # 블록 마스크 정확도: Grounding DINO 1순위 brick 박스로 SAM 프롬프트 → HSV 정답 마스크와 IoU
    rows = []
    for i in range(N_EPISODES):
        name = f"ep{i:02d}_start.png"
        f = frame(name)
        bricks = sorted((d for d in detect(f) if d["label"] == BRICK), key=lambda d: -d["score"])
        gt_m = brick_mask(f)
        if not bricks:
            rows.append({"frame": name, "mask_iou": 0.0})
            continue
        m = sam.predict(f, bboxes=[bricks[0]["box"]], device=DEVICE, verbose=False)[0].masks.data[0].numpy().astype(bool)
        rows.append({"frame": name, "mask_iou": round(float((m & gt_m).sum() / (m | gt_m).sum()), 3)})
    ious = [r["mask_iou"] for r in rows]
    return {
        "model": "Grounded-SAM 2 = Grounding DINO-T + SAM 2.1-B (SAM 3 대체)", "weights": weights.name, "run_as": "substitute",
        "size_mb": round(weights.stat().st_size / 2**20, 1), "params_m": n_params(sam.model),
        "latency_ms_sam_only": bench(run, n=5, warmup=1),
        "example": fig, "example_masks": [{"label": d["label"], "pixels": int(m.sum())} for d, m in zip(dets, masks)],
        "brick_mask": {"frames": len(rows), "mean_iou": round(float(np.mean(ious)), 3),
                       "ratio_iou_ge_0.5": round(float(np.mean([v >= 0.5 for v in ious])), 3), "per_frame": rows},
    }


# ---------- 4) SigLIP 2 ----------
def run_siglip() -> dict:
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(SIGLIP_ID, revision=SIGLIP_REV)
    model = AutoModel.from_pretrained(SIGLIP_ID, revision=SIGLIP_REV).to(DEVICE).eval()
    tags = ["start", "end"]
    x1, y1, x2, y2 = SIGLIP_ROI
    conditions = {  # 조건 이름: (이미지 전처리, 라벨 문장)
        "full_frame": (lambda img: img, SIGLIP_LABELS),
        "box_roi": (lambda img: img[y1:y2, x1:x2], SIGLIP_ROI_LABELS),
    }

    @torch.no_grad()
    def classify(img: np.ndarray, labels: dict) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = proc(text=[labels[t] for t in tags], images=pil, padding="max_length", max_length=64,
                      return_tensors="pt").to(DEVICE)
        return torch.sigmoid(model(**inputs).logits_per_image)[0].numpy()  # SigLIP: 라벨별 독립 sigmoid 확률

    def annotate(img, p, names):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 58), (0, 0, 0), -1)
        for k, prob in enumerate(p):
            val = f"{prob:.4f}" if prob >= 1e-3 else f"{prob:.1e}"
            txt = f"{names[k]}: {val}" + ("  <- pred" if k == np.argmax(p) else "")
            cv2.putText(out, txt, (8, 22 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    result = {
        "model": "SigLIP 2 Base/16-224", "weights": f"{SIGLIP_ID}@{SIGLIP_REV[:7]}", "run_as": "original",
        "params_m": n_params(model), "size_mb": round(n_params(model) * 4 / 1.048576, 1),
        "latency_ms": bench(lambda: classify(frame(EXAMPLE), SIGLIP_LABELS)), "roi": SIGLIP_ROI, "examples": [],
    }
    for cond, (prep, labels) in conditions.items():
        rows = []
        for i in range(N_EPISODES):
            for tag in tags:
                p = classify(prep(frame(f"ep{i:02d}_{tag}.png")), labels)
                rows.append({"frame": f"ep{i:02d}_{tag}.png", "truth": tag, "pred": tags[int(np.argmax(p))],
                             "probs": p.round(4).tolist()})
        acc = lambda rs: round(sum(r["pred"] == r["truth"] for r in rs) / len(rs), 3)
        result[cond] = {
            "labels": labels, "frames": len(rows), "accuracy": acc(rows),
            "accuracy_start": acc([r for r in rows if r["truth"] == "start"]),
            "accuracy_end": acc([r for r in rows if r["truth"] == "end"]),
            "per_frame": rows,
        }
        names = ["on table", "in box"] if cond == "full_frame" else ["empty box", "brick in box"]
        s_img, e_img = prep(frame("ep00_start.png")), prep(frame("ep00_end.png"))
        if cond == "box_roi":  # 잘라낸 영역이 작아서 보기 좋게 3배 확대 (판정에는 원본 크롭 사용)
            s_vis, e_vis = (cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) for im in (s_img, e_img))
        else:
            s_vis, e_vis = s_img, e_img
        result["examples"].append(save_pair(
            f"siglip_{cond}.jpg",
            annotate(s_vis, classify(s_img, labels), names), annotate(e_vis, classify(e_img, labels), names),
            f"SigLIP 2 [{cond}] ep0 first frame (truth: {names[0]})", f"SigLIP 2 [{cond}] ep0 last frame (truth: {names[1]})"))
    return result


# ---------- 5) OCR (Tesseract) ----------
def tesseract(path: Path, lang: str, fmt: str = "") -> str:
    cmd = ["tesseract", str(path), "stdout", "-l", lang] + ([fmt] if fmt else [])
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def cer(pred: str, gold: str) -> float:
    """문자 오류율 근사: 1 - 유사도(공백 정규화)."""
    norm = lambda s: " ".join(s.split())
    return round(1 - SequenceMatcher(None, norm(pred), norm(gold)).ratio(), 4)


def ocr_boxes(path: Path, lang: str) -> np.ndarray:
    img = cv2.imread(str(path))
    for line in tesseract(path, lang, "tsv").splitlines()[1:]:
        f = line.split("\t")
        if len(f) == 12 and f[11].strip() and float(f[10]) > 0:
            x, y, w, h = map(int, f[6:10])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 255), 2)
    return img


def run_ocr() -> dict:
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    en, ko = DATA / "ocr/phototest.tif", DATA / "ocr/korean.png"
    gold = (DATA / "ocr/phototest.gold.txt").read_text()
    en_text, ko_text = tesseract(en, "eng"), tesseract(ko, "kor+eng")
    ko_inv = DATA / "ocr/korean_inverted.png"  # 흰 글씨→검은 글씨 전처리 비교
    cv2.imwrite(str(ko_inv), 255 - cv2.cvtColor(cv2.imread(str(ko)), cv2.COLOR_BGR2GRAY))
    ko_inv_text = tesseract(ko_inv, "kor+eng")
    ko_title = "Tesseract word boxes (kor+eng)" if ko_text.strip() else "Tesseract (kor+eng): no text recognized"
    figs = [
        save_pair("ocr_english.jpg", cv2.imread(str(en)), ocr_boxes(en, "eng"), "Input (Tesseract phototest.tif)", "Tesseract word boxes (eng)"),
        save_pair("ocr_korean.jpg", cv2.imread(str(ko)), ocr_boxes(ko, "kor+eng"), "Input (EasyOCR korean.png)", ko_title),
    ]
    return {
        "model": "Tesseract OCR (LSTM)", "version": version, "run_as": "original",
        "size_mb": round(sum((TESSDATA / f"{l}.traineddata").stat().st_size for l in ("eng", "kor")) / 2**20, 1),
        "latency_ms": bench(lambda: tesseract(en, "eng"), n=5, warmup=1),
        "document_english_cer": cer(en_text, gold), "scene_korean_cer": cer(ko_text, OCR_SCENE_GOLD),
        "scene_korean_cer_inverted": cer(ko_inv_text, OCR_SCENE_GOLD),
        "english_text": en_text.strip(), "korean_text": ko_text.strip(), "korean_text_inverted": ko_inv_text.strip(),
        "korean_gold": OCR_SCENE_GOLD,
        "examples": figs,
    }


# ---------- 6) MediaPipe ----------
MP_VISIBLE_HANDS = 2  # 공식 샘플 3장 모두 사람이 눈으로 센 '보이는 손' 개수 = 2


def run_mediapipe() -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    import mediapipe_arm_hand_landmarks as mh

    models, samples = mh.ensure_assets()  # 공식 샘플 3장: male_full_height_hands / woman_hands / pose
    pose_o = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models["pose"])), num_poses=1)
    hand_o = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models["hand"])), num_hands=2)
    rows = []
    with vision.PoseLandmarker.create_from_options(pose_o) as pose_lm, vision.HandLandmarker.create_from_options(hand_o) as hand_lm:
        for path in samples:
            bgr = cv2.imread(str(path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            poses, hands = mh.detect_arms(pose_lm, image), mh.detect_hands(hand_lm, image)
            extra = mh.recover_missing_hands(hand_lm, rgb, poses, hands)
            rows.append({"image": path.name, "visible_hands": MP_VISIBLE_HANDS, "hands_full_image": len(hands),
                         "hands_with_wrist_crop": len(hands) + len(extra), "persons": len(poses)})
            if path.name == "male_full_height_hands.jpg":
                lat_hand = bench(lambda: mh.detect_hands(hand_lm, image))
                lat_pose = bench(lambda: mh.detect_arms(pose_lm, image))
                vis = bgr.copy()
                mh.draw_arms(vis, poses)
                mh.draw_hands(vis, hands)
                fig = save_pair("mediapipe.jpg", bgr, vis, "Input (MediaPipe test image)", "Pose arm(6) + Hand(21x2) landmarks")
    total = MP_VISIBLE_HANDS * len(rows)
    return {
        "model": "MediaPipe Hand Landmarker + Pose Landmarker (full)", "version": mp.__version__, "run_as": "original",
        "size_mb": round(sum(p.stat().st_size for p in models.values()) / 2**20, 1),
        "latency_ms_hand": lat_hand, "latency_ms_pose": lat_pose, "example": fig,
        "hand_detection": {"images": len(rows), "visible_hands": total,
                           "recall_full_image": round(sum(r["hands_full_image"] for r in rows) / total, 3),
                           "recall_with_wrist_crop": round(sum(r["hands_with_wrist_crop"] for r in rows) / total, 3),
                           "per_image": rows},
    }


# ---------- 실행 ----------
if __name__ == "__main__":
    # 인자 없이: 전체 실행 / 인자로 단계 이름(yoloe dinox sam3 siglip ocr mediapipe)을 주면 그 단계만 다시 돌려 JSON에 덮어씀
    only = set(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    gt = {k: v for k, v in json.loads((DATA / "gt_boxes.json").read_text()).items() if k.endswith("_start.png")}
    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if only and res_path.exists() else {}
    results["env"] = {
        "machine": platform.machine(), "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__, "device": DEVICE, "torch_threads": torch.get_num_threads(),
        "dataset": "lerobot/svla_so101_pickplace@f641879", "episodes": N_EPISODES, "prompts": PROMPTS, "iou_thresh": IOU_THRESH,
    }
    gd = {}
    def gdino():
        if not gd:
            gd["model"], gd["detect"] = load_gdino()
        return gd["model"], gd["detect"]

    steps = {
        "yoloe": lambda: run_yoloe(gt),
        "dinox": lambda: run_gdino(gt, *gdino()),
        "sam3": lambda: run_grounded_sam2(gdino()[1]),
        "siglip": run_siglip, "ocr": run_ocr, "mediapipe": run_mediapipe,
    }
    for key, fn in steps.items():
        if only and key not in only:
            continue
        print(f"[run] {key}", flush=True)
        results[key] = fn()
        brief = {k: v for k, v in results[key].items() if k not in ("english_text", "korean_text", "korean_text_inverted", "example_detections")}
        brief = {k: ({a: b for a, b in v.items() if a != "per_frame"} if isinstance(v, dict) else v) for k, v in brief.items()}
        print(json.dumps(brief, ensure_ascii=False, indent=1), flush=True)
    res_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"saved: {res_path}")
