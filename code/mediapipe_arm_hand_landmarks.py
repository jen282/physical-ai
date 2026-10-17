"""
TeachArm 손 추적 비교군용: MediaPipe로 팔(어깨-팔꿈치-손목)과 손(21점) 랜드마크 시각화

사용 흐름:
  0) ensure_assets         : 공식 모델(.task)과 공식 샘플 이미지를 내려받아 캐시 (최초 1회)
  1) detect_arms           : PoseLandmarker(33점) 중 팔 관절 6점(11~16)만 사용
  2) detect_hands          : HandLandmarker로 전체 이미지에서 양손 21점 + 왼손/오른손 판별
  3) recover_missing_hands : 손이 작아 2)에서 놓친 경우, 팔꿈치→손목 방향으로 손 위치를 추정해
                             잘라낸 뒤 재검출 (MediaPipe Holistic과 같은 아이디어)
  4) draw_*                : 왼쪽=주황, 오른쪽=하늘색으로 그려 팔과 손의 짝을 구분
  5) main                  : 결과를 창으로 띄우고 code/mediapipe_output/ 에 저장

샘플 데이터 출처 (모두 Google 공식 배포본):
  - male_full_height_hands.jpg, pose.jpg
      github.com/google-ai-edge/mediapipe 의 third_party/external_files.bzl 에 등록된 테스트 이미지
  - woman_hands.jpg
      github.com/google-ai-edge/mediapipe-samples 의 hand_landmarker Python 예제 노트북에서 쓰는 이미지
  - 모델: storage.googleapis.com/mediapipe-models (MediaPipe 공식 모델 배포 경로)

실행:
  python code/mediapipe_arm_hand_landmarks.py                  # 공식 샘플 3장
  python code/mediapipe_arm_hand_landmarks.py my.jpg --no-show # 내 이미지, 창 없이 저장만

주의:
  - mediapipe 1.x 는 mp.solutions(legacy) API가 없어 Tasks API(mediapipe.tasks.python.vision)를 사용.
  - HolisticLandmarker(팔+손 통합 모델)는 1.1.0 IMAGE 모드에서 이미지 크기가 바뀌면 에러가 나서 Pose+Hand 조합을 씀.
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "mediapipe_assets"
OUTPUT_DIR = ROOT / "mediapipe_output"

MODELS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
SAMPLES = {
    "male_full_height_hands.jpg": "https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/male_full_height_hands.jpg",
    "woman_hands.jpg": "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
    "pose.jpg": "https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pose.jpg",
}

# Pose 33점 중 팔: 11/12 어깨, 13/14 팔꿈치, 15/16 손목 (홀수=왼쪽, 짝수=오른쪽, 사람 기준)
ARM_IDS = {"Left": (11, 13, 15), "Right": (12, 14, 16)}
SHOULDERS = (11, 12)
VIS_THRESH = 0.5  # 가려진 관절(visibility 낮음)은 그리지 않음

COLORS = {"Left": (0, 140, 255), "Right": (255, 200, 0)}  # BGR: 주황 / 하늘색
NEUTRAL = (200, 200, 200)


# ---------- 0) 모델·샘플 다운로드 ----------
def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def ensure_assets() -> tuple[dict[str, Path], list[Path]]:
    models = {k: download(u, ASSET_DIR / "models" / Path(u).name) for k, u in MODELS.items()}
    samples = [download(u, ASSET_DIR / "samples" / name) for name, u in SAMPLES.items()]
    return models, samples


# ---------- 1) 팔 (Pose) ----------
def detect_arms(landmarker: vision.PoseLandmarker, image: mp.Image) -> list:
    """사람별 33점 리스트. 팔 관절만 그리기 단계에서 골라 씀."""
    return landmarker.detect(image).pose_landmarks


# ---------- 2) 손 (Hand, 전체 이미지) ----------
def detect_hands(landmarker: vision.HandLandmarker, image: mp.Image) -> list[tuple[str, list, str]]:
    """[(Left|Right, 21점, 'full')]. 핸드니스는 '사람 기준' 왼손/오른손."""
    res = landmarker.detect(image)
    return [(h[0].category_name, lms, "full") for h, lms in zip(res.handedness, res.hand_landmarks)]


# ---------- 3) 손 (손목 주변 크롭 재검출) ----------
def to_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def detect_hand_near_wrist(landmarker: vision.HandLandmarker, rgb: np.ndarray,
                           wrist: tuple[int, int], elbow: tuple[int, int]) -> list | None:
    """손 중심 ≈ 손목에서 팔뚝 방향으로 0.5배 떨어진 곳. 한 변 = 팔뚝 길이 × 2 인 정사각형으로 크롭."""
    h, w = rgb.shape[:2]
    fx, fy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    half = int(np.hypot(fx, fy))
    cx, cy = int(wrist[0] + 0.5 * fx), int(wrist[1] + 0.5 * fy)
    x0, y0, x1, y1 = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    if not res.hand_landmarks:
        return None
    cw, ch = x1 - x0, y1 - y0  # 크롭 좌표 → 원본 정규화 좌표로 복원
    return [NormalizedLandmark(x=(x0 + lm.x * cw) / w, y=(y0 + lm.y * ch) / h, z=lm.z)
            for lm in res.hand_landmarks[0]]


def recover_missing_hands(landmarker: vision.HandLandmarker, rgb: np.ndarray,
                          poses: list, hands: list) -> list[tuple[str, list, str]]:
    """팔은 보이는데 그 손목 근처에 검출된 손이 없으면 크롭 재검출. 좌우는 팔(Pose) 기준으로 지정."""
    h, w = rgb.shape[:2]
    found = []
    for lms in poses:
        for side, (_, e, wr) in ARM_IDS.items():
            if min(lms[e].visibility, lms[wr].visibility) < VIS_THRESH:
                continue
            wrist, elbow = to_px(lms[wr], w, h), to_px(lms[e], w, h)
            forearm = np.hypot(wrist[0] - elbow[0], wrist[1] - elbow[1])
            if any(np.hypot(*np.subtract(to_px(hl[0], w, h), wrist)) < 0.5 * forearm
                   for _, hl, _ in hands + found):
                continue  # 이미 이 팔의 손을 찾음
            hl = detect_hand_near_wrist(landmarker, rgb, wrist, elbow)
            if hl:
                found.append((side, hl, "crop"))
    return found


# ---------- 4) 그리기 ----------
def draw_arms(img: np.ndarray, poses: list) -> None:
    h, w = img.shape[:2]
    thick = max(2, min(h, w) // 250)
    for lms in poses:
        ok = lambda i: lms[i].visibility > VIS_THRESH
        a, b = SHOULDERS
        if ok(a) and ok(b):
            cv2.line(img, to_px(lms[a], w, h), to_px(lms[b], w, h), NEUTRAL, thick, cv2.LINE_AA)
        for side, (s, e, wr) in ARM_IDS.items():
            for i, j in ((s, e), (e, wr)):
                if ok(i) and ok(j):
                    cv2.line(img, to_px(lms[i], w, h), to_px(lms[j], w, h), COLORS[side], thick * 2, cv2.LINE_AA)
            for i in (s, e, wr):
                if ok(i):
                    p = to_px(lms[i], w, h)
                    cv2.circle(img, p, thick * 3, COLORS[side], -1, cv2.LINE_AA)
                    cv2.circle(img, p, thick * 3, (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)


def draw_hands(img: np.ndarray, hands: list[tuple[str, list, str]]) -> None:
    h, w = img.shape[:2]
    thick = max(1, min(h, w) // 400)
    for side, lms, src in hands:
        vision.drawing_utils.draw_landmarks(
            img, lms, vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=vision.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=thick + 2),
            connection_drawing_spec=vision.drawing_utils.DrawingSpec(color=COLORS.get(side, NEUTRAL), thickness=thick + 1),
        )
        x, y = to_px(lms[0], w, h)  # 0번 = 손목
        label = f"{side} hand" + (" (crop)" if src == "crop" else "")
        cv2.putText(img, label, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS.get(side, NEUTRAL), 2, cv2.LINE_AA)


def draw_legend(img: np.ndarray, n_persons: int, n_hands: int) -> None:
    cv2.rectangle(img, (0, 0), (260, 70), (0, 0, 0), -1)
    cv2.putText(img, "Left arm/hand", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Left"], 2, cv2.LINE_AA)
    cv2.putText(img, "Right arm/hand", (135, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Right"], 2, cv2.LINE_AA)
    cv2.putText(img, f"persons: {n_persons}  hands: {n_hands}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def print_summary(name: str, poses: list, hands: list, w: int, h: int) -> None:
    print(f"\n[{name}] persons={len(poses)} hands={len(hands)}")
    for p, lms in enumerate(poses):
        for side, ids in ARM_IDS.items():
            pts = [f"{to_px(lms[i], w, h)}" if lms[i].visibility > VIS_THRESH else "hidden" for i in ids]
            print(f"  person{p} {side:<5} arm  shoulder/elbow/wrist(px): {' / '.join(pts)}")
    for side, lms, src in hands:
        print(f"  {side:<5} hand  wrist(px)={to_px(lms[0], w, h)}  index_tip(px)={to_px(lms[8], w, h)}  [{src}]")


# ---------- 5) 실행 ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="MediaPipe 팔·손 랜드마크 시각화")
    ap.add_argument("images", nargs="*", type=Path, help="입력 이미지 (생략 시 공식 샘플 사용)")
    ap.add_argument("--no-show", action="store_true", help="창을 띄우지 않고 파일로만 저장")
    args = ap.parse_args()

    models, samples = ensure_assets()
    images = args.images or samples
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pose_opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models["pose"])), num_poses=2)
    hand_opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models["hand"])), num_hands=2)

    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
         vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:
        for path in images:
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f"\n[{path}] 이미지를 읽을 수 없어 건너뜀")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            poses = detect_arms(pose_lm, mp_image)
            hands = detect_hands(hand_lm, mp_image)
            hands += recover_missing_hands(hand_lm, rgb, poses, hands)

            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            draw_arms(canvas, poses)
            draw_hands(canvas, hands)
            draw_legend(canvas, len(poses), len(hands))

            h, w = canvas.shape[:2]
            print_summary(path.name, poses, hands, w, h)
            out = OUTPUT_DIR / f"{path.stem}_landmarks.jpg"
            cv2.imwrite(str(out), canvas)
            print(f"  saved: {out}")

            if not args.no_show:
                cv2.imshow("MediaPipe arm + hand landmarks (any key: next, q: quit)", canvas)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
