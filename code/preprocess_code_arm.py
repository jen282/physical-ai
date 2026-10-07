"""
TeachArm 데이터 전처리 핵심 모듈 (SO-101 + LeRobot + ACT 기준)

사용 흐름 (실행 순서):
  0) load_episodes        : LeRobot parquet 로드 (v2.1 / v3.0 공통: episode_index 컬럼으로 그룹핑)
  1) check_sync           : 타임스탬프 간격(dt) 점검 → 프레임 드랍/루프 지연 탐지
  2) estimate_action_lag  : action[t] ↔ state[t+k] 상호상관으로 추종 지연(프레임) 측정
  3) filter_episodes      : 실패/길이 이상/드랍 과다 에피소드 제외
  4) clean_trajectory     : Hampel(스파이크 제거) → (옵션) Savitzky-Golay → 정지 구간 트리밍 인덱스 산출
  5) image_quality        : 블러(Laplacian 분산)/밝기 지표 → 품질 이상 프레임 탐지
  6) compute_stats        : 필터링 이후 데이터로만 정규화 통계 재계산

주의: 컬럼 이름("observation.state", "action")은 LeRobot 기본값 기준. 버전에 따라 확인 필요.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 30
GRIPPER_IDX = 5  # SO-101: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]


# ---------- 0) 로드 ----------
def load_episodes(root: str) -> dict[int, pd.DataFrame]:
    files = sorted(Path(root).glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return {int(k): g.sort_values("frame_index").reset_index(drop=True)
            for k, g in df.groupby("episode_index")}


def to_array(ep: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(ep[col].to_numpy()).astype(np.float64)  # (T, 6)


# ---------- 1) 동기화 점검 ----------
def check_sync(ep: pd.DataFrame, fps: int = FPS) -> dict:
    dt = np.diff(ep["timestamp"].to_numpy())
    nominal = 1.0 / fps
    return {
        "dt_mean_ms": dt.mean() * 1e3,
        "dt_std_ms": dt.std() * 1e3,
        "late_ratio": float((dt > 1.5 * nominal).mean()),   # 지연/드랍 의심 비율
        "non_monotonic": int((dt <= 0).sum()),              # 역전/중복
    }


# ---------- 2) action → state 추종 지연 ----------
def estimate_action_lag(action: np.ndarray, state: np.ndarray, max_lag: int = 10) -> int:
    """관절별 속도의 상호상관이 최대가 되는 k (프레임). SO-101에서는 보통 1~4."""
    va, vs = np.diff(action[:, :5], axis=0), np.diff(state[:, :5], axis=0)
    scores = []
    for k in range(max_lag + 1):
        a, s = va[: len(va) - k], vs[k:]
        num = (a * s).sum()
        den = np.sqrt((a ** 2).sum() * (s ** 2).sum()) + 1e-9
        scores.append(num / den)
    return int(np.argmax(scores))


# ---------- 3) 에피소드 필터링 ----------
def filter_episodes(eps: dict, success: dict[int, bool], max_late_ratio: float = 0.05) -> list[int]:
    lengths = np.array([len(e) for e in eps.values()])
    med = np.median(lengths)
    mad = np.median(np.abs(lengths - med)) + 1e-9
    keep = []
    for idx, ep in eps.items():
        n = len(ep)
        if not success.get(idx, True):
            continue                                          # 실패 에피소드 제외
        if abs(n - med) / (1.4826 * mad) > 3.5:
            continue                                          # 길이 이상치 (robust z-score)
        if check_sync(ep)["late_ratio"] > max_late_ratio:
            continue                                          # 프레임 드랍 과다
        keep.append(idx)
    return keep


# ---------- 4) 궤적 클리닝 ----------
def hampel(x: np.ndarray, half_window: int = 3, n_sigma: float = 3.0) -> np.ndarray:
    """스파이크만 국소 중앙값으로 치환. 정상 신호는 건드리지 않음."""
    y = x.copy()
    T = len(x)
    for t in range(T):
        lo, hi = max(0, t - half_window), min(T, t + half_window + 1)
        win = x[lo:hi]
        med = np.median(win, axis=0)
        mad = 1.4826 * np.median(np.abs(win - med), axis=0) + 1e-9
        out = np.abs(x[t] - med) > n_sigma * mad
        y[t, out] = med[out]
    return y


def clean_trajectory(arr: np.ndarray, smooth: bool = False) -> np.ndarray:
    y = hampel(arr)
    if smooth:  # 떨림이 실제로 관찰될 때만. 그리퍼 채널은 제외 (파지 타이밍 보존)
        arm = [i for i in range(arr.shape[1]) if i != GRIPPER_IDX]
        y[:, arm] = savgol_filter(y[:, arm], window_length=5, polyorder=2, axis=0)
    return y


def idle_trim_range(action: np.ndarray, vel_thresh: float = 0.3,
                    min_still: int = 15, margin: int = 10) -> tuple[int, int]:
    """시작/끝 정지 구간을 잘라낼 [start, end) 인덱스. vel_thresh는 관절 단위(도 또는 정규화값)/프레임."""
    speed = np.linalg.norm(np.diff(action, axis=0), axis=1)
    moving = np.where(speed > vel_thresh)[0]
    if len(moving) == 0:
        return 0, len(action)
    start = moving[0] - margin if moving[0] >= min_still else 0
    end = moving[-1] + margin if len(action) - moving[-1] >= min_still else len(action)
    return max(0, start), min(len(action), end + 1)


# ---------- 5) 이미지 품질 ----------
def image_quality(frames_bgr) -> np.ndarray:
    """frames: (T,H,W,3) uint8. 반환: (T,2) = [Laplacian 분산(선명도), 평균 밝기]."""
    import cv2
    out = []
    for f in frames_bgr:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        out.append([cv2.Laplacian(g, cv2.CV_64F).var(), g.mean()])
    return np.array(out)


def flag_bad_frames(q: np.ndarray, blur_ratio: float = 0.4, bright_z: float = 3.0) -> np.ndarray:
    """에피소드 중앙값 대비 상대 기준 (절대 임계값은 카메라마다 달라서 비권장)."""
    blur_bad = q[:, 0] < blur_ratio * np.median(q[:, 0])
    b = q[:, 1]
    bright_bad = np.abs(b - np.median(b)) > bright_z * (1.4826 * np.median(np.abs(b - np.median(b))) + 1e-9)
    return blur_bad | bright_bad


# ---------- 6) 정규화 통계 (필터링 이후에만 계산) ----------
def compute_stats(eps: dict, keep: list[int], col: str) -> dict:
    x = np.concatenate([to_array(eps[i], col) for i in keep])
    return {"mean": x.mean(0), "std": x.std(0) + 1e-6, "min": x.min(0), "max": x.max(0)}


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    eps = load_episodes(root)
    print(f"episodes: {len(eps)}")
    for i, ep in list(eps.items())[:5]:
        a, s = to_array(ep, "action"), to_array(ep, "observation.state")
        print(i, check_sync(ep), "lag:", estimate_action_lag(a, s), "trim:", idle_trim_range(a))
