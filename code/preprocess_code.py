"""
PCB 리플로우 후 외관검사(AOI) 전처리 파이프라인 — 단일 파일 버전
=====================================================================

추론 순서
  0  획득 품질 게이트
  1  결함 화소 보정 + 다크/플랫필드 보정        (화소 단위 연산, 보간 없음)
  2  렌즈 왜곡 보정                               (프레임 보간은 여기서 1회뿐)
  3  전역 정렬(피듀셜) → 이미지가 아니라 CAD 좌표를 변환
  4  색 정규화(CAD 마스크 기반 Lab a*,b* 이동)   → 통계만 계산, 적용은 ROI에만
  5  CAD ROI 추출 + 시차 보정(Δr = r·h/WD)
  6  ROI 국부 정렬 → 정수 화소로 다시 잘라내기(보간 없음), 부품 몸체는 정렬에서 제외
  7  포화 마스크 채널 (원본 기준, 지우지 않음)
  8  (선택) 경량 노이즈 제거 (기본 꺼짐, 커널 3px 상한)
  9  (선택) 룰 기반 분기: CLAHE + 화이트 탑햇
  10 골든 차분 채널
  11 모델 입력 규격화 (B,G,R,SAT,DIFF 5채널)
  +  별도 경로: 골든 Envelope 기반 랜덤 이물 후보 + 플럭스 의심 표시

의존성: numpy, opencv-python (또는 opencv-python-headless)

사용
  python pcb_aoi_preproc.py              # 합성 데이터 검증 데모 실행
  python pcb_aoi_preproc.py --out x.png  # 시각화 저장 경로 지정

라이브러리로 사용하는 순서
  cfg = PipelineConfig(); cfg.optics.working_distance_mm = <실측값>
  calib = Calibration()
  calib.dark, calib.gain, calib.hot_map = build_dark_flat(darks, flats)
  K, dist, rms = calibrate_lens(checkers); calib.map1, calib.map2 = build_undistort_maps(K, dist, size)
  pre = Preprocessor(board_cad, cfg, calib)
  calib.fiducial_template = build_fiducial_template(golden_gray, fid_px, diameter_px)
  calib.ref_sharpness = sharpness(golden_gray)
  calib.ref_ab = build_reference_ab(pre, good_raws, shot)
  calib.golden_rois[shot.shot_id] = build_golden_rois(pre, good_raws, shot)
  calib.envelopes[shot.shot_id] = build_envelope(pre, good_raws, shot)
  res = pre.process_shot(raw, shot)              # 피듀셜 있는 촬영
  res2 = pre.process_shot(raw2, shot2, T=res.T)  # 피듀셜 없는 촬영은 보드 변환 T 전달

조명 교체·렌즈 재조정·기판 리비전 변경 시 보정 기준값 전체를 다시 생성할 것.
"""
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2


# ############################################################################
# 설정값 (추천 초기 파라미터)
# ############################################################################

@dataclass
class OpticsConfig:
    mm_per_px: float = 0.02                 # 5MP, FOV 50x40mm 기준
    working_distance_mm: float = 110.0      # [추정치] 반드시 실측값으로 교체 (시차 계산에 사용)
    image_size: Tuple[int, int] = (2448, 2048)  # (w, h)


@dataclass
class GateConfig:                            # 0단계: 획득 품질 게이트
    sharpness_ratio_min: float = 0.70        # 기준 라플라시안 분산 대비 70% 미만 → 재촬영
    saturation_ratio_warn: float = 0.05      # 포화 화소 5% 초과 → 경고
    saturation_level: int = 250


@dataclass
class FlatFieldConfig:                       # 1단계: 결함화소 + 다크/플랫
    flat_sigma_px: float = 50.0              # 플랫 프레임 평활 σ
    hot_pixel_k: float = 8.0                 # median + k·MAD 초과 화소를 핫픽셀로 판정
    gain_clip: Tuple[float, float] = (0.5, 2.0)


@dataclass
class FiducialConfig:                        # 3단계: 전역 정렬
    search_radius_px: int = 50               # 공칭 위치 ±50px 탐색
    ncc_min: float = 0.80
    max_residual_px: float = 1.5             # 재투영 잔차 초과 시 재촬영


@dataclass
class ColorNormConfig:                       # 4단계: 색 정규화
    erode_px: int = 10                       # 솔더마스크 마스크 침식량(풀해상도 기준)
    max_shift: float = 15.0                  # a*, b* 이동량 상한
    stats_downscale: int = 4                 # 통계는 1/4 축소 이미지로 계산
    apply_sigma_ab: float = 12.0             # 솔더마스크와 비슷한 색의 화소에만 보정 적용(가중치 σ)


@dataclass
class RoiConfig:                             # 5단계: CAD ROI
    margin_mm: float = 0.5                   # 패드 외곽 + 0.5mm(25px)


@dataclass
class LocalAlignConfig:                      # 6단계: 국부 정렬
    method: str = "ecc"                      # "ecc" | "phase" | "none"
    iterations: int = 50
    eps: float = 1e-4
    gauss_filt: int = 3
    max_shift_px: float = 8.0                # 이보다 큰 보정량은 신뢰하지 않음(전역 정렬 결과 유지)
    ecc_only_fine_pitch: bool = False        # True면 파인피치 IC에만 ECC, 나머지는 phase (경량화)


@dataclass
class DenoiseConfig:                         # 8단계: (선택) 노이즈 제거
    mode: str = "off"                        # "off" | "gaussian" | "bilateral"
    gaussian_sigma: float = 0.6
    bilateral_d: int = 5
    bilateral_sigma_color: float = 15.0
    bilateral_sigma_space: float = 2.0
    max_kernel_px: int = 3                   # 커널 3px 초과 금지 (브릿지 7~8px 보존)


@dataclass
class RuleBranchConfig:                      # 9단계: (선택) 룰 기반 분기
    enabled: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    tophat_kernel_px: int = 15               # 약 0.3mm


@dataclass
class ModelInputConfig:                      # 11단계: 모델 입력 규격화
    small_size: int = 128
    tile_size: int = 384
    tile_overlap: int = 32
    min_scale: float = 0.67                  # 축소 시 배율 하한(브릿지 5px 이상 유지)
    # 학습셋(전처리 후) 기준 채널 평균/표준편차. 학습 후 채워 넣을 것.
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None


@dataclass
class EnvelopeConfig:                        # 별도 경로: 랜덤 위치 이물
    tolerance: int = 20                      # 골든 min/max 범위 밖 허용 여유(gray level)
    spatial_tol_px: int = 1                  # 정렬 잔차 허용(min/max를 공간적으로 ±1px 확장)
    min_area_px: int = 12                    # 약 0.1mm 결함의 절반 면적
    roi_exclude_dilate_px: int = 5
    flux_L_min: float = 170.0                # 플럭스 의심: 밝고(L) 황색 기미(b*)
    flux_b_min: float = 140.0                # 8bit Lab 기준(b*+128)


@dataclass
class PipelineConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    flat: FlatFieldConfig = field(default_factory=FlatFieldConfig)
    fiducial: FiducialConfig = field(default_factory=FiducialConfig)
    color: ColorNormConfig = field(default_factory=ColorNormConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    local_align: LocalAlignConfig = field(default_factory=LocalAlignConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    rule: RuleBranchConfig = field(default_factory=RuleBranchConfig)
    model_input: ModelInputConfig = field(default_factory=ModelInputConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)

# ############################################################################
# CAD 기하 정보와 좌표 변환
# ############################################################################

@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def corners(self) -> np.ndarray:
        return np.array([[self.x0, self.y0], [self.x1, self.y0],
                         [self.x1, self.y1], [self.x0, self.y1]], np.float64)


@dataclass
class Component:
    ref: str                      # Ref. Designator (예: R12, U3)
    pads: List[Rect]
    body: Rect
    height_mm: float
    polarized: bool = False
    fine_pitch: bool = False
    input_mode: str = "pad"       # "pad"(소형, 패딩) | "tile"(대형 IC, 타일 분할)

    def bbox(self) -> Rect:
        rs = self.pads + [self.body]
        return Rect(min(r.x0 for r in rs), min(r.y0 for r in rs),
                    max(r.x1 for r in rs), max(r.y1 for r in rs))


@dataclass
class BoardCAD:
    width_mm: float
    height_mm: float
    components: List[Component]
    fiducials: List[Tuple[float, float]]       # 피듀셜 중심(mm)
    silkscreen: List[Rect] = field(default_factory=list)
    fiducial_diameter_mm: float = 1.0


@dataclass
class Shot:
    """갠트리 촬영 1회. origin_mm = 이미지 (0,0) 화소 중심의 공칭 보드 좌표."""
    shot_id: int
    origin_mm: Tuple[float, float]
    mm_per_px: float
    image_size: Tuple[int, int]                # (w, h)

    def A_nom(self) -> np.ndarray:
        s = 1.0 / self.mm_per_px
        return np.array([[s, 0, -self.origin_mm[0] * s],
                         [0, s, -self.origin_mm[1] * s]], np.float64)

    def contains_mm(self, x, y, margin_mm=0.0) -> bool:
        w, h = self.image_size
        x0, y0 = self.origin_mm
        return (x0 + margin_mm <= x <= x0 + w * self.mm_per_px - margin_mm and
                y0 + margin_mm <= y <= y0 + h * self.mm_per_px - margin_mm)


def to3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0, 0, 1]])


def compose(A_nom: np.ndarray, T: np.ndarray) -> np.ndarray:
    """보드 mm → 이미지 px 최종 2x3 행렬."""
    return (to3(A_nom) @ to3(T))[:2]


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


IDENTITY_T = np.array([[1, 0, 0], [0, 1, 0]], np.float64)


@dataclass
class RoiGeom:
    ref: str
    x0: int
    y0: int
    x1: int
    y1: int
    body_poly: np.ndarray        # ROI 좌표계 기준 부품 몸체(바닥+시차 반영 상단) 다각형들
    parallax_px: np.ndarray      # 부품 상단의 방사 방향 시차 벡터(px)
    component: Component

    @property
    def size(self):
        return self.x1 - self.x0, self.y1 - self.y0


def _roi_points(comp: Component, M: np.ndarray, center: np.ndarray,
                wd_mm: float, margin_mm: float):
    bb = comp.bbox()
    m = margin_mm
    box_px = apply(M, Rect(bb.x0 - m, bb.y0 - m, bb.x1 + m, bb.y1 + m).corners())
    body_px = apply(M, comp.body.corners())
    # 시차: 몸체 상단은 광축 중심에서 바깥쪽으로 Δr = r·h/WD 만큼 밀려 보임
    parallax = (body_px.mean(0) - center) * comp.height_mm / wd_mm
    top_px = body_px + parallax
    return np.vstack([box_px, top_px]), body_px, top_px, parallax


def compute_rois(cad: BoardCAD, shot: Shot, T: np.ndarray, wd_mm: float,
                 margin_mm: float) -> List[RoiGeom]:
    """
    CAD 좌표로 ROI 계산 + 시차 보정.
    ROI 크기는 공칭 배치(T=I) 기준으로 고정하고 위치(중심)만 실제 T로 계산합니다.
    → 보드마다 ROI 크기가 같아서 골든 ROI와 화소 단위 비교가 가능합니다.
    """
    M_nom = compose(shot.A_nom(), IDENTITY_T)
    M = compose(shot.A_nom(), T)
    w, h = shot.image_size
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    rois = []
    for comp in cad.components:
        bb = comp.bbox()
        if not shot.contains_mm((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2):
            continue
        pts_n, _, _, _ = _roi_points(comp, M_nom, center, wd_mm, margin_mm)
        rw = int(np.ceil(np.ptp(pts_n[:, 0]))) + 2
        rh = int(np.ceil(np.ptp(pts_n[:, 1]))) + 2
        pts, body_px, top_px, parallax = _roi_points(comp, M, center, wd_mm, margin_mm)
        c = (pts.min(0) + pts.max(0)) / 2
        x0 = int(np.floor(c[0] - rw / 2))
        y0 = int(np.floor(c[1] - rh / 2))
        off = np.array([x0, y0], np.float64)
        rois.append(RoiGeom(comp.ref, x0, y0, x0 + rw, y0 + rh,
                            body_poly=np.stack([body_px - off, top_px - off]),
                            parallax_px=parallax, component=comp))
    return rois


def body_mask(roi: RoiGeom, dilate_px: int = 2) -> np.ndarray:
    """ROI 안의 부품 몸체 영역(바닥~시차 상단까지 볼록 껍질)=255."""
    w, h = roi.size
    mask = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(roi.body_poly.reshape(-1, 2).astype(np.float32))
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * dilate_px + 1,) * 2, np.uint8))
    return mask


def solder_mask_region(cad: BoardCAD, shot: Shot, T: np.ndarray, scale: float,
                       erode_px: float, out_size: Tuple[int, int]) -> np.ndarray:
    """
    '순수 솔더마스크' 영역 마스크(색 통계용).
    보드 영역 − (패드 ∪ 부품 몸체 ∪ 실크 ∪ 피듀셜) 을 erode_px 만큼 침식.
    scale: 출력 해상도 / 풀해상도 (예: 0.25)
    """
    M = compose(shot.A_nom(), T)
    S = np.diag([scale, scale, 1.0])
    Ms = (S @ to3(M))[:2]
    w, h = out_size
    mask = np.zeros((h, w), np.uint8)
    board = Rect(0, 0, cad.width_mm, cad.height_mm)
    cv2.fillConvexPoly(mask, np.round(apply(Ms, board.corners())).astype(np.int32), 255)
    excl = []
    for c in cad.components:
        excl += c.pads + [c.body]
    excl += cad.silkscreen
    r = cad.fiducial_diameter_mm  # 피듀셜 주변 클리어런스 포함
    excl += [Rect(x - r, y - r, x + r, y + r) for x, y in cad.fiducials]
    for rect in excl:
        cv2.fillConvexPoly(mask, np.round(apply(Ms, rect.corners())).astype(np.int32), 0)
    k = max(1, int(round(erode_px * scale)))
    mask = cv2.erode(mask, np.ones((2 * k + 1,) * 2, np.uint8))
    return mask

# ############################################################################
# 보정 기준값(Calibration) 생성·저장
# ############################################################################

@dataclass
class Calibration:
    # 1단계
    dark: Optional[np.ndarray] = None          # float32 HxWx3
    gain: Optional[np.ndarray] = None          # float32 HxWx3
    hot_map: Optional[np.ndarray] = None       # bool HxW
    # 2단계
    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None
    # 0단계
    ref_sharpness: Optional[float] = None
    # 3단계
    fiducial_template: Optional[np.ndarray] = None   # gray uint8
    # 4단계: 기준 로트 솔더마스크 (a, b) 평균 (8bit Lab 단위)
    ref_ab: Optional[Tuple[float, float]] = None
    # 10단계: shot_id -> {ref: golden BGR ROI}
    golden_rois: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    # 별도 경로: shot_id -> (min, max) gray (공칭 좌표계)
    envelopes: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def save(self, path: str):
        d = {}
        for k in ["dark", "gain", "hot_map", "map1", "map2", "fiducial_template"]:
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.ref_sharpness is not None:
            d["ref_sharpness"] = np.array(self.ref_sharpness)
        if self.ref_ab is not None:
            d["ref_ab"] = np.array(self.ref_ab)
        for sid, g in self.golden_rois.items():
            for ref, img in g.items():
                d[f"golden__{sid}__{ref}"] = img
        for sid, (lo, hi) in self.envelopes.items():
            d[f"envlo__{sid}"] = lo
            d[f"envhi__{sid}"] = hi
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        z = np.load(path)
        c = cls()
        for k in ["dark", "gain", "hot_map", "map1", "map2", "fiducial_template"]:
            if k in z:
                setattr(c, k, z[k])
        if "ref_sharpness" in z:
            c.ref_sharpness = float(z["ref_sharpness"])
        if "ref_ab" in z:
            c.ref_ab = tuple(float(v) for v in z["ref_ab"])
        for k in z.files:
            if k.startswith("golden__"):
                _, sid, ref = k.split("__", 2)
                c.golden_rois.setdefault(int(sid), {})[ref] = z[k]
            elif k.startswith("envlo__"):
                sid = int(k.split("__")[1])
                c.envelopes[sid] = (z[k], z[f"envhi__{sid}"])
        return c


# ---------------------------------------------------------------- 1단계

def build_dark_flat(dark_frames: List[np.ndarray], flat_frames: List[np.ndarray],
                    flat_sigma_px: float = 50.0, hot_k: float = 8.0,
                    gain_clip=(0.5, 2.0)):
    """
    dark_frames: 렌즈캡 장착, 실제 노출/게인 조건, 20장 권장 (가능하면 운용 온도에서)
    flat_frames: 균일 회색판(18% 그레이 카드 등)을 실제 조명으로 촬영, 20장 권장
    반환: dark(float32), gain(float32), hot_map(bool)
    """
    D = np.mean(np.stack(dark_frames).astype(np.float32), axis=0)
    F = np.mean(np.stack(flat_frames).astype(np.float32), axis=0)

    # 핫픽셀: 다크 평균에서 국부 중앙값 대비 크게 튀는 화소
    Dg = D.max(axis=2) if D.ndim == 3 else D
    resid = Dg - cv2.medianBlur(np.clip(Dg, 0, 255).astype(np.uint8), 3).astype(np.float32)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    hot_map = resid > np.median(resid) + hot_k * 1.4826 * mad

    FD = F - D
    # 큰 σ로 평활: 회색판의 먼지·미세무늬·핫픽셀이 게인맵에 새겨지지 않도록 저주파 성분만 남김
    FD_s = cv2.GaussianBlur(FD, (0, 0), flat_sigma_px, borderType=cv2.BORDER_REFLECT)
    FD_s = np.maximum(FD_s, 1e-3)
    mean = FD_s.reshape(-1, FD_s.shape[-1]).mean(0) if FD_s.ndim == 3 else FD_s.mean()
    gain = np.clip(mean / FD_s, *gain_clip).astype(np.float32)
    return D.astype(np.float32), gain, hot_map


# ---------------------------------------------------------------- 2단계

def calibrate_lens(checker_images: List[np.ndarray], pattern_size=(9, 6),
                   square_mm: float = 2.0):
    """체커보드 15장 이상 권장. 반환: K, dist, rms(재투영 오차 px, 목표 0.3 이하)."""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_mm
    objpoints, imgpoints = [], []
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for img in checker_images:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        size = g.shape[::-1]
        ok, corners = cv2.findChessboardCorners(g, pattern_size)
        if not ok:
            continue
        corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), crit)
        objpoints.append(objp)
        imgpoints.append(corners)
    if len(objpoints) < 10:
        raise RuntimeError(f"체커보드 검출 {len(objpoints)}장: 최소 10장(권장 15장) 필요")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return K, dist, rms


def build_undistort_maps(K, dist, image_size):
    """맵은 1회 사전 계산. 고정소수점(CV_16SC2)으로 remap 속도 향상."""
    w, h = image_size
    return cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)


def identity_maps(image_size):
    """왜곡 보정 없이 사용할 때(캘리브레이션 전 임시)."""
    w, h = image_size
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.convertMaps(mx, my, cv2.CV_16SC2)


# ---------------------------------------------------------------- 0, 3단계

def sharpness(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    _, sd = cv2.meanStdDev(lap)
    return float(sd[0, 0] ** 2)


def build_fiducial_template(golden_gray: np.ndarray, center_px, diameter_px: float,
                            pad_factor: float = 1.6) -> np.ndarray:
    """골든 보드(보정 완료 이미지)에서 피듀셜 주변을 잘라 템플릿 생성."""
    r = int(round(diameter_px * pad_factor / 2))
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))
    return golden_gray[cy - r:cy + r + 1, cx - r:cx + r + 1].copy()

# ############################################################################
# 전처리 파이프라인 본체 (0~11단계 + 랜덤 이물 경로)
# ############################################################################

# ===================================================================== 결과 구조

@dataclass
class RoiResult:
    ref: str
    geom: RoiGeom
    bgr: np.ndarray                      # 정렬·색 정규화·(노이즈 제거) 완료 ROI
    sat: np.ndarray                      # 포화 마스크 uint8 {0,255}
    diff: Optional[np.ndarray]           # 골든 차분 uint8 (골든 없으면 None)
    rule: Optional[Dict[str, np.ndarray]]
    model_inputs: List[Tuple[np.ndarray, dict]]   # (CHW float32, meta)
    local_shift: Tuple[float, float]
    align_method: str
    sat_ratio: float
    notes: List[str] = field(default_factory=list)


@dataclass
class ShotResult:
    ok: bool
    shot_id: int
    gate: dict
    T: Optional[np.ndarray] = None
    fiducial_residual_px: Optional[float] = None
    fiducials: List[dict] = field(default_factory=list)
    color: Optional[dict] = None
    rois: List[RoiResult] = field(default_factory=list)
    foreign_candidates: List[dict] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)


class _Timer:
    def __init__(self, store, key):
        self.store, self.key = store, key

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        self.store[self.key] = self.store.get(self.key, 0.0) + (time.perf_counter() - self.t) * 1e3


# ===================================================================== 유틸

def crop_padded(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """이미지 밖으로 나가는 부분은 0으로 채워 고정 크기 ROI를 보장."""
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    out = img[cy0:cy1, cx0:cx1]
    if (cx0, cy0, cx1, cy1) == (x0, y0, x1, y1):
        return out.copy()
    return cv2.copyMakeBorder(out, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1,
                              cv2.BORDER_CONSTANT, value=0)


def _subpixel_peak(res: np.ndarray, loc: Tuple[int, int]) -> Tuple[float, float]:
    """정규상관 맵 최댓값 주변 포물선 피팅으로 서브픽셀 보정."""
    x, y = loc
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        den = l - 2 * c + r
        if den < 0:
            dx = 0.5 * (l - r) / den
    if 0 < y < res.shape[0] - 1:
        u, c, d = res[y - 1, x], res[y, x], res[y + 1, x]
        den = u - 2 * c + d
        if den < 0:
            dy = 0.5 * (u - d) / den
    return float(dx), float(dy)


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama 닫힌해: 회전+균일배율+평행이동 (2점 이상)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    E = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        E[1, 1] = -1
    R = U @ E @ Vt
    scale = np.trace(np.diag(S) @ E) / (s ** 2).sum() * len(src)
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


# ===================================================================== 본체

class Preprocessor:
    def __init__(self, cad: BoardCAD, cfg: PipelineConfig, calib: Calibration):
        self.cad, self.cfg, self.calib = cad, cfg, calib
        self._hot_idx = None
        self._prepare_hot_pixels()
        if cfg.denoise.mode == "gaussian" and cfg.denoise.max_kernel_px < 3:
            raise ValueError("가우시안 커널 3px 미만은 의미 없음")

    # ------------------------------------------------------------ 0단계
    def gate(self, raw: np.ndarray) -> dict:
        g = self.cfg.gate
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        s = sharpness(gray)
        mx = cv2.max(cv2.max(raw[..., 0], raw[..., 1]), raw[..., 2])
        sat = float(np.count_nonzero(mx >= g.saturation_level)) / mx.size
        ratio = None if not self.calib.ref_sharpness else s / self.calib.ref_sharpness
        ok = ratio is None or ratio >= g.sharpness_ratio_min
        return dict(ok=ok, sharpness=s, sharpness_ratio=ratio, saturation_ratio=sat,
                    saturation_warn=sat > g.saturation_ratio_warn)

    # ------------------------------------------------------------ 1단계
    def _prepare_hot_pixels(self):
        hm = self.calib.hot_map
        if hm is None or not hm.any():
            return
        ys, xs = np.nonzero(hm)
        H, W = hm.shape
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        ny = np.clip(ys[:, None] + np.array([o[0] for o in offs]), 0, H - 1)
        nx = np.clip(xs[:, None] + np.array([o[1] for o in offs]), 0, W - 1)
        self._hot_idx = (ys, xs, ny, nx)

    def radiometric(self, raw: np.ndarray) -> np.ndarray:
        """I_c = (I − D) · gain,  gain = mean(F − D)/(F − D). 이후 핫픽셀만 8-이웃 중앙값으로 치환."""
        c = self.calib
        img = raw.astype(np.float32)
        if c.dark is not None:
            img = cv2.subtract(img, c.dark)
        if c.gain is not None:
            img = cv2.multiply(img, c.gain)
        out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
        if self._hot_idx is not None:
            ys, xs, ny, nx = self._hot_idx
            out[ys, xs] = np.median(out[ny, nx], axis=1).astype(np.uint8)
        return out

    # ------------------------------------------------------------ 2단계
    def undistort(self, img: np.ndarray, nearest: bool = False) -> np.ndarray:
        if self.calib.map1 is None:
            return img
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(img, self.calib.map1, self.calib.map2, interp,
                         borderMode=cv2.BORDER_CONSTANT)

    # ------------------------------------------------------------ 3단계
    def estimate_board_transform(self, gray: np.ndarray, shot: Shot,
                                 T_guess: np.ndarray = IDENTITY_T):
        """
        반환: (T 또는 None, 잔차 px, 검출 목록)
        T: 공칭 보드 mm → 실제 보드 mm. 이 shot에 피듀셜이 2개 미만이면 None
        (다른 shot에서 구한 T를 process_shot(T=...)로 전달해 재사용).
        """
        fc = self.cfg.fiducial
        tpl = self.calib.fiducial_template
        if tpl is None:
            raise RuntimeError("피듀셜 템플릿이 없습니다(calibration.build_fiducial_template).")
        th, tw = tpl.shape
        M = compose(shot.A_nom(), T_guess)
        H, W = gray.shape
        R = fc.search_radius_px
        dets = []
        for (fx, fy) in self.cad.fiducials:
            if not shot.contains_mm(fx, fy, margin_mm=self.cad.fiducial_diameter_mm):
                continue
            p = apply(M, [[fx, fy]])[0]
            x0 = int(round(p[0] - (tw - 1) / 2 - R))
            y0 = int(round(p[1] - (th - 1) / 2 - R))
            x1, y1 = x0 + tw + 2 * R, y0 + th + 2 * R
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue
            res = cv2.matchTemplate(gray[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            d = dict(nominal_mm=(fx, fy), ncc=float(mx), ok=mx >= fc.ncc_min)
            if d["ok"]:
                sx, sy = _subpixel_peak(res, loc)
                d["px"] = (x0 + loc[0] + sx + (tw - 1) / 2, y0 + loc[1] + sy + (th - 1) / 2)
            dets.append(d)
        good = [d for d in dets if d["ok"]]
        if len(good) < 2:
            return None, None, dets
        nom = np.array([d["nominal_mm"] for d in good])
        px = np.array([d["px"] for d in good])
        act_mm = px * shot.mm_per_px + np.array(shot.origin_mm)   # A_nom 역변환
        T = _similarity(nom, act_mm)
        pred = apply(compose(shot.A_nom(), T), nom)
        resid = float(np.sqrt(((pred - px) ** 2).sum(1)).max())
        return T, resid, dets

    # ------------------------------------------------------------ 4단계
    def color_stats(self, img: np.ndarray, shot: Shot, T: np.ndarray) -> dict:
        cc = self.cfg.color
        ds = cc.stats_downscale
        H, W = img.shape[:2]
        small = cv2.resize(img, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        mask = solder_mask_region(self.cad, shot, T, 1.0 / ds, cc.erode_px,
                                    (small.shape[1], small.shape[0]))
        n = int((mask > 0).sum())
        if n < 100:
            return dict(ok=False, n_px=n, shift=(0.0, 0.0), ab=None)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
        sel = lab[mask > 0].astype(np.float32)
        ab = (float(sel[:, 1].mean()), float(sel[:, 2].mean()))
        shift = (0.0, 0.0)
        if self.calib.ref_ab is not None:
            shift = tuple(float(np.clip(r - c, -cc.max_shift, cc.max_shift))
                          for r, c in zip(self.calib.ref_ab, ab))
        return dict(ok=True, n_px=n, ab=ab, L=float(sel[:, 0].mean()), shift=shift)

    def apply_color_shift(self, bgr: np.ndarray, color: dict) -> np.ndarray:
        """
        솔더마스크와 비슷한 색(현재 로트 평균 a*,b* 근처)일수록 큰 가중치로 이동.
        납땜(무채색)·실크(흰색)·부품(검정)의 색은 거의 건드리지 않음.
        """
        if not color or not color.get("ok") or color["shift"] == (0.0, 0.0):
            return bgr
        da, db = color["shift"]
        a0, b0 = color["ab"][0] - 128.0, color["ab"][1] - 128.0     # 8bit → a*, b*
        # float32 Lab에서 처리: 8bit Lab 왕복의 양자화 노이즈가 미세 결함 대비(CNR)를 깎지 않도록
        lab = cv2.cvtColor(bgr.astype(np.float32) * (1.0 / 255.0), cv2.COLOR_BGR2Lab)
        s = self.cfg.color.apply_sigma_ab
        w = np.exp(-((lab[..., 1] - a0) ** 2 + (lab[..., 2] - b0) ** 2) / (2 * s * s))
        lab[..., 1] += w * da
        lab[..., 2] += w * db
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255.0
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ 6단계
    def local_align(self, img: np.ndarray, geom: RoiGeom,
                    golden_bgr: np.ndarray) -> Tuple[float, float, str]:
        """
        골든 ROI 대비 평행이동량(dx, dy) 추정.
        부품 몸체는 마스크로 제외 → 기판 쪽 특징(패드 외곽, 실크, 비아)으로만 정렬.
        (몸체로 정렬하면 '부품 틀어짐' 불량이 정렬로 보정되어 사라짐)
        """
        lc = self.cfg.local_align
        method = lc.method
        if method == "none":
            return 0.0, 0.0, "none"
        if lc.ecc_only_fine_pitch and not geom.component.fine_pitch:
            method = "phase"
        crop = crop_padded(img, geom.x0, geom.y0, geom.x1, geom.y1)
        tg = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ig = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bmask = body_mask(geom, dilate_px=2)
        valid = cv2.bitwise_not(bmask)
        dx = dy = 0.0
        used = method
        if method == "ecc":
            warp = np.eye(2, 3, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, lc.iterations, lc.eps)
            try:
                _, warp = cv2.findTransformECC(tg, ig, warp, cv2.MOTION_TRANSLATION, crit,
                                               valid, lc.gauss_filt)
                dx, dy = float(warp[0, 2]), float(warp[1, 2])
            except cv2.error:
                used = "phase(ecc_fail)"
                method = "phase"
        if method == "phase":
            # 위상 상관은 마스크 미지원 → 몸체 영역을 각자의 평균 밝기로 채워 영향 제거
            m = valid > 0
            tg2, ig2 = tg.copy(), ig.copy()
            tg2[~m] = tg[m].mean()
            ig2[~m] = ig[m].mean()
            win = cv2.createHanningWindow((tg.shape[1], tg.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(tg2, ig2, win)
        if np.hypot(dx, dy) > lc.max_shift_px:
            return 0.0, 0.0, used + "(rejected)"
        return float(dx), float(dy), used

    # ------------------------------------------------------------ 8단계
    def denoise(self, bgr: np.ndarray) -> np.ndarray:
        d = self.cfg.denoise
        if d.mode == "off":
            return bgr
        if d.mode == "gaussian":
            return cv2.GaussianBlur(bgr, (3, 3), d.gaussian_sigma)
        if d.mode == "bilateral":
            if d.bilateral_d > d.max_kernel_px + 2:
                raise ValueError("양방향 필터 지름이 너무 큼: 미세 결함 손실 위험")
            return cv2.bilateralFilter(bgr, d.bilateral_d, d.bilateral_sigma_color,
                                       d.bilateral_sigma_space)
        raise ValueError(d.mode)

    # ------------------------------------------------------------ 9단계
    def rule_branch(self, bgr: np.ndarray) -> Dict[str, np.ndarray]:
        r = self.cfg.rule
        L = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[..., 0]
        clahe = cv2.createCLAHE(clipLimit=r.clahe_clip, tileGridSize=(r.clahe_tile,) * 2)
        Lc = clahe.apply(L)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (r.tophat_kernel_px,) * 2)
        tophat = cv2.morphologyEx(Lc, cv2.MORPH_TOPHAT, k)
        return dict(clahe_L=Lc, tophat=tophat)

    # ------------------------------------------------------------ 10단계
    @staticmethod
    def golden_diff(bgr: np.ndarray, golden_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if golden_bgr is None:
            return None
        return cv2.absdiff(bgr, golden_bgr).max(axis=2)

    # ------------------------------------------------------------ 11단계
    def model_inputs(self, bgr, sat, diff, comp: Component) -> List[Tuple[np.ndarray, dict]]:
        mc = self.cfg.model_input
        h, w = bgr.shape[:2]
        if diff is None:
            diff = np.zeros((h, w), np.uint8)
        x = np.dstack([bgr.astype(np.float32) / 255.0,
                       sat.astype(np.float32) / 255.0,
                       diff.astype(np.float32) / 255.0])          # H×W×5 (B,G,R,SAT,DIFF)
        if mc.mean is not None and mc.std is not None:
            x = (x - np.array(mc.mean, np.float32)) / np.array(mc.std, np.float32)

        def pad_to(a, S):
            ph, pw = S - a.shape[0], S - a.shape[1]
            return cv2.copyMakeBorder(a, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

        S = mc.small_size
        if comp.input_mode == "pad":
            if h <= S and w <= S:
                return [(pad_to(x, S).transpose(2, 0, 1).copy(), dict(x=0, y=0, scale=1.0, valid=(h, w)))]
            s = min(S / h, S / w)
            if s >= mc.min_scale:
                xr = cv2.resize(x, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                if xr.ndim == 2:
                    xr = xr[..., None]
                return [(pad_to(xr, S).transpose(2, 0, 1).copy(),
                         dict(x=0, y=0, scale=s, valid=xr.shape[:2]))]
            # 축소 배율 하한 미달 → 타일 방식으로 전환 (미세 결함 보존)
        Tz, step = mc.tile_size, mc.tile_size - mc.tile_overlap
        ys = list(range(0, max(h - Tz, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - Tz, 0) + 1, step)) or [0]
        if ys[-1] + Tz < h:
            ys.append(h - Tz)
        if xs[-1] + Tz < w:
            xs.append(w - Tz)
        out = []
        for ty in ys:
            for tx in xs:
                t = x[max(ty, 0):ty + Tz, max(tx, 0):tx + Tz]
                out.append((pad_to(t, Tz).transpose(2, 0, 1).copy(),
                            dict(x=max(tx, 0), y=max(ty, 0), scale=1.0, valid=t.shape[:2])))
        return out

    # ------------------------------------------------------------ 0~4단계 묶음
    def prepare(self, raw: np.ndarray, shot: Shot, T: Optional[np.ndarray] = None,
                timings: Optional[dict] = None):
        """반환: img(보정 BGR), sat(포화 마스크), T, 잔차, 피듀셜, color(dict)"""
        t = timings if timings is not None else {}
        lvl = self.cfg.gate.saturation_level
        with _Timer(t, "1_radiometric"):
            # 포화는 '원본' 기준: 보정 전에 이미 255였던 화소만 정보 손실로 간주
            mx = cv2.max(cv2.max(raw[..., 0], raw[..., 1]), raw[..., 2])
            _, sat_raw = cv2.threshold(mx, lvl - 1, 255, cv2.THRESH_BINARY)
            img = self.radiometric(raw)
        with _Timer(t, "2_undistort"):
            img = self.undistort(img)
            sat = self.undistort(sat_raw, nearest=True)
        resid, fids = None, []
        if T is None:
            with _Timer(t, "3_global_align"):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                T, resid, fids = self.estimate_board_transform(gray, shot)
        color = None
        if T is not None:
            with _Timer(t, "4_color_stats"):
                color = self.color_stats(img, shot, T)
        return img, sat, T, resid, fids, color

    def process_roi(self, img, sat_full, geom: RoiGeom, golden: Optional[np.ndarray],
                    color: dict, t: dict, rule: bool = True, make_inputs: bool = True) -> RoiResult:
        notes = []
        dx = dy = 0.0
        method = "none(no_golden)"
        if golden is not None:
            with _Timer(t, "6_local_align"):
                dx, dy, method = self.local_align(img, geom, golden)
        with _Timer(t, "5_roi_extract"):
            ix, iy = int(round(dx)), int(round(dy))
            g = RoiGeom(geom.ref, geom.x0 + ix, geom.y0 + iy, geom.x1 + ix, geom.y1 + iy,
                          geom.body_poly, geom.parallax_px, geom.component)
            # 보간 없이 정수 화소로 다시 잘라냄 (잔차 ≤ 0.5px)
            bgr = crop_padded(img, g.x0, g.y0, g.x1, g.y1)
        with _Timer(t, "4_color_apply"):
            bgr = self.apply_color_shift(bgr, color)
        with _Timer(t, "7_saturation"):
            s = crop_padded(sat_full, g.x0, g.y0, g.x1, g.y1)
            s = cv2.dilate(s, np.ones((3, 3), np.uint8))
            sat_ratio = float((s > 0).mean())
        with _Timer(t, "8_denoise"):
            bgr = self.denoise(bgr)
        rb = None
        if rule and self.cfg.rule.enabled and (g.component.fine_pitch or golden is None):
            with _Timer(t, "9_rule_branch"):
                rb = self.rule_branch(bgr)
        with _Timer(t, "10_golden_diff"):
            diff = self.golden_diff(bgr, golden)
        mi = []
        if make_inputs:
            with _Timer(t, "11_model_input"):
                mi = self.model_inputs(bgr, s, diff, g.component)
        if golden is None:
            notes.append("골든 ROI 없음: 국부 정렬·차분 생략")
        return RoiResult(g.ref, g, bgr, s, diff, rb, mi, (dx, dy), method, sat_ratio, notes)

    # ------------------------------------------------------------ 전체
    def process_shot(self, raw: np.ndarray, shot: Shot, T: Optional[np.ndarray] = None,
                     run_envelope: bool = True) -> ShotResult:
        t: Dict[str, float] = {}
        t0 = time.perf_counter()
        with _Timer(t, "0_gate"):
            gate = self.gate(raw)
        res = ShotResult(ok=False, shot_id=shot.shot_id, gate=gate, timings_ms=t)
        if not gate["ok"]:
            res.messages.append("초점 품질 미달 → 재촬영")
            return res
        if gate["saturation_warn"]:
            res.messages.append(f"포화 화소 {gate['saturation_ratio']:.1%}: 조명/노출 점검 필요")

        img, sat, T, resid, fids, color = self.prepare(raw, shot, T, t)
        res.fiducials, res.fiducial_residual_px = fids, resid
        if T is None:
            res.messages.append("피듀셜 2개 미만 검출 → 다른 shot의 T를 전달하거나 재촬영")
            return res
        if resid is not None and resid > self.cfg.fiducial.max_residual_px:
            res.messages.append(f"피듀셜 잔차 {resid:.2f}px > 허용치 → 재촬영")
            return res
        if resid is not None and len([f for f in fids if f["ok"]]) == 2:
            res.messages.append("피듀셜 2개: 잔차 검증 불가(3개 이상 권장)")
        res.T, res.color = T, color

        with _Timer(t, "5_roi_geometry"):
            geoms = compute_rois(self.cad, shot, T, self.cfg.optics.working_distance_mm,
                                   self.cfg.roi.margin_mm)
        golden = self.calib.golden_rois.get(shot.shot_id, {})
        for g in geoms:
            res.rois.append(self.process_roi(img, sat, g, golden.get(g.ref), color, t))

        if run_envelope and shot.shot_id in self.calib.envelopes:
            with _Timer(t, "E_envelope"):
                res.foreign_candidates = self.detect_foreign(img, shot, T)
        t["total"] = (time.perf_counter() - t0) * 1e3
        res.ok = True
        return res

    # ===================================================== 별도 경로: 랜덤 이물
    def to_nominal(self, img: np.ndarray, shot: Shot, T: np.ndarray,
                   interp=cv2.INTER_LINEAR) -> np.ndarray:
        """실제 이미지를 공칭 배치 좌표계로 워핑 (이 경로에서만 사용: 이물 후보는 0.1mm 이상)."""
        A = to3(shot.A_nom())
        W = (A @ to3(T) @ np.linalg.inv(A))[:2]          # 공칭 px → 실제 px
        h, w = img.shape[:2]
        return cv2.warpAffine(img, W, (w, h), flags=interp | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def detect_foreign(self, img: np.ndarray, shot: Shot, T: np.ndarray) -> List[dict]:
        ec = self.cfg.envelope
        lo, hi = self.calib.envelopes[shot.shot_id]
        nom = self.to_nominal(img, shot, T)
        gray = cv2.cvtColor(nom, cv2.COLOR_BGR2GRAY).astype(np.int16)
        out = ((gray < lo.astype(np.int16) - ec.tolerance) |
               (gray > hi.astype(np.int16) + ec.tolerance)).astype(np.uint8) * 255
        # 부품 ROI는 ROI 경로가 담당 → 제외 (공칭 좌표계이므로 T=I로 계산)
        for g in compute_rois(self.cad, shot, IDENTITY_T, self.cfg.optics.working_distance_mm,
                                self.cfg.roi.margin_mm):
            d = ec.roi_exclude_dilate_px
            cv2.rectangle(out, (g.x0 - d, g.y0 - d), (g.x1 + d, g.y1 + d), 0, -1)
        n, lab_img, stats, cents = cv2.connectedComponentsWithStats(out, connectivity=8)
        lab = cv2.cvtColor(nom, cv2.COLOR_BGR2Lab)
        cands = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < ec.min_area_px:
                continue
            m = lab_img[y:y + h, x:x + w] == i
            L, a, b = (float(lab[y:y + h, x:x + w][..., k][m].mean()) for k in range(3))
            flux = L >= ec.flux_L_min and b >= ec.flux_b_min
            # 미검 최소화: 플럭스 의심도 버리지 않고 '등급만 낮춰' 보고
            cands.append(dict(bbox_nominal_px=(int(x), int(y), int(w), int(h)), area=int(area),
                              centroid=tuple(cents[i]), Lab=(L, a, b), flux_suspect=flux))
        return cands


# ===================================================================== 기준값 생성 (골든 보드 사용)

def build_reference_ab(pre: Preprocessor, raws: List[np.ndarray], shot: Shot) -> Tuple[float, float]:
    """기준 로트 양품 보드들의 솔더마스크 평균 a*, b*."""
    saved = pre.calib.ref_ab
    pre.calib.ref_ab = None
    abs_ = []
    for raw in raws:
        _, _, T, _, _, color = pre.prepare(raw, shot)
        if color and color["ok"]:
            abs_.append(color["ab"])
    pre.calib.ref_ab = saved
    return tuple(np.mean(abs_, axis=0).tolist())


def build_golden_rois(pre: Preprocessor, raws: List[np.ndarray], shot: Shot,
                      Ts: Optional[List[np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """
    양품 보드 N장(20장 이상 권장) → 부품별 골든 ROI(중앙값).
    추론과 '같은 코드'로 0~8단계를 거친 ROI를 사용합니다(Train-Serve 불일치 방지).
    1차: 첫 보드를 임시 기준으로 나머지를 국부 정렬 → 2차: 중앙값.
    """
    prepared = [pre.prepare(r, shot, None if Ts is None else Ts[i]) for i, r in enumerate(raws)]
    t = {}
    stacks: Dict[str, List[np.ndarray]] = {}
    ref0: Dict[str, np.ndarray] = {}
    for k, (img, sat, T, _, _, color) in enumerate(prepared):
        if T is None:
            continue
        geoms = compute_rois(pre.cad, shot, T, pre.cfg.optics.working_distance_mm,
                               pre.cfg.roi.margin_mm)
        for g in geoms:
            r = pre.process_roi(img, sat, g, ref0.get(g.ref), color, t,
                                rule=False, make_inputs=False)
            if g.ref not in ref0:
                ref0[g.ref] = r.bgr
            stacks.setdefault(g.ref, []).append(r.bgr)
    return {ref: np.median(np.stack(v), axis=0).astype(np.uint8) for ref, v in stacks.items()}


def build_envelope(pre: Preprocessor, raws: List[np.ndarray], shot: Shot):
    """
    양품 보드 화소별 min/max 범위. 정렬 잔차(±1px)로 실크·패드 가장자리가
    과검되지 않도록 공간 방향으로도 범위를 넓힘(min은 침식, max는 팽창).
    """
    grays = []
    for raw in raws:
        img, _, T, _, _, _ = pre.prepare(raw, shot)
        if T is None:
            continue
        grays.append(cv2.cvtColor(pre.to_nominal(img, shot, T), cv2.COLOR_BGR2GRAY))
    st = np.stack(grays)
    k = 2 * pre.cfg.envelope.spatial_tol_px + 1
    ker = np.ones((k, k), np.uint8)
    return cv2.erode(st.min(0), ker), cv2.dilate(st.max(0), ker)

# ############################################################################
# 학습 전용 증강 · 희소 불량 합성 · 데이터 분할
# ############################################################################

@dataclass
class AugConfig:
    shift_px: int = 3                    # 국부 정렬 잔차 흉내
    rot_deg: float = 1.0                 # 틀어짐 허용 한계보다 작게
    brightness: float = 0.10             # ±10%
    contrast: float = 0.10
    ab_shift_range: Tuple[float, float] = (6.0, 6.0)   # 실제 로트 간 편차 측정값으로 교체
    noise_sigma: Tuple[float, float] = (1.0, 2.0)
    blur_sigma_max: float = 0.5
    p_rot: float = 0.5
    p_flip: float = 0.5
    p_blur: float = 0.2
    p_color: float = 0.7
    p_noise: float = 0.5


FORBIDDEN = {"scale", "strong_blur", "cutout", "random_erasing", "elastic"}


def augment_roi(bgr: np.ndarray, polarized: bool, rng: np.random.Generator,
                cfg: AugConfig = AugConfig(), sat: Optional[np.ndarray] = None,
                solder_mask_ab: Optional[Tuple[float, float]] = None):
    """
    bgr: 전처리 완료 ROI. sat: 포화 마스크(기하 변환만 동일하게 적용).
    solder_mask_ab: 기준 솔더마스크 (a,b). 주어지면 그 색 근처 화소만 색조 이동.
    반환: (bgr_aug, sat_aug)
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()
    s = None if sat is None else sat.copy()

    # --- 기하 (작은 이동/회전: 반사 패턴 왜곡을 줄이기 위해 회전은 확률적으로만)
    ang = rng.uniform(-cfg.rot_deg, cfg.rot_deg) if rng.random() < cfg.p_rot else 0.0
    tx, ty = rng.integers(-cfg.shift_px, cfg.shift_px + 1, size=2)
    if ang != 0.0 or tx or ty:
        M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), ang, 1.0)
        M[:, 2] += (tx, ty)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if s is not None:
            s = cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # --- 반전: 무극성 부품만
    if not polarized and rng.random() < cfg.p_flip:
        code = int(rng.integers(-1, 2))         # -1: 양축, 0: 상하, 1: 좌우
        out = cv2.flip(out, code)
        if s is not None:
            s = cv2.flip(s, code)

    geo = out.copy()   # 기하 변환만 적용된 상태(포화 화소 복원용)

    # --- 광학적
    f = out.astype(np.float32)
    f = (f - 128) * (1 + rng.uniform(-cfg.contrast, cfg.contrast)) + 128
    f = f * (1 + rng.uniform(-cfg.brightness, cfg.brightness))
    out = np.clip(f, 0, 255).astype(np.uint8)

    if rng.random() < cfg.p_color:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2Lab).astype(np.float32)
        da = rng.uniform(-cfg.ab_shift_range[0], cfg.ab_shift_range[0])
        db = rng.uniform(-cfg.ab_shift_range[1], cfg.ab_shift_range[1])
        if solder_mask_ab is not None:
            wgt = np.exp(-((lab[..., 1] - solder_mask_ab[0]) ** 2 +
                           (lab[..., 2] - solder_mask_ab[1]) ** 2) / (2 * 12.0 ** 2))
        else:
            wgt = 1.0
        lab[..., 1] += wgt * da
        lab[..., 2] += wgt * db
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    if rng.random() < cfg.p_blur:
        sig = rng.uniform(0.2, cfg.blur_sigma_max)
        out = cv2.GaussianBlur(out, (3, 3), sig)

    if rng.random() < cfg.p_noise:
        sig = rng.uniform(*cfg.noise_sigma)
        out = np.clip(out.astype(np.float32) + rng.normal(0, sig, out.shape), 0, 255).astype(np.uint8)

    # 원래 포화였던 화소는 포화 상태 유지 (밝기 감소로 '정보가 있는 것처럼' 보이지 않도록)
    if s is not None:
        out[s > 0] = geo[s > 0]
    return out, s


def synth_reverse_polarity(roi_bgr: np.ndarray, body_rect_px: Tuple[int, int, int, int]) -> np.ndarray:
    """
    역삽 불량 합성: 극성 부품의 몸체 영역만 180° 회전(np.rot90 2회 → 보간 없음).
    body_rect_px: ROI 좌표계 (x0, y0, x1, y1). 패드·납땜은 그대로 유지.
    """
    x0, y0, x1, y1 = body_rect_px
    out = roi_bgr.copy()
    out[y0:y1, x0:x1] = np.rot90(roi_bgr[y0:y1, x0:x1], 2)
    return out


def split_by_group(groups, val_ratio=0.2, seed=0):
    """
    보드(또는 로트) 단위 분할. groups: 샘플별 보드ID/로트ID 리스트.
    같은 보드의 6개 촬영·같은 부품 크롭이 학습/검증에 나뉘어 들어가는 누수를 방지.
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    nval = max(1, int(round(len(uniq) * val_ratio)))
    val_groups = set(uniq[:nval].tolist())
    val = np.array([x in val_groups for x in g])
    return np.nonzero(~val)[0], np.nonzero(val)[0]

# ############################################################################
# 검증 지표
# ############################################################################

def flat_uniformity(img: np.ndarray, border_frac: float = 0.02, block: int = 64) -> float:
    """균일 회색판 이미지의 블록 평균 기준 (max−min)/max. 목표 ≤ 0.05."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    b = int(min(h, w) * border_frac)
    g = g[b:h - b, b:w - b].astype(np.float32)
    small = cv2.resize(g, (max(1, g.shape[1] // block), max(1, g.shape[0] // block)),
                       interpolation=cv2.INTER_AREA)
    return float((small.max() - small.min()) / max(small.max(), 1e-6))


def lab_mean_to_cielab(ab_or_lab: Sequence[float]) -> np.ndarray:
    """OpenCV 8bit Lab(L*255/100, a+128, b+128) → CIE L*a*b*."""
    v = np.asarray(ab_or_lab, np.float64)
    if v.size == 3:
        return np.array([v[0] * 100 / 255, v[1] - 128, v[2] - 128])
    return np.array([v[0] - 128, v[1] - 128])


def delta_e76(lab1, lab2) -> float:
    """CIE76 ΔE. 입력은 CIE L*a*b* (또는 a*,b*만)."""
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def cnr(img: np.ndarray, defect_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    """대비 대 노이즈 비: |μ_d − μ_b| / σ_b.  단계 전후로 떨어지면 안 됨."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g.astype(np.float32)
    d, b = g[defect_mask > 0], g[bg_mask > 0]
    return float(abs(d.mean() - b.mean()) / (b.std() + 1e-6))


def rule_of_three_upper(n_defects_tested: int) -> float:
    """미검 0건일 때 실제 미검률의 95% 신뢰 상한 ≈ 3/n."""
    return 3.0 / max(n_defects_tested, 1)


def defects_needed_for(target_fnr: float) -> int:
    return int(np.ceil(3.0 / target_fnr))


def zero_miss_operating_point(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    불량 점수(높을수록 불량) 기준, 미검 0을 유지하는 최대 임계값과 그때의 과검률.
    반환: (threshold, FPR)
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if not labels.any():
        raise ValueError("불량 샘플이 없습니다")
    thr = scores[labels].min()
    fpr = float((scores[~labels] >= thr).mean()) if (~labels).any() else 0.0
    return float(thr), fpr


def fnr_fpr(scores, labels, thr) -> Tuple[float, float]:
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pred = scores >= thr
    fnr = float((~pred[labels]).mean()) if labels.any() else 0.0
    fpr = float(pred[~labels].mean()) if (~labels).any() else 0.0
    return fnr, fpr


# ############################################################################
# 합성 데이터 검증 데모
# ############################################################################

def run_demo(out_png: str = "demo_panel.png"):
    """합성 PCB로 전 과정을 검증. 현장 조건(비네팅 25%, 핫픽셀, 납땜 포화, 로트 색 편차,
    위치·회전 편차, 국부 휨, 시차, 0.15mm 브릿지, 이물·플럭스)을 흉내 냄.
    ※ 렌즈 왜곡은 합성하지 않음(항등 맵)."""

    RNG = np.random.default_rng(7)
    W, H = 2448, 2048
    MMPX = 0.02
    WD = 110.0

    LOTS = {"A": (45, 125, 35), "B": (30, 128, 55)}   # BGR 솔더마스크 (B 로트: 노란 기미)


    # ============================================================ CAD 정의

    def chip0402(ref, x, y, polarized=False):
        pads = [Rect(x - 0.65, y - 0.28, x - 0.2, y + 0.28), Rect(x + 0.2, y - 0.28, x + 0.65, y + 0.28)]
        return Component(ref, pads, Rect(x - 0.5, y - 0.25, x + 0.5, y + 0.25), 0.35, polarized)


    def qfn(ref, x, y, n=10, pitch=0.4, lead_w=0.25, body=5.0):
        pads = []
        half = body / 2
        off = (np.arange(n) - (n - 1) / 2) * pitch
        for o in off:   # 리드 간 간격 = 0.4 − 0.25 = 0.15mm (≈7.5px)
            pads.append(Rect(x + o - lead_w / 2, y - half - 0.3, x + o + lead_w / 2, y - half + 0.3))
            pads.append(Rect(x + o - lead_w / 2, y + half - 0.3, x + o + lead_w / 2, y + half + 0.3))
            pads.append(Rect(x - half - 0.3, y + o - lead_w / 2, x - half + 0.3, y + o + lead_w / 2))
            pads.append(Rect(x + half - 0.3, y + o - lead_w / 2, x + half + 0.3, y + o + lead_w / 2))
        return Component(ref, pads, Rect(x - half + 0.35, y - half + 0.35, x + half - 0.35, y + half - 0.35),
                           0.9, polarized=True, fine_pitch=True, input_mode="tile")


    def ecap(ref, x, y):
        pads = [Rect(x - 3.4, y - 0.8, x - 1.4, y + 0.8), Rect(x + 1.4, y - 0.8, x + 3.4, y + 0.8)]
        return Component(ref, pads, Rect(x - 3.15, y - 3.15, x + 3.15, y + 3.15), 8.0, polarized=True,
                           input_mode="pad")


    def make_cad():
        comps = []
        k = 0
        for j in range(3):
            for i in range(5):
                k += 1
                comps.append(chip0402(f"R{k}", 8 + i * 3.0, 10 + j * 3.0))
        comps.append(qfn("U1", 27, 20))
        comps.append(ecap("C1", 40, 32))
        comps.append(chip0402("D1", 12, 32, polarized=True))
        silk = [Rect(7, 7.6, 9.5, 8.0), Rect(22, 15.6, 25, 16.0), Rect(35, 26.0, 38, 26.4)]
        return BoardCAD(100, 80, comps, fiducials=[(5, 5), (46, 5), (5, 38)], silkscreen=silk,
                          fiducial_diameter_mm=2.0)


    SHOT = Shot(0, origin_mm=(1.0, 1.0), mm_per_px=MMPX, image_size=(W, H))


    # ============================================================ 렌더러

    def random_T(max_t=0.5, max_deg=0.3):
        a = np.deg2rad(RNG.uniform(-max_deg, max_deg))
        cx, cy = 25, 20                     # 회전 중심(보드 위 임의점)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        t = np.array([cx, cy]) - R @ [cx, cy] + RNG.uniform(-max_t, max_t, 2)
        return np.hstack([R, t[:, None]])


    def _poly(canvas, pts_px, color):
        p = np.round(pts_px * 16).astype(np.int32)
        cv2.fillConvexPoly(canvas, p, color, lineType=cv2.LINE_AA, shift=4)


    def render_scene(cad, T, lot="A", bridge=False, warp_px=(0.0, 0.0), foreign=False, flux=False):
        """반환: float32 장면(255 초과 = 포화될 반사), 진실값 dict"""
        M = compose(SHOT.A_nom(), T)
        sc = np.zeros((H, W, 3), np.float32)
        sc[:] = LOTS[lot]
        # 솔더마스크 미세 텍스처(정렬·노이즈 현실감)
        tex = cv2.GaussianBlur(RNG.normal(0, 6, (H // 4, W // 4)).astype(np.float32), (0, 0), 2)
        sc += cv2.resize(tex, (W, H))[..., None]
        # 비아
        for vx in np.arange(4, 48, 2.5):
            for vy in [44.0, 3.0]:
                c = apply(M, [[vx, vy]])[0]
                cv2.circle(sc, tuple(np.round(c).astype(int)), 9, (60, 90, 110), -1, cv2.LINE_AA)
        for r in cad.silkscreen:
            _poly(sc, apply(M, r.corners()), (235, 235, 235))
        for fx, fy in cad.fiducials:
            c = apply(M, [[fx, fy]])[0]
            cv2.circle(sc, tuple(np.round(c).astype(int)), int(1.0 / MMPX), (110, 170, 215), -1, cv2.LINE_AA)
        center = np.array([(W - 1) / 2, (H - 1) / 2])
        truth = {}
        for comp in cad.components:
            Mc = M.copy()
            if comp.ref == "U1":            # 국부 휨: QFN 영역만 추가 이동
                Mc[:, 2] += warp_px
            for pad in comp.pads:
                pp = apply(Mc, pad.corners())
                _poly(sc, pp, (175, 175, 180))
                inner = pp.mean(0) + (pp - pp.mean(0)) * 0.45
                _poly(sc, inner, (320, 320, 320))          # 경면 하이라이트 → 포화
            body = apply(Mc, comp.body.corners())
            par = (body.mean(0) - center) * comp.height_mm / WD
            col = {"U1": (28, 28, 30), "C1": (90, 60, 45)}.get(comp.ref, (45, 45, 50))
            _poly(sc, body + par, col)                      # 위에서 본 상단(시차 반영)
            if comp.polarized:
                b = comp.body
                mark = Rect(b.x0, b.y0, b.x0 + (b.x1 - b.x0) * 0.2, b.y1)
                _poly(sc, apply(Mc, mark.corners()) + par, (200, 200, 200))
        if bridge:
            u1 = next(c for c in cad.components if c.ref == "U1")
            top = sorted([p for p in u1.pads if p.y0 < 20 - 2.5], key=lambda p: p.x0)
            a, b = top[4], top[5]                            # 상단 5,6번 리드 사이
            br = Rect(a.x1, a.y0 + 0.1, b.x0, a.y1 - 0.1)
            Mc = M.copy(); Mc[:, 2] += warp_px
            _poly(sc, apply(Mc, br.corners()), (215, 215, 220))
            gap_ok = Rect(top[1].x1, a.y0 + 0.1, top[2].x0, a.y1 - 0.1)     # 정상 간격(비교용)
            truth["bridge_px"] = apply(Mc, br.corners())
            truth["gap_px"] = apply(Mc, gap_ok.corners())
        if foreign:
            c = apply(M, [[18, 24]])[0]
            cv2.circle(sc, tuple(np.round(c).astype(int)), 6, (20, 20, 20), -1, cv2.LINE_AA)   # 0.24mm 이물
            truth["foreign_px"] = c
        if flux:
            c = apply(M, [[33, 40]])[0]
            cv2.ellipse(sc, tuple(np.round(c).astype(int)), (14, 9), 20, 0, 360, (150, 215, 230), -1, cv2.LINE_AA)
            truth["flux_px"] = c
        truth["T"] = T
        return sc, truth


    # 카메라 고정 패턴(보드마다 동일)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    VIGNETTE = (1.0 - 0.25 * r2)[..., None]          # 모서리 25% 감광
    DARK = np.full((H, W, 3), 4.0, np.float32)
    hy, hx = RNG.integers(0, H, 300), RNG.integers(0, W, 300)
    DARK[hy, hx] += 90                                 # 핫픽셀(38℃ 하우징)


    def camera(scene):
        img = scene * VIGNETTE + DARK + RNG.normal(0, 1.5, scene.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


    # ============================================================ 메인

    def main(out_png="demo_panel.png"):
        t_all = time.perf_counter()
        cad = make_cad()
        cfg = PipelineConfig()
        cfg.optics.working_distance_mm = WD
        calib = Calibration()
        print("=" * 70)
        print("[보정 기준값 생성]")

        # 1단계: 다크/플랫
        darks = [camera(np.zeros((H, W, 3), np.float32)) for _ in range(8)]
        flats = [camera(np.full((H, W, 3), 170, np.float32)) for _ in range(8)]
        calib.dark, calib.gain, calib.hot_map = build_dark_flat(darks, flats, cfg.flat.flat_sigma_px,
                                                                cfg.flat.hot_pixel_k)
        print(f"  핫픽셀 검출 {int(calib.hot_map.sum())}개 (주입 300개, 일부 중복 위치 포함)")
        # 2단계: 렌즈 왜곡 (합성 데이터에는 왜곡 없음 → 항등 맵)
        calib.map1, calib.map2 = identity_maps((W, H))

        pre = Preprocessor(cad, cfg, calib)
        test_flat = camera(np.full((H, W, 3), 170, np.float32))
        u0 = flat_uniformity(test_flat)
        u1 = flat_uniformity(pre.radiometric(test_flat))
        print(f"  플랫 균일도 편차: 보정 전 {u0:.1%} → 보정 후 {u1:.1%} (목표 ≤ 5%)")

        # 3단계: 피듀셜 템플릿 (공칭 위치에 놓은 골든 보드에서)
        sc0, _ = render_scene(cad, IDENTITY_T, "A")
        g0 = cv2.cvtColor(pre.undistort(pre.radiometric(camera(sc0))), cv2.COLOR_BGR2GRAY)
        fpx = apply(SHOT.A_nom(), [cad.fiducials[0]])[0]
        calib.fiducial_template = build_fiducial_template(g0, fpx, diameter_px=cad.fiducial_diameter_mm / MMPX)
        calib.ref_sharpness = sharpness(g0)

        # 4·10단계 + Envelope: 기준 로트(A) 양품 보드
        goods = []
        for _ in range(8):
            sc, _ = render_scene(cad, random_T(), "A", warp_px=RNG.uniform(-2, 2, 2))
            goods.append(camera(sc))
        calib.ref_ab = build_reference_ab(pre, goods, SHOT)
        calib.golden_rois[SHOT.shot_id] = build_golden_rois(pre, goods, SHOT)
        calib.envelopes[SHOT.shot_id] = build_envelope(pre, goods, SHOT)
        print(f"  기준 솔더마스크 (a,b) = ({calib.ref_ab[0]:.1f}, {calib.ref_ab[1]:.1f}), "
              f"골든 ROI {len(calib.golden_rois[0])}개")

        # ============================== 검사: B 로트, 브릿지 + 국부 휨 + 이물 + 플럭스
        print("=" * 70)
        print("[검사 실행: B 로트 · 브릿지 · 국부 휨(+3,−2px) · 이물 · 플럭스]")
        T_true = random_T()
        warp = np.array([3.0, -2.0])
        sc, truth = render_scene(cad, T_true, "B", bridge=True, warp_px=warp, foreign=True, flux=True)
        raw = camera(sc)
        pre.process_shot(raw, SHOT)                    # 워밍업(최초 호출 오버헤드 제외)
        res = pre.process_shot(raw, SHOT)
        print(f"  성공: {res.ok}  메시지: {res.messages}")

        # (3) 전역 정렬 정확도
        probe = np.array([[8, 10], [27, 20], [40, 32], [45, 40]], float)
        err = np.linalg.norm(apply(compose(SHOT.A_nom(), res.T), probe) -
                             apply(compose(SHOT.A_nom(), T_true), probe), axis=1)
        print(f"  [3] 피듀셜 잔차 {res.fiducial_residual_px:.3f}px, CAD 투영 오차 최대 {err.max():.3f}px (목표 ≤ 1px)")

        # (4) 색 정규화 효과
        col = res.color
        ref_lab = lab_mean_to_cielab(calib.ref_ab)
        cur_lab = lab_mean_to_cielab(col["ab"])
        post_ab = np.array(col["ab"]) + np.array(col["shift"])
        print(f"  [4] 솔더마스크 ΔE(a*b*): 보정 전 {delta_e76(cur_lab, ref_lab):.2f} → "
              f"보정 후 {delta_e76(lab_mean_to_cielab(post_ab), ref_lab):.2f}")
        # ROI 실측: 정규화 후 ROI의 솔더마스크 화소 색
        r1 = next(r for r in res.rois if r.ref == "R1")
        lab = cv2.cvtColor(r1.bgr, cv2.COLOR_BGR2Lab)[:6, :, 1:].reshape(-1, 2).mean(0)
        print(f"      R1 ROI 가장자리(솔더마스크) 실측 a*b* = ({lab[0]-128:.1f}, {lab[1]-128:.1f}) / "
              f"기준 ({calib.ref_ab[0]-128:.1f}, {calib.ref_ab[1]-128:.1f})")

        # (6) 국부 정렬: 주입한 휨(+3,−2) 복원 여부 (부호 규약 검증)
        u1r = next(r for r in res.rois if r.ref == "U1")
        print(f"  [6] U1 국부 정렬량 ({u1r.local_shift[0]:+.2f}, {u1r.local_shift[1]:+.2f})px, "
              f"방법={u1r.align_method} / 주입 (+3.00, −2.00)")
        others = [np.hypot(*r.local_shift) for r in res.rois if r.ref != "U1"]
        print(f"      휨 없는 부품들의 국부 정렬량 최대 {max(others):.2f}px "
              f"(정수 크롭 양자화 ±0.5px + 전역 정렬 잔차 수준)")

        # (6번 검증) 결함 보존: 브릿지 CNR — 원본 vs 최종 ROI
        def masks_in(geom_x0, geom_y0, shape):
            dm = np.zeros(shape, np.uint8); bm = np.zeros(shape, np.uint8)
            off = np.array([geom_x0, geom_y0])
            cv2.fillConvexPoly(dm, np.round(truth["bridge_px"] - off).astype(np.int32), 255)
            cv2.fillConvexPoly(bm, np.round(truth["gap_px"] - off).astype(np.int32), 255)
            k = np.ones((3, 3), np.uint8)
            return cv2.erode(dm, k), cv2.erode(bm, k)
        g = u1r.geom
        raw_roi = raw[g.y0:g.y1, g.x0:g.x1]
        dm, bm = masks_in(g.x0, g.y0, raw_roi.shape[:2])
        c_raw = cnr(raw_roi, dm, bm)
        c_fin = cnr(u1r.bgr, dm, bm)
        print(f"      브릿지 CNR(정상 간격 대비): 원본 {c_raw:.1f} → 전처리 후 {c_fin:.1f} "
              f"({'보존' if c_fin >= 0.9 * c_raw else '저하!'})")
        dd = u1r.diff.astype(float)
        print(f"  [10] 골든 차분: 브릿지 영역 평균 {dd[dm > 0].mean():.0f}, "
              f"정상 간격 평균 {dd[bm > 0].mean():.0f}")
        print(f"  [7] U1 포화 화소 비율 {u1r.sat_ratio:.1%}")

        # (11) 모델 입력
        shapes = {r.ref: (len(r.model_inputs), r.model_inputs[0][0].shape) for r in res.rois
                  if r.ref in ("R1", "U1", "C1", "D1")}
        print(f"  [11] 모델 입력(개수, CHW): {shapes}")

        # 별도 경로: 이물·플럭스
        print(f"  [E] 이물 후보 {len(res.foreign_candidates)}개:")
        for c in res.foreign_candidates:
            print(f"      중심 {tuple(round(v) for v in c['centroid'])}, 면적 {c['area']}px, "
                  f"플럭스 의심={c['flux_suspect']}")
        fp, xp = np.round(truth['foreign_px']).astype(int), np.round(truth['flux_px']).astype(int)
        print(f"      (진실값(실제 px): 이물 ({fp[0]}, {fp[1]}), 플럭스 ({xp[0]}, {xp[1]}) "
              f"— 후보는 공칭 좌표라 보드 이동량만큼 차이)")

        # 과검 확인: B 로트 양품
        sc_g, _ = render_scene(cad, random_T(), "B", warp_px=RNG.uniform(-2, 2, 2))
        res_g = pre.process_shot(camera(sc_g), SHOT)
        u1g = next(r for r in res_g.rois if r.ref == "U1")
        print(f"  [대조] B 로트 양품 U1 골든 차분 99퍼센타일 {np.percentile(u1g.diff, 99):.0f} "
              f"(불량 보드 브릿지 영역 평균 {dd[dm > 0].mean():.0f}), 이물 후보 {len(res_g.foreign_candidates)}개")

        # 시간
        print("=" * 70)
        print("[처리 시간 ms — 검증용 컨테이너 CPU 1코어 기준, GPU 실장비와 다름 · 촬영 1회 · ROI %d개]" % len(res.rois))
        for k, v in sorted(res.timings_ms.items()):
            print(f"  {k:<16} {v:8.1f}")

        # 경량화 옵션: 파인피치 IC만 ECC, 나머지는 위상 상관
        cfg.local_align.ecc_only_fine_pitch = True
        res_fast = pre.process_shot(raw, SHOT)
        u1f = next(r for r in res_fast.rois if r.ref == "U1")
        r1f = next(r for r in res_fast.rois if r.ref == "R1")
        print(f"  [경량화] ecc_only_fine_pitch=True → 6_local_align {res_fast.timings_ms['6_local_align']:.1f}ms, "
              f"total {res_fast.timings_ms['total']:.1f}ms")
        print(f"           U1 {u1f.align_method} ({u1f.local_shift[0]:+.2f}, {u1f.local_shift[1]:+.2f}), "
              f"R1 {r1f.align_method} ({r1f.local_shift[0]:+.2f}, {r1f.local_shift[1]:+.2f})")
        cfg.local_align.ecc_only_fine_pitch = False
        cfg.local_align.method = "phase"
        u1p = next(r for r in pre.process_shot(raw, SHOT).rois if r.ref == "U1")
        print(f"           (부호 검증) U1 위상 상관 단독: ({u1p.local_shift[0]:+.2f}, {u1p.local_shift[1]:+.2f})")
        cfg.local_align.method = "ecc"

        # 증강·합성 데모
        aug, s_aug = augment_roi(u1r.bgr, polarized=True, rng=RNG, sat=u1r.sat,
                                         solder_mask_ab=calib.ref_ab)
        c1 = next(r for r in res.rois if r.ref == "C1")
        body = c1.geom.body_poly.reshape(-1, 2)
        bx0, by0 = np.floor(body.min(0)).astype(int); bx1, by1 = np.ceil(body.max(0)).astype(int)
        rev = synth_reverse_polarity(c1.bgr, (bx0, by0, bx1, by1))

        # 시각화 저장
        def lab_(im, t):
            im = im.copy() if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return im
        sz = (380, 380)
        tiles = [lab_(cv2.resize(raw_roi, sz, interpolation=cv2.INTER_NEAREST), "raw U1 (bridge)"),
                 lab_(cv2.resize(u1r.bgr, sz, interpolation=cv2.INTER_NEAREST), "preprocessed"),
                 lab_(cv2.resize(cv2.applyColorMap(u1r.diff, cv2.COLORMAP_JET), sz,
                                 interpolation=cv2.INTER_NEAREST), "golden diff"),
                 lab_(cv2.resize(u1r.sat, sz, interpolation=cv2.INTER_NEAREST), "saturation mask"),
                 lab_(cv2.resize(aug, sz, interpolation=cv2.INTER_NEAREST), "train augment"),
                 lab_(cv2.resize(rev, sz, interpolation=cv2.INTER_NEAREST), "synth reverse C1")]
        panel = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(out_png, panel)
        print("=" * 70)
        print(f"시각화 저장: {out_png}   (데모 전체 {time.perf_counter() - t_all:.1f}s)")

    main(out_png)


# ############################################################################
# 실행 진입점
# ############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PCB AOI 전처리 파이프라인 합성 데이터 검증 데모")
    ap.add_argument("--out", default="demo_panel.png", help="시각화 이미지 저장 경로")
    args = ap.parse_args()
    run_demo(args.out)