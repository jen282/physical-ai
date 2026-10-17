#!/usr/bin/env bash
# Colab 런타임 공통 셋업. 모든 노트북에서 Drive 연결과 저장소 받기 다음에 실행합니다.
#   1. Python 버전 확인   2. GPU 확인   3. ffmpeg 확인
#   4. 학습 패키지 설치   5. Drive 폴더 생성   6. 실행 환경 기록
# 런타임이 끊길 때마다 다시 실행해도 됩니다. 이미 설치된 패키지는 pip가 건너뜁니다.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVE_ROOT="${TEACHARM_DRIVE_ROOT:-/content/drive/MyDrive/teacharm}"
REQ="$REPO_DIR/code/training/requirements-train.txt"

echo "[1/6] Python"
python - <<'EOF'
import sys
v = sys.version.split()[0]
assert sys.version_info >= (3, 12), f"LeRobot 0.6.1은 Python 3.12 이상이 필요합니다 (현재 {v})"
print(f"   {v}")
EOF

echo "[2/6] GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
else
  echo "   GPU 없음 (CPU 런타임). 학습(02번)이 아니면 CPU로 충분합니다."
fi

echo "[3/6] ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null
fi
ffmpeg -version | head -1 | sed 's/^/   /'

echo "[4/6] 학습 패키지 ($REQ)"
torch_version() { python -c "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null || echo "없음"; }
TORCH_BEFORE="$(torch_version)"
python -m pip install -q -r "$REQ"
TORCH_AFTER="$(torch_version)"
echo "   torch $TORCH_BEFORE -> $TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "   ⚠ Colab 기본 torch가 LeRobot 요구 범위 밖이라 다시 설치했습니다."
  echo "     이 셀 전에 torch를 import했다면 '런타임 → 세션 다시 시작' 후 처음부터 다시 실행하세요."
fi

echo "[5/6] Drive 폴더 ($DRIVE_ROOT)"
if [ ! -d /content/drive/MyDrive ]; then
  echo "   Drive가 연결되지 않았습니다. 노트북에서 drive.mount('/content/drive')를 먼저 실행하세요." >&2
  exit 1
fi
mkdir -p "$DRIVE_ROOT"/{data,outputs,onnx,env}
echo "   data/ outputs/ onnx/ env/ 준비됨"

echo "[6/6] 실행 환경 기록"
REPO_DIR="$REPO_DIR" DRIVE_ROOT="$DRIVE_ROOT" python - <<'EOF'
import importlib.metadata as md
import json, os, platform, subprocess
from datetime import datetime, timezone, timedelta

import torch

def git(*args):
    return subprocess.run(["git", "-C", os.environ["REPO_DIR"], *args],
                          capture_output=True, text=True).stdout.strip()

now = datetime.now(timezone(timedelta(hours=9)))
env = {
    "recorded_at": now.isoformat(timespec="seconds"),
    "git_commit": git("rev-parse", "HEAD"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    **{pkg: md.version(pkg) for pkg in ("lerobot", "torchvision", "onnx", "onnxruntime", "wandb")},
}
path = os.path.join(os.environ["DRIVE_ROOT"], "env", f"env_{now:%Y%m%d-%H%M%S}.json")
with open(path, "w") as f:
    json.dump(env, f, indent=2, ensure_ascii=False)
print(f"   {path}")
for k, v in env.items():
    print(f"   {k}: {v}")
EOF

echo "셋업 완료"
