#!/usr/bin/env bash
# Prepare a FastWAM + LIBERO environment.
# Login-node safe: downloads and pip only. Does not train or preprocess the backbone.
#
#   ./scripts/setup_libero.sh eval    # env + LIBERO + released HF weights (default)
#   ./scripts/setup_libero.sh train   # same + LIBERO LeRobot training demos
set -euo pipefail

MODE="${1:-eval}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/setup_libero.sh [eval|train]

  eval   (default) FastWAM venv, LIBERO sim, released libero_uncond_2cam224 weights
  train  same as eval, plus yuanty/LIBERO-fastwam LeRobot demos

Optional env:
  FASTWAM_SCRATCH  if set, symlink data/ checkpoints/ runs/ evaluate_results/ there
  HF_TOKEN         Hugging Face token
EOF
  exit 0
fi
if [[ "${MODE}" != "eval" && "${MODE}" != "train" ]]; then
  echo "Error: mode must be eval or train, got: ${MODE}" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${FASTWAM_SCRATCH:-}" ]]; then
  if [[ "${FASTWAM_SCRATCH}" != /* ]]; then
    echo "Error: FASTWAM_SCRATCH must be an absolute path, got: ${FASTWAM_SCRATCH}" >&2
    exit 1
  fi
  scratch_parent="$(dirname "${FASTWAM_SCRATCH}")"
  if [[ ! -d "${scratch_parent}" || ! -w "${scratch_parent}" ]]; then
    echo "Error: cannot create FASTWAM_SCRATCH=${FASTWAM_SCRATCH}" >&2
    echo "Parent ${scratch_parent} is missing or not writable." >&2
    echo "This usually means \$SCRATCH is unset, so \"\$SCRATCH/fastwam\" became \"/fastwam\"." >&2
    echo "Fix: unset FASTWAM_SCRATCH   # store files under this repo" >&2
    echo "  or: export FASTWAM_SCRATCH=/real/scratch/dir/fastwam" >&2
    exit 1
  fi
  echo "[setup] linking data/checkpoints/runs/evaluate_results -> ${FASTWAM_SCRATCH}"
  mkdir -p "${FASTWAM_SCRATCH}/data" \
    "${FASTWAM_SCRATCH}/checkpoints" \
    "${FASTWAM_SCRATCH}/runs" \
    "${FASTWAM_SCRATCH}/evaluate_results"
  ln -sfn "${FASTWAM_SCRATCH}/data" data
  ln -sfn "${FASTWAM_SCRATCH}/checkpoints" checkpoints
  ln -sfn "${FASTWAM_SCRATCH}/runs" runs
  ln -sfn "${FASTWAM_SCRATCH}/evaluate_results" evaluate_results
fi

mkdir -p checkpoints data/libero_mujoco3.3.2

export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not on PATH after install. Add ~/.local/bin to PATH and retry." >&2
  exit 1
fi

echo "[setup] Python 3.10 venv"
uv python install 3.10
if [[ ! -d "${ROOT}/.venv" ]]; then
  uv venv --python 3.10 "${ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

echo "[setup] torch 2.7.1+cu128"
uv pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[setup] FastWAM (pip -e .)"
uv pip install -e "${ROOT}" --extra-index-url https://download.pytorch.org/whl/cu128

LIBERO_DIR="${ROOT}/third_party/LIBERO"
if [[ ! -d "${LIBERO_DIR}/.git" ]]; then
  echo "[setup] cloning LIBERO"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
else
  echo "[setup] LIBERO already cloned"
fi

echo "[setup] LIBERO into the FastWAM env (no LIBERO torch/numpy pins)"
uv pip install -e "${LIBERO_DIR}"
uv pip install "robosuite==1.4.0" "bddl==1.0.1" easydict
uv pip install mujoco==3.3.2

python - <<'PY'
import torch
print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()}")
import libero.libero  # noqa: F401
print("[setup] import libero.libero ok")
PY

CKPT_DIR="${ROOT}/checkpoints/fastwam_release"
CKPT="${CKPT_DIR}/libero_uncond_2cam224.pt"
STATS="${CKPT_DIR}/libero_uncond_2cam224_dataset_stats.json"
if [[ ! -f "${CKPT}" || ! -f "${STATS}" ]]; then
  echo "[setup] downloading released FastWAM LIBERO weights"
  huggingface-cli download yuanty/fastwam \
    libero_uncond_2cam224.pt \
    libero_uncond_2cam224_dataset_stats.json \
    --local-dir "${CKPT_DIR}"
else
  echo "[setup] released weights already present: ${CKPT}"
fi

if [[ "${MODE}" == "train" ]]; then
  DATA_ROOT="${ROOT}/data/libero_mujoco3.3.2"
  NEED_DATA=0
  for split in libero_spatial_no_noops_lerobot libero_object_no_noops_lerobot \
    libero_goal_no_noops_lerobot libero_10_no_noops_lerobot; do
    if [[ ! -d "${DATA_ROOT}/${split}" ]]; then
      NEED_DATA=1
      break
    fi
  done
  if [[ "${NEED_DATA}" -eq 1 ]]; then
    echo "[setup] downloading yuanty/LIBERO-fastwam LeRobot demos"
    huggingface-cli download yuanty/LIBERO-fastwam \
      --repo-type dataset \
      --include "libero_*_no_noops_lerobot.tar.gz" \
      --local-dir "${DATA_ROOT}"
    shopt -s nullglob
    tars=("${DATA_ROOT}"/libero_*_no_noops_lerobot.tar.gz)
    if [[ ${#tars[@]} -eq 0 ]]; then
      echo "Error: no LIBERO tar.gz files in ${DATA_ROOT}" >&2
      exit 1
    fi
    for f in "${tars[@]}"; do
      echo "[setup] extracting $(basename "${f}")"
      tar -xzf "${f}" -C "${DATA_ROOT}"
    done
  else
    echo "[setup] LIBERO LeRobot demos already extracted under ${DATA_ROOT}"
  fi
fi

echo
echo "[setup] done (mode=${MODE})"
echo "  activate:  source ${ROOT}/.venv/bin/activate"
if [[ "${MODE}" == "eval" ]]; then
  echo "  next:      sbatch scripts/slurm/eval_libero.sbatch"
else
  echo "  next:      sbatch scripts/slurm/train_libero.sbatch"
  echo "             then eval with CKPT=runs/.../checkpoints/weights/step_XXXXXX.pt"
fi
