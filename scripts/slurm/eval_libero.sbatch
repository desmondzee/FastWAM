#!/usr/bin/env bash
# Evaluate FastWAM on LIBERO. Edit the #SBATCH header for your cluster.
#
#   sbatch scripts/slurm/eval_libero.sbatch
#   CKPT=runs/.../checkpoints/weights/step_XXXXXX.pt sbatch scripts/slurm/eval_libero.sbatch
#
# Optional env: CKPT, DATASET_STATS, NUM_GPUS, SIGMA_SHIFT
#SBATCH --job-name=fastwam-libero-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j-eval.out
# Uses the cluster default partition/account. Override on the command line:
#   sbatch --partition=NAME scripts/slurm/eval_libero.sbatch
# 1 GPU, 64G RAM. A30 is 24GB VRAM.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [[ ! -f experiments/libero/run_libero_manager.py ]]; then
  echo "Error: run sbatch from the FastWAM repo root." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Error: .venv missing. Run ./scripts/setup_libero.sh eval first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/env_libero.sh

resolve_ckpt() {
  if [[ -n "${CKPT:-}" ]]; then
    printf '%s\n' "${CKPT}"
    return
  fi
  local candidate
  for candidate in \
    checkpoints/fastwam_release/libero_uncond_2cam224.pt \
    checkpoints/libero_uncond_2cam224.pt \
    ./libero_uncond_2cam224.pt; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "Error: no checkpoint found. Set CKPT= or run ./scripts/setup_libero.sh eval." >&2
  exit 1
}

resolve_stats() {
  local ckpt_path="$1"
  if [[ -n "${DATASET_STATS:-}" ]]; then
    printf '%s\n' "${DATASET_STATS}"
    return
  fi
  local stem="${ckpt_path%.pt}_dataset_stats.json"
  if [[ -f "${stem}" ]]; then
    printf '%s\n' "${stem}"
    return
  fi
  local dir
  dir="$(cd "$(dirname "${ckpt_path}")" && pwd)"
  local i
  for i in 1 2 3 4; do
    if [[ -f "${dir}/dataset_stats.json" ]]; then
      printf '%s\n' "${dir}/dataset_stats.json"
      return
    fi
    dir="$(dirname "${dir}")"
  done
  echo "Error: no dataset stats next to ${ckpt_path}. Set DATASET_STATS=." >&2
  exit 1
}

CKPT_PATH="$(resolve_ckpt)"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Error: checkpoint does not exist: ${CKPT_PATH}" >&2
  exit 1
fi
STATS_PATH="$(resolve_stats "${CKPT_PATH}")"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE}}"
else
  NUM_GPUS="${NUM_GPUS:-1}"
fi

SIGMA_ARGS=()
if [[ -n "${SIGMA_SHIFT:-}" ]]; then
  SIGMA_ARGS+=("EVALUATION.sigma_shift=${SIGMA_SHIFT}")
elif [[ "${CKPT_PATH}" == *fastwam_release* ]]; then
  # Original released FastWAM LIBERO ckpts were trained with action shift 5.0
  SIGMA_ARGS+=("EVALUATION.sigma_shift=5.0")
fi

echo "[eval] ckpt=${CKPT_PATH}"
echo "[eval] stats=${STATS_PATH}"
echo "[eval] num_gpus=${NUM_GPUS}"

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="${CKPT_PATH}" \
  EVALUATION.dataset_stats_path="${STATS_PATH}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  "${SIGMA_ARGS[@]}"
