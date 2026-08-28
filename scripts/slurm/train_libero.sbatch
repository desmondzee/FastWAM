#!/usr/bin/env bash
# Train FastWAM on LIBERO (uncond 2cam). Edit the #SBATCH header for your cluster.
#
#   sbatch scripts/slurm/train_libero.sbatch
#
# Optional env: NPROC_PER_NODE, DIFFSYNTH_MODEL_BASE_PATH, DIFFSYNTH_DOWNLOAD_SOURCE
#SBATCH --job-name=fastwam-libero-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j-train.out
# Uses the cluster default partition/account. Override with sbatch --partition=NAME
# 1 GPU, 64G RAM. Single-A30 training may still OOM; lower batch_size if it does.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [[ ! -f scripts/train_zero1.sh ]]; then
  echo "Error: run sbatch from the FastWAM repo root (missing scripts/train_zero1.sh)." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Error: .venv missing. Run ./scripts/setup_libero.sh train first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(pwd)/checkpoints}"
mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE}}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

BACKBONE="${DIFFSYNTH_MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
# Hydra config uses a repo-relative path; keep a copy (or symlink) at the default location.
REPO_BACKBONE="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [[ ! -f "${BACKBONE}" && ! -f "${REPO_BACKBONE}" ]]; then
  echo "[train] building ActionDiT backbone (needs GPU + Wan2.2 weights)"
  python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output "${REPO_BACKBONE}" \
    --device cuda \
    --dtype bfloat16
elif [[ -f "${BACKBONE}" && ! -f "${REPO_BACKBONE}" ]]; then
  mkdir -p checkpoints
  ln -sfn "${BACKBONE}" "${REPO_BACKBONE}"
fi

CACHE_DIR="data/text_embeds_cache/libero"
if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null || true)" ]]; then
  echo "[train] precomputing T5 text embeddings"
  python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
fi

echo "[train] nproc_per_node=${NPROC_PER_NODE}"
bash scripts/train_zero1.sh "${NPROC_PER_NODE}" task=libero_uncond_2cam224_1e-4

echo "[train] checkpoints: runs/libero_uncond_2cam224_1e-4/<run_id>/checkpoints/weights/"
echo "[train] eval with: CKPT=runs/libero_uncond_2cam224_1e-4/<run_id>/checkpoints/weights/step_XXXXXX.pt sbatch scripts/slurm/eval_libero.sbatch"
