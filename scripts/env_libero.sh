# shellcheck shell=bash
# Source from the FastWAM repo (after `source .venv/bin/activate`).
# Puts vendored LIBERO on PYTHONPATH so `import libero` works even if
# `uv pip install -e third_party/LIBERO` did not register the package.
_FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case ":${PYTHONPATH:-}:" in
  *":${_FASTWAM_ROOT}/third_party/LIBERO:"*) ;;
  *) export PYTHONPATH="${_FASTWAM_ROOT}/third_party/LIBERO${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${_FASTWAM_ROOT}/checkpoints}"
unset _FASTWAM_ROOT
