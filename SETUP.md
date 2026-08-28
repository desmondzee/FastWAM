# FastWAM LIBERO setup (HPC)

This fork’s entry point for **LIBERO only**. RoboTwin is unchanged and unused here.

Architecture alone is not enough. Training needs the FastWAM LeRobot demos. Eval needs the LIBERO simulator plus a checkpoint.

| Mode | `setup_libero.sh` | Then submit |
| --- | --- | --- |
| **eval** (default) | FastWAM env + LIBERO + released weights | `scripts/slurm/eval_libero.sbatch` |
| **train** | Same env + LIBERO + training demos | `scripts/slurm/train_libero.sbatch`, then eval with `CKPT=` |

Setup is CPU/network (login node). Backbone preprocess and training run on GPU in Slurm.

## 1. Clone this fork

```bash
git clone https://github.com/desmondzee/FastWAM.git
cd FastWAM
```

`origin` is this fork. `upstream` is [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM). Do not push to `upstream`.

## 2. Scratch (HPC)

Do not keep datasets or checkpoints in `$HOME`. Either symlink by hand:

```bash
mkdir -p "$SCRATCH/fastwam"/{data,checkpoints,runs,evaluate_results}
ln -sfn "$SCRATCH/fastwam/data" data
ln -sfn "$SCRATCH/fastwam/checkpoints" checkpoints
ln -sfn "$SCRATCH/fastwam/runs" runs
ln -sfn "$SCRATCH/fastwam/evaluate_results" evaluate_results
```

or set `FASTWAM_SCRATCH` before setup (the script creates the same layout).

Disk: LIBERO demos ~5GB compressed; Wan2.2 + ActionDiT backbone tens of GB; training writes extra under `runs/`.

## 3. Install

Needs [uv](https://docs.astral.sh/uv/) and Python 3.10. CUDA 12.8-compatible module if the cluster uses modules.

```bash
# released FastWAM weights (~eval only; no training dataset)
./scripts/setup_libero.sh eval

# also download the paper’s LIBERO LeRobot demos
./scripts/setup_libero.sh train
```

Both modes install FastWAM into `.venv`, clone LIBERO into `third_party/LIBERO` (gitignored), and pin `mujoco==3.3.2`. That follows this repo, not LIBERO’s own Python 3.8 / torch 1.11 conda.

`train` is the authors’ preprocessed demos ([yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)), not the original LIBERO `.hdf5` dump.

Optional: `export HF_TOKEN=...` if Hugging Face rate-limits you. Wan downloads use Hugging Face (`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`), not ModelScope.

## 4. Existing scripts (wrapped, not replaced)

- `scripts/train_zero1.sh` — train. Writes `runs/<task>/<timestamp>/`.
- `scripts/preprocess_action_dit_backbone.py` — GPU, train-only. Builds ActionDiT init from Wan2.2.
- `scripts/precompute_text_embeds.py` — GPU, train-only. T5 cache.
- `experiments/libero/run_libero_manager.py` — LIBERO eval.

## 5. Where checkpoints live

All of this is gitignored (`runs/`, `checkpoints`, `*.pt`).

After training:

```text
runs/libero_uncond_2cam224_1e-4/<timestamp>/
  checkpoints/weights/step_XXXXXX.pt
  checkpoints/state/step_XXXXXX/
  dataset_stats.json
  config.yaml
```

Saves every 2000 steps and again when the run finishes.

Eval looks for a `.pt` in this order:

1. `CKPT` (your `runs/.../checkpoints/weights/step_XXXXXX.pt`)
2. `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (`setup eval` download)
3. `checkpoints/libero_uncond_2cam224.pt`
4. `./libero_uncond_2cam224.pt`

Stats: `DATASET_STATS`, or `<stem>_dataset_stats.json` next to the ckpt, or `dataset_stats.json` in a parent directory (the train run dir).

Released paper weights: [yuanty/fastwam](https://huggingface.co/yuanty/fastwam) (`libero_uncond_2cam224.pt` + matching `*_dataset_stats.json`). Those use `EVALUATION.sigma_shift=5.0`. Your own `runs/` ckpts use the config default (`1.0`) unless you set `SIGMA_SHIFT`.

## 6. Slurm

Edit `#SBATCH` partition / account / GPU count at the top of each file, then:

```bash
# eval released weights (after setup eval)
sbatch scripts/slurm/eval_libero.sbatch

# train (after setup train). Preprocess + T5 cache run in-job if missing.
sbatch scripts/slurm/train_libero.sbatch

# eval a trained run
CKPT=runs/libero_uncond_2cam224_1e-4/<timestamp>/checkpoints/weights/step_XXXXXX.pt \
  sbatch scripts/slurm/eval_libero.sbatch
```

Override GPU count with `NPROC_PER_NODE` (train) or `NUM_GPUS` (eval).

## 7. Manual commands (same as the wrappers)

```bash
source .venv/bin/activate
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.sigma_shift=5.0 \
  MULTIRUN.num_gpus=8
```
