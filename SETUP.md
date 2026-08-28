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

Only set `FASTWAM_SCRATCH` to a path that already exists and is writable. On many clusters `$SCRATCH` is **unset**; then `export FASTWAM_SCRATCH="$SCRATCH/fastwam"` becomes `/fastwam` and setup fails.

Check first:

```bash
echo "SCRATCH=${SCRATCH-<unset>}"
ls -ld "$SCRATCH" 2>/dev/null || true
```

If `$SCRATCH` is empty, either skip scratch (files stay in the clone) or pick a real directory:

```bash
unset FASTWAM_SCRATCH
# or, example only — use your cluster’s scratch:
# export FASTWAM_SCRATCH="$HOME/scratch/fastwam"
```

If you do have a writable scratch root:

```bash
export FASTWAM_SCRATCH="/path/that/exists/fastwam"
```

The setup script will create `data`, `checkpoints`, `runs`, and `evaluate_results` under that path and symlink them into the repo.

Disk: eval-only weights are a few GB; LIBERO demos ~5GB compressed; Wan2.2 + ActionDiT backbone tens of GB; training writes extra under `runs/`.

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

## 6. Submit eval as a Slurm job

Setup stays in the shell (`./scripts/setup_libero.sh eval`). The GPU eval is `sbatch`, which queues a job and returns your prompt.

Submit from the **login node** (the host you first SSH to), not after `ssh` onto a GPU box like `hetg5`. Compute nodes often cannot talk to the controller (`sinfo: Unexpected message received`).

```bash
cd ~/FastWAM
git pull
# setup must already have succeeded on a shared filesystem so the job sees .venv

sbatch scripts/slurm/eval_libero.sbatch
squeue -u $USER
tail -f slurm-*-eval.out
```

That requests **1 GPU** and **64G RAM** on the default partition. No `--account` (this cluster’s accounting plugin is disabled). If Slurm says you must name a partition:

```bash
sbatch --partition=PARTITION_FROM_SINFO scripts/slurm/eval_libero.sbatch
```

If `sinfo` still fails on the login node, try `module load slurm` (or whatever the site docs use) and run `sinfo` again.

Interactive smoke test (blocks your shell; 1 GPU) is still:

```bash
source .venv/bin/activate
source scripts/env_libero.sh
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  gpu_id=0
```

## 7. Manual train / full eval

```bash
source .venv/bin/activate
source scripts/env_libero.sh

bash scripts/train_zero1.sh 1 task=libero_uncond_2cam224_1e-4

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.sigma_shift=5.0 \
  MULTIRUN.num_gpus=1
```
