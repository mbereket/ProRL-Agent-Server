# SWE-Gym Slime GRPO

Train **Qwen3.5-4B** with async **GRPO** on the 293 **SWE-Gym** training tasks:
a **codex** agent solves each task inside an Apptainer sandbox, **Polar**
records the token-level trajectory and grades the patch with the SWE-Gym test
harness, and **Slime** (Megatron + SGLang) does the policy update.

Tested layouts: one node with 8 GPUs (4 train, 4 serve) and two nodes (8 train,
8 serve). Larger engine pools follow the same pattern; Qwen3.5-9B is a
`MODEL_ARGS_FILE=model_args_9b.sh` swap. `launch_e2e.sh` also sets up the
environment itself, including on clusters whose driver, toolkit, or container
runtime are older than the pinned stack needs (see below).

> Unlike the rollout demos (calculator / count_stars / swebench_verified), this
> path serves the model with **SGLang**: Slime owns the inference engines and
> syncs freshly trained weights into them every step (GPU-to-GPU NCCL).

## Quick start (single node)

```bash
export WANDB_API_KEY=<your-key>          # optional
bash examples/swegym_slime_grpo/launch_e2e.sh
```

`launch_e2e.sh` is idempotent: re-run it after fixing whatever it reports and
it resumes. Everything it creates lands under `WORKROOT` (default: `<repo>/tmp`).
Cache variables you already export (`HF_HOME`, `UV_CACHE_DIR`,
`APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`) are respected; unset ones are placed
under `WORKROOT`.

What it does, in order:

| Step | Script | Notes |
|---|---|---|
| Preflight | `setup/preflight.sh` | Machine facts → decisions; fails fast on anything it cannot fix |
| Python stack | `setup/install_python_stack.sh` | venv, Polar, `sglang==0.5.13`, torch family on one CUDA build |
| CUDA user space | `setup/ensure_cuda_userspace.sh` | forward-compat libs if the driver is old; toolkit if `nvcc` is missing |
| Apptainer | `setup/ensure_apptainer.sh` | uses `POLAR_APPTAINER_BIN` / PATH, else unprivileged install |
| Checkouts | `launch_e2e.sh` | Slime v0.3.0 + its canonical Megatron commit and patch, router-token patch |
| Training stack | `setup/ensure_training_stack.sh` | Transformer Engine matched to torch's CUDA major, FLA, flash-attn on B200 |
| Assets | `prepare_data.py`, `prepare_apptainer_images.py` | 293-task JSONL, per-task SIFs, pinned agent CLIs |
| Checkpoint | `launch_e2e.sh`, `convert_weights.sh` | full HF snapshot download, HF → Megatron torch_dist |
| Train | `run.sh` | Polar services + Ray + Slime `train_async.py` |

### What preflight checks

| Check | If it fails |
|---|---|
| GPUs visible, driver's native CUDA version | driver < CUDA 13 → `NEED_CUDA_COMPAT` (forward-compat libraries, no root needed) |
| CUDA 13 `nvcc` on PATH or under `CUDA_HOME` | `NEED_CUDA_TOOLKIT` (conda-forge toolkit via pixi under `WORKROOT`); pixi itself is bootstrapped if missing |
| `gcc`/`g++` | fatal: Transformer Engine builds from source |
| apptainer/singularity | `NEED_APPTAINER` (unprivileged install; needs `cpio` and `rpm2cpio` or `busybox`); fatal if user namespaces are also disabled |
| `uv` | bootstrapped into `WORKROOT/bin` |
| `git curl tar xz envsubst` | fatal |
| `WORKROOT` writable; `HOME` writable | fatal / caches go under `WORKROOT` |
| Ports 8080, 8100, 9000, 8265, 6379 free | fatal with the env var to change |
| github.com, pypi.org, huggingface.co, download.pytorch.org reachable, plus the hosts any enabled fix downloads from | fatal with what depends on it |

Run it alone with `bash examples/swegym_slime_grpo/setup/preflight.sh`.

### Why the environment is pinned the way it is

The stack has a hard CUDA-version requirement that comes from the software, not
from your hardware. The launcher's job is to satisfy it on whatever machine it
lands on.

- **The pinned `sglang==0.5.13` only resolves against CUDA 13.** Its dependency
  tree pins `cuda-python` 13.x and `flash-attn-4` (pre-release only), which
  makes every cu12x torch build unresolvable. So torch is always installed from
  the `cu130` index with `--prerelease=allow`. The repo's default
  `torch-backend=auto` would pick a torch build from the *driver* instead, and
  on an older driver that yields a cpu or cu12x torch whose CUDA-only
  companions are silently dropped; the first symptom is
  `operator torchvision::nms does not exist` at import.
- **Your driver decides whether forward-compat libraries are needed.** A CUDA 13
  build needs an R580-class driver. If `nvidia-smi` reports a lower native CUDA
  version (dfw: 535.x, CUDA 12.2), preflight sets `NEED_CUDA_COMPAT` and
  `ensure_cuda_userspace.sh` puts NVIDIA's cuda-compat 13.1 user-space driver
  on `LD_LIBRARY_PATH`. Nothing system-wide changes; a matmul gate verifies it.
- **Your toolkit decides whether one is installed.** Transformer Engine builds
  its torch bindings from source and needs a CUDA 13 `nvcc`. If none is on PATH
  or under `CUDA_HOME`, a conda-forge toolkit 13.1 is installed via pixi under
  `WORKROOT`. Both compat and toolkit are 13.1 rather than 13.0 because TE's
  cu13 core needs a ≥13.1 cuBLASLt; the cu130 torch runs on any 13.x user space.
- **Transformer Engine must match torch's CUDA major.** `2.14.0[core-cu13]`
  for cu13 torch; the older `2.5.0` pin ships a cu12 core only and fails at
  import with `libcublas.so.12` missing.
- **Four resolution pins** live in `setup/constraints.txt`, applied through
  `UV_OVERRIDE` so no later `uv pip install` can undo them: `nvidia-cublas`
  ≥ 13.6 (TE symbol), `numpy<2` (slime asserts 1.x), `scipy==1.13.1` (1.18 uses
  numpy-2-only API), `wandb==0.22.3` (0.29 removed a function slime calls).
- **Megatron** is pinned to the commit slime v0.3.0's own Dockerfile uses, plus
  its companion patch; the `26.04-alpha.rc1` tag no longer ships a module slime
  imports.

On a machine with a current driver and toolkit, only the first, fourth, and
fifth points apply and preflight reports every fix as "not needed".

## Multi-node

The trainer takes `ACTOR_NUM_GPUS` GPUs and every other GPU in the Ray cluster
serves an SGLang engine. On more than one node the trainer must take whole
nodes (`ACTOR_NUM_GPUS` a multiple of the node size), because slime v0.3.0
assigns engine addresses per whole node; `run.sh` rejects other layouts. So
2 nodes gives 8 train / 8 serve and 3 nodes gives 8 train / 16 serve.
Generation is normally the bottleneck (see `perf/wait_time_ratio`), so add
nodes on the engine side. Long traces fit via context parallelism
(`per-trace cap = MAX_TOKENS_PER_GPU × CP`; TP × CP must divide the actor GPUs).

**Slurm:**

```bash
export WORKROOT=/shared/fs/prorl            # shared by all nodes
bash examples/swegym_slime_grpo/multinode/sbatch_launch.sh --nodes 2 --partition <p> --account <a>
```

This runs `multinode/head_entry.sh` on the first node, which `srun`s
`multinode/ray_worker_join.sh` on the others and then runs `launch_e2e.sh`.

**Without slurm:** on every worker node run
`bash multinode/ray_worker_join.sh <head-ip>`; on the head run

```bash
NUM_NODES=2 RAY_HEAD_IP=<head-ip> POLAR_BIND_HOST=0.0.0.0 POLAR_PUBLIC_HOST=<head-ip> \
  bash examples/swegym_slime_grpo/launch_e2e.sh
```

`run.sh` waits for all `NUM_NODES` Ray nodes before submitting the job.

## Knobs

All are environment variables with the single-node defaults shown.

| Knob | Default | Meaning |
|---|---|---|
| `WORKROOT` | `<repo>/tmp` | Root for checkouts, caches, toolchains, checkpoints |
| `HF_CHECKPOINT`, `MODEL_ARGS_FILE` | `Qwen/Qwen3.5-4B`, `model_args.sh` | Model; use `model_args_9b.sh` for Qwen3.5-9B (`TP_SIZE=4`) |
| `NUM_NODES`, `GPUS_PER_NODE` | 1, detected | Ray cluster size |
| `ACTOR_NUM_GPUS`, `ROLLOUT_NUM_GPUS` | 4 (one node on multi-node), all remaining | Trainer GPUs (whole nodes when multi-node) and engine GPUs |
| `TP_SIZE`, `CONTEXT_PARALLEL_SIZE` | 2, 1 | Megatron parallelism (`head_entry.sh` defaults CP to all train GPUs / TP) |
| `MAX_TOKENS_PER_GPU`, `SGLANG_CONTEXT_LENGTH` | 30000, 50000 | 16384 is the safe value on H100-80GB for the 4B model |
| `ROLLOUT_MAX_PROMPT_LEN`, `ROLLOUT_MAX_RESPONSE_LEN` | 32000, 16000 | Slime rollout length caps |
| `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `NUM_EPOCH`, `SAVE_INTERVAL` | 4, 16, 1, 10 | Batch and schedule |
| `POLAR_ROLLOUT_PORT`, `POLAR_GATEWAY_PORT`, `SGLANG_ROUTER_PORT`, `RAY_DASHBOARD_PORT`, `RAY_GCS_PORT` | 8080, 8100, 9000, 8265, 6379 | Service ports |
| `POLAR_BIND_HOST`, `POLAR_PUBLIC_HOST`, `RAY_HEAD_IP` | 127.0.0.1 | Set by `head_entry.sh` for multi-node |
| `POLAR_APPTAINER_BIN`, `APPTAINER_IMAGE_DIR`, `AGENT_CLI_DIR` | auto | Container runtime and prepared assets |
| `SETUP_ENV` | 1 | `0` skips preflight and all `setup/` scripts (bring your own environment) |
| `BUILD_MAX_JOBS` | 8 | Parallelism cap for TE / flash-attn source builds |
| `RUN_TRAINING` | 1 | `0` stops after setup and conversion |

Agent harness (`codex` by default), timeouts and evaluator live in
`polar_config.yaml`; gateway worker counts in `topology.yaml`; the SWE-Gym
split in `sample_tasks.py`. The codex CLI is installed at the version the
harness enforces (`prepare_apptainer_images.py`).

## Watching a run

The Slime driver's training output (step metrics, checkpoint saves) goes to
the Ray job, not to your shell. Follow it with wandb, or
`cat $SAVE_DIR/latest_checkpointed_iteration.txt`. To inspect live agent
sessions start the dashboard from the repo root:

```bash
uv run polar dashboard -c $WORKROOT/swegym_slime_grpo/topology.yaml
```

## Files

| File | Purpose |
|---|---|
| `launch_e2e.sh` | One-shot entry: environment + setup + run |
| `setup/` | Preflight and environment scripts (see table above), `constraints.txt` |
| `run.sh` | Polar services + Ray + Slime training job |
| `convert_weights.sh` | HF checkpoint → Megatron torch_dist |
| `model_args.sh`, `model_args_9b.sh` | Qwen3.5-4B / 9B Megatron args |
| `topology.yaml`, `polar_config.yaml` | Polar templates rendered by `run.sh` |
| `multinode/` | `sbatch_launch.sh`, `head_entry.sh`, `ray_worker_join.sh` |
| `prepare_data.py`, `prepare_apptainer_images.py`, `sample_tasks.py` | Data, SIF images, agent CLIs |
