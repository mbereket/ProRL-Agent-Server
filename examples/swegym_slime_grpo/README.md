# SWE-Gym Slime GRPO

Train **Qwen3.5-4B** with async **GRPO** on the 293 **SWE-Gym** training tasks:
a **codex** agent solves each task inside an Apptainer sandbox, **Polar**
records the token-level trajectory and grades the patch with the SWE-Gym test
harness, and **Slime** (Megatron + SGLang) does the policy update.

Tested layouts: one node with 8 GPUs (4 train, 4 serve) and two nodes (8 train,
8 serve); each is a config in `configs/`. `launch.sh` also sets up the
environment itself, including on clusters whose driver, toolkit, or container
runtime are older than the locked stack needs (see below).

> Unlike the rollout demos (calculator / count_stars / swebench_verified), this
> path serves the model with **SGLang**: Slime owns the inference engines and
> syncs freshly trained weights into them every step (GPU-to-GPU NCCL).

## Quick start

```bash
export WANDB_API_KEY=<your-key>          # optional

# one 8-GPU node: 4 trainer GPUs + 4 SGLang engines
bash examples/swegym_slime_grpo/launch.sh examples/swegym_slime_grpo/configs/qwen35-4b-1node.yaml

# two nodes under slurm (node count comes from the config)
bash examples/swegym_slime_grpo/slurm_launch.sh \
    --config examples/swegym_slime_grpo/configs/qwen35-4b-2node.yaml --partition <p> --account <a>

# see what a config resolves to without touching GPUs
bash examples/swegym_slime_grpo/launch.sh examples/swegym_slime_grpo/configs/smoke-1node.yaml --dry-run
```

`launch.sh` is idempotent: re-run it after fixing whatever it reports and it
resumes. Everything it creates lands under `WORKROOT` (default: `<repo>/tmp`).
Cache variables you already export (`HF_HOME`, `UV_CACHE_DIR`,
`APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`) are respected; unset ones are placed
under `WORKROOT`.

## Run configs

A run is one YAML file in `configs/`. All keys have defaults (the single-node
recipe); unknown keys are rejected.

```yaml
name: qwen35-4b-1node                   # RUN_ID, wandb group, checkpoint dir name

model:
  hf_checkpoint: Qwen/Qwen3.5-4B
  model_args_file: model_args.sh        # Megatron args in internal/ (model_args_9b.sh for 9B)

cluster:
  num_nodes: 1
  actor_num_gpus: 4                     # trainer GPUs; every other GPU serves an SGLang engine
  tp_size: 2                            # TP x CP must divide actor_num_gpus
  context_parallel_size: 1

rollout:
  batch_size: 4                         # prompts per step
  n_samples_per_prompt: 16              # GRPO group size
  num_epoch: 1
  max_prompt_len: 32000
  max_response_len: 16000
  sglang_context_length: 50000

training:
  max_tokens_per_gpu: 16384             # longest trainable trace = this x context_parallel_size
  lr: 1e-6
  use_kl_loss: true
  kl_loss_coef: 0.001
  grpo_std_normalization: false         # mean-only advantages (Dr.GRPO); true scales by group std
  save_interval: 10
  extra_train_args: ""                  # appended to train_async.py verbatim

eval:
  prompt_data: ""                       # "<name> <path.jsonl>" enables a held-out eval
  interval: 10
  n_samples_per_prompt: 1

wandb:
  project: polar-swegym-grpo
  group: <name>
```

Shipped configs:

| Config | Layout | Notes |
|---|---|---|
| `smoke-1node.yaml` | 1 node, 4 train / 4 serve | 2 prompts x 4 samples, 2 steps; end-to-end check |
| `qwen35-4b-1node.yaml` | 1 node, TP2 | validated single-node recipe on H100-80GB |
| `qwen35-4b-2node.yaml` | 2 nodes, 8 train (TP2 x CP4) / 8 serve | 65k-token traces for codex's long sessions |
| `qwen35-9b-2node.yaml` | 2 nodes, 8 train (TP4 x CP2) / 8 serve | Qwen3.5-9B |

On more than one node the trainer must take whole nodes (`actor_num_gpus` a
multiple of the node size), because slime v0.3.0 assigns engine addresses per
whole node. Generation is normally the bottleneck (see `perf/wait_time_ratio`),
so add nodes on the engine side. The agent harness (`codex`), its timeouts and
the evaluator live in `internal/polar_config.yaml`; gateway worker counts in
`internal/topology.yaml`; the SWE-Gym split in `internal/sample_tasks.py`.

## What the launcher does

`launch.sh` resolves the config into environment variables and runs
`internal/pipeline.sh`, in order:

| Step | Script | Notes |
|---|---|---|
| Preflight | `internal/setup/preflight.sh` | Machine facts → decisions; fails fast on anything it cannot fix |
| Python stack | `internal/setup/install_python_stack.sh` | `uv sync --frozen` of `internal/setup/stack/uv.lock` |
| CUDA user space | `internal/setup/ensure_cuda_userspace.sh` | forward-compat libs if the driver is old; toolkit if `nvcc` is missing |
| Apptainer | `internal/setup/ensure_apptainer.sh` | uses `POLAR_APPTAINER_BIN` / PATH, else unprivileged install |
| Checkouts | `internal/pipeline.sh` | Slime v0.3.0 + its canonical Megatron commit and patch, router-token patch; installed editable over the lock |
| Training stack | `internal/setup/ensure_training_stack.sh` | Transformer Engine torch bindings built against the locked core, flash-attn on B200 |
| Assets | `internal/prepare_data.py`, `internal/prepare_apptainer_images.py` | 293-task JSONL, per-task SIFs, pinned agent CLIs |
| Checkpoint | `internal/pipeline.sh`, `internal/convert_weights.sh` | full HF snapshot download, HF → Megatron torch_dist |
| Train | `internal/run.sh` | Polar services + Ray + Slime `train_async.py` |

### The environment

All python packages come from one lock, `internal/setup/stack/uv.lock`,
generated from `internal/setup/stack/pyproject.toml`. The set is anchored on `sglang==0.5.13` (the
version Polar's token-metadata patch targets), which fixes torch 2.11+cu130 and
CUDA 13; Slime v0.3.0 is locked as a git dependency so its requirements resolve
together with everything else; Transformer Engine is the 2.12 cu13 core, whose
prebuilt library runs on the cuBLAS torch ships. Three constraints
(`numpy<2`, `scipy<1.14`, `wandb<0.29`) keep the resolver on versions Slime's
code can use; there are no overrides.

Two things sit on top of the lock because they cannot be wheels: the patched
Slime and Megatron checkouts (editable, `--no-deps`) and the Transformer Engine
torch bindings (built from source against the locked torch). The sync is
`--inexact`, so re-running the launcher keeps them.

To change a version, edit `internal/setup/stack/pyproject.toml` and run
`uv lock` in that directory (from any OS; the lock is restricted to linux x86_64), then
re-run the launcher.

Machine-side, preflight decides what is missing and installs it under
`WORKROOT` without touching the system: CUDA forward-compat libraries when the
driver predates CUDA 13, a conda-forge CUDA 13.0 toolkit (via pixi) when there
is no `nvcc` for the TE build, and an unprivileged Apptainer when there is no
container runtime.

## Machine settings

Anything about the machine rather than the experiment stays an environment
variable, exported before `launch.sh` or `slurm_launch.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `WORKROOT` | `<repo>/tmp` | Root for checkouts, caches, toolchains, checkpoints; shared filesystem on multi-node |
| `POLAR_ROLLOUT_PORT`, `POLAR_GATEWAY_PORT`, `SGLANG_ROUTER_PORT`, `RAY_DASHBOARD_PORT`, `RAY_GCS_PORT` | 8080, 8100, 9000, 8265, 6379 | Service ports |
| `POLAR_APPTAINER_BIN`, `APPTAINER_IMAGE_DIR`, `AGENT_CLI_DIR` | auto | Container runtime and prepared assets |
| `SETUP_ENV` | 1 | `0` skips preflight and all `internal/setup/` scripts (bring your own environment) |
| `BUILD_MAX_JOBS` | 8 | Parallelism cap for TE / flash-attn source builds |
| `RUN_TRAINING` | 1 | `0` stops after setup and conversion |
| `RUN_ID` | config `name` | Override to keep several runs of one config apart |

Multi-node without slurm: on every worker node run
`bash internal/ray_worker_join.sh <head-ip>`; on the head run

```bash
RAY_HEAD_IP=<head-ip> POLAR_BIND_HOST=0.0.0.0 POLAR_PUBLIC_HOST=<head-ip> \
  bash examples/swegym_slime_grpo/launch.sh configs/qwen35-4b-2node.yaml
```

## Watching a run

The Slime driver's training output (step metrics, checkpoint saves) goes to
the Ray job, not to your shell. Follow it with wandb, or
`cat $SAVE_DIR/latest_checkpointed_iteration.txt`. To inspect live agent
sessions start the dashboard from the repo root:

```bash
.venv/bin/polar dashboard -c $WORKROOT/swegym_slime_grpo/topology.yaml
```

## Files

| Path | Purpose |
|---|---|
| `launch.sh` | Entry point: `launch.sh configs/<run>.yaml [--dry-run]` |
| `configs/` | Run configs (model, GPU layout, rollout and training hyperparameters) |
| `slurm_launch.sh` | Submit a config as a multi-node sbatch job |
| `internal/` | Everything the launcher runs for you: `pipeline.sh`, `run.sh`, `convert_weights.sh`, `head_entry.sh`, `ray_worker_join.sh`, data and image prep, Polar templates, Megatron model args |
| `internal/setup/` | Preflight and environment scripts; `stack/` holds the locked python environment |
