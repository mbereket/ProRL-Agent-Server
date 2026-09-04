# Harbor Slime GRPO

Train a coding agent with GRPO on **any directory of Harbor tasks**, multi-node,
with Apptainer sandboxes: a harness (mini-SWE-agent by default) solves each task
inside the task's own container, **Polar** records the token-level trajectory and
grades it with the task's `tests/test.sh`, and **Slime** (Megatron + SGLang) does
the policy update. One YAML run config describes the whole run. Ships with
[TMax-15k](https://arxiv.org/abs/2606.23321) (14.6k terminal tasks with prebuilt
images) as the reference dataset; swapping in your own tasks means pointing the
config at a different directory.

Tested layout: two 8-GPU H100 nodes, Qwen3.5-9B, trainer TP4 x CP2 on one node
and 8 SGLang engines on the other. One node (4 train / 4 serve) works for smokes.

## The task contract

A task directory is a [Harbor task](https://www.harborframework.com/docs/tasks):

```
<tasks>/manifest.json            optional: {"tasks": [{"directory": ..., "source_id": ...}]}
<tasks>/harbor/<task>/           (or <tasks>/<task>/ without a manifest)
    instruction.md               the prompt the agent receives
    task.toml                    [environment] docker_image (pullable ref) + workdir,
                                 optional agent_path_prepend (first on the agent's PATH, e.g. a conda env)
                                 [agent] timeout_sec, [verifier] timeout_sec
    tests/test.sh                verifier; writes 0..1 to /logs/verifier/reward.txt
    environment/files/setup.sh   optional staging run before the agent starts, with
                                 WORKDIR and HARBOR_STAGING (=environment/files) set
```

`docker_image` must be pullable (`apptainer pull docker://...`): this example does
not build images. `prepare_tasks.py` turns the directory into Slime's prompt JSONL
and the image list; `prepare_images.sh` pulls the SIFs; Polar mounts the task
directory read-only at `/harbor_data`, runs `setup.sh` if present, runs the agent,
then uploads `tests/` and runs `test.sh` (the agent never sees the tests).

## Quick start

Two datasets ship with the example. **SWE-Gym-Lite** (230 Python bug-fix tasks
from 11 repositories, prebuilt images, the Harbor SWE-Gym adapter's format) is
the smaller starting point; **TMax-15k** (14.6k terminal tasks) is the larger
reference set.

```bash
export WORKROOT=/shared/fs/harbor-grpo            # shared by all nodes
export WANDB_API_KEY=<key>                        # optional

# 1. Materialize a task directory (needs the venv: run launch.sh once with RUN_TRAINING=0, or any python with `datasets`)
.venv/bin/python examples/harbor_slime_grpo/datasets/swegym_lite.py --output $WORKROOT/tasks/swegym-lite
.venv/bin/python examples/harbor_slime_grpo/datasets/tmax15k.py --output $WORKROOT/tasks/tmax15k

# 2. Check what a config resolves to (prompts, rendered Polar config) without touching GPUs
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/swegym-lite-smoke-1node.yaml --dry-run

# 3a. Single-node smoke (environment setup, images, harness, checkpoint conversion, 2 steps)
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/swegym-lite-smoke-1node.yaml

# 3b. Two nodes under slurm (node count comes from the config)
bash examples/harbor_slime_grpo/slurm_launch.sh \
    --config examples/harbor_slime_grpo/configs/swegym-lite-qwen35-9b-2node.yaml --partition <p> --account <a>
```

Shipped configs: `swegym-lite-smoke-1node`, `swegym-lite-qwen35-9b-2node`,
`tmax15k-smoke-1node`, `tmax15k-qwen35-9b-2node`.

`launch.sh` is idempotent: environment setup, checkouts, image pulls and the
checkpoint conversion are skipped when already present. Image pulls and the
harness install need network; on clusters whose compute nodes have no egress,
run the launcher once on a login node with `RUN_TRAINING=0` to prepare assets.

## The run config

A run is one YAML file in `configs/`. Every key has a default; `name` and
`tasks.dir` are required; unknown keys are rejected. The sections mirror the
swegym example (`model`, `cluster`, `rollout`, `training`, `eval`, `wandb`) plus
the two Harbor-specific ones, `tasks` and `harness`.

```yaml
name: tmax15k-qwen35-9b-2node
tasks:
  dir: ${WORKROOT}/tasks/tmax15k    # the task directory (env vars expand; relative to this file)
  dataset: tmax15k                  # optional: datasets/<name>.py creates dir when it is missing
  dataset_args: ""                  # extra flags for that script (e.g. --limit 10)
  n: 32                             # random sample of tasks (omit for all)
  seed: 0
  # task_ids_file / exclude_ids_file: one directory name or source_id per line
harness:
  name: mini_swe_agent              # codex | opencode | claude_code | qwen_code | pi | hermes | mini_swe_agent
  model_name: openai/gpt-5.4        # cosmetic; the gateway serves the trained model
  settings: {step_limit: 64, cost_limit: 0}   # passed to the Polar harness preset
  session_timeout: 3000             # per-session budget: agent + verifier + margin
  request_timeout: 3600
  max_run_workers: 16               # concurrent sandboxes per sandbox host
  max_async_level: 1
model:
  hf_checkpoint: Qwen/Qwen3.5-9B
  model_args_file: model_args_9b.sh # Megatron args in internal/ (model_args.sh for Qwen3.5-4B)
  end_of_turn_token_id: 248046
cluster:
  num_nodes: 2
  actor_num_gpus: 8                 # whole nodes when multi-node; every other GPU serves an engine
  tp_size: 4                        # TP x CP must divide actor_num_gpus
  context_parallel_size: 2
  sandbox_nodes: all                # head | all: hosts whose CPUs run agent sandboxes + verifiers
rollout:
  batch_size: 8                     # prompts per step
  n_samples_per_prompt: 16
  num_epoch: 30
  max_prompt_len: 24000
  max_response_len: 8000
  sglang_context_length: 32768      # keep equal to the trace cap
training:
  sync: true                        # train.py (on-policy) or train_async.py (1 step off-policy + TIS)
  max_tokens_per_gpu: 16384         # longest trainable trace = this x context_parallel_size
  lr: 1e-6
  use_kl_loss: false
  grpo_std_normalization: false     # false = mean-only advantages
  group_id_scope: trajectory
  timeout_reward_zero: true
  drop_zero_variance_groups: true
  save_interval: 5
  extra_train_args: ""              # appended to the slime train script verbatim
eval:
  prompt_data: ""                   # "<name> <path.jsonl>" enables a held-out eval
  interval: 10
  n_samples_per_prompt: 1
wandb:
  project: harbor-slime-grpo
  group: <name>
```

`launch.sh --dry-run` prints every variable the pipeline receives, builds the
prompt list and renders the Polar config, so the mapping is never hidden.

### Bring your own tasks

1. Put them in the layout above. Existing Harbor datasets export into it directly
   (`harbor datasets download <name> --export -o <dir>`) as long as `task.toml`
   names a pullable `docker_image`; datasets that ship only a `Dockerfile` need
   the images built and pushed first (see `examples/tmax-15k/build_images.py`).
2. Point `tasks.dir` at the directory. Nothing else changes.
3. If the tasks need staging (data copied into the workdir, services started),
   add `environment/files/setup.sh`; it runs in the container before the agent.

Rewards are whatever `test.sh` writes. Slime sees a scalar in [0, 1]; the bridge
drops groups whose rewards are all equal (`drop_zero_variance_groups`) and scores
timeouts as 0 (`timeout_reward_zero`), both configurable.

## Multi-node

The trainer takes `actor_num_gpus` GPUs and every other GPU in the Ray cluster
serves an SGLang engine. On more than one node the trainer must take whole nodes
(slime v0.3.0 assigns engine addresses per node), so 2 nodes = 8 train / 8 serve,
3 nodes = 8 train / 16 serve. `internal/head_entry.sh <config>` runs on the first
node: it starts `internal/ray_worker_join.sh` on the others with `srun`, exports
the head IP and bind hosts, then runs `launch.sh`. `run.sh` waits for all Ray
nodes before submitting the job.

Sandboxes are CPU work (the agent CLI, the task container, the verifier) and
run wherever a Polar gateway node runs. `cluster.sandbox_nodes: head` keeps one
gateway on the head; `all` puts one on every node (`node-01` head, `node-02`..
workers), each with `harness.max_run_workers` slots, and the rollout server
dispatches to the least-loaded healthy one. Under slurm `run.sh` starts the
worker gateways itself with `srun`; without slurm it uses `ssh <host>`, or start
them by hand: `polar serve_gateway -c $WORKROOT/harbor_slime_grpo/<run>/topology.yaml
--node-id node-0N` on each worker, and export `WORKER_HOSTS`/`WORKER_IPS`
(comma lists) before `launch.sh`. Without slurm, run `ray_worker_join.sh <head-ip>`
on each worker and `RAY_HEAD_IP=<ip> POLAR_BIND_HOST=0.0.0.0
POLAR_PUBLIC_HOST=<ip> bash launch.sh <config>` on the head.

Ports (`POLAR_ROLLOUT_PORT` 8080, `POLAR_GATEWAY_PORT` 8100, `SGLANG_ROUTER_PORT`
9000, Ray 8265/6379) are environment knobs; preflight refuses ports already in use.

## Sandboxes: Apptainer, harness, environment

- **Images.** One SIF per distinct `docker_image`, pulled once into
  `APPTAINER_IMAGE_DIR` (`$WORKROOT/harbor_sif_images`). `HARBOR_SIF_SEED_DIR`
  reuses SIFs from an existing Harbor cache. Docker Hub rate limits apply; set
  `APPTAINER_DOCKER_USERNAME/PASSWORD` for large pulls.
- **Harness.** Built once by `internal/prepare_harness.sh` into `$WORKROOT/harbor_harness`
  and bind-mounted read-only at the same path in every container, so task images
  need nothing preinstalled: Node CLIs under `node/`, mini-swe-agent as a uv tool
  with its own Python 3.12. No per-trial install, no egress from the sandbox.
- **Container environment.** `internal/polar_config.yaml` sets `HOME=/polar/session/home`,
  prepends the harness to `PATH`, mounts `/harbor_data`, uses host networking, and
  writes to a host-backed overlay (Polar's Apptainer runtime). Edit the template
  for image-specific needs (extra `PATH` entries, `LD_LIBRARY_PATH`, GPUs).
- **Length.** The trainer drops any trace longer than
  `max_tokens_per_gpu x context_parallel_size` (a fully masked placeholder takes its
  place). Keep `sglang_context_length` equal to that cap so the agent cannot produce
  a trace the trainer will censor; raise the cap with more CP (more trainer GPUs) or
  more tokens per GPU (memory permitting; 16384 fits H100-80GB for 9B at TP4).

## Watching a run

Per-session agent logs and artifacts are under
`$WORKROOT/harbor_slime_grpo/<run>/sessions/session-<id>/logs/agent/` on every
sandbox host. Slime's step metrics go to wandb and the Ray job log, not to your shell. Watch
`$SAVE_ROOT/<run>/latest_checkpointed_iteration.txt`, wandb, or the Polar
dashboard (`polar dashboard -c $WORKROOT/harbor_slime_grpo/<run>/topology.yaml`).
Useful metrics: `polar/reward_mean`, `polar/dropped_*` (should be 0 except
zero-variance), `polar/rollout_success_rate`, and the TIS statistics (`tis`,
`tis_abs`, `tis_clipfrac`): with `sync: true` and weight sync every step, `tis`
should sit at 1.000 and `tis_clipfrac` at 0. Anything else points at a
sampling/trainer mismatch or a tokenization mismatch in prefix merging.

## Differences from the TMax paper's training run

This is a reusable base, not a reproduction. Compared with the recipe in the TMax
paper (Table 13 and section 4.1):

| | TMax paper | this example |
|---|---|---|
| harness | their mini-SWE-agent-derived Vanillux agent | Polar's `mini_swe_agent` preset (or any listed harness) |
| algorithm | DPPO (binary TV mask, threshold 0.1), token-level loss | GRPO with TIS, per-trajectory aggregation, clip 0.2/0.28 |
| async | 4 async steps, active sampling | synchronous by default (`sync: false` gives 1-step async) |
| batch | 8 prompts x 32 samples | 8 x 16 (config) |
| length | 2k prompt + 65k response, 16k per turn, 64 steps | trace cap `max_tokens_per_gpu x CP`; 64 steps |
| LM head | FP32 (they report it as important for Qwen3.5 mismatch) | bf16; watch `tis_*` |
| infra | open-instruct + vLLM, 2 train + 6 inference nodes | Polar + Slime (Megatron + SGLang), 1 train + 1 inference node |
| data | all 14.6k tasks | `tasks.n` random sample (32 in the config) |

## Files

| Path | Purpose |
|---|---|
| `launch.sh` | Entry point: `launch.sh configs/<run>.yaml [--dry-run]` |
| `slurm_launch.sh` | Submit a config as a multi-node sbatch job (node count from the config) |
| `configs/` | Run configs: 2-node reference and 1-node smoke per dataset |
| `datasets/swegym_lite.py`, `datasets/swegym_test.sh.tmpl` | SWE-Gym / SWE-Gym-Lite HF dataset -> task directory, with the Harbor adapter's verifier |
| `datasets/tmax15k.py` | TMax-15k HF dataset -> task directory |
| `internal/` | Everything the launcher runs for you: `pipeline.sh`, `run.sh`, `convert_weights.sh`, `head_entry.sh`, `ray_worker_join.sh`, `prepare_tasks.py`, `prepare_images.sh`, `prepare_harness.sh`, Polar templates, Megatron model args |
| `internal/setup/` | Preflight and environment scripts; `stack/` holds the locked python environment (shared recipe with the swegym example) |
