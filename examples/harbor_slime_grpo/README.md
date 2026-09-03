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

## Quick start (TMax-15k)

```bash
export WORKROOT=/shared/fs/harbor-grpo            # shared by all nodes
export WANDB_API_KEY=<key>                        # optional

# 1. Materialize the task directory (HF dataset: prompts, tests, prebuilt image tags)
uv run python examples/harbor_slime_grpo/datasets/tmax15k.py --output $WORKROOT/tasks/tmax15k

# 2. Check the plan without touching GPUs
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/tmax15k-qwen35-9b-2node.yaml --dry-run

# 3a. Single node smoke (environment setup, images, harness, checkpoint conversion, 2 steps)
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/tmax15k-smoke-1node.yaml

# 3b. Two nodes under slurm
bash examples/harbor_slime_grpo/multinode/sbatch_launch.sh \
    --config examples/harbor_slime_grpo/configs/tmax15k-qwen35-9b-2node.yaml \
    --nodes 2 --partition <p> --account <a>
```

`launch.sh` is idempotent: environment setup, checkouts, image pulls and the
checkpoint conversion are skipped when already present. Image pulls and the
harness install need network; on clusters whose compute nodes have no egress,
run the launcher once on a login node with `RUN_TRAINING=0` to prepare assets.

## The run config

```yaml
name: tmax15k-qwen35-9b-2node
tasks:
  dir: ${WORKROOT}/tasks/tmax15k    # the task directory (env vars expand)
  n: 32                             # random sample of tasks (omit for all)
  seed: 0
  # task_ids_file / exclude_ids_file: one directory name or source_id per line
harness:
  name: mini_swe_agent              # codex | opencode | claude_code | qwen_code | pi | hermes | mini_swe_agent
  settings: {step_limit: 64, cost_limit: 0}   # passed to the Polar harness preset
model:
  hf_checkpoint: Qwen/Qwen3.5-9B
  model_args_file: model_args_9b.sh # model_args.sh for Qwen3.5-4B
  end_of_turn_token_id: 248046
training:
  sync: true                        # train.py (on-policy) or train_async.py (1 step off-policy + TIS)
  tp_size: 4
  context_parallel_size: 2
  actor_num_gpus: 8                 # whole nodes when multi-node; the rest serve
  rollout_batch_size: 8             # prompts per step
  n_samples_per_prompt: 16
  num_epoch: 30
  max_tokens_per_gpu: 16384         # trace cap = this x context_parallel_size
  sglang_context_length: 32768      # keep equal to the trace cap
  ...                               # see configs/ for the full list and defaults
```

`launch.sh --dry-run` prints every variable `run.sh` receives and the rendered
Polar config, so the mapping is never hidden.

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
3 nodes = 8 train / 16 serve. `multinode/head_entry.sh <config>` runs on the first
node: it starts `multinode/ray_worker_join.sh` on the others with `srun`, exports
the head IP and bind hosts, then runs `launch.sh`. `run.sh` waits for all Ray
nodes before submitting the job. Without slurm, run `ray_worker_join.sh <head-ip>`
on each worker and `NUM_NODES=2 RAY_HEAD_IP=<ip> POLAR_BIND_HOST=0.0.0.0
POLAR_PUBLIC_HOST=<ip> bash launch.sh <config>` on the head.

Ports (`POLAR_ROLLOUT_PORT` 8080, `POLAR_GATEWAY_PORT` 8100, `SGLANG_ROUTER_PORT`
9000, Ray 8265/6379) are environment knobs; preflight refuses ports already in use.

## Sandboxes: Apptainer, harness, environment

- **Images.** One SIF per distinct `docker_image`, pulled once into
  `APPTAINER_IMAGE_DIR` (`$WORKROOT/harbor_sif_images`). `HARBOR_SIF_SEED_DIR`
  reuses SIFs from an existing Harbor cache. Docker Hub rate limits apply; set
  `APPTAINER_DOCKER_USERNAME/PASSWORD` for large pulls.
- **Harness.** Built once by `prepare_harness.sh` into `$WORKROOT/harbor_harness`
  and bind-mounted read-only at the same path in every container, so task images
  need nothing preinstalled: Node CLIs under `node/`, mini-swe-agent as a uv tool
  with its own Python 3.12. No per-trial install, no egress from the sandbox.
- **Container environment.** `polar_config.yaml` sets `HOME=/polar/session/home`,
  prepends the harness to `PATH`, mounts `/harbor_data`, uses host networking, and
  writes to a host-backed overlay (Polar's Apptainer runtime). Edit the template
  for image-specific needs (extra `PATH` entries, `LD_LIBRARY_PATH`, GPUs).
- **Length.** The trainer drops any trace longer than
  `max_tokens_per_gpu x context_parallel_size` (a fully masked placeholder takes its
  place). Keep `sglang_context_length` equal to that cap so the agent cannot produce
  a trace the trainer will censor; raise the cap with more CP (more trainer GPUs) or
  more tokens per GPU (memory permitting; 16384 fits H100-80GB for 9B at TP4).

## Watching a run

Slime's step metrics go to wandb and the Ray job log, not to your shell. Watch
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

| File | Purpose |
|---|---|
| `launch.sh` | One entry: run config -> environment -> tasks/images/harness -> checkpoint -> `run.sh`; `--dry-run` |
| `run.sh` | Polar services + Ray + Slime (`train.py` or `train_async.py`) |
| `configs/` | Run configs (2-node reference, 1-node smoke) |
| `datasets/tmax15k.py` | TMax-15k HF dataset -> task directory (the only TMax-specific file) |
| `prepare_tasks.py` | Task directory -> prompt JSONL + image list, with sampling and id filters |
| `prepare_images.sh`, `prepare_harness.sh` | SIF pulls; harness directory (Node CLIs, mini-swe-agent) |
| `polar_config.yaml`, `topology.yaml` | Polar templates (`@TOKENS@` from the config, `${VARS}` from `run.sh`) |
| `convert_weights.sh`, `model_args*.sh` | HF -> Megatron torch_dist; Qwen3.5-9B / 4B args |
| `setup/` | Preflight and environment scripts (venv, CUDA user space, Apptainer, Transformer Engine) |
| `multinode/` | `sbatch_launch.sh`, `head_entry.sh`, `ray_worker_join.sh` |
