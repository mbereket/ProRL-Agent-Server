# Harbor Slime GRPO

GRPO on **any directory of Harbor tasks** with **any supported agent harness**,
multi-node, Apptainer sandboxes. Polar sits between the agent sandbox and the
SGLang engines, records token-level trajectories and rewards; Slime (Megatron +
SGLang) updates the policy. One YAML run config in `configs/` drives a run.

## Requirements

- **Linux x86_64 with NVIDIA GPUs.** The locked stack is CUDA 13 / torch 2.11
  and needs glibc 2.35+; an older kernel driver is fine (forward-compat
  libraries are installed automatically).
- **A shared filesystem** (`WORKROOT`) mounted at the same path on every node.
  Installs, task images, checkpoints and caches go there (compiler caches to
  node-local `/tmp`); cache variables you export yourself (`HF_HOME`,
  `UV_CACHE_DIR`, `APPTAINER_CACHEDIR`, ...) are honored.
- **Apptainer or Singularity**, or user namespaces enabled so an unprivileged
  Apptainer can be installed for you. Task images are pulled, never built.
- **Slurm** for multi-node runs (single node needs none).
- **Network during setup**: GitHub, PyPI, Hugging Face, your image registry.
  Training itself needs no egress.
- **Host tools**: `gcc`/`g++`, `git`, `curl`, `tar`, `xz`. Python, `uv`, the
  CUDA toolkit and Apptainer are installed under `WORKROOT` if missing.

`internal/setup/preflight.sh` checks all of this and says exactly what is missing.

### What has been tested

| | tested |
|---|---|
| hardware | H100-80GB nodes (8 GPUs), slurm; 1, 2 and 4 nodes |
| models | Qwen3.5-9B (all runs below); Qwen3.5-4B (earlier stack version) |
| harnesses | codex (SWE-Gym, bio-synth), opencode (BixBench-Hypothesis, judge-graded), mini_swe_agent (smoke) |
| datasets | SWE-Gym-Lite via `datasets/swegym_lite.py`; bio-synth task sets published on HF in Harbor layout |
| layouts | TP4 (1 node smoke); TP4 x CP2, 32k traces (2 nodes); TP4 x CP4, 128k traces with optimizer offload (4 nodes) |
| training modes | synchronous (reference); asynchronous `max_async_level: 2` on 4 nodes, 16 steps |
| not yet | non-Qwen models, B200, multi-node without slurm |

## Setup and first run

```bash
export WORKROOT=/shared/fs/harbor-grpo
export WANDB_API_KEY=<key>                  # optional; without it wandb runs offline

# 1. A directory of Harbor tasks. Registry datasets download directly; datasets
#    published on HF in Harbor layout download with hf; SWE-Gym needs the converter.
harbor datasets download <name> -o $WORKROOT/tasks                                  # -> $WORKROOT/tasks/<name>/
hf download <org>/<dataset> --repo-type dataset --local-dir $WORKROOT/tasks/<name>
python examples/harbor_slime_grpo/datasets/swegym_lite.py --output $WORKROOT/tasks/swegym-lite

# 2. Pull the task images as SIFs (needs the registry: login node). A task directory
#    pulls everything it references; a run config pulls only its selected tasks.
bash examples/harbor_slime_grpo/prepare_images.sh $WORKROOT/tasks/swegym-lite
bash examples/harbor_slime_grpo/prepare_images.sh examples/harbor_slime_grpo/configs/smoke-1node.yaml

# 3. Run. First run also builds the venv, the harness and converts the checkpoint.
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/smoke-1node.yaml --dry-run     # resolve + render only
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/smoke-1node.yaml               # single node, no slurm
bash examples/harbor_slime_grpo/launch.sh examples/harbor_slime_grpo/configs/swegym-lite-9b-2node.yaml \
    --slurm --partition <p> --account <a> --time 04:00:00                                                    # sbatch, node count from the config
```

`launch.sh <cfg> --setup-only` does everything but training (use it on a login
node when compute nodes have no network). Resuming: submit the same config
again; the run is keyed by `name` and reloads its latest checkpoint. Do not
change `rollout.num_steps` on resume (the LR schedule is sized from it).

Where things end up:

| Path | Contents |
|---|---|
| `$WORKROOT/harbor_slime_grpo/<name>/` | per run: `env.sh`, `train.jsonl` (prompts), rendered `polar_config.yaml`, `topology.yaml`, `train_args.sh`, `sessions/`, `rollout_results/` |
| `$WORKROOT/ckpt/harbor_slime_grpo/<name>/` | Megatron checkpoints, `latest_checkpointed_iteration.txt` |
| `$WORKROOT/joblogs/` | sbatch stdout (setup and Ray driver log; step metrics go to wandb) |
| `$WORKROOT/tasks/`, `harbor_sif_images/`, `harbor_harness/`, `checkpoints/` | shared assets, built once |

## The run config

One YAML file. `name` and `tasks.dir` are required, every other key has a
default, unknown keys are rejected. `internal/render.py` holds the schema and
every default; `${WORKROOT}` and other env vars expand in values; relative
paths resolve against the config file.

```yaml
name: my-run                          # run id: per-run dir and checkpoint dir are named after it

tasks:                                # WHAT to train on
  dir: ${WORKROOT}/tasks/swegym-lite  # directory of Harbor tasks (see "The task contract")
  n: 32                               # random subset (omit = all); seed: 0
  task_ids_file: ref8.txt             # or an explicit list (one dir name / source_id per line); exclude_ids_file likewise
  mount_root: ${WORKROOT}/tasks       # mount this instead of dir (tasks referencing shared data outside dir)

harness:                              # WHICH agent solves the tasks
  name: codex                         # codex | opencode | mini_swe_agent
  cli_version: ""                     # pin the CLI version installed into the shared harness dir
  model_name: openai/gpt-5.4          # name the CLI asks for; the gateway always serves the trained model
  settings: {}                        # harness-specific settings passed to the Polar preset
  thinking: true                      # Qwen3 models: chat-template thinking for every request (unset = template default, off)
  session_timeout: 1500               # per-attempt budget: agent + verifier + margin (seconds)
  request_timeout: 1500               # per-LLM-request timeout at the gateway
  max_run_workers: 16                 # concurrent sandboxes per sandbox node
  max_async_level: 1                  # rollout steps the sampler may run ahead (>1 needs training.sync: false)
  path_prepend: ""                    # first on the agent PATH inside every sandbox (e.g. an image's conda env)
  ld_library_path: ""                 # LD_LIBRARY_PATH inside the sandbox
  keep_sessions: false                # keep per-session dirs (agent logs, verifier output); millions of inodes on long runs

model:
  hf_checkpoint: Qwen/Qwen3.5-9B
  model_args_file: qwen3_5_9b.sh      # Megatron architecture args: internal/model_args/{qwen3_5_9b,qwen3_5_4b}.sh or an absolute path
  sglang_tool_call_parser: qwen3_coder  # must match the model's tool-call format; a mismatch returns tool calls as plain text
  load_dir: ""                        # start from another run's checkpoint dir instead of the base model
  torch_dist_dir: ""                  # converted reference checkpoint (default ${WORKROOT}/checkpoints/<model>_torch_dist)

cluster:                              # GPU layout. Trainer takes actor_num_gpus, every other GPU serves an SGLang engine.
  num_nodes: 2
  actor_num_gpus: 8                   # whole nodes when num_nodes > 1
  tp_size: 4                          # TP x CP must divide actor_num_gpus
  context_parallel_size: 2
  sandbox_nodes: all                  # head | all: nodes whose CPUs run sandboxes and verifiers (one Polar gateway each)

rollout:                              # HOW MUCH is sampled per step
  batch_size: 8                       # tasks per step
  n_samples_per_prompt: 16            # attempts per task -> 128 sessions per step
  num_steps: 100                      # training steps; 0 = eval only (needs eval.prompt_data)
  # num_epoch: 4                      # alternative: passes over the task set
  sglang_context_length: 32768        # longest trace the agent can build; keep equal to the trainer's trace cap
  max_prompt_len: 8000                # initial task prompts longer than this are skipped
  max_response_len: 24000             # feeds slime's response-length metrics only

training:
  sync: true                          # true: train.py (on-policy); false: train_async.py, generation overlaps training (TIS-corrected)
  max_tokens_per_gpu: 16384           # trace cap = this x context_parallel_size (32k here); 16384 fits H100-80GB for 9B TP4
  optimizer_cpu_offload: false        # Adam states on host; needed for 32768 tok/GPU (128k traces at CP4)
  lr: 1e-6
  use_kl_loss: false                  # kl_loss_coef: 0.001
  grpo_std_normalization: false       # false = mean-only advantages
  loss_denominator: trainable_units   # trainable_units | global_batch
  group_id_scope: trajectory          # trajectory: every attempt weighs the same; prompt: token-mean over a task's attempts
  drop_zero_variance_groups: true     # skip tasks whose attempts all got the same reward; false for overfit runs
  timeout_reward_zero: true           # sessions that hit session_timeout get reward 0
  overlong_policy: zero_reward_train  # zero_reward_train | drop: attempts that ran out of context
  save_interval: 5
  checkpoint_keep_every: 0            # >0: delete saved iterations that are not multiples of this (latest kept)
  extra_train_args: ""                # appended to the slime command line verbatim

eval:
  prompt_data: "held ${RUN_DIR}/train.jsonl"   # "<name> <jsonl>" enables a held-out eval; ${RUN_DIR} = this run's dir
  interval: 10
  n_samples_per_prompt: 1

judge:                                # LLM judge for rubric-graded tasks (empty = none)
  model: openai/<judge-model>
  api_base: https://<endpoint>/v1
  api_key_env: JUDGE_API_KEY          # host env var; injected into the sandbox and the verifier

wandb:
  project: harbor-slime-grpo
  group: <name>
```

`launch.sh <cfg> --dry-run` prints the resolved summary, writes `train.jsonl`
and shows the rendered Polar config and slime arguments, so nothing is hidden.

## The task contract

A task directory is a [Harbor task](https://www.harborframework.com/docs/tasks):

```
<tasks>/manifest.json            optional: {"tasks": [{"directory": ..., "source_id": ...}]}
<tasks>/harbor/<task>/           (or <tasks>/<task>/ without a manifest, as `harbor datasets download` writes)
    instruction.md               the prompt the agent receives
    task.toml                    [environment] docker_image (pullable ref) + workdir,
                                 optional agent_path_prepend (first on the agent's PATH, e.g. a conda env)
                                 [agent] timeout_sec, [verifier] timeout_sec
    tests/test.sh                verifier; writes a reward in [0, 1] to /logs/verifier/reward.txt
    environment/files/setup.sh   optional staging run before the agent starts, with
                                 WORKDIR and HARBOR_STAGING (=environment/files) set
```

- **Images must be pullable.** `docker_image` is fetched with `apptainer pull`
  by `prepare_images.sh`, one SIF per distinct image named the way Harbor's own
  singularity cache names them, so an existing Harbor cache works as
  `APPTAINER_IMAGE_DIR`. Datasets that ship only a `Dockerfile` need their
  images built, pushed, and named in `task.toml` first; this example never
  builds. Private registries: `APPTAINER_DOCKER_USERNAME/PASSWORD`.
- **The verifier is the reward.** `test.sh` runs inside the task image after the
  agent finishes; the agent never sees `tests/`. Rubric-graded tasks can call
  the `judge` model from `test.sh` (its address and key arrive as `RUBRIC_MODEL*`).
- **Nothing preinstalled in the image.** The harness (agent CLI, node or its own
  python) is built once into `$WORKROOT/harbor_harness` and bind-mounted
  read-only at the same path in every sandbox. If the agent needs the image's
  toolchain on PATH, set `agent_path_prepend` per task or `harness.path_prepend`.

`datasets/swegym_lite.py` is the one converter shipped here: SWE-Gym is not in
the Harbor registry and its Harbor form ships Dockerfiles, so it writes the
layout above from the HF dataset with the prebuilt SWE-Gym images and the
Harbor adapter's verifier.

Check a new dataset with `launch.sh <cfg> --dry-run`, then a 1-node smoke
(`rollout.batch_size: 2`, `n_samples_per_prompt: 4`, `num_steps: 2`) before
committing GPUs. For long-trace tasks, an eval-only run (`num_steps: 0` with
`eval.prompt_data`) at a generous `sglang_context_length` shows the
trace-length distribution and picks the trace cap.

## Multi-node

The trainer takes `actor_num_gpus` GPUs and every other GPU in the Ray cluster
serves an SGLang engine. On more than one node the trainer must take whole
nodes (slime assigns engine addresses per node): 2 nodes = 8 train / 8 serve,
3 nodes = 8 train / 16 serve. `internal/head_entry.sh <config>` runs on the
first node of the allocation: it starts `internal/ray_worker_join.sh` on the
others with `srun`, exports the head IP and bind hosts, then runs `launch.sh`.
`run.sh` waits for all Ray nodes before submitting the job.

Sandboxes are CPU work (agent CLI, task container, verifier) and run wherever a
Polar gateway runs. `cluster.sandbox_nodes: all` puts one gateway on every node
(`node-01` head, `node-02`.. workers, started with `srun`), each with
`harness.max_run_workers` slots; `head` keeps one on the head. Multi-node
without slurm is untested.

Ports (`POLAR_ROLLOUT_PORT` 8080, `POLAR_GATEWAY_PORT` 8100,
`SGLANG_ROUTER_PORT` 9000, Ray 8265/6379) are environment knobs; preflight
refuses ports already in use.

## Trace length

The trainer drops any trace longer than `max_tokens_per_gpu x
context_parallel_size` (a fully masked placeholder takes its place; see
`overlong_policy`). Keep `sglang_context_length` equal to that cap so the agent
cannot produce a trace the trainer will censor; raise the cap with more CP (more
trainer GPUs) or more tokens per GPU (memory permitting).

## Watching a run

Per-session agent logs and artifacts (with `harness.keep_sessions`) are under
`$WORKROOT/harbor_slime_grpo/<run>/sessions/` on every sandbox host. Slime's
step metrics go to wandb and the Ray driver log (in the job log), not to your
shell. Watch `$WORKROOT/ckpt/harbor_slime_grpo/<run>/latest_checkpointed_iteration.txt`,
wandb, or the Polar dashboard (`polar dashboard -c $WORKROOT/harbor_slime_grpo/<run>/topology.yaml`).
Useful metrics: `polar/reward_mean`, `polar/dropped_*` (0 except zero-variance),
`polar/rollout_success_rate`, and the TIS statistics (`tis`, `tis_abs`,
`tis_clipfrac`): with `sync: true` and weight sync every step, `tis` should sit
at 1.000 and `tis_clipfrac` at 0. Anything else points at a sampler/trainer
mismatch or a tokenization mismatch in prefix merging.

## Files

| Path | Purpose |
|---|---|
| `launch.sh` | Entry point: setup (idempotent), prompts, render, run; `--dry-run`, `--setup-only`, `--slurm` |
| `prepare_images.sh` | Pull task images as SIFs for a task directory or a run config |
| `configs/` | `smoke-1node.yaml`, `swegym-lite-9b-2node.yaml` |
| `datasets/swegym_lite.py` (+ `swegym_test.sh.tmpl`) | SWE-Gym / SWE-Gym-Lite HF dataset -> task directory |
| `internal/render.py` | Config schema and defaults; renders `polar_config.yaml`, `topology.yaml` and the slime argument list per run |
| `internal/run.sh` | Processes: Polar rollout server + gateways, Ray, `ray job submit` |
| `internal/prepare_tasks.py` | Task directory -> prompt JSONL, image list, SIF presence check |
| `internal/prepare_harness.sh`, `internal/convert_weights.sh` | Shared harness dir; HF -> torch_dist conversion |
| `internal/head_entry.sh`, `internal/ray_worker_join.sh` | Slurm multi-node entry and worker loop |
| `internal/polar_config.yaml`, `internal/topology.yaml` | Polar templates (the task template is where sandbox env, mounts and the verifier are defined) |
| `internal/model_args/` | Megatron architecture args per model |
| `internal/setup/` | Preflight and environment scripts; `stack/` holds the locked python environment |
