# Simplification plan

Working notes for cutting harbor_slime_grpo down. Nothing here is applied yet;
each section records the current state, the target, and the edits.

Target setup story for a user:

```bash
# 1. stage a Harbor task directory (any layout harbor writes; pullable docker_image per task)
harbor datasets download <name> -o $WORKROOT/tasks            # registry datasets
hf download <repo> --repo-type dataset --local-dir $WORKROOT/tasks/<name>   # datasets published in Harbor layout

# 2. pull the images those tasks reference into the shared SIF dir
bash examples/harbor_slime_grpo/prepare_images.sh $WORKROOT/tasks/<name>

# 3. point a config at the directory and launch
```

## 1. Dataset staging: remove all dataset code

Current: `datasets/{swegym_lite,tmax15k,bbh_max_v2}.py` (~560 lines) plus the
`tasks.dataset` / `tasks.dataset_args` config keys and the auto-materialize
block in `internal/pipeline.sh` that runs them when `tasks.dir` is missing.

Why they exist: SWE-Gym and TMax in their Harbor-registry form ship a
Dockerfile per task and Apptainer can only pull, so the scripts rewrite them
from HF sources that name prebuilt images. `bbh_max_v2.py` is a plain
`snapshot_download` plus a validation `prepare_tasks.py` repeats anyway.

Target: the pipeline consumes a Harbor task directory and nothing else.
`internal/prepare_tasks.py` already accepts both layouts (manifest + `harbor/`,
or plain dirs found by globbing `task.toml`), so a raw `harbor datasets
download --export` output works today.

Edits:
- Delete `datasets/bbh_max_v2.py`: staging is `hf download
  mbereket-nvidia/bio-synth-bbh-max-v2-train-v2 --repo-type dataset --local-dir <dir>`
  (TODO note already in the file's docstring).
- Keep `datasets/swegym_lite.py` + `swegym_test.sh.tmpl` in the example as the
  one documented special case: SWE-Gym is not in the Harbor registry and its
  Harbor form ships Dockerfiles, so a converter is genuinely needed. It becomes
  a plain "run once to create the task directory" script, not a config hook.
- Drop TMax: `datasets/tmax15k.py` and `configs/tmax15k-*.yaml`. Materialized,
  never trained.
- Remove `tasks.dataset` / `dataset_args` from `internal/config_to_env.py`,
  the materialize block in `pipeline.sh`, and the `launch.sh` error hint.
- README: replace "Adding a New Dataset" three-way list with the two download
  commands above, and state plainly that registry datasets that ship only a
  Dockerfile must have their images built and pushed first (no Docker on the
  cluster).

## 2. Image pulling: one user-facing script

Current: `prepare_tasks.py` writes `images.txt` (`ref<TAB>sif`) for the
selected subset; `internal/prepare_images.sh` pulls that list 4-wide into
`$WORKROOT/harbor_sif_images`, with our own hashed SIF names
(`sif_name_for`), a `HARBOR_SIF_SEED_DIR` symlink bridge to Harbor's cache
naming, and a completeness check. Runs inside `launch.sh` setup, gated by
`PREPARE_IMAGES`.

Harbor has no standalone pull command (SIF conversion happens only inside
the singularity environment's `start()`), so a script stays. Harbor's cache
naming is `<ref with / and : -> _>.sif`, with `:latest` appended when the ref
has no tag, flock + temp file.

Target: `prepare_images.sh <task_dir> [sif_dir]` at the example top level.
Reads distinct `docker_image` values straight from the task directory (same
discovery as `prepare_tasks.py`), pulls into the shared SIF dir with Harbor's
naming, parallel, temp-name-then-rename, fails if anything is missing.

Edits:
- Adopt Harbor's SIF naming in `prepare_tasks.py` (`sif_name_for` becomes the
  two-replace rule); delete the `HARBOR_SIF_SEED_DIR` code since a Harbor cache
  dir is then usable as `APPTAINER_IMAGE_DIR` directly.
- Move `internal/prepare_images.sh` to the top level, take a task directory
  instead of `images.txt`; drop `images.txt` as an artifact.
- Decide whether `launch.sh` still pulls implicitly. Leaning: keep an implicit
  pull of the selected subset for the smoke path, but document the explicit
  script as the normal step (big datasets, login-node network).

## Open

- Keep pulls per dataset directory (explicit script) vs per selected subset
  (current implicit behavior). SWE-Gym full is ~2400 images.

## 3. Everything else (draft, not agreed yet)

Line budget today, lockfile excluded: ~2700. What needs correctness review is
small: the Polar task template, the slime argument list, prompt JSONL
construction, checkpoint load/resume. The rest is portability and convenience.

| layer | files | lines | plan |
|---|---|---|---|
| environment install | `internal/setup/*`, checkout + Megatron block in `pipeline.sh`, `stack/` | ~750 | OPEN: image vs in-place (below) |
| config plumbing | `config_to_env.py`, `@TOKENS@` render, `envsubst`, `expand_gateway_nodes.py` | ~300 | one `render.py` |
| orchestration | `run.sh` | 497 | trim to ~250 |
| multi-node | `slurm_launch.sh`, `head_entry.sh`, `ray_worker_join.sh` | 152 | keep |
| asset prep | `prepare_tasks.py`, `prepare_images.sh`, `prepare_harness.sh`, `convert_weights.sh`, `model_args/` | ~420 | keep; trim harness list |
| feature knobs | judge, fp32 head, prune loop, eval, num_epoch, sandbox_nodes, ssh fallback, setup toggles | ~150 | delete |

### 3a. One render step

Today: run config -> ~70 env vars (`config_to_env.py`) -> `@TOKENS@`
substitution (python in `pipeline.sh`) -> `${VARS}` via `envsubst` (`run.sh`)
-> `expand_gateway_nodes.py`. Defaults duplicated between the schema and
`run.sh`'s `${X:-default}` fallbacks.

Target: one `internal/render.py` that reads the run config plus runtime facts
(head IP, worker IPs, ports, run dir) and writes `polar_config.yaml`,
`topology.yaml` (all gateway nodes) and the slime argument list. Every default
lives there. `run.sh` only manages processes: polar servers, ray, `ray job
submit`.

### 3b. Feature inventory (decided 09-06)

Agreed: render step (3a) yes; no image packaging for now (3c); keep the
in-place setup scripts and simplify them separately later.

Optional knobs, each with what it does and where it lives. Core knobs (batch
size, TP/CP, lr, nodes) are not candidates and are omitted.

| # | feature | what | implementation | status |
|---|---|---|---|---|
| 1 | `judge.*` | LLM judge for rubric tasks | RUBRIC_* env into sandbox + evaluator env; key from host env at render | keep |
| 2 | `model.fp32_lm_head` | fp32 trainer output layer | custom model provider flag; sampler flag via extra_train_args | DROP |
| 3 | `training.checkpoint_keep_every` | prune non-multiple iterations | 5-min loop in run.sh + prune_checkpoints.sh | keep |
| 4 | `eval.*` | held-out eval every N steps | 3 slime flags + ${RUN_DIR} substitution | keep |
| 5 | `num_steps: 0` | eval-only run | adds --lr-decay-iters 1 --no-load-optim --no-load-rng | keep |
| 6 | `rollout.num_epoch` | steps as passes over tasks | --num-epoch instead of --num-rollout | keep |
| 7 | `cluster.sandbox_nodes` | gateways on head or all nodes | worker IPs from head_entry, srun (or ssh) per host, expand_gateway_nodes.py | keep head\|all; DROP ssh fallback (slurm srun only) |
| 8 | `model.load_dir` | start from another run's ckpt | load dir override | keep |
| 9 | `model.torch_dist_dir` | converted ref ckpt location | path override | keep |
| 10 | `tasks.mount_root` | mount parent of task dir | relative task paths | keep |
| 11 | `tasks.n/seed`, `task_ids_file`, `exclude_ids_file` | subset selection | prepare_tasks.py | keep |
| 12 | `harness.thinking` | force chat-template thinking | enable_thinking on gateway inference | keep |
| 13 | `harness.keep_sessions` | keep session dirs | env var into gateway | keep |
| 14 | `harness.path_prepend`, `ld_library_path` | sandbox PATH/libs | template env block | keep |
| 15 | `harness.cli_version` | pin CLI version | per-harness env var into prepare_harness.sh | keep |
| 16 | `harness.settings` | preset settings JSON | passthrough into template | keep |
| 17 | `training.sync` + `max_async_level` | sync vs async trainer | train.py vs train_async.py; Polar async level | keep |
| 18 | `training.optimizer_cpu_offload` | Adam on host | 4 Megatron flags | keep |
| 19 | bridge knobs (loss_denominator, group_id_scope, overlong_policy, timeout_reward_zero, drop_zero_variance_groups) | loss/reward shaping | one flag or Polar key each | keep |
| 20 | `training.extra_train_args` | passthrough slime flags | whitespace split | keep |
| 21 | resume | reload latest own ckpt else ref @ rollout 0 | latest_checkpointed_iteration.txt check | keep |
| 22 | compiler caches on node-local /tmp + warm seed | avoid network-fs corruption | 7 env vars into Ray runtime env | keep |
| 23 | setup toggles (SETUP_ENV, INSTALL_*, PREPARE_*, CONVERT_WEIGHTS, RUN_TRAINING) | skip setup steps | env gates in pipeline.sh | DROP; one --setup-only flag |
| 24 | `--dry-run` | render without GPUs | flag through launch/pipeline | keep |
| 25 | `HARBOR_SIF_SEED_DIR` | reuse Harbor cache SIFs | symlink in prepare_images.sh | DROP |
| 26 | concurrent-job setup lock | serialize shared installs | flock in pipeline.sh | keep |

Removal list (everything else stays): fp32_lm_head; ssh gateway fallback; setup toggles -> `--setup-only`; HARBOR_SIF_SEED_DIR;
harnesses other than codex, opencode, mini_swe_agent; qwen3_8b.sh; TMax script + configs;
bbh_max_v2.py; tasks.dataset hook.

Also: `prepare_harness.sh` stays (task images ship no agent CLI; the shared
harness dir is bind-mounted into every sandbox); trim to codex, opencode, mini_swe_agent.

### 3c. Environment: decided, not now

Prebuilt training image rejected for now: Polar gateways spawn Apptainer
sandboxes, and Apptainer inside Apptainer is fragile (user-namespace nesting,
setuid helpers). The clean form would be gateways on the host in a CPU venv and
Ray + slime inside the container; untested, deferred. Setup scripts stay and
get simplified on their own later.

### Target layout

```
harbor_slime_grpo/
  README.md
  simplification.md
  configs/{smoke-1node,swegym-lite-9b-2node}.yaml
  launch.sh                 # local, or --slurm; idempotent setup then run
  prepare_images.sh         # <task_dir> -> SIFs (section 2)
  datasets/swegym_lite.py + swegym_test.sh.tmpl
  internal/
    render.py               # config + runtime facts -> polar yamls + slime args
    run.sh                  # processes only
    prepare_tasks.py
    prepare_harness.sh      # codex, opencode, mini_swe_agent
    convert_weights.sh, model_args/{qwen3_5_9b,qwen3_5_4b}.sh
    head_entry.sh, ray_worker_join.sh
    setup/                  # in-place install scripts (unchanged for now)
```

Rough size: ~1200 lines after 3a and the approved removals.

## Implementation rules (09-06)

- as few files as possible; do not split for tidiness
- no handling of unlikely edge cases: fail loudly when the machine is not set up as expected
- branch: simplify-harbor-example; smoke on the cluster before merging
