# harbor_miles configs: recipe > layout > dataset > experiment

A run is ONE short experiment file. The launcher (`setup/config.sh` `hm_load_config`, called by `launch/node_entry.sh` and
`launch/node_peer.sh`) loads four layers, later wins, then derives what no layer set:

| layer | file | holds |
|---|---|---|
| 1 recipe | `recipe.env` | the current best settings for every run: LoRA r32/a64 all-linear, merged serving, no MTP loss, LR 3e-5, fully async (staleness <= 2), zero-variance groups dropped, overlong = reward 0, codex, CPU reserve 16, save every step, chaining. Evidence: `miles-work/FINDINGS.md`. |
| 2 layout | `layouts/<name>.env` (`LAYOUT_PRESET=`) | model (`models/*.env`), nodes, trainer GPUs / TP / CP, engine TP / memory, in-flight sessions per engine (the measured KV knee), the largest cap the trainer fits (`MAX_CAP`) |
| 3 dataset | `datasets/<name>.env` (`DATASET=`) | task packages, default task list, verifier env (`HM_REQUIRE_ENV`), agent timeout, agent process env |
| 4 experiment | `experiments/<name>.env` | run name, tasks, **cap (`CAP`, required: no default)**, batch (RBS x NS), steps, LR and whatever else this run changes |

Derived when not set (an explicit value in any layer wins): `MAX_SEQ_LEN` from `CAP`; `MTPG = MAX_SEQ_LEN / CP`;
`LOGPROB_CHUNK` 1024 when one sequence puts >= 96k tokens on a trainer GPU, else 4096; `TRAIN_ALLOC_CONF`
(`expandable_segments:True` for the trainer via Miles `--train-env-vars`) at caps >= 96k unless `PYTORCH_CUDA_ALLOC_CONF` is
exported job-wide; `ASYNC_CONCURRENCY = engines x INFLIGHT_PER_ENGINE`; `HM_SANDBOXES_PER_NODE = ceil(in-flight / nodes)`.
Checks before any GPU work: the job's node count equals the layout's `NODES`, the cluster allows that many nodes
(aws-iad/ord/draco: 1), the cap fits the trainer (`MAX_CAP`), TP <= `MAX_TP`, headwise CP divides the GDN heads; a warning when
in-flight sessions exceed the engines' knee.

```bash
# experiments/my-run.env
RUN_NAME=de4-27b-lr5e-5-r1
LAYOUT_PRESET=27b-2n
DATASET=de4-v1-k1
CAP=96k
LR=5e-5
```

Always dry-render before submitting (submit.py does it automatically for node_entry.sh configs):
`tools/dry_render.sh --cluster dfw configs/experiments/my-run.env` prints the layer chain + derived knobs and writes the exact
Miles args. Experiment files are KEY=VALUE lines (sourced with `set -a`); `LAYOUT_PRESET`/`DATASET` are read from the file
itself, so set them there, not via `--env`.

## Presets

| layout | nodes | trainer | engines | in flight | caps | where |
|---|---|---|---|---|---|---|
| `27b-2n` | 2 | 4 GPUs TP4 CP1 (node 0) | 3 x TP4 | 105 (35/engine) | 64k, 96k | hel, dfw |
| `27b-1n-split` | 1 | 4 GPUs TP4 CP1 | 1 x TP4 | 35 | 64k, 96k | every pool |
| `27b-1n-colo` | 1 | 8 GPUs TP4 x CP2 headwise, colocated, sync | 2 x TP4 (MEMF .75) | 56 (28/engine: MODEL, the de4 knee scaled by KV; adjust after aws-iad 7599199, which ran 32) | 96k (64k: set CP=1) | every pool |
| `27b-2n-128k` | 2 | 8 GPUs TP4 x CP2 headwise (node 0) | 2 x TP4 (node 1) | 70 | up to 128k | hel, dfw |
| `27b-3n-128k` | 3 | 8 GPUs TP4 x CP2 headwise (node 0) | 4 x TP4 (nodes 1-2) | 140 | up to 128k | hel/dfw batch (3 nodes > interactive cap) |
| `27b-1n-rollout` | 1 | none (`ROLLOUT_ONLY=1`, base weights) | 2 x TP4 | 70 | any (set CAP; base passes use 128k) | every pool |
| `9b-1n` | 1 | 4 GPUs TP4 CP1 | 2 x TP2 | 64 (32/engine) | 64k | every pool |

Datasets: `de4-v1-k1` (97 generated data-analysis tasks, NVINF_API_KEY required, thread caps), `swegym-lite-v3`
(rand48-s0 by default). Task lists: `tasks/` (`de4-overfit8.txt` = the 8 overfit tasks, `de4-smoke2-short.txt` = the 2
shortest tasks for smokes, `de4-v1-k1-half{A,B}.txt`, `swegym-rand48-s0.txt`, `swegym-overfit8.txt`).

## Experiments here

| file | what |
|---|---|
| `diag-de4-27b-overfit8-r3.env`, `q27-de4-overfit8-2n-lr1e4.env`, `de4-codex-27b-bs128-2n.env` | the three 27B de4 runs running on 2026-10-01, re-expressed; dry-render equivalent to their originals (`archive/de4-27b/EQUIVALENCE.md`) |
| `de4-27b-smoke-2n-128k.env` | THE harbor-miles validation smoke: `27b-2n-128k` end to end, 2 long de4 tasks x 4, 2 steps, 1800 s agents |
| `de4-27b-smoke-2n.env`, `de4-27b-smoke-1n.env` | plumbing smokes at 96k (2 short tasks x 4, 2 steps, 900 s agents, zero-variance groups kept) |
| `de4-27b-96k-2n.env`, `de4-27b-128k-2n.env`, `de4-27b-128k-3n.env`, `de4-27b-96k-1n-colo.env` | templates for real de4 runs per cap / footprint |
| `de4-27b-eval-r3-smoke.env` | decoupled eval of saved adapters (`EVAL_FROM_RUN`, `EVAL_STEP`, `EVAL_WATCH_EVERY`): the trainer must match the source run's trainer GPUs (per-rank shards). Re-expresses `archive/de4-27b/de4-codex-27b-eval-r3-smoke.env` on the 96k-derived trainer settings; the differences (max tokens/GPU, logprob chunk, allocator, NUM_ROLLOUT, 35 vs 26 in flight) are inert for an eval-only job (no training step, weights-only adapter view, 16 sessions) |
| `de4-27b-base128k.env` | base pass (rollout-only) on all de4 tasks x 4 |
| `9b-swegym-overfit8.env` | D2 replication on the recipe (9B SWE-Gym overfit) |

`archive/`: superseded configs, verbatim (see `archive/README.md`).
