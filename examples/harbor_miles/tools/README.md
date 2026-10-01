# harbor_miles tools

## Before a run: render and compare configs (local, no cluster)

| tool | what | usage |
|---|---|---|
| `dry_render.sh` | run the real launcher with Slurm/GPU/Ray/Miles/Harbor stubbed; writes the exact Miles args (= `args-<job>.txt`), train command, patch sets, agent servers per node, task list, job env, and checks the layout (nodes, cap, cluster). `cluster/submit.py` runs it before every `node_entry.sh` submission. | `tools/dry_render.sh [--cluster dfw] [--nodes N] [--ref GIT_REF] CONFIG [K=V ...]` |
| `config_equiv.sh` / `config_equiv.py` | prove a re-expressed config launches the same run as the original (old config on its own commit vs new config here; exit 0 = only paths / run name differ) | `tools/config_equiv.sh OLD_REF OLD_CONFIG NEW_CONFIG [--nodes N]` |

## During / after a run: analysis (`analysis/`, details in `analysis/README.md`)

| tool | answers |
|---|---|
| `analysis/fetch_run.sh` | mirror a run's small files (trials, args, chain/jobs logs, GPU/node monitors) + its job logs locally |
| `analysis/per_task_success.py` | did training move success? per step, task-balanced, per-task paired vs a base window (DIAG's method; overlong = 0) |
| `analysis/step_report.py` | per-step wall / rollout wait / train time, train tok/s, response length, reward, overlong, trainer peak GPU memory |
| `analysis/profile_steps.py` | per-step groups kept/dropped, staleness, TITO mismatch, outcome mix, overlong token share, evals |
| `analysis/batchpoint.py` | per-step systems record: engine running requests / KV usage / queue, sandboxes and node CPU/memory |
| `analysis/base_rates.py` | per-task base pass rates from a base pass; picks overfit tasks (mixed outcomes) |
| `analysis/session_lengths.py` | session length / turns / decode distribution, overflow and lost successes per cap (Harbor trial dirs) |
| `analysis/dump_lengths.py` | the same from Miles rollout/eval dumps (`dumps/*.pt`, needs torch) |
| `analysis/frontier.py` | MODEL: step time and steps/node-hour per layout, cap and batch (`analysis/specs/de4-27b-provisional.json`) |
| `analysis/watch_run.py` | live watcher: job state, launched args, per-step lines, trials per agent server, first error |
| `analysis/hmruns.py` | shared helpers: run / job-log resolution (new `miles/runs` + legacy roots), metric parsing, step assignment |
| `analyze_run.py` | older single-run summary from a job log + trials (per-step reward, KL, timings) |

## Data staging (CPU jobs; see miles-work/EXPERIMENTS.md "Staging a new dataset")

| tool | what |
|---|---|
| `prepare_data.py` | task packages -> Miles prompt jsonl (one row per task; the driver calls it) |
| `stage_sifs.sh`, `stage_job.sh` | pull task images as SIFs with Harbor's cache names ahead of time (no pulls during runs) |
| `fleet_stage.sh` | make aws-iad / ord / draco ready (tasks, uv, model, agent toolchains, SIFs) |
| `fleet_stage_de4.sh`, `fleet_assemble_de4.sh` | stage de4 tasks + its SIF on a FLEET cluster (chunked upload for aws-iad) |

## Harness validation (historical; no Miles)

`hv1_job.sh` .. `hv4_job.sh` (Harbor-on-Apptainer bring-up, teardown, harness replay), `validate_harbor.py`, `log_proxy.py` +
`replay_check.py` (what a harness replays per turn: session matcher decision), `slime_sglang_env.sh` (borrowed SGLang, pre-Miles).
