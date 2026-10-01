# Run analysis tools

Python 3.10+ stdlib (remote reads: cluster-tools; `dump_lengths.py`: torch). Each tool's docstring has the details and
its provenance.

**Run spec** (every tool's RUN_SPEC): a local mirrored run dir (`trials-*.jsonl`, `args-*.txt`, `gpu-*.csv`, `node-*.csv`,
`chain.log`, job logs in `joblogs/`), or `CLUSTER:RUN_NAME` (first of `<user_root>/miles/runs/RUN`, then the legacy
`miles/{path-b,qwen27b,shared/hm}/runs/RUN`), or `CLUSTER:/abs/run/dir`. Remote specs read over SFTP (the tool re-runs
itself under the cluster-tools venv when needed). Job logs are found by the run's job ids in `<user_root>/miles/joblogs/` (new) and the legacy `<HM_ROOT>/joblogs/`.
Log timestamps are parsed in `HM_LOG_TZ` (default `America/Los_Angeles`, the clusters' log clock).
CAVEAT: verified only on dfw/hel logs (they line up with the trials' `t_end` epochs only in Pacific time); aws-iad, ord and draco
are unverified: if every trial lands in "in progress" or step windows look shifted, set `HM_LOG_TZ` (e.g. `UTC`).

| tool | answers | usage |
|---|---|---|
| `fetch_run.sh` | mirror a run's small files + its job logs locally | `fetch_run.sh CLUSTER RUN [LOCAL_DIR]` |
| `per_task_success.py` | did training move success, per step and task-paired vs a base window? (overlong = 0) | `per_task_success.py RUN_SPEC [--windows 0-1,8-12] [--split train] [--per-task]` |
| `step_report.py` | where does each step's time go; trainer peak GPU memory | `step_report.py RUN_SPEC [JOBID ...] [--trainer-gpus 0-3]` |
| `profile_steps.py` | per-step filtering, staleness, TITO mismatch, outcome mix, overlong token share, evals | `profile_steps.py RUN_SPEC [--jobs ID ...] [--group-size N]` |
| `batchpoint.py` | are engines / sandboxes / nodes saturated per step (SGLang decode stats, node CSVs)? | `batchpoint.py --run RUN_SPEC [--rbs N]` or `--rbs N JOBLOG... [--nodes CSV...] [--trials JSONL...]` |
| `base_rates.py` | per-task base pass rates; which tasks to overfit on (mixed outcomes) | `base_rates.py RUN_SPEC [RUN_SPEC ...] [--pick 8] [--json OUT]` |
| `session_lengths.py` | session length / turns / decode distribution and overflow per cap, from Harbor trial dirs | `session_lengths.py RUN_SPEC ... [--caps 64k,96k] [--expected N] [--cache FILE]` |
| `dump_lengths.py` | the same from Miles eval dumps (`dumps/eval_0.pt`, local) | `dump_lengths.py eval.pt ... [--caps ...] [--rollout-cap 128k] [--json OUT]` |
| `frontier.py` | MODEL of step time vs layout (nodes, trainer, engines) per cap and batch | `frontier.py [specs/de4-27b-provisional.json]` (no arg: SWE-Gym 27B) |
| `watch_run.py` | live watcher: job state, args, per-step lines, trials per agent server, first error | `watch_run.py CLUSTER JOBID RUN [--local-mirror DIR] [--interval 180]` |
| `hmruns.py` | shared helpers (run/job-log resolution, metric parsing, step assignment); CLI for fetch_run.sh | `hmruns.py resolve\|jobids\|joblogs SPEC` |

**Typical workflow**

```bash
T=examples/harbor_miles/tools/analysis
$T/fetch_run.sh hel my-run                                     # -> ./runs-mirror/hel/my-run (+ joblogs/)
python3 $T/per_task_success.py runs-mirror/hel/my-run --per-task  # success per step + paired windows
python3 $T/step_report.py runs-mirror/hel/my-run                  # timing / memory per step
```
