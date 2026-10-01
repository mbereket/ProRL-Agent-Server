# harbor_miles — agentic RL on Harbor tasks with Miles, sandboxes on Apptainer

Miles (radixark/miles) trains; Harbor (harbor-framework/harbor, `harbor-miles-v0.20.0`) runs each
trajectory as a Harbor trial in an Apptainer sandbox; Miles' TITO session server records the exact
tokens/logprobs the agent's model calls produced. No Polar, no Docker.

```
 node 0 (head)                                   node 1..N-1
 ┌───────────────────────────────────────────┐   ┌─────────────────────────────────┐
 │ Miles SIF (miles_runtime/ray_node.sh)     │   │ Miles SIF: Ray worker            │
 │  trainer (Megatron, LoRA) · SGLang engines│   │  SGLang engines / trainer ranks  │
 │  router · TITO session servers :30000+    │   │                                  │
 │  RolloutExecutor ── hm_agent.run ──┐      │   │                                  │
 │        least-loaded /run dispatch  │      │   │                                  │
 ├────────────────────────────────────┼──────┤   ├─────────────────────────────────┤
 │ host: Harbor agent server :18300 ◀─┼──────┼───┼▶ host: Harbor agent server :18300│
 │   └ apptainer sandboxes (opencode, │      │   │   └ apptainer sandboxes          │
 │     MCP, verifier) ── model calls ─┘──────┼───┼──── to the head's session URL     │
 └───────────────────────────────────────────┘   └─────────────────────────────────┘
```

* One Harbor agent server per node (`launch/start_agent_server.sh`, host venv, singularity backend):
  sandboxes run on the node whose server took the trial. The Miles agent function
  (`miles_side/hm_agent.py`) dispatches each trial to the least-loaded server
  (`miles_side/hm_dispatch.py`; exact in-flight counts — every call goes through the one
  RolloutExecutor process). A server that refuses connections is benched for 60 s; trials that fail
  before the agent ran (no model calls yet) are re-dispatched.
* Rewards: Harbor verifier (`tests/test.sh`, e.g. the BBH MCP server's judge) → `metadata.reward`.
  `miles_side/hm_rollout.py`: reward hook, GRPO post-process that drops infrastructure failures
  from both the loss and the group baseline, per-rollout Harbor metrics (sync mode).
* Session: TITO v1 (linear) + `--session-message-matcher loose_tool_call`: opencode replays its
  history verbatim except tool-call arguments, which it re-serializes as compact JSON (0/74
  byte-identical, 73/74 JSON-identical in hv1) — `strict` would treat every turn as a divergence.
  opencode's title-generation side calls are disabled in the agent config.

## Harbor patches (`harbor/PIN` + `harbor/patches/`, applied by `setup/ensure_harbor.sh`)

| patch | what |
|---|---|
| 0001 singularity | offline bootstrap: the in-sandbox exec server runs on a host python (fastapi+uvicorn) bind-mounted read-only — no apt/pip per sandbox start (hel: 9-11 s start at 32-64 concurrent); node-local `--overlay` dirs instead of the 64 MB `--writable-tmpfs`; prebuilt-SIF lookup (Harbor cache names and polar's `ref-<sha>` names) in read-only dirs; 600 s start window; task files bound read-only; server log + `sandbox_startup.json` per trial; teardown kills the whole container process tree (apptainer's FUSE/fakeroot helpers otherwise outlive a forced stop) |
| 0002 agent server | `HARBOR_ENV_TYPE=singularity`; each trial in its own process group — `/flush` and `/flush_all` SIGTERM the worker (trial cancelled → sandbox stopped cleanly), SIGKILL the group after 45 s; per-request `agent_import_path` / `agent_kwargs` / `agent_env`; `/stats`; failure attribution (`exception_type`, `agent_started`) and context-overflow → `SequenceLengthLimitExceeded` |

Agents (`harbor_miles_agents/`): `opencode_agents.PreinstalledOpenCode` (the image's opencode, no
per-trial npm install), `BbhOpenCode` (starts the BBH MCP server in agent setup, where the task env
with the judge key is applied; stops opencode after `submit_answer`, same as eval-v2).

## Launch

```bash
cd examples/harbor_miles/cluster
~/Desktop/code/cluster-tools/.venv/bin/python submit.py --cluster hel --partition interactive \
    --nodes 1 --gpus 8 --cpus 96 --mem 1200G --hours 4 --name learn1n \
    --extra-pkg ../../miles_runtime launch/node_entry.sh configs/learn-bbh8-async-1n.env
```

`launch/node_entry.sh` runs once per node: agent server on the host, then `miles_runtime/ray_node.sh`
(STACK's runtime: Miles SIF, Ray; head runs `launch/train_driver.sh`). Everything is in the config
env (`configs/*.env`); outputs in `<HM_ROOT>/runs/<RUN_NAME>/` (`trials-<job>.jsonl` one line per
trial, `trials/<host>/` full Harbor trial dirs, `ckpt/`, `args-<job>.txt`). A rerun with the same
`RUN_NAME` resumes from `ckpt/`. `tools/analyze_run.py <job.log> <trials.jsonl>` summarizes.

Validation jobs (no Miles): `tools/hv1_job.sh` (SGLang + agent server: nop wave, real opencode
trials, flush), `tools/hv2_job.sh` (64-wide nop wave + flush of sleeping sandboxes; leak check).
