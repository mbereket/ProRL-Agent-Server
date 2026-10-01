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

## Miles patches (`miles_patches/miles/`, applied at the pinned Miles commit by `miles_runtime/mrun`)

| patch | what |
|---|---|
| 0001 responses | session server `/v1/responses` (codex) with exact replay |
| 0002 lora resume | a bridge-LoRA resume keeps `--start-rollout-id` and its dataset state |
| 0003 codec | non-finite rollout logprobs survive the session wire (diagnostic `hm codec nonfinite` line) so they reach hm_rollout's HARD STOP |
| 0004 lora optimizer state | the LoRA checkpoint saves each rank's fp32 masters + Adam moments (`optimizer_state_rank*.pt`; Megatron's DistributedOptimizer.state_dict() omits them, and its load left torch.empty moments: NaN after every resume, FINDINGS F66); masters refreshed after the adapter copy; exact verification on load; a checkpoint without optimizer state refuses to resume; non-finite grad_norm skips the step and aborts; finite-state check after every LoRA step; no scheduler double count. `MILES_LORA_KEEP_OPTIMIZER_STATE` (default 2) prunes older optimizer shards |

Agents (`harbor_miles_agents/`): `opencode_agents.PreinstalledOpenCode` (the image's opencode, no
per-trial npm install), `BbhOpenCode` (starts the BBH MCP server in agent setup, where the task env
with the judge key is applied; stops opencode after `submit_answer`, same as eval-v2).

## Launch (branch `harbor-miles`; full guide: miles-work/EXPERIMENTS.md)

A run is one short experiment file on top of the recipe (`configs/README.md`: recipe > layout > dataset > experiment):

```bash
cat > configs/experiments/de4-27b-lr5e-5-r1.env <<'EOT'
RUN_NAME=de4-27b-lr5e-5-r1
LAYOUT_PRESET=27b-2n
DATASET=de4-v1-k1
CAP=96k
LR=5e-5
EOT
tools/dry_render.sh --cluster dfw configs/experiments/de4-27b-lr5e-5-r1.env     # layers, derived knobs, exact Miles args
cd cluster && ~/Desktop/code/cluster-tools/.venv/bin/python submit.py --cluster dfw --partition interactive \
    --nodes 2 --gpus 8 --cpus 96 --mem 1200G --hours 4 --name de4-lr5e-5 \
    --extra-pkg ../../miles_runtime launch/node_entry.sh configs/experiments/de4-27b-lr5e-5-r1.env
```

`launch/node_entry.sh` runs on node 0 (the slurm-compose step): it loads the config layers, takes the run lock, queues the
chain successor (`HM_CHAIN_MAX`), starts the other nodes' agent servers (`launch/node_peer.sh`) and its own, then
`miles_runtime/ray_node.sh` (Miles SIF, Ray; the head runs `launch/train_driver.sh`). Outputs: ONE run root per cluster,
`<user root>/miles/runs/<RUN_NAME>/` (`trials-<job>.jsonl` one line per trial, `trials/<host>/` Harbor trial dirs, `ckpt/`,
`args-<job>.txt`, `config-<job>.txt`, `jobs.log`, `chain.log`, `gpu-<job>.csv`, `node-<job>-<host>.csv`); Slurm logs in
`<user root>/miles/joblogs/`. A rerun with the same `RUN_NAME` resumes from `ckpt/`. Analysis: `tools/README.md`.

Validation jobs (no Miles): `tools/hv1_job.sh` (SGLang + agent server: nop wave, real opencode
trials, flush), `tools/hv2_job.sh` (64-wide nop wave + flush of sleeping sandboxes; leak check).
