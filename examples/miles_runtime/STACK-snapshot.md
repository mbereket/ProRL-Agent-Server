# STACK.md — the Miles runtime on our clusters (owner: stack agent)

Status: **usable on dfw and hel** (dfw: SIF, Ray 1+2 nodes, NCCL over IB validated; hel: SIF, imports, nested apptainer validated). aws-iad: SIF + compat + nested apptainer + Ray validated (job 7587420). **Benchmark results: §9.**
Code: ProRL-Agent-Server branch **`miles-stack`**, dir `examples/miles_runtime/` (README there).
Last updated: 2026-09-30 20:25 PT. **Use the latest `miles-stack` head** (≥ `52074b8b`; < `52074b8b`: Qwen3.5 + context parallelism crashes, Ray ports inside dfw's ephemeral range; < `5c1b3bc8`: multi-node ray_node.sh only started the head; < `025a9936`: earlier mrun crashes with ENOSPC on dfw/aws-iad `$HOME`; < `6f487a5d`: Ray cannot start).

## 1. What the runtime is

One Apptainer SIF per cluster of a digest-pinned image — **no venv build, no forks**:

| pin | value |
|---|---|
| image | `radixark/miles@sha256:30bca3fc51a870a34aac7105627211ba42673dc2b7447f110b356d84bac1031b` (tag `dev-202609302131` == `:latest` on 09-30), cu13 x86_64 |
| Miles | `/root/miles` @ `8f409c7c49e4` (main, 2026-09-30) — editable install |
| SGLang | `/sgl-workspace/sglang` @ `6e25d1bd57a5` (sglang-miles, `0.5.21.dev66`) — editable (`python/`) |
| Megatron-LM | `/root/Megatron-LM` @ `a84b105473ec` (radixark `miles-main`) — editable |
| other | torch 2.13.0+cu130, TE 2.17.0, flash_attn 2.7.4.post1 (+FA3), ray 2.58.0, transformers 5.12.1, mbridge 0.15.1, peft 0.18.1, triton 3.7.1, NCCL 2.29.7, Python 3.12 (venv at `/opt/sglang`) |
| Apptainer | 1.5.3 unprivileged install (none of hel/dfw/aws-iad has one on PATH) |
| CUDA compat | `590.48.01_cuda13.1` forward-compat libcuda, auto-used when the kernel driver is < 580 |

Why a SIF and not a venv: the image bakes ~15 prebuilt/patched wheels (TE 2.17 w/ patches, FA2/FA3,
apex, flash-mla, fla patch, sgl-model-gateway, Megatron-Bridge pin, ...) — reproducing that as a uv
venv buys nothing and would drift from what Miles CI tests.

## 2. Per-cluster facts

| cluster | root (`MILES_STACK_ROOT`) | driver | compat | SIF |
|---|---|---|---|---|
| hel | `/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/mbereket/miles/stack` | 580.126 (CUDA 13.0 native) | no | ✅ 17.4 GB (job 1521940) |
| dfw | `/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/stack` | 535.216 (CUDA 12.2) | **yes** (auto) | ✅ 17.4 GB |
| aws-iad | `/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/stack` (aws-iad's own lustre) | 575 (CUDA 12.9) | **yes** (auto) | ✅ 17.4 GB (job 7584859; pull took ~2 h on lustre tmp) |

aws-iad networking is EFA (`uverbs*`), not IB: cross-node NCCL needs the aws-ofi-nccl plugin + libfabric,
which the Miles image does not ship → assume **single-node only on aws-iad** (multi-node would fall back to
TCP sockets) until someone layers the plugin in.

SIF path on each: `$MILES_STACK_ROOT/images/miles-dev-202609302131-30bca3fc51a8.sif`
(+ `.versions` manifest). Apptainer: `$MILES_STACK_ROOT/apptainer/1.5.3/bin/apptainer`.

## 3. How to run things

Get the scripts into your job (either): check out ProRL `miles-stack` (or merge it into your branch)
and use `examples/miles_runtime/`, or export the directory as a slurm-compose package (what
`jobs/submit.py` does: it appears as `$SCOMPOSE_PKGS/miles_runtime/`).

```bash
# one command inside the runtime (GPU, /lustre bound, host network)
$MR/mrun -- python3 -c 'import torch, miles, sglang; print(torch.cuda.device_count())'

# with setup-time patches (pinned upstream + patch files):
$MR/mrun --patches my-patches -- python3 /root/miles/train.py ...
#   my-patches/{miles,sglang,megatron}/*.patch (-p1 from each repo root) + optional pip-requirements.txt

# multi-node Ray under Slurm: call ONCE on the first node (a slurm-compose step's command runs in the batch
# script on node 0, NOT under srun -- despite `step_type: srun`). The head spawns one worker per other node
# with `srun --overlap -w <node>`, waits for all nodes, runs DRIVER, then stops Ray and the workers.
$MR/ray_node.sh [--patches D] [--pythonpath P] -- bash driver.sh
#   inside DRIVER: RAY_ADDRESS, MASTER_ADDR, MILES_NUM_NODES are set; `python3 /root/miles/train.py ...`
#   or Miles' python launchers with MILES_SCRIPT_EXTERNAL_RAY=1. Worker logs: <owner root>/joblogs/ray-worker-<job>-<node>.log

# submit from the laptop (slurm-compose; hel/dfw/aws-iad):
python3 $MR/jobs/submit.py --cluster dfw --partition interactive --nodes 1 --gpus 8 --hours 2 \
    --name my-run --owner path-a [--pkg ~/my/code] -- bash '$SCOMPOSE_PKGS/miles_runtime/ray_node.sh' -- bash '$SCOMPOSE_PKGS/code/driver.sh'
#   --owner X puts logs/exports under <user_root>/miles/X/ ; MILES_STACK_ROOT is exported to the job.
```

**Base patch set (always applied by mrun, `patches/base/`)** — fixes to the pinned image itself:
- `miles/0001-qwen3_vl-explicit-cp-mrope-position-ids.patch`: **Qwen3.5 with context parallelism** (bridge mode
  builds the Qwen3-VL model) crashed with `ValueError: Pre-sharded packed CP inputs require explicit rank-local 3D
  MRoPE position_ids` — the pinned Megatron-Bridge wants explicit position ids; Miles computed them but injected
  them too late. Patch passes them explicitly (validation running: bench arms-64k).
- `opentelemetry-api==1.44.0` (pip overlay): the image ships api 1.45.0 with sdk 1.44.0 → Ray's dashboard
  agent dies with `ImportError: _ExtendedAttributes`, the raylet times out, and **`ray start` fails**
  ("The current node timed out during startup"). Anything that starts Ray inside the SIF needs this.

`ray_node.sh` details: job-unique Ray port block `65010 + (jobid % 12) * 40` .. +8 (`RAY_PORT_BASE` overrides):
**dfw's ephemeral port range is 9000–65000**, so any fixed port below 65000 can be taken by a random outbound
socket (a block at 47520 lost its object-manager port that way); Miles' dynamic SGLang ports (from 20000) are
also inside it — an engine occasionally hits EADDRINUSE at startup and Miles restarts it (seen, recovered); Ray defaults (6379/8265) collide on shared
nodes. Dashboard **on** by default (works with the otel pin; needed by `ray.util.state`, `ray job submit`,
Miles `--pin-rollout-manager-to-head`); `RAY_DASHBOARD=0` disables.
Validated: dfw job 19587294 (`RAY_OK`).

**`$HOME` inside mrun is node-local scratch** (`/tmp/miles-$USER-$SLURM_JOB_ID/home`, mounted at your real
`$HOME` path): tilelang (`~/.tilelang`), flashinfer/sglang (`~/.cache`), triton, `~/.apptainer` all write
there at import/JIT time and the real `$HOME` is full on dfw (ENOSPC killed every training arm) and over
quota on aws-iad. `~/.netrc ~/.secrets ~/.gitconfig ~/.config/wandb ~/.huggingface` are bound read-only.
`MRUN_REAL_HOME=1` restores the real home. Put anything persistent on lustre.

`mrun` details: unsets host `LD_LIBRARY_PATH/PYTHONPATH/VIRTUAL_ENV`; sets `PYTHONPATH=[--pythonpath..]:
[overlay pydeps]:/root/Megatron-LM`, `CUDA_DEVICE_MAX_CONNECTIONS=1`; Triton/Inductor/XDG/CUDA JIT caches
and `~/.cache` go to node-local `/tmp/miles-$USER-$SLURM_JOB_ID/` (lustre caches shared by jobs corrupt;
`$HOME` is full on dfw / over quota on aws-iad). Extra apptainer flags via `MRUN_APPTAINER_ARGS`.

## 4. Host-side services next to the containerized trainer (path-a Polar, path-b Harbor)

The SIF shares the host network, `/tmp`, `/dev/shm`, `$HOME` and `/lustre`. Run sandbox-launching
services (Polar gateway, Harbor agent server) **on the host in their own venv**, pointing at the
trainer's SGLang router / session server on the node IP. Rollout code that runs *inside* Miles
(e.g. a Polar bridge `--rollout-function-path`) needs its package on `PYTHONPATH`
(`mrun --pythonpath` / `ray_node.sh --pythonpath`) and pure-python deps via a patch dir's
`pip-requirements.txt`. Nested Apptainer (sandbox started from inside the SIF): see §6.

## 5. Model weights and data

- Qwen3.5-9B HF: `$MILES_STACK_ROOT/models/Qwen3.5-9B` (staged by `bench/prepare.sh`; on hel copied
  from `prorl-harbor/hf_home/.../snapshots/c2022362...`). Bridge mode (`--megatron-to-hf-mode bridge
  --hf-checkpoint <dir>`) loads HF directly — no torch_dist conversion needed for LoRA.
- dapo-math-17k / aime-2024: `$MILES_STACK_ROOT/datasets/`.

## 6. Validation results

**aws-iad** job 7587420 (2x H100, driver 575 + compat, `bench/validate.sh`): imports OK, bf16 matmul 724 TFLOP/s,
`--pid` OK, nested Apptainer `exec` **and** `instance` OK (needs mrun's scratch `$HOME` — real `$HOME` is over
quota), `ray_node.sh` + dashboard `RAY_OK`. EFA devices (`uverbs*`) visible; no aws-ofi-nccl in the image, so
treat aws-iad as single-node. ip_local_port_range 9000–65000 (same as dfw).

**hel** job 1521940 (driver 580, no compat): imports OK, IB devices visible, nested Apptainer
`exec` **and** `instance start/exec/stop` from inside the SIF work (`nested-instance-ok`).

**dfw multi-node** job 19587728 (2 nodes x 8 H100, batch_short): torchrun inside the SIF on both nodes,
NCCL picks `NET/IB/.../GDRDMA`; all-reduce busbw 254 / 385 / **440 GB/s** at 64 MB / 256 MB / 1 GB
(16 ranks). `ray_node.sh` 2-node bring-up → `RAY_OK nodes 2 GPU 16`. No NCCL env tuning needed.

dfw job 19586913 (2x H100, driver 535 + compat), `setup.sh compat manifest smoke pidns nested`:
- torch 2.13/cu130 sees the GPUs through the forward-compat libcuda; `import sglang, miles, megatron.core,
  transformer_engine, megatron.bridge` OK (38 s cold on lustre).
- `/dev/infiniband` visible and libibverbs present in the image (cross-node NCCL-over-IB test: pending, job 19587067).
- `apptainer exec --pid` (own PID namespace) works — `ray_node.sh` uses it so a finished run leaves no GPU stragglers.
- **Nested Apptainer works**: from inside the Miles SIF, the lustre-installed unprivileged apptainer can
  `exec` a busybox SIF (`nested-exec-ok`). `instance start` failed only because `~/.apptainer` is on the
  full dfw `$HOME`; mrun now sets `APPTAINER_CONFIGDIR` to node-local scratch (re-test pending).
  => path-b's in-process Harbor (singularity backend inside the rollout worker) is viable; path-a can
  still run the Polar gateway on the host.

## 7. Minimal working launch flags (Qwen3.5 line)

Validated end-to-end on dfw (1 node x 8 H100): `bench/grpo.sh` builds exactly this (see its `args.txt` per run).
Model args come from Miles: `read -ra MODEL_ARGS <<< "$(python3 /root/miles/miles/utils/external_utils/model_args_utils.py qwen3.5-9B)"`
(`qwen3.8-27B` for the 27B; same `miles_plugins.models.qwen3_5` spec).

```bash
python3 /root/miles/train.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint $MILES_STACK_ROOT/models/Qwen3.5-9B --megatron-to-hf-mode bridge \   # HF loaded directly, no torch_dist
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 16384 \
  --advantage-estimator grpo --kl-loss-coef 0 --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash --num-gpus-per-node 8 --actor-num-nodes 1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 \
  --colocate --actor-num-gpus-per-node 8 \
  # LoRA:
  --lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 --target-modules all-linear --no-gradient-accumulation-fusion \
  --sglang-max-lora-rank 32 --lora-base-cpu-backup \        # base-cpu-backup: colocated only
  ... rollout/data flags (--prompt-data, --rollout-batch-size, --n-samples-per-prompt, --rollout-max-response-len, ...)
```

- Full FT instead of LoRA: drop the LoRA lines, `--lr 1e-6`; at `--max-tokens-per-gpu 16384` TP2 **OOMs on 80 GB at
  step 1** (Adam state) → add `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`
  (or TP4 / smaller `--max-tokens-per-gpu`; Miles' own recipe uses 9216).
- Disaggregated: replace `--colocate --actor-num-gpus-per-node 8` with `--actor-num-gpus-per-node N --rollout-num-gpus M
  --update-weight-transfer-mode broadcast` (and no `--lora-base-cpu-backup`).
- `--use-rollout-logprobs` skips the separate old-logprob forward pass (~25 % of train time); benchmark pending.

## 8. Gotchas

- dfw/aws-iad drivers are older than CUDA 13: without the compat libcuda torch reports
  "driver too old (found version 12020)". mrun handles it automatically (`MILES_CUDA_COMPAT=0/1` to override).
- flashinfer/sglang (`~/.cache/sglang`) and tilelang (`~/.tilelang/cache`) write under `$HOME` at import; dfw `$HOME`
  is full (ENOSPC) — mrun gives the container a scratch `$HOME` (see §3).
- Synthetic/replayed rollout dumps: `Sample.rollout_id` is Miles' *trajectory* id — samples sharing it must share
  one reward (`ValueError: all samples in rollout N must share one reward`). Leave it unset for one-sample-per-trajectory.
- Colocated LoRA needs `--lora-base-cpu-backup` (Miles' 35B-A3B recipe: rollout/train KL ~1.0 without it vs ~1e-3).
- LoRA weight sync supports only `--update-weight-transfer-mode broadcast` (not p2p / disk-delta);
  disaggregated LoRA needs bridge mode and pipeline-parallel 1.
- hel: SGLang custom all-reduce is broken → `--sglang-disable-custom-all-reduce` whenever engine TP > 1.
- slurm-compose `step_type: srun` steps run once in the batch script on node 0 (no srun) — anything
  per-node must fan out with its own `srun --overlap` (ray_node.sh does).
- Ray heartbeat tolerance raised by `ray_node.sh` (`RAY_health_check_failure_threshold=30`, 30 s timeout).
- hel interactive QoS: 2 nodes / 16 GPUs per user across ALL our agents.

## 9. Benchmark results (dfw, 1 node x 8 H100 80GB, Qwen3.5-9B, bridge mode, code `52074b8b`+)

Raw: `$MILES_STACK_ROOT/bench/q35-9b-v1/<arm>/` (args.txt, train.log, metrics.jsonl, summary.json); local copy
`miles-work/stack/results/dfw/q35-9b-v1/`. "train" = Megatron fwd+bwd+optimizer (`perf/actor_train_time`);
steady state = median excluding step 0 (step 0 pays ~100 s of TileLang GDN JIT per job).

**A. Training side, length-controlled** (identical fixed-length synthetic batches, 1.05 M tokens/step, TP2, full recompute):

| arm | config | actor_train s | tok/s (node) | step s | peak GB/GPU |
|---|---|---|---|---|---|
| t16k-full | full, TP2, 16k tok/GPU | OOM at step 1 (Adam state) | – | – | 80 |
| t16k-full-off | full, TP2, CPU Adam | 23.4 | 45 k | 30.3 | 72 |
| t16k-full-tp4 | full, TP4, no offload | 24.1 | 44 k | 32.1 | 53 |
| t16k-lora | LoRA r32 all-linear, TP2 | **20.4** | **51 k** | 28.7 | **52** |
| t16k-lora-rlp | + `--use-rollout-logprobs` | 20.5 | 51 k | **21.7** (-24 %) | 53 |
| t16k-full-off-rlp | full CPU Adam + rollout logprobs | 23.3 | 45 k | 24.4 | 72 |
| c64k-lora-cp4 | LoRA, 64k samples, TP2 CP4 (CP patch) | 31.7 | 33 k | 33.2 | 64 |
| c64k-lora-tp8 | LoRA, 64k, TP8 no CP | 38.6 | 27 k | 40.1 | 50 |
| c64k-full-off-cp4 | full CPU Adam, 64k, TP2 CP4 | OOM (logits) | – | – | 77 |
| **with `--log-probs-chunk-size 4096`** (fp32 [tokens x vocab/TP] cast was every OOM below) | | | | | |
| k16k-lora-tp2-ck | LoRA TP2 16k | 20.5 | 51 k | 21.7 | **40** (was 53) |
| k16k-lora-tp1-ck | LoRA **TP1** 16k (OOM without chunking) | **17.9** | **59 k** | 19.0 | 74 |
| k64k-lora-tp2cp2-ck | LoRA 64k TP2 CP2 | 31.1 | 34 k | 32.4 | 77 |
| k64k-full-off-cp4-ck | full CPU Adam 64k TP2 CP4 | 33.9 | 31 k | 35.3 | 72 |
| k128k-lora-cp4-ck | LoRA **128k** samples, TP2 CP4 | 37.3 | 28 k | 38.4 | 78 (tight) |
| k128k-lora-cp8-ck | LoRA 128k TP1 CP8 | OOM | – | – | 80 |

Same comparison on *real* rollouts (replay of the full arm's dapo-math dumps, ~2.1 M tok/step, `--max-tokens-per-gpu 9216`):
full 43.8 s vs LoRA 44.1 s actor_train; peak 69 GB vs **35 GB**.

**B. End-to-end GRPO, colocated** (dapo-math, 32 prompts x 8, 8k max response — ~all responses truncated at 8k for both
arms, so lengths match: 8.07 k vs 8.02 k mean; `--max-tokens-per-gpu 9216`):

| arm | engine TP | rollout s | decode tok/s | train s | sleep+wake+sync s | step s |
|---|---|---|---|---|---|---|
| full-colo | 1 | 92.6 | 22.3 k | 47.8 | ~16 | **172** |
| lora-colo | 1 | 123 | 16.7 k | 47.4 | ~14 | 214 |
| etp2-lora | 2 | 102 | 20.2 k | 55.6 | | 199 |
| etp4-full | 4 | **68.8** | 30.1 k | 47.3 | | **146** |
| lora-disagg (4 train + 4 rollout GPUs, sync) | 1 | 227 | 9.1 k | 86.8 | wsync 0.35 | 346 |
| full-disagg (4+4) | 1 | – | – | OOM (4 trainer GPUs, no offload) | | |
| etp2-full | 2 | 73.0 | 28.2 k | 59.9 | | 178 |
| etp2-lora-triton (`--sglang-lora-backend triton`) | 2 | 93.1 | 22.2 k | 52.6 | | 185 |
| etp4-lora (csgmv) | 4 | 97.9 | 21.2 k | 47.8 | | 181 |

**C. Correctness**
- **The rollout uses the trained adapter** (colocated *and* disaggregated): with 2 optimizer steps per rollout
  (LR 3e-4), one update moves the policy by KL≈0.03 on its own batch (`train/ppo_kl`, step 1), yet at the first step
  of the next rollout the SGLang-served logprobs match the updated trainer to KL≈2.3e-4 (|Δlogp|≈0.0066) — the same
  numerical floor as step 0. Over 5–10 steps the floor stays 0.006–0.009 (lora-hot, lora-move-colo/disagg).
- Same floor for full FT (0.0077–0.0082), so LoRA adds no train/rollout mismatch.
- `--lora-train-only` (rollout on frozen base) **crashes** in colocated mode (train actor dies in the first
  weight sync) — Miles bug in an off-path mode; don't use it.
- **SGLang engine TP>1 works for Qwen3.5 in this image**: etp2-lora and etp4-full have the same train/rollout
  mismatch as TP1 (0.0080–0.0082) and sane rewards — the sglang#21039 garbage-output bug the Miles recipe pins TP1
  for is not present at sglang `6e25d1b`. TP4 engines are 1.35x faster on rollout than TP1 (full FT).

**D. What this means**
- LoRA does **not** make the training math faster: at equal parallelism it is 0–15 % faster per token (no weight
  grads/Adam, but activation recompute + the 248k-vocab logits dominate). Its real wins are **memory** (35–52 GB vs
  69–80 GB/GPU) — full FT needs CPU Adam or TP4 at 16k/GPU and does not fit 64k samples on one node — and a
  **~7x smaller weight sync** in disaggregated mode (0.35 s vs ~2.4 s).
- SGLang LoRA serving (rank 32) costs **21–30 % decode throughput** at equal engine TP (TP1 16.7 k vs 22.3 k;
  TP2 20.2 k csgmv / 22.2 k triton vs 28.2 k; TP4 21.2 k vs 30.1 k tok/s). `--sglang-lora-backend triton` beats the
  default csgmv by ~10 % at TP2; LoRA engines barely gain from TP2→TP4. **End-to-end on one colocated node,
  LoRA steps are ~20 % slower than full FT** (best: full TP4 engines 146 s vs LoRA 181–185 s).
- Long context: chunked logprobs (`--log-probs-chunk-size 4096`) are mandatory at ≥16k tokens/GPU with this 248k
  vocab; with them LoRA trains 128k-token samples on a single node (TP2 CP4), full FT needs CPU Adam from 16k up.
- **Per-GPU training throughput here is ~10x the current polar-slime runs** (Miles: 28–59 k tok/s per 8-GPU node,
  285–373 "TFLOPs"/GPU by the same 3x-fwd formula, for both LoRA and full FT; polar-slime bbh 64k on 32 trainer GPUs
  logged 6–12 k tok/s, 17–35 TFLOPs). Caveat: synthetic uniform-length batches vs real multi-turn traces — path-a
  should confirm on replayed agentic traces. If it holds, the stack (TileLang GDN kernels, recompute/offload layout),
  not LoRA, is the big speedup.

**E. Recommended layout for Qwen3.5-9B LoRA agentic RL at 64k–128k** (measured pieces above; untested as a whole)
- Trainer: LoRA r32 `all-linear`, `--megatron-to-hf-mode bridge`, TP2 + CP2 (64k, `--max-tokens-per-gpu 32768`)
  or CP4 (128k, 32768), full recompute, `--log-probs-chunk-size 4096`, `--use-rollout-logprobs`, Adam on GPU (no
  offload needed), base patch set (CP fix). One node (8 GPUs) trains 1 M tokens in ~31 s (64k) / ~37 s (128k).
- Rollout: SGLang engines **TP2, `--sglang-lora-backend triton`**, `--sglang-max-lora-rank 32`; colocated
  (`--colocate --lora-base-cpu-backup`) for single-node experiments; for multi-node agentic runs, disaggregate
  (`--update-weight-transfer-mode broadcast`, PP=1) with fully-async rollouts — adapter sync is 0.35 s, so most
  GPUs can serve rollouts while 1 node trains.
- If rollout throughput is the bottleneck and memory allows (≤32k), full FT with TP4 engines is ~20 % faster per
  step than LoRA on one node; LoRA wins when context length, GPU count for training, or weight-sync size bind.
- `--use-rollout-logprobs` removes the separate old-logprob forward pass: −24 % train step for both arms. With one
  optimizer step per rollout and the measured mismatch (KL≈2.5e-4) this is a safe default.
- Synchronous disaggregation is a loss on one node (no overlap); it only pays with fully-async rollouts.
