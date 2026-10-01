# STACK.md — the Miles runtime on our clusters (owner: stack agent)

Status: **usable on dfw and hel** (dfw: SIF, Ray 1+2 nodes, NCCL over IB validated; hel: SIF, imports, nested apptainer validated). aws-iad: SIF + compat + nested apptainer + Ray validated (job 7587420). **Benchmark results: §9.**
Code: ProRL-Agent-Server branch **`miles-stack`**, dir `examples/miles_runtime/` (README there).
Last updated: 2026-10-01 05:35 PT. **Use the latest `miles-stack` head** (≥ `52074b8b`; < `52074b8b`: Qwen3.5 + context parallelism crashes, Ray ports inside dfw's ephemeral range; < `5c1b3bc8`: multi-node ray_node.sh only started the head; < `025a9936`: earlier mrun crashes with ENOSPC on dfw/aws-iad `$HOME`; < `6f487a5d`: Ray cannot start).

> **[2026-10-01 06:20] MTP auxiliary loss — FIXED in the base patch set (miles-stack `a49a88ab`, base `0004`): pull `miles-stack`.**
> Bridge mode built the HF config's MTP layer (Qwen3.5/3.8 `mtp_num_hidden_layers: 1`) although Miles logs `mtp_num_layers None` /
> `enable_mtp_training False`; its next-token CE over ALL tokens (scale 0.2, not advantage-weighted) was attached to the decoder
> output by `MTPLossAutoScaler`, i.e. mixed into every LoRA/RL update. Every bridge-mode run before `a49a88ab` had it.
> - 27B, synthetic data whose RL gradient is ~0 (dummy rollout logprobs -> importance ratios ~e^-10, clipped): grad_norm
>   **0.21 with MTP vs 0.0000 without** — the whole update was the MTP loss (random tokens: its worst case).
> - 9B real SWE-Gym traces, identical data and weights at step 0: grad_norm 0.0769 -> 0.0716.
> - Memory/speed: 9B 64k TP4 66.3 -> 56.1 GB, 44.5 -> 40.0 s/step (useful MFU 20.7 -> 23.0 %); 27B 64k TP4 74.2 -> 62.4 GB,
>   153 -> 141 s; 27B 128k TP4·CP2-hw OOM -> fits. Params per TP rank -60.8 M (9B, TP4).
> - Merged serving end-to-end without MTP (dfw 19610045, merged-move-colo shape): served-vs-trainer KL 2.3-3.7e-4 after
>   updates (stale engine: ~0.03) -> export fine. `patches/no-mtp` is now an empty alias. `--enable-mtp-training` restores MTP.

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
- `miles/0002` (resident colocation weight sync), `miles/0003` (layer-aware MFU metrics), and **`miles/0004`: drop the
  HF-config MTP layer unless `--enable-mtp-training`** (see the note at the top; it leaked an MTP loss into every update).
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
- Training throughput of this stack (synthetic fixed-length batches): 28–59 k tok/s per 8-GPU node for both
  LoRA and full FT. Real-trace validation (SWE-Gym traces from path-a/path-b runs) is pending.

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

**F. SGLang LoRA serving knobs (dfw job 19593033, standalone servers, TP1, r32 all-linear adapter, forced decode)**

| config | 2k prompt + 2k gen, 64 conc | 32k prompt + 1k gen, 16 conc |
|---|---|---|
| base (no LoRA) | 4478 tok/s | 581 tok/s |
| csgmv chunk 16 (default) / 32 / 64 / 128 | 3420 / 3409 / 3401 / 3405 (−24 %) | 456 / 469 / 468 / 471 (−19–22 %) |
| triton | 3545 (−21 %) | 407 (−30 %) |
| torch_native | 3263 (−27 %) | 471 (−19 %) |

No serving knob closes the gap (CUDA graphs are already on with LoRA; `max_loras_per_batch` is already 1;
`max_lora_rank` = actual rank 32). triton is best for decode-heavy, csgmv for prefill-heavy shapes.
→ prototype `--lora-serve-merged` (patch set `patches/lora-serve-merged/`, validation running: bench arms-merged).

## 10. Train LoRA, serve merged (`--lora-serve-merged`) — VALIDATED, recommended default for LoRA

Patch set `patches/lora-serve-merged/` (miles-stack `90b4b4be`; stack it after the base set:
`mrun --patches <runtime>/patches/lora-serve-merged ...`, bench: `PATCH_DIR=bench/patches:patches/lora-serve-merged`).
Each weight sync exports **W + (α/r)·B·A** through Megatron-Bridge (`export_hf_weights(merge_adapter_weights=True)`)
and pushes full weights via the normal full-model path (CUDA IPC colocated, NCCL broadcast disaggregated). SGLang
runs a plain model: no LoRA kernels, no `lora_path`. Model-agnostic (any bridge-mode model, incl. Qwen3.8-27B).

| (Qwen3.5-9B, 16 prompts x 8, 4k responses, LR 3e-4, 2 optimizer steps/rollout) | adapter serving | **merged serving** |
|---|---|---|
| colocated: rollout s (steady) | 49.4 | **35.9** (−27 %) |
| colocated: weight sync s | ~2.0 | 2.8 |
| disaggregated 4 train + 4 rollout GPUs: rollout s | 56–58 | **41.9** (−26 %) |
| disaggregated: weight sync s | 0.30–0.49 | 0.52–0.55 |
| KL(served ‖ trainer) at each new rollout, vs one-update policy movement KL≈0.03 | 2.1–2.6e-4 | 2.1–2.8e-4 |

The served-vs-trainer KL stays at the numerical floor after every update (a stale or unmerged engine would
show ≈0.03), in both layouts. The patch also keys the trainer's colocated sleep/wake memory layout on "LoRA is
trained" instead of "LoRA is served" (this also fixes the `--lora-train-only` colocated crash).

**Recommended LoRA flags (9B and 27B)**
```
--lora-rank 32 --lora-alpha 32|64 --lora-dropout 0 --target-modules all-linear --megatron-to-hf-mode bridge
--lora-serve-merged                      # NOT --sglang-max-lora-rank / --lora-base-cpu-backup / --sglang-lora-backend
--use-rollout-logprobs --log-probs-chunk-size 4096 --no-gradient-accumulation-fusion
# disaggregated: --update-weight-transfer-mode broadcast (pipeline-parallel 1)
```
Agent gateways (Polar/Harbor) must **not** send `lora_path` under `--lora-serve-merged` (plain engine).
End-to-end (dfw 19595394; dapo 32x8, 8k responses, colocated, 4 steps): merged serving == full-model serving.

| arm | engine TP | rollout s | step s | train/rollout abs-diff of log p | KL(rollout‖train) |
|---|---|---|---|---|---|
| full FT (etp2-full / etp4-full) | 2 / 4 | 72–74 / 66–72 | 164 / 143–149 | 0.0077–0.0082 | 2.7–2.9e-4 |
| LoRA, adapter serving (etp2-lora) | 2 | 102 | 199 | 0.0080–0.0081 | ~2.9e-4 |
| **LoRA, merged serving** (m2-etp2 / m2-etp4) | 2 / 4 | **73–78 / 65–71** | **154 / 146–151** | 0.0078–0.0089 | 2.8–3.2e-4 |
| LoRA merged + `--sglang-mamba-ssm-dtype bfloat16` | 2 | 68–70 | 148–150 | **0.0187–0.0212** | **1.4–1.6e-3** |

**bf16 mamba SSM state is NOT free for 9B against the trainer**: 7 % faster rollout but 2.4x the train/rollout
logprob mismatch and 5x the KL (the 27B decode-vs-prefill self-consistency check cannot see this). Not in the
recommended flags; if used, pair it with importance correction (TIS/ICEPOP) and re-measure for 27B against the
trainer, not engine self-consistency.
64k trainer layout default: see §11 (provisional).

## 11. 64k LoRA trainer layout on REAL agentic traces — DECIDED: TP4 / CP1 (4 and 8 trainer GPUs)

Data: path-a's SWE-Gym codex dumps (smoke2, 64k cap), 3 dumps merged into steps of all 24 trajectories =
**892k tokens/step, lengths 9k–65.5k (median 36k, p90 65k)**; replayed with `--load-debug-rollout-data`
(`bench/compose_replay.py`), LoRA r32/α64, path-a's loss flags (`--calculate-per-token-loss
--use-rollout-logprobs --log-probs-chunk-size 4096 --reward-key score`), full recompute, THD packing.

| 8 trainer GPUs (DP = 8/(TP·CP)) | GDN CP mode | max tok/GPU | steady actor_train s/step | tok/s/node | tok/s/GPU | peak GB/GPU* |
|---|---|---|---|---|---|---|
| **TP4, no CP** (DP2) | – | 65536 | **22.2** | **40.2 k** | **5.0 k** | 65 |
| TP2·CP2 | chunkwise | 32768 | 27.2–28.0 in one job, **OOM at step 1** in another (same config) | 32 k | 4.0 k | 77–79 (edge) |
| TP2·CP2 | headwise | 32768 | 26.0–28.4 | 31–34 k | 4.1 k | 79 (edge) |
| TP2·CP2 (path-a: chunkwise + `--attention-backend auto`) | chunkwise | 32768 | **OOM** | – | – | 78 |
| TP2·CP4 | headwise | 16384 | 28.7 | 31.1 k | 3.9 k | 50 |
| TP8, no CP | – | 65536 | 32.4 | 27.5 k | 3.4 k | 36 |
| TP4·CP2 | headwise | 32768 | 36.1 | 24.7 k | 3.1 k | 52 |
| TP1·CP4 | headwise | 16384 | OOM | – | – | 79 |

*nvidia-smi used memory (includes the PyTorch allocator cache).

**Default for 8 trainer GPUs, 64k cap: TP4, no context parallelism, `--max-tokens-per-gpu 65536`**
(22.2 s per 892k-token real step = 40 k tok/s/node, 65 GB peak). CP adds communication without saving enough
memory to pay for itself at 64k; TP4+SP already puts only 16k tokens of activations per GPU.
- headwise vs chunkwise GDN CP: **same speed** at TP2·CP2 on these traces (26–28 s); both sit at the 80 GB edge
  (chunkwise OOMed in 1 of 2 identical runs). Headwise is not the 64k fix — dropping CP is. Use headwise when CP
  is unavoidable (128k), since it avoids the naive TP-gather fallback and has the same speed.
- path-a's exact config (+ `--attention-backend auto`) OOMs: use the default flash backend.
- **INVALID: every earlier "4-GPU" bench result** (arms-real64k-4gpu `r4-*`, any `GPUS=4` arm before miles-stack
  `168cd373`) actually trained on 8 GPUs with DP doubled: `ray_node.sh` overwrote the driver's `GPUS` env. Fixed
  in `168cd373` (same fix as miles-fleet `b53b7fee`). **Pull miles-stack ≥ 168cd373** and check
  `--actor-num-gpus-per-node` in each arm's args.txt.
- **4 trainer GPUs (true, hel 1524737)**, same real steps:

| 4 GPUs | warm actor_train s/step | tok/s | useful MFU | peak GB |
|---|---|---|---|---|
| **TP4, no CP** (`--max-tokens-per-gpu 65536`) | **44.2** | **20.2 k** | **20.9 %** | 65 |
| TP2·CP2 headwise | 51.2 (+16 %) | 17.4 k | 18.0 % | 78 (edge) |
| TP2·CP2 chunkwise | OOM | | | 75 |
| TP1·CP4 headwise | OOM | | | 79 |

**D3 decision (64k cap, 4 or 8 trainer GPUs): TP4, CP1, `--max-tokens-per-gpu 65536`, full recompute, flash
attention backend, `--log-probs-chunk-size 4096`.** Throughput is 5.0 k tok/s/GPU on real SWE-Gym traces at both
4 and 8 GPUs (scales linearly with DP). Use CP (headwise) only past 64k/sample. Warm 1-node runs are
rollout-bound (path-a: 191 s train vs 515 s rollout), so further trainer tuning has low value; TP2×DP2 at
64k/GPU is the one remaining check (hel batch 1524978).
- Chunkwise CP (Megatron's default) with THD packing + sequence parallelism goes through the naive
  "TP gather → CP all-to-all → TP scatter" layout conversion around every GDN layer: much slower and more
  memory (OOM here). Headwise keeps attention's zigzag layout and splits GDN heads across CP instead.
- Headwise needs `(TP·CP)` to divide the GDN key heads (9B: 16; 27B: check `linear_num_key_heads`).
- Step 0 of every *new shape* pays TileLang/Triton JIT (86–545 s here); runtime `50c1cda3`+ persists the JIT
  caches across jobs (seed/publish to `$MILES_STACK_ROOT/jitcache/`).

## 12. DRAFT recommended recipe (pending D1 path, D2 learning, D4 session caps, D5 27B footprint)

**Runtime**: ProRL `miles-stack` ≥ `7519dd6e` (includes FLEET's A100/`MR_NODE_GPUS` fixes), SIF
`radixark/miles@sha256:30bca3fc…`; base patch set auto-applied (otel pin, Qwen3.5 CP MRoPE fix, resident-colocation
fix, layer-aware MFU metrics, MTP-layer drop — needs miles-stack ≥ `a49a88ab`); add `--patches <runtime>/patches/lora-serve-merged`. JIT caches persist per
cluster automatically (`$MILES_STACK_ROOT/jitcache/<image>-sm90/latest.tar`, seeded at `ray_node.sh` start, published
at its end): 9B TP4 real-trace step 0 **454 s cold → 137 s warm** in a new job (steady 44 s; hel 1524737 vs 1525159).
Jobs that don't launch through `ray_node.sh` can call `mr_jit_seed` / `mr_jit_publish` from `lib.sh`.

**LoRA + serving (9B and 27B)** — validated (§10):
```
--lora-rank 32 --lora-alpha 64 --lora-dropout 0 --target-modules all-linear --megatron-to-hf-mode bridge
--lora-serve-merged --no-gradient-accumulation-fusion
--use-rollout-logprobs --log-probs-chunk-size 4096 --calculate-per-token-loss
--attention-backend flash            # NOT auto (OOM on real 64k traces)
```
Not recommended (measured): adapter serving (−21–30 % decode), `--sglang-mamba-ssm-dtype bfloat16` (5x train/rollout
KL on 9B), chunkwise GDN CP at 64k (OOM/edge), `--lora-train-only`, fp8 weights for rollout (27B: 5x mismatch).

**Trainer @64k (D3, decided)**: `--tensor-model-parallel-size 4 --context-parallel-size 1 --sequence-parallel
--max-tokens-per-gpu 65536 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1`.
9B: 5.0 k tok/s/GPU on real SWE-Gym traces (4 GPUs: 20 k tok/s; 8 GPUs: 40 k tok/s), useful MFU ~21 %.
27B: Megatron TP ≤ 4 (4 query groups) → TP4/CP1 too (qwen27b D5 run on it).
**128k (P1)**: 9B needs 8 trainer GPUs: TP8 (43 s per 1.05 M tokens) or TP4·CP2 `--linear-cp-mode headwise` (48 s);
nothing fits 128k on 4 GPUs.

**Rollout (D4)**: plain engines (merged serving); engine TP>1 is correct for the Qwen3.5 line
(9B TP2/TP4 = same mismatch floor as TP1; 27B TP2/TP4 same). For disaggregated weight sync:
`--update-weight-transfer-mode broadcast` (9B full-weight broadcast 0.5 s on one node).
**Engine TP and session caps (D4, measured: aws-iad 7592005, `bench/agent_suite.sh`)** — 9B, 4 engine GPUs (the
rollout half of a 1-node run), merged/plain engines at mem-fraction 0.85, simulated agentic sessions (6k prompt + 16
turns of 1.4k tool tokens + 600 generated; context grows to ~38k; prefix-cache reuse like router session affinity):

| engines | in-flight sessions | generated tok/s | turns/s | turn latency p50 / p90 s | prefix-cache hit |
|---|---|---|---|---|---|
| 4 x TP1 | 48 | 3.8 k | 6.4 | 7.5 / 8.7 | 0.91 |
| **2 x TP2** | 48 | **4.7 k** | 7.8 | 6.1 / 7.3 | 0.91 |
| 4 x TP1 | 96 | 5.9 k | 9.8 | 9.4 / 11.4 | 0.91 |
| **2 x TP2** | **96** | **6.8 k** | **11.3** | 8.2 / 10.7 | 0.92 |
| 4 x TP1 | 160 | 3.4 k (collapse) | 5.6 | 13.5 / 65 | 0.33 |
| 2 x TP2 | 160 | 3.4 k (collapse) | 5.7 | 13.5 / 48 | 0.41 |

At the real 64k shape (hel 1525224, cancelled after 3 of 4 levels to yield to path-b's D2 baseline; 8k prompt + 37 turns
x (960 generated + 560 tool), contexts grow to ~64k):

| engines | in-flight | generated tok/s | turns/s | turn latency p50 / p90 s | prefix-cache hit |
|---|---|---|---|---|---|
| 4 x TP1 | 32 / 48 / 64 | 3.1 k / 3.5 k / 3.9 k | 3.2 / 3.7 / 4.1 | 9.9 / 13.0 / 13.1 (p90 20.9 at 64) | 0.97 / 0.97 / **0.91** |
| **2 x TP2** | 32 / 48 / **64** | 3.9 k / 4.2 k / **5.2 k** | 4.1 / 4.4 / **5.4** | 7.8 / 10.9 / 11.5 | 0.97 / 0.97 / 0.97 |

- **Use TP2 engines** (+15–33 % vs TP1 at equal GPUs): a TP2 engine holds 1.97 M KV tokens vs 0.85 M per TP1 GPU
  (+16 % per GPU, weights split) and decodes faster.
- **Cap in-flight sessions per engine by KV: sessions × mean *live* context ≲ 0.9 × KV_tokens** (live context
  averages ~half the peak over a session's life; the peak-based bound below is conservative — 2 x TP2 still had a 0.97
  cache hit at 32 sessions/engine with 64k peaks). Past it the radix/prefix cache
  thrashes (hit 0.91 → 0.33–0.41) and throughput halves — same cliff as 27B. 9B TP2 @ mem 0.85: ~45 sessions/engine at
  ~38k peak context, ~27/engine at 64k peak; TP1: ~20 and ~12. Mamba state slots cap running requests too (TP1: 96,
  TP2: 225 at these settings), not binding here.
- Raising in-flight from 48 to 96 on 4 engine GPUs gave +45 % generated tok/s with p50 turn latency 6→8 s; path-a/b
  run 48 in flight with engines at 11–12 running requests, so raising the cap (to the KV bound) is the cheapest 1-node
  speedup.

**Layouts (D1/D5 pending; current state of the real runs, 2026-10-01 00:15)**

| model | 1 node (8 GPU) — what the real runs use now | measured | next change (recipe) |
|---|---|---|---|
| 9B @64k | trainer 4 GPUs TP4/CP1/65536 + **2 x TP2 engines**, async (staleness ≤ 2), 8x8 sessions/step, 48 in flight, session-affine routing | path-a r2 (dfw 19597971): rollout-bound, ~5–10 steps/h, train 8.7–14.4k tok/s on 2.1–2.9M tok/step, engines 10–13 running, KV max .33–.45, cache hit .97–.98 | **`--lora-serve-merged`** (both paths still serve the adapter with triton: −20–26 % rollout); in-flight 48 → KV bound (~96 at these contexts; path-b arm D testing) |
| 27B @64k | trainer 4 GPUs TP4/CP1/65536 + 1 x TP4 engine, in-flight ≈ 26 | qwen27b r2 (dfw 19599407) running; trainer ~3.5x slower per token than 9B | same merged-serving switch; cap from TP4 KV (1.9M tokens) |
| 9B / 27B multi-node | 2 nodes: trainer node (8 GPUs, 9B TP4×DP2) + engine node (4 x TP2 for 9B) | path-a 2-node arm ready (P1) | pending D5 |

**Switching a run to merged serving** (patch set `patches/lora-serve-merged`): add `--lora-serve-merged`; remove
`--sglang-max-lora-rank`, `--sglang-lora-backend`, `--lora-base-cpu-backup`. Miles' session server (path-b) drops
`lora_path` automatically (the session config carries the flag); **path-a's Polar gateway must stop injecting
`lora_path`** (SGLang without `--enable-lora` rejects such requests: `ValueError: LoRA adapter 'miles_lora' was requested, but LoRA is not enabled`). Weight sync becomes a full-weight broadcast (9B 0.5 s).
Warm 1-node 9B runs are rollout-bound (path-a: train ~190 s vs rollout ~515 s per 64-session step), so the levers in
order are: merged serving, in-flight sessions up to the KV bound, engine GPUs (trainer:engine split), then trainer.

## 13. 27B LoRA trainer layouts for de4 (in progress; for qwen27b's trade-off table)

Qwen3.8-27B, LoRA r32 all-linear, bridge mode, full recompute, `--use-rollout-logprobs`, logprob chunk 4096,
`--max-tokens-per-gpu` = context/CP, train_only replay, H100 80GB. Megatron TP <= 4 (4 query groups); GDN 16 key / 48 value
heads -> headwise TP*CP must divide 16. Steady state = steps 1-2 (step 0 includes JIT/warm-up).
**Data**: SYNTHETIC = fixed-length samples at the cap (90 % response); REAL = DIAG's 9B-generated SWE-Gym traces, base policy
(shared tokenizer; 96k dumps: max 98.3k, 128k dump: max 103.7k), 16 samples/step (0.62-0.98 M tokens) incl. the longest group.
**MTP off** = base patch 0004 (all rows below except the two marked "on").

| context | layout (trainer GPUs) | data | s/step | tokens/step | tok/s | useful MFU | peak GB | result |
|---|---|---|---|---|---|---|---|---|
| 64k | TP4 (4), MTP on | synthetic | 153 | 1.05 M | 6.85 k | 24.8 % | 74.2 | fits |
| 64k | TP4 (4) | synthetic | **141** | 1.05 M | **7.43 k** | 26.9 % | **62.4** | fits |
| 96k | TP4 (4) | synthetic | 116.5 | 0.79 M | 6.75 k | 27.8 % | 72.7 | fits |
| 96k | TP4 (4) | **REAL** | — | | | | 78.9 | **OOM** (backward: 11.25 GiB alloc, 10.3 GiB free) |
| 96k | TP4 (4) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **REAL** | 150-189 | 0.62-0.77 M | **4.1 k** | 14.5 % | **77.2** | **fits, but slow** (vs 6.75 k synthetic: allocator pressure at the edge) |
| 96k | **TP4 (4) + expandable segments + `--log-probs-chunk-size 1024`** | **REAL** | **83-104** | 0.62-0.77 M | **7.45 k** | 26.4 % | 77.2 | **fits at full speed** |
| 96k | TP4 x DP2 (8) | synthetic | 59.7 | 0.79 M | 13.2 k | 27.1 % | 72.8 | fits on synthetic only: DP keeps the 4-GPU TP4 per-replica footprint, which OOMs on REAL 96k |
| 128k | TP4 (4) | synthetic | — | | | | 78.9 | OOM (~1 GiB short, logprob chunk) |
| 128k | TP4 (4) | **REAL** | — | | | | 78.7 | OOM (same place) |
| 128k | TP4 (4) + expandable segments + logprob chunk 1024 | REAL and synthetic | — | | | | 79.0 | **OOM** (114 MiB free) — 128k on 4 GPUs is out |
| 128k | TP2·CP2 headwise (4) | **REAL** | — | | | | | OOM (12 GiB [tokens/CP x vocab/TP] logits alloc) |
| 128k | TP4·CP2 headwise (8), MTP on | synthetic | — | | | | 78.5 | OOM |
| 128k | TP4·CP2 headwise (8) | synthetic | **115.6** | 1.05 M | **9.07 k** | 20.9 % | **77.8** | fits (tight) |
| 128k | TP4·CP2 headwise (8) | **REAL** | 94.3 (step 2; step 1: 266) | 0.98 M | 10.4 k | 19.3 % | **79.0** | fits at the edge |
| 128k | TP4 x DP2 (8) | synthetic | — | | | | | OOM (as on 4 GPUs: CP1 at 128k does not fit) |
| 96k | TP4·CP2 headwise (8) + expandable segments | **REAL** | 136-225 | 0.62-0.77 M | 3.4-4.6 k | 6-8 % | **56.3** | fits with **~24 GB headroom** (slow: see JIT note) |
| 128k | TP4·CP2 headwise (8) + expandable segments | **REAL** | 98 (step 2; steps 0-1: 262, 228) | 0.98 M | 9.9 k (step 2) | 18.5 % | **68.3** | fits, ~11 GB headroom |
| 128k | TP2·CP4 headwise (8) + expandable segments | **REAL** | — | | | | 77.8 | OOM in backward (recompute) |
| 192k | TP4·CP2 headwise (8) + expandable segments | synthetic | — | | | | 77.2 | OOM (logprob chunk, 0.4 GiB free) |
| 192k | TP2·CP4 headwise (8) + expandable segments | synthetic | — | | | | 77.7 | OOM (logprob chunk, 0.2 GiB free) |
| 192k | 16 GPUs (TP4·CP4) | — | | | | | | **not tested** — dropped: de4's longest observed 27B session is ~93k |

**192k needs ≥ 16 trainer GPUs (all 8-GPU layouts OOM); 16-GPU untested.** 2-node DP layouts (TP4 x DP4, TP4·CP2 x DP2) were
also dropped: data parallelism leaves the per-replica footprint unchanged, so they fit exactly where the 4- and 8-GPU
replicas above fit, at ~2x the tokens/s.

**Recipe for 4 trainer GPUs (1-node 4 trainer + 1 TP4 engine), 27B LoRA, MTP off (base 0004):**
- **64k**: TP4 / CP1 / `--max-tokens-per-gpu 65536` — 7.4 k tok/s, 62 GB (synthetic); comfortable.
- **96k**: TP4 / CP1 / `--max-tokens-per-gpu 98304` **plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in the
  environment Ray's trainer workers inherit (set it before `ray start`) — REAL traces, warm JIT: 7.2-7.45 k tok/s, 26 % useful
  MFU, 77.2 GB (cold: ~4.4 k, see the speed note below). Without expandable segments it OOMs — also with
  `--log-probs-chunk-size 1024` (dfw 19613127), which changes neither memory fit nor speed. Little headroom: watch
  per-step peak memory in the first steps of a real run.
- **128k**: does not fit on 4 GPUs (OOM with every setting tried). Needs 8 trainer GPUs, TP4·CP2 headwise **with expandable
  segments** (REAL: 68.3 GB; 79.0 GB without). TP2·CP4 headwise OOMs on REAL 128k.
- **Margin, 8 GPUs TP4·CP2 headwise + expandable segments, REAL (same cluster/settings): 96k 56.3 GB vs 128k 68.3 GB.**
- **192k**: needs ≥ 16 trainer GPUs (all 8-GPU layouts OOM); 16-GPU untested.

**Speed: the 4.1-4.4 k vs 7.2-7.45 k tok/s gap on REAL 96k (4 GPUs) is JIT/autotune warm-up, not the logprob chunk**
(dfw 19613127, identical arms back to back, expandable segments only, chunk 4096): run a (cold for these shapes) 4.4 k tok/s
(141-175 s/step, useful MFU 15-16 %); run b, same data, starting from a's JIT cache: **7.2-7.3 k tok/s** (86-106 s/step, 25-26 %).
Peak 77.2 GB in both. So chunk 1024 is not a speed lever; warm 4-GPU 96k = ~7.3 k tok/s.
**Open risk for real runs:** run a stayed cold on all 3 steps, and each step held different samples (new sequence lengths ->
new packed microbatch sizes). If kernel compile/autotune keys on those sizes, a production run (new lengths every step) would
sit near the cold ~4.4 k tok/s rather than 7.3 k. The 9B real-trace numbers (§11) replayed the SAME 24 samples every step, so
they were warm. Diagnostic queued (P1): fresh 6-step composition with `TRITON_PRINT_AUTOTUNING=1`.

Older reading (kept for the record): 96k fits the synthetic worst case but not real packed steps with default settings. The per-microbatch cost that grows with context is the LM-head logits, [tokens/CP x vocab/TP] (vocab
248k: 12 GiB in bf16 at 96k on TP4) plus their gradient; TP cannot go above 4 for 27B, so CP (more GPUs) is the lever.
The REAL 128k TP4·CP2-hw row has no headroom (79.0 GB) and one slow step (266 s vs 94 s); treat it as "fits only with the
rescue settings" until the expandable-segments rerun lands.
In flight: hel 1527310 (4-GPU 96k/128k rescue: expandable segments, logprob chunk 1024; then REAL 128k TP4·CP2-hw + exp,
synthetic 192k), aws-iad 7598871 (8 GPUs: synthetic 192k TP4·CP2-hw / TP2·CP4-hw, REAL 96k TP4·CP2-hw, REAL 128k TP2·CP4-hw / TP4·CP2-hw,
all with expandable segments — the 96k vs 128k TP4·CP2-hw peaks are the margin argument for the cap).

