# Harbor Miles GRPO

GRPO on a directory of Harbor tasks with Polar (agent sandboxes, token capture, verifier) and
[Miles](https://github.com/radixark/miles) (Megatron trainer + SGLang engines), **LoRA by default**,
full fine-tune with `lora.rank: 0`. Same run-config schema and task contract as
`../harbor_slime_grpo` (task prep, harness, Polar templates are shared with it); this file covers
only what differs.

## Layout

| piece | where it runs |
|---|---|
| Polar rollout server, one gateway per sandbox node, Apptainer task sandboxes | host, small venv (`$WORKROOT/polar_venv`) |
| Ray, Miles trainer (Megatron + Megatron-Bridge LoRA), SGLang engines + model gateway | pinned Miles image (`../miles_runtime`: `mrun`, `ray_node.sh`) on every node |
| bridge (`src/slime_bridge`, runs Miles or Slime) | inside Miles' rollout executor, pinned to the Ray head |

Rollouts: agent CLI → Polar gateway (adds `lora_path=miles_lora` and the session id as
`X-SMG-Routing-Key`) → SGLang model gateway (`manual` policy: one session sticks to one engine) →
engine. The bridge refuses to start unless every gateway (rendered topology and live
`/admin/inference/status`) sends the trainer's adapter name.

## Run

```bash
# laptop, through the cluster layer (bio-synth train/polar-slime):
bash cluster/bootstrap.sh --clusters hel submit --example harbor_miles_grpo --repo-ref miles-path-a \
    --workroot <user_root>/miles/path-a --config experiments/<exp>/<run>.yaml --partition batch --hours 4
# or inside an allocation, on its first node:
bash examples/harbor_miles_grpo/internal/head_entry.sh <run>.yaml
```

Setup is idempotent (Miles SIF via `../miles_runtime/setup.sh`, polar venv, task SIFs, harness). Task
SIFs live in `$WORKROOT/harbor_sif_images`; images another work root's image dir already has
(`APPTAINER_IMAGE_DIR` from the cluster profile) are symlinked, the rest pulled. A `harness.dir`
outside `WORKROOT` is used read-only. Resume: resubmit; LoRA runs restart from the newest complete
`iter_N/adapter` (adapter + optimizer state; HF PEFT `adapter_model.safetensors` is saved next to it).

## Config additions (vs harbor_slime_grpo)

```yaml
lora: {rank: 32, alpha: 64, dropout: 0.0, target_modules: all-linear, exclude_modules: ""}   # rank 0 = full FT
cluster: {router_policy: manual}           # manual|consistent_hashing = session-affine; round_robin as before
rollout: {sglang_mem_fraction: 0.8}
training:
  sync: false                              # default async (train_async.py), TIS-corrected
  lr: 1e-5                                 # LoRA; ~1e-6 for full FT
  loss_aggregation: token_mean             # token_mean | trajectory_mean
  normalize_advantages: false
  qkv_format: thd                          # packed + dynamic batch (GDN supports thd and CP)
  save_hf_interval: 0                      # >0: also export a merged HF model
eval: {before_train: true}
```

Removed (Slime-only): `loss_denominator`, `checkpoint_keep_every`, `--dynamic-history`.

## Algorithm choices (rationale in `internal/render.py`)

- Every trace of a Polar trajectory is one Miles `Sample` with the same `rollout_id` (= Slime's
  `group_id`): the step schedule counts trajectories, not traces.
- Loss = sum of per-token PG terms over the step's trainable tokens / their count
  (`--calculate-per-token-loss`): unbiased w.r.t. the trajectory objective and independent of how
  prefix merging splits a trajectory into traces; no per-trajectory length normalization (Dr. GRPO bias).
- Advantage = reward − leave-one-trajectory-out group mean, no std scaling, no batch whitening.
- TIS on (also logs `train/train_rollout_logprob_abs_diff`).

## Measured (Qwen3.5-9B, H100)

- Smoke (dfw 19588031): LoRA r32 sync, codex: adapter sync 0.3–2.5 s, adapter save 0.6 s, trainer
  9.3 GB/GPU at TP2; warm step 12.9k train tok/s on 4 GPUs.
- Session-affine routing vs round_robin (dfw 19588618/19588628, 32 codex sessions, 4 engines):
  uncached prefill 1.54M vs 5.89M tokens (97% vs 88% cache hit), rollout 127 s vs 166 s.
