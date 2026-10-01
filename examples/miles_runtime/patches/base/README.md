Runtime-level fixes to the pinned image, applied by every `mrun` call (MILES_NO_BASE_PATCHES=1 skips).
- pip-requirements.txt: opentelemetry-api 1.44.0 to match the image's opentelemetry-sdk 1.44.0 (Ray agent crash otherwise).
- miles/0001-qwen3_vl-explicit-cp-mrope-position-ids.patch: Qwen3.5 (bridge = Qwen3-VL model) with context
  parallelism: the pinned Megatron-Bridge requires explicit rank-local 3D MRoPE position_ids for pre-sharded
  THD CP input; Miles builds them but only injected them through a get_rope_index hook (too late) ->
  ValueError "Pre-sharded packed CP inputs require explicit rank-local 3D MRoPE position_ids." Pass them explicitly.
- miles/0002: colocated LoRA with --no-offload-train read weights from a non-existent memory-saver backup (AssertionError: TorchMemorySaver observes invalid LD_PRELOAD).
- miles/0003: layer-aware training FLOPs -> perf/actor_train_{useful,hw}_{tflops,mfu}, perf/flops_core_frac (GDN layers linear, LoRA: no base weight grads, recompute counted in hw). Legacy perf/actor_train_tflops (3 x all-softmax fwd) overstates 9B LoRA useful work ~1.9x on real 64k traces.
- miles/0004: bridge mode built the HF config's MTP layer (Qwen3.5/3.6/3.8: mtp_num_hidden_layers=1) although Miles has
  enable_mtp_training False; its next-token CE (scale 0.2, all tokens, not advantage-weighted) leaked into every RL/LoRA
  gradient via MTPLossAutoScaler, and its fp32 [tokens x vocab/TP] logits cost ~10 GB at 64k. Drop it unless
  --enable-mtp-training. Measured: 27B zero-advantage grad_norm 0.21 -> 0; 9B 64k -10 GB, -10 % step; merged serving e2e OK.
