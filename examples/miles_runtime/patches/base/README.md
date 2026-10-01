Runtime-level fixes to the pinned image, applied by every `mrun` call (MILES_NO_BASE_PATCHES=1 skips).
- pip-requirements.txt: opentelemetry-api 1.44.0 to match the image's opentelemetry-sdk 1.44.0 (Ray agent crash otherwise).
- miles/0001-qwen3_vl-explicit-cp-mrope-position-ids.patch: Qwen3.5 (bridge = Qwen3-VL model) with context
  parallelism: the pinned Megatron-Bridge requires explicit rank-local 3D MRoPE position_ids for pre-sharded
  THD CP input; Miles builds them but only injected them through a get_rope_index hook (too late) ->
  ValueError "Pre-sharded packed CP inputs require explicit rank-local 3D MRoPE position_ids." Pass them explicitly.
