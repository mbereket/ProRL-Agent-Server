# A100 (sm80) support for the Miles runtime.
# megatron/0001: Transformer Engine prefers FlashAttention 4 (CuTe DSL) when installed; on sm80 FA4 fails to
# compile Qwen3.5 GQA head_dim-256 attention and cuDNN fused attention is unavailable for that shape, so hide
# FA4 below sm90 -> TE uses FlashAttention 2 (fleet job 35884731: FA2 21.9 ms fwd+bwd vs unfused 178.9 ms).
# No-op on H100/B200. Use: run.sh SUITE ARMS patches/a100[:other], or mrun --patches patches/a100.
