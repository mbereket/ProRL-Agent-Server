# shellcheck shell=bash
# Qwen3.5-9B Megatron model args for Miles (mirrors miles/scripts/models/qwen3.5-9B.py).
# Raw mode (full fine-tune) builds the model from these args and the Miles
# qwen3_5 spec (GatedDeltaNet + gated full attention, fp32 A_log). Under
# --megatron-to-hf-mode bridge (LoRA) Megatron-Bridge builds the model from the
# HF config and these only satisfy Megatron's argparse (the spec is inert).
MODEL_ARGS=(
    --spec "miles_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 12288
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate
)
