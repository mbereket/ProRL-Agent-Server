# shellcheck shell=bash
# Qwen3-8B (dense) Megatron model args. Select with MODEL_ARGS_FILE=model_args_qwen3_8b.sh.
#
# Mirrors slime/scripts/models/qwen3-8B.sh (HF config: 36 layers, hidden 4096,
# ffn 12288, 32 heads / 8 kv groups, head_dim 128, rope_theta 1e6, untied
# embeddings, vocab 151936). Standard GPT spec: no --spec, no GatedDeltaNet.
MODEL_ARGS=(
    --swiglu
    --num-layers 36
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
    --untie-embeddings-and-output-weights
)
