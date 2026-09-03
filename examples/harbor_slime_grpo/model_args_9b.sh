# shellcheck shell=bash
# Qwen3.5-9B Megatron model args. Select with MODEL_ARGS_FILE=model_args_9b.sh.
#
# Mirrors slime/scripts/models/qwen3.5-9B.sh. Differs from the 4B args only in
# hidden/ffn sizes and untied embeddings (HF tie_word_embeddings=false for 9B).
# --spec wires in the hybrid GatedDeltaNet + full-attention layer layout.
MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 12288
    --use-gated-attention
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
