Opt-in (A/B pending): drop HF-config MTP layers in bridge mode unless --enable-mtp-training.
Without it, Qwen3.5/3.6/3.8 (mtp_num_hidden_layers=1) train an MTP cross-entropy loss (scale 0.2, all tokens,
unweighted by advantage) whose gradient flows into the decoder LoRA via MTPLossAutoScaler, and whose fp32
[tokens x vocab/TP] logits OOM 27B at 128k.
