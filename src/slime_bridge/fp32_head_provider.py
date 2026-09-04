"""fp32 LM head for slime's Megatron actor and reference model.

Selected with ``--custom-model-provider-path slime_bridge.fp32_head_provider.model_provider``
(the example sets it from ``model.fp32_lm_head: true``). Builds the model with
slime's default provider, then on the post-process stage:

* replaces ``output_layer.weight`` with an fp32 Parameter holding the same
  values, keeping the attributes Megatron set on the bf16 one
  (``tensor_model_parallel``, ``partition_dim``, ``partition_stride``,
  ``allreduce``, ``sequence_parallel``, ``is_embedding_or_output_parameter``, ...);
* registers a forward pre-hook that casts the incoming hidden states to fp32,
  so ``ColumnParallelLinear``'s bare ``torch.matmul`` runs in fp32 (mixed
  dtypes would raise) and the logits reach slime's loss as fp32 without a
  post-GEMM upcast.

Everything downstream already handles an fp32 parameter next to bf16 ones:
DDP buckets by dtype, the distributed optimizer keeps fp32 params without a
separate master copy, weight sync ships tensors in their own dtype and the
sampler copies into its bf16 head storage. The sampler's own fp32 GEMM is
``--sglang-enable-fp32-lm-head``.
"""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger(__name__)


def _upcast_output_layer(layer: torch.nn.Module) -> None:
    old = layer.weight
    if old.dtype == torch.float32:
        return
    new = torch.nn.Parameter(old.detach().to(torch.float32), requires_grad=old.requires_grad)
    for key, value in old.__dict__.items():
        if not key.startswith("_"):
            setattr(new, key, value)
    layer.weight = new
    if getattr(layer, "bias", None) is not None and layer.bias.dtype != torch.float32:
        old_b = layer.bias
        new_b = torch.nn.Parameter(old_b.detach().to(torch.float32), requires_grad=old_b.requires_grad)
        for key, value in old_b.__dict__.items():
            if not key.startswith("_"):
                setattr(new_b, key, value)
        layer.bias = new_b

    def _cast_input(module, args, kwargs):
        if args and isinstance(args[0], torch.Tensor) and args[0].dtype != torch.float32:
            args = (args[0].to(torch.float32),) + tuple(args[1:])
        return args, kwargs

    layer.register_forward_pre_hook(_cast_input, with_kwargs=True)
    logger.info("fp32 LM head: output_layer.weight %s -> float32, input cast hook installed", tuple(old.shape))


def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    from megatron.training.global_vars import get_args
    from slime.backends.megatron_utils.model_provider import _get_model_provider_func

    args = get_args()
    # slime dispatches here because custom_model_provider_path is set; build the
    # stock model with a copy that has it cleared.
    base_args = argparse.Namespace(**vars(args))
    base_args.custom_model_provider_path = None
    provider = _get_model_provider_func(base_args)
    model = provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    output_layer = getattr(model, "output_layer", None)
    if post_process and output_layer is not None and getattr(output_layer, "weight", None) is not None:
        _upcast_output_layer(output_layer)
    return model
