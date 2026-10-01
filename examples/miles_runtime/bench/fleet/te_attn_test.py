"""Which Transformer Engine attention backend works on this GPU for Qwen3.5's gated attention shape?

Qwen3.5-9B full-attention layers: 16 query heads, 4 KV heads (GQA), head_dim 256, causal, packed varlen (thd).
Runs fwd+bwd once per env setting in a subprocess and reports the backend TE picked + ok/error + time.
"""
import json
import os
import subprocess
import sys
import time

CHILD = r'''
import os, time, torch
import transformer_engine.pytorch as te
if os.environ.get("HIDE_FA4") == "1":
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils
    FlashAttentionUtils.v4_is_installed = False
torch.manual_seed(0)
H, G, D = 16, 4, 256
lens = [int(x) for x in os.environ.get("SEQS", "8192,4096,2048").split(",")]
T = sum(lens)
cu = torch.tensor([0] + list(__import__("itertools").accumulate(lens)), dtype=torch.int32, device="cuda")
q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(T, G, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(T, G, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
attn = te.DotProductAttention(H, D, num_gqa_groups=G, attn_mask_type="padding_causal", qkv_format="thd").cuda()
def step():
    o = attn(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
    o.float().sum().backward()
step(); torch.cuda.synchronize()
t = time.time()
for _ in range(5): step()
torch.cuda.synchronize()
print("RESULT ok %.1f ms/fwdbwd" % ((time.time() - t) / 5 * 1e3))
'''

VARIANTS = {
    "default": {},
    "no_flash(fused cuDNN)": {"NVTE_FLASH_ATTN": "0"},
    "no_fused(flash only)": {"NVTE_FUSED_ATTN": "0"},
    "fa2 (FA4 hidden)": {"HIDE_FA4": "1", "NVTE_FUSED_ATTN": "0"},
    "unfused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "SEQS": "4096,2048"},
}

if __name__ == "__main__":
    import torch  # noqa: F401  (import torch before TE: TE first pulls a mismatched libnccl)
    import transformer_engine, flash_attn
    print("TE", transformer_engine.__version__, "flash_attn", flash_attn.__version__, "torch", __import__("torch").__version__)
    try:
        import cudnn  # noqa
        print("cudnn frontend present")
    except Exception:
        pass
    import torch
    print("cudnn", torch.backends.cudnn.version(), "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    out = {}
    for name, extra in VARIANTS.items():
        env = dict(os.environ, NVTE_DEBUG="1", NVTE_DEBUG_LEVEL="2", **extra)
        t = time.time()
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=900)
        txt = p.stdout + p.stderr
        backend = [l for l in txt.splitlines() if "Selected backend" in l or "Running with" in l or "Available backends" in l]
        res = [l for l in txt.splitlines() if l.startswith("RESULT")]
        err = [l for l in txt.splitlines() if "Error" in l and "ImportError: _Ext" not in l][-3:]
        out[name] = dict(rc=p.returncode, result=res[-1] if res else None, backend=backend[-3:], err=err, s=round(time.time() - t, 1))
        print(f"== {name}: rc={p.returncode} {res[-1] if res else ''}")
        for l in backend[-3:]:
            print("   ", l[-300:])
        for l in err:
            print("   ERR", l[-300:])
    print(json.dumps(out, indent=1)[:6000])
