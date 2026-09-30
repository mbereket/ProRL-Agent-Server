"""Cross-node NCCL all-reduce bandwidth (torchrun). Prints bus bandwidth per message size and the transport."""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for mb in (64, 256, 1024):
    n = mb * 1024 * 1024 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.time()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    busbw = (2 * (world - 1) / world) * (n * 2) / dt / 1e9
    if rank == 0:
        print(f"allreduce {mb} MB world {world}: {dt*1e3:.2f} ms, busbw {busbw:.1f} GB/s", flush=True)
dist.destroy_process_group()
