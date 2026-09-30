# miles_runtime — pinned Miles (radixark/miles) runtime for our Slurm clusters

One Apptainer SIF of a digest-pinned `radixark/miles` image per cluster, a wrapper that runs
commands inside it (with optional setup-time patches), a multi-node Ray launcher, and the
Qwen3.5 LoRA/full-FT benchmark. Model-agnostic: nothing here is specific to one model except
`bench/` defaults. See `miles-work/STACK.md` for the validated flags, numbers and gotchas.

| file | what |
|---|---|
| `pins.env` | image repo/tag/digest, Apptainer version |
| `setup.sh` | idempotent per-cluster setup on a compute node: Apptainer, SIF pull, version manifest, smoke tests |
| `mrun` | `mrun [--patches DIR] [--bind ..] [--pythonpath ..] -- CMD` — run CMD in the SIF (`--nv`, lustre bound) |
| `ray_node.sh` | one call per Slurm node: head starts Ray + runs the driver, others join; each node session in its own PID namespace |
| `jobs/submit.py` | local: submit a job via slurm-compose (`--cluster hel --partition interactive ... -- CMD`) |
| `bench/` | `run.sh SUITE ARMS_FILE [PATCH_DIR]`, `grpo.sh` (knob-driven Qwen3.5 GRPO driver), `parse_metrics.py`, synthetic fixed-length batches |

## Layout on each cluster (`MILES_STACK_ROOT`, default `<lustre user root>/miles/stack`)

```
apptainer/1.5.3/bin/apptainer        unprivileged Apptainer (hel/dfw have none on PATH)
images/miles-<tag>-<digest12>.sif    the runtime (+ .versions manifest)
overlays/<hash>/                      patched trees built by mrun --patches (content-hashed)
models/<name>/                        HF checkpoints (bridge mode loads HF directly; no torch_dist)
datasets/, bench/, joblogs/
```

## Patches (pinned upstream + patch files, no forks)

A patch set is a directory:

```
my-patches/
  miles/0001-foo.patch        # -p1 relative to the Miles repo root      (image: /root/miles)
  sglang/0001-bar.patch       # -p1 relative to the SGLang repo root     (image: /sgl-workspace/sglang, python/ only)
  megatron/0001-baz.patch     # -p1 relative to Megatron-LM root        (image: /root/Megatron-LM, megatron/ only)
  pip-requirements.txt        # optional pure-python extras (pip --no-deps --target, prepended to PYTHONPATH)
```

`mrun --patches my-patches -- ...` (or `MILES_PATCHES=dir1:dir2`) builds
`overlays/<sha of sif+patch contents>/` once (flock-guarded, ~1 min) and bind-mounts the patched
trees over the image's editable installs. Patches must apply with `--fuzz=0` against the pinned
image; a patch that does not apply fails loudly. Generate patches against the commits recorded in
`images/<name>.versions`.

## Typical job

```bash
# local
python jobs/submit.py --cluster hel --partition interactive --nodes 1 --gpus 8 --hours 3 --name my-run \
    --pkg /path/to/my/code -- bash '$SCOMPOSE_PKGS/miles_runtime/ray_node.sh' --pythonpath '$SCOMPOSE_PKGS/code' \
    -- bash '$SCOMPOSE_PKGS/code/driver.sh'
```

The driver runs on the head node inside the SIF with `RAY_ADDRESS`/`MASTER_ADDR` set; it can call
`python3 /root/miles/train.py ...` directly or Miles' python launchers with `MILES_SCRIPT_EXTERNAL_RAY=1`.

## Host-side services next to a containerized trainer

The SIF runs with the host network and host `/tmp`, `/dev/shm`, `$HOME`, `/lustre`. Services that
launch Apptainer sandboxes (Polar gateway, a Harbor agent server) can therefore run **on the host**
(their own venv) and talk HTTP to the trainer's SGLang router / session server over localhost or the
node IP — no nesting needed. Nested Apptainer (sandbox launched from *inside* the SIF) is tested by
`setup.sh nested`; see STACK.md for the result on each cluster.
