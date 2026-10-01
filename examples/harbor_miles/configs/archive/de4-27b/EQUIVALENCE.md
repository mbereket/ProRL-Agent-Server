# Dry-render equivalence of the re-expressed running configs (2026-10-01 ~08:00 PT)

Each ORIGINAL config was rendered with the launcher of the commit it was submitted from, the re-expression with harbor-miles
(`tools/config_equiv.sh OLD_REF OLD_CONFIG NEW_CONFIG --nodes 2`, cluster dfw). Compared: the Miles args, train command,
runtime patch sets, agent servers per node, prepare_data inputs incl. the task list content, and every job-env variable some
launcher/agent/rollout code reads. Verdict for all three: EQUIVALENT; the only differences are the run-dir path (new shared
run root <user root>/miles/runs instead of miles/path-b/runs or miles/qwen27b/runs) and, for the hedge, HM_ROOT (setup root:
same Harbor PIN + patches, different venv location).

Not covered by the render (same for old and new): the Miles model args (printed by the pinned Miles inside the SIF from MODEL_TYPE).
Code differences outside the config, both inert for these runs: harbor-miles carries per-trial cancel by file (miles-path-b
1fdfa6d6; r3's miles-diag 7f76aa95 predates it: acts only when RUN_DIR/cancel/<id> is created), and the runtime's MTP-drop patch
moved from patches/no-mtp to base 0004 (identical content, 100 % rename; the hedge already ran with this arrangement).

## DIAG r3 = origin/miles-diag 7f76aa95 configs/diag-de4-27b-overfit8-2n-r3.env -> configs/experiments/diag-de4-27b-overfit8-r3.env

```
1. Miles args: old 149 tokens / 80 flags, new 149 tokens / 80 flags
   SAME flags and values and order; 2 token(s) differ only by run dir / run name
     cosmetic: /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/path-b/runs/diag-de4-27b-overfit8-r3/ckpt  ->  /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/runs/diag-de4-27b-overfit8-r3/ckpt
     cosmetic: /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/path-b/runs/diag-de4-27b-overfit8-r3/data/train.jsonl  ->  /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/runs/diag-de4-27b-overfit8-r3/data/train.jsonl
2. train command: SAME (train_async.py --fully-async)
3. runtime patch sets: SAME: --pythonpath <EXAMPLE>/miles_side --pythonpath <EXAMPLE> --patches <RUNTIME>/patches/lora-serve-merged --patches <RUNTIME>/patches/no-mtp --patches <EXAMPLE>/miles_patches
4. agent servers (2 node(s)): SAME
     node 0: sandboxes 40, pinned 80cpus, agent timeout 3600 s
     node 1: sandboxes 40, pinned 80cpus, agent timeout 3600 s
5. prepare_data: args SAME; task list content SAME (ids_sha256=f9b4471e449ae160 ids_count=8)
6. job environment (variables the launcher/agent/rollout code consumes):
   read only by train_driver.sh, whose output (Miles args + train command) is identical: LORA_TARGETS None->'all-linear', N_ENGINES None->'3', TRAIN_ALLOC_CONF None->''
   cosmetic (paths / run name): HARBOR_AGENT_SERVERS_FILE, HM_CONFIG_FILE, HM_CONFIG_LAYERS, HM_RUNS_ROOT, HM_TRIAL_LOG, RUN_DIR, TASK_IDS_FILE
   derivation inputs (new layered config; their effect is in the args/env above): CAP=96k, DATASET=de4-v1-k1, GDN_KEY_HEADS=16, INFLIGHT_PER_ENGINE=26, KNEE_INFLIGHT_PER_ENGINE=35, LAYOUT_PRESET=27b-2n, MAX_CAP=98304, MAX_TP=4, NODES=2
   set on one side, read by no code (no effect): HARBOR_MAX_SEQ_LEN=98304 (old)

VERDICT: EQUIVALENT (only cosmetic differences: Miles args: 2 path token(s) (run dir); env: HARBOR_AGENT_SERVERS_FILE, HM_CONFIG_FILE, HM_CONFIG_LAYERS, HM_RUNS_ROOT, HM_TRIAL_LOG, RUN_DIR, TASK_IDS_FILE)
```

## QWEN27B LR-1e-4 hedge = miles-qwen27b d48fac8a (local branch) configs/q27-de4-overfit8-2n-lr1e4.env (HM_ROOT miles/qwen27b) -> configs/experiments/q27-de4-overfit8-2n-lr1e4.env

```
1. Miles args: old 151 tokens / 81 flags, new 151 tokens / 81 flags
   SAME flags and values and order; 2 token(s) differ only by run dir / run name
     cosmetic: /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/qwen27b/runs/q27-de4-overfit8-2n-lr1e4/ckpt  ->  /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/runs/q27-de4-overfit8-2n-lr1e4/ckpt
     cosmetic: /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/qwen27b/runs/q27-de4-overfit8-2n-lr1e4/data/train.jsonl  ->  /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/runs/q27-de4-overfit8-2n-lr1e4/data/train.jsonl
2. train command: SAME (train_async.py --fully-async)
3. runtime patch sets: SAME: --pythonpath <EXAMPLE>/miles_side --pythonpath <EXAMPLE> --patches <RUNTIME>/patches/lora-serve-merged --patches <RUNTIME>/patches/no-mtp --patches <EXAMPLE>/miles_patches
4. agent servers (2 node(s)): SAME
     node 0: sandboxes 40, pinned 80cpus, agent timeout 3600 s
     node 1: sandboxes 40, pinned 80cpus, agent timeout 3600 s
5. prepare_data: args SAME; task list content SAME (ids_sha256=f9b4471e449ae160 ids_count=8)
6. job environment (variables the launcher/agent/rollout code consumes):
   read only by train_driver.sh, whose output (Miles args + train command) is identical: EXTRA '--train-env-vars {"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'->None, LORA_TARGETS None->'all-linear', N_ENGINES None->'3', TRAIN_ALLOC_CONF None->'expandable_segments:True'
   cosmetic (paths / run name): APPTAINER_CACHEDIR, APPTAINER_CONFIGDIR, HARBOR_AGENT_SERVERS_FILE, HF_HOME, HM_CONFIG_FILE, HM_CONFIG_LAYERS, HM_ROOT, HM_RUNS_ROOT, HM_TRIAL_LOG, RUN_DIR, TASK_IDS_FILE, UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, XDG_CACHE_HOME
   derivation inputs (new layered config; their effect is in the args/env above): CAP=96k, DATASET=de4-v1-k1, GDN_KEY_HEADS=16, INFLIGHT_PER_ENGINE=26, KNEE_INFLIGHT_PER_ENGINE=35, LAYOUT_PRESET=27b-2n, MAX_CAP=98304, MAX_TP=4, NODES=2
   set on one side, read by no code (no effect): HARBOR_MAX_SEQ_LEN=98304 (old)

VERDICT: EQUIVALENT (only cosmetic differences: Miles args: 2 path token(s) (run dir); env: APPTAINER_CACHEDIR, APPTAINER_CONFIGDIR, HARBOR_AGENT_SERVERS_FILE, HF_HOME, HM_CONFIG_FILE, HM_CONFIG_LAYERS, HM_ROOT, HM_RUNS_ROOT, HM_TRIAL_LOG, RUN_DIR, TASK_IDS_FILE, UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, XDG_CACHE_HOME)
```

## batch point 2 = origin/miles-path-b b8a7f86a configs/de4-codex-27b-bs128-2n.env -> configs/experiments/de4-codex-27b-bs128-2n.env

```
1. Miles args: old 149 tokens / 80 flags, new 149 tokens / 80 flags
   SAME flags and values and order; 2 token(s) differ only by run dir / run name
     cosmetic: /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/path-b/runs/de4-codex-27b-bs128-2n-r1/ckpt  ->  /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/runs/de4-codex-27b-bs128-2n-r1/ckpt
     cosmetic: /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/path-b/runs/de4-codex-27b-bs128-2n-r1/data/train.jsonl  ->  /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/runs/de4-codex-27b-bs128-2n-r1/data/train.jsonl
2. train command: SAME (train_async.py --fully-async)
3. runtime patch sets: SAME: --pythonpath <EXAMPLE>/miles_side --pythonpath <EXAMPLE> --patches <RUNTIME>/patches/lora-serve-merged --patches <RUNTIME>/patches/no-mtp --patches <EXAMPLE>/miles_patches
4. agent servers (2 node(s)): SAME
     node 0: sandboxes 53, pinned 80cpus, agent timeout 3600 s
     node 1: sandboxes 53, pinned 80cpus, agent timeout 3600 s
5. prepare_data: args SAME; task list content SAME (ids_sha256=f9b4471e449ae160 ids_count=8)
6. job environment (variables the launcher/agent/rollout code consumes):
   read only by train_driver.sh, whose output (Miles args + train command) is identical: LORA_TARGETS None->'all-linear', N_ENGINES None->'3', TRAIN_ALLOC_CONF None->''
   cosmetic (paths / run name): HARBOR_AGENT_SERVERS_FILE, HM_CONFIG_FILE, HM_CONFIG_LAYERS, HM_RUNS_ROOT, HM_TRIAL_LOG, RUN_DIR, TASK_IDS_FILE
   derivation inputs (new layered config; their effect is in the args/env above): CAP=96k, DATASET=de4-v1-k1, GDN_KEY_HEADS=16, INFLIGHT_PER_ENGINE=35, KNEE_INFLIGHT_PER_ENGINE=35, LAYOUT_PRESET=27b-2n, MAX_CAP=98304, MAX_TP=4, NODES=2
   set on one side, read by no code (no effect): HARBOR_MAX_SEQ_LEN=98304 (old)

VERDICT: EQUIVALENT (only cosmetic differences: Miles args: 2 path token(s) (run dir); env: HARBOR_AGENT_SERVERS_FILE, HM_CONFIG_FILE, HM_CONFIG_LAYERS, HM_RUNS_ROOT, HM_TRIAL_LOG, RUN_DIR, TASK_IDS_FILE)
```
