#!/usr/bin/env python3
"""Run config -> everything the shell scripts need, for the Miles trainer. The only place defaults live.

    render.py env <run.yaml>   print `export KEY=VALUE` lines (launch.sh evals them)
    render.py run <run.yaml>   write <RUN_DIR>/{polar_config.yaml,topology.yaml,train_args.sh}

Same config schema as harbor_slime_grpo (a config of that example runs here after
removing the Slime-only training keys), plus a `lora:` section and the Miles
algorithm/layout knobs. Unknown keys are an error. Schema (defaults in SCHEMA):

  name: run-id
  tasks:    dir (required), mount_root, n, seed, task_ids_file, exclude_ids_file
  harness:  name, model_name, dir, settings, session_timeout, request_timeout, max_run_workers,
            max_async_level, thinking, keep_sessions, path_prepend, ld_library_path, cli_version
  model:    hf_checkpoint, model_args_file, torch_dist_dir, load_dir, sglang_tool_call_parser
  lora:     rank (0 = full fine-tune), alpha, dropout, target_modules, exclude_modules
  cluster:  num_nodes, actor_num_gpus, tp_size, context_parallel_size, sandbox_nodes (head|all),
            gpus_per_engine, router_policy
  rollout:  batch_size, n_samples_per_prompt, num_steps | num_epoch, max_prompt_len,
            max_response_len, sglang_context_length, sglang_mem_fraction
  training: sync, max_tokens_per_gpu, qkv_format (auto|thd|bshd), lr, loss_aggregation, normalize_advantages,
            use_kl_loss, kl_loss_coef, grpo_std_normalization, optimizer_cpu_offload,
            group_id_scope, timeout_reward_zero, overlong_policy, drop_zero_variance_groups,
            save_interval, save_hf_interval, extra_train_args
  eval:     prompt_data ("<name> <path>", ${RUN_DIR} allowed), interval, n_samples_per_prompt, before_train
  judge:    model, api_base, api_key_env
  wandb:    project, group
  submit:   reaper_exempt_mins, reaper_reason, reaper_desc  (for the submitting layer; ignored here)

Algorithm defaults, and why (they differ from the Slime example on purpose):
  * loss_aggregation token_mean: sum of per-token policy-gradient terms over every
    trainable token of the step, divided by the step's trainable-token count
    (Miles --calculate-per-token-loss). This is the policy gradient of the
    trajectory-level objective up to a per-step constant: a trajectory's weight
    grows with its sampled tokens, as in sum_t grad log pi(a_t|s_t). The per-
    trajectory token mean (trajectory_mean) divides each trajectory by its own
    length, which biases toward long failures and short successes (Dr. GRPO) --
    and agent trajectories range from 1k to the full trace cap. Both are invariant
    to how prefix merging splits a trajectory into traces; token_mean needs no
    grouping at all, trajectory_mean groups by Miles rollout_id.
  * normalize_advantages false: token-level whitening across the batch shifts every
    advantage by the token-weighted batch mean, which under any length-varying
    aggregation is a length-dependent reward, and rescales the step by a batch
    statistic. The advantage is the leave-one-trajectory-out baseline (unbiased,
    bridge reward_post_process) without std scaling.
  * Asynchronous by default (training.sync false): agent rollouts are long-tailed
    (up to the session timeout), so a synchronous step waits for its slowest
    session while every GPU idles. Off-policy drift is corrected by TIS.
  * Session-affine routing (cluster.router_policy manual): the gateway sends the
    session id in X-SMG-Routing-Key; every turn of a session hits the engine that
    holds its prefix in cache.
"""
from __future__ import annotations

import copy
import os
import shlex
import string
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "..", "harbor_slime_grpo", "internal"))  # polar templates
PATH = object()  # marker: a filesystem path (expands ${ENV} and ~, relative to the config file)
LORA_ADAPTER_NAME = "miles_lora"  # miles.utils.lora.utils.LORA_ADAPTER_NAME
ROUTING_KEY_HEADER = "X-SMG-Routing-Key"  # read by the SGLang model gateway's manual/consistent_hashing policies

SCHEMA = {
    "tasks": {
        "dir": PATH,
        "mount_root": PATH,
        "n": None,
        "seed": 0,
        "task_ids_file": PATH,
        "exclude_ids_file": PATH,
    },
    "harness": {
        "name": "codex",             # codex | opencode | mini_swe_agent
        "model_name": "openai/gpt-5.4",
        "dir": PATH,
        "settings": {},
        "session_timeout": 3000,
        "request_timeout": 3600,
        "max_run_workers": 16,       # concurrent sandboxes per sandbox node
        "max_async_level": 2,        # groups in flight = batch_size x this (a warm pool across steps)
        "thinking": None,
        "keep_sessions": False,
        "path_prepend": "",
        "ld_library_path": "",
        "cli_version": "",
    },
    "model": {
        "hf_checkpoint": "Qwen/Qwen3.5-9B",
        "model_args_file": "qwen3_5_9b.sh",
        "torch_dist_dir": PATH,      # full fine-tune only: converted base checkpoint (raw mode)
        "load_dir": PATH,            # start from this checkpoint dir instead of own save dir / base
        "sglang_tool_call_parser": "qwen3_coder",
    },
    "lora": {
        "rank": 32,                  # 0 = full fine-tune (raw mode, torch_dist base, Adam over all weights)
        "alpha": 64,
        "dropout": 0.0,
        "target_modules": "all-linear",  # attention (incl. GDN projections) + MLP, Miles HF target groups
        "exclude_modules": "",
    },
    "cluster": {
        "num_nodes": 1,
        "actor_num_gpus": 4,         # trainer GPUs; whole nodes when num_nodes > 1. Every other GPU serves an engine
        "tp_size": 2,
        "context_parallel_size": 1,
        "sandbox_nodes": "all",
        "gpus_per_engine": 1,
        "router_policy": "manual",   # manual | consistent_hashing (session affinity) | round_robin | cache_aware
    },
    "rollout": {
        "batch_size": 8,
        "n_samples_per_prompt": 16,
        "num_steps": None,
        "num_epoch": None,
        "max_prompt_len": 8000,
        "max_response_len": 24000,
        "sglang_context_length": 32768,
        "sglang_mem_fraction": 0.8,
    },
    "training": {
        "sync": False,               # false: train_async.py (generation overlaps training, TIS-corrected)
        "max_tokens_per_gpu": 16384, # trace cap = this x context_parallel_size
        "qkv_format": "auto",        # auto | thd (packed) | bshd (one sample per micro-batch). auto: bshd under
                                     # LoRA (Megatron-Bridge builds megatron-core GatedDeltaNet, which rejects
                                     # packed sequences), thd in raw mode (Miles' qwen3_5 spec)
        "lr": "1e-5",                # LoRA; full fine-tune wants ~1e-6
        "loss_aggregation": "token_mean",  # token_mean | trajectory_mean
        "normalize_advantages": False,
        "use_kl_loss": False,
        "kl_loss_coef": 0.001,
        "grpo_std_normalization": False,
        "optimizer_cpu_offload": False,
        "group_id_scope": "trajectory",    # trajectory_mean unit: trajectory | prompt
        "timeout_reward_zero": True,
        "overlong_policy": "zero_reward_train",
        "drop_zero_variance_groups": True,
        "save_interval": 5,
        "save_hf_interval": 0,       # >0: also export a merged HF model every N steps (eval-v2 input)
        "extra_train_args": "",
    },
    "eval": {
        "prompt_data": "",
        "interval": 10,
        "n_samples_per_prompt": 1,
        "before_train": True,        # step-0 eval (the base model under LoRA)
    },
    "judge": {
        "model": "",
        "api_base": "",
        "api_key_env": "",
    },
    "wandb": {
        "project": "harbor-miles-grpo",
        "group": "",
    },
    "submit": {
        "reaper_exempt_mins": 0,
        "reaper_reason": "other",
        "reaper_desc": "",
    },
}
HARNESSES = ("codex", "opencode", "mini_swe_agent")
ROUTER_POLICIES = ("manual", "consistent_hashing", "round_robin", "cache_aware", "random", "power_of_two")
SESSION_AFFINE = ("manual", "consistent_hashing")


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def load(path: str) -> dict:
    """Read the config, apply defaults, resolve paths. Returns {section: {key: value}} plus 'name'."""
    cfg_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "name" not in raw:
        die(f"{path}: 'name' is required")
    unknown = set(raw) - set(SCHEMA) - {"name"}
    if unknown:
        die(f"{path}: unknown top-level keys {sorted(unknown)}; known: {sorted(SCHEMA)}")
    cfg = {"name": str(raw["name"])}
    for section, fields in SCHEMA.items():
        block = raw.get(section) or {}
        if set(block) - set(fields):
            die(f"{path}: unknown keys in '{section}': {sorted(set(block) - set(fields))}; known: {sorted(fields)}")
        out = {}
        for key, default in fields.items():
            v = block.get(key, None if default is PATH else copy.deepcopy(default))
            if isinstance(v, str):
                v = os.path.expandvars(v)
            if default is PATH and v:
                v = os.path.expanduser(str(v))
                v = v if os.path.isabs(v) else os.path.normpath(os.path.join(cfg_dir, v))
            out[key] = v
        cfg[section] = out
    t, h, c, r, tr, lo = (cfg[k] for k in ("tasks", "harness", "cluster", "rollout", "training", "lora"))
    if not t["dir"]:
        die(f"{path}: tasks.dir is required")
    if (r["num_steps"] is None) == (r["num_epoch"] is None):
        die(f"{path}: set exactly one of rollout.num_steps and rollout.num_epoch")
    if c["sandbox_nodes"] not in ("head", "all"):
        die(f"{path}: cluster.sandbox_nodes must be head or all")
    if c["router_policy"] not in ROUTER_POLICIES:
        die(f"{path}: cluster.router_policy must be one of {ROUTER_POLICIES}")
    if h["name"] not in HARNESSES:
        die(f"{path}: harness.name must be one of {HARNESSES}")
    if r["num_steps"] == 0 and not cfg["eval"]["prompt_data"]:
        die(f"{path}: rollout.num_steps: 0 (eval only) needs eval.prompt_data")
    if h["max_async_level"] > 1 and tr["sync"]:
        die(f"{path}: harness.max_async_level > 1 needs training.sync: false")
    if tr["loss_aggregation"] not in ("token_mean", "trajectory_mean"):
        die(f"{path}: training.loss_aggregation must be token_mean or trajectory_mean")
    if tr["qkv_format"] not in ("auto", "thd", "bshd"):
        die(f"{path}: training.qkv_format must be auto, thd or bshd")
    if tr["qkv_format"] == "auto":
        tr["qkv_format"] = "bshd" if int(lo["rank"]) > 0 else "thd"
    if int(lo["rank"]) < 0:
        die(f"{path}: lora.rank must be >= 0")
    return cfg


def lora_enabled(cfg: dict) -> bool:
    return int(cfg["lora"]["rank"]) > 0


def derived(cfg: dict) -> dict:
    workroot = os.environ.get("WORKROOT") or die("WORKROOT is not set")
    run_id = os.environ.get("RUN_ID") or cfg["name"]
    m = cfg["model"]
    args_file = m["model_args_file"]
    if not os.path.isabs(args_file):
        args_file = os.path.join(HERE, "model_args", args_file)
    return {
        "RUN_NAME": cfg["name"],
        "RUN_ID": run_id,
        "RUN_DIR": f"{workroot}/harbor_miles_grpo/{run_id}",
        "SAVE_DIR": f"{workroot}/ckpt/harbor_miles_grpo/{run_id}",
        "HARNESS_DIR": cfg["harness"]["dir"] or f"{workroot}/harbor_harness",
        "APPTAINER_IMAGE_DIR": os.environ.get("APPTAINER_IMAGE_DIR") or f"{workroot}/harbor_sif_images",
        "TORCH_DIST_DIR": m["torch_dist_dir"] or f"{workroot}/checkpoints/{m['hf_checkpoint'].rstrip('/').rsplit('/', 1)[-1]}_miles_torch_dist",
        "MODEL_ARGS_FILE": args_file,
        "TRAIN_SCRIPT": "train.py" if cfg["training"]["sync"] else "train_async.py",
        "LORA": "1" if lora_enabled(cfg) else "0",
    }


def mode_env(cfg: dict) -> None:
    d = derived(cfg)
    t, h, m, c, tr, lo = cfg["tasks"], cfg["harness"], cfg["model"], cfg["cluster"], cfg["training"], cfg["lora"]
    subset = f" (n={t['n']}, seed {t['seed']})" if t["n"] is not None else ""
    steps = f"steps {cfg['rollout']['num_steps']}" if cfg["rollout"]["num_steps"] is not None else f"epochs {cfg['rollout']['num_epoch']}"
    mode = f"LoRA r{lo['rank']}/a{lo['alpha']} {lo['target_modules']}" if lora_enabled(cfg) else "full fine-tune"
    env = {
        **d,
        "TASKS_DIR": t["dir"],
        "TASKS_MOUNT_ROOT": t["mount_root"] or t["dir"],
        "TASKS_N": "" if t["n"] is None else t["n"],
        "TASKS_SEED": t["seed"],
        "TASK_IDS_FILE": t["task_ids_file"] or "",
        "EXCLUDE_IDS_FILE": t["exclude_ids_file"] or "",
        "HARNESS": h["name"],
        "HARNESS_CLI_VERSION": h["cli_version"],
        "HF_CHECKPOINT": m["hf_checkpoint"],
        "NUM_NODES": c["num_nodes"],
        "SANDBOX_NODES": c["sandbox_nodes"],
        "POLAR_KEEP_SESSION_DIRS": "1" if h["keep_sessions"] else "",
        "JUDGE_API_KEY_ENV": cfg["judge"]["api_key_env"],
        "SUMMARY": (
            f"{cfg['name']} (RUN_ID {d['RUN_ID']}): {h['name']} on {t['dir']}{subset}; {m['hf_checkpoint']} {mode}; "
            f"{c['num_nodes']} node(s), trainer {c['actor_num_gpus']} GPUs TP{c['tp_size']} x CP{c['context_parallel_size']}, "
            f"engines TP{c['gpus_per_engine']} router {c['router_policy']}, sandboxes on {c['sandbox_nodes']}; "
            f"{cfg['rollout']['batch_size']} x {cfg['rollout']['n_samples_per_prompt']} per step, {steps}; "
            f"trace cap {tr['max_tokens_per_gpu'] * c['context_parallel_size']} tok; {d['TRAIN_SCRIPT']}, {tr['loss_aggregation']}"
        ),
    }
    for k, v in env.items():
        print(f"export {k}={shlex.quote(str(v))}")


def runtime_facts() -> dict:
    e = os.environ
    return {
        "HEAD_IP": e.get("RAY_HEAD_IP", "127.0.0.1"),
        "BIND_HOST": e.get("POLAR_BIND_HOST", "127.0.0.1"),
        "WORKER_IPS": [ip for ip in e.get("WORKER_IPS", "").split(",") if ip],
        "ROLLOUT_PORT": int(e.get("POLAR_ROLLOUT_PORT", 8080)),
        "GATEWAY_PORT": int(e.get("POLAR_GATEWAY_PORT", 8100)),
        "ROUTER_PORT": int(e.get("SGLANG_ROUTER_PORT", 9000)),
        "GPUS_PER_NODE": int(e.get("GPUS_PER_NODE", 8)),
        "DRY_RUN": e.get("DRY_RUN") == "1",
        "CALLBACK_HOST": e.get("POLAR_CALLBACK_HOST", "127.0.0.1"),
    }


def end_of_turn_token_id(hf_checkpoint: str) -> int:
    """Id of the token that closes an assistant turn (<|im_end|> on ChatML models);
    Polar's prefix-merging builder splits each completion's prompt at it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    special = set(tok.all_special_ids)
    eot = next((t for t in reversed(ids) if t in special), tok.eos_token_id)
    print(f"end-of-turn token: {tok.convert_ids_to_tokens(eot)!r} = {eot}", file=sys.stderr)
    return int(eot)


def model_args(args_file: str) -> list[str]:
    out = subprocess.run(["bash", "-c", f'source {shlex.quote(args_file)} && printf "%s\\n" "${{MODEL_ARGS[@]}}"'],
                         check=True, capture_output=True, text=True)
    return out.stdout.splitlines()


def gpu_layout(cfg: dict, f: dict) -> tuple[int, int, int]:
    """(actor_nodes, actor_gpus_per_node, rollout_gpus). Engines take every non-trainer GPU."""
    c = cfg["cluster"]
    gpn, actor, nodes = f["GPUS_PER_NODE"], c["actor_num_gpus"], c["num_nodes"]
    if nodes > 1 and actor % gpn:
        die(f"cluster.actor_num_gpus={actor} must be a multiple of the {gpn} GPUs per node when num_nodes > 1")
    actor_nodes, actor_gpus_per_node = (actor // gpn, gpn) if actor >= gpn else (1, actor)
    rollout_gpus = nodes * gpn - actor_nodes * actor_gpus_per_node
    if rollout_gpus < 1:
        die(f"no GPUs left for rollout ({nodes} x {gpn} total, {actor} train)")
    if actor % (c["tp_size"] * c["context_parallel_size"]):
        die(f"tp_size x context_parallel_size must divide actor_num_gpus ({actor})")
    if rollout_gpus % c["gpus_per_engine"] or gpn % c["gpus_per_engine"]:
        die(f"cluster.gpus_per_engine={c['gpus_per_engine']} must divide the engine GPUs ({rollout_gpus}) and the GPUs per node ({gpn})")
    return actor_nodes, actor_gpus_per_node, rollout_gpus


def latest_lora_adapter(root: str) -> str | None:
    """Newest Miles LoRA checkpoint under root: <root>/iter_NNNNNNN/adapter with rank-0 shards.

    Bridge LoRA saves write only the adapter dir (per-rank adapter + optimizer
    shards), no latest_checkpointed_iteration.txt, so resume points
    --lora-adapter-path at the newest one; Miles restores the adapter, the
    optimizer and the iteration (start = iteration + 1) from it.
    """
    if not root or not os.path.isdir(root):
        return None
    best = None
    for name in os.listdir(root):
        if not (name.startswith("iter_") and name[5:].isdigit()):
            continue
        adapter = os.path.join(root, name, "adapter")
        if os.path.isfile(os.path.join(adapter, "adapter_megatron_rank0.pt")) and \
                os.path.isfile(os.path.join(adapter, "training_state_rank0.pt")):
            best = max(best or (-1, ""), (int(name[5:]), adapter))
    return best[1] if best else None


def checkpoint_args(cfg: dict, d: dict, f: dict) -> list[str]:
    """Where the policy starts: resume (own save dir) > model.load_dir > the base model."""
    m = cfg["model"]
    latest = "latest_checkpointed_iteration.txt"
    if lora_enabled(cfg):
        # The frozen base always comes from --hf-checkpoint (Megatron-Bridge); only the adapter resumes.
        own = latest_lora_adapter(d["SAVE_DIR"])
        if own:
            return ["--lora-adapter-path", own]
        if m["load_dir"]:
            warm = latest_lora_adapter(m["load_dir"]) or die(f"model.load_dir has no LoRA adapter checkpoint: {m['load_dir']}")
            # Warm start: adapter weights of another run, but this run counts its own steps from 0.
            return ["--lora-adapter-path", warm, "--start-rollout-id", "0", "--no-load-optim"]
        # The adapter starts at B = 0, i.e. exactly the base policy.
        return ["--start-rollout-id", "0"]
    if os.path.isfile(os.path.join(d["SAVE_DIR"], latest)):
        return ["--load", d["SAVE_DIR"]]  # resume: Miles derives start_rollout_id from the checkpoint
    if m["load_dir"]:
        os.path.isfile(os.path.join(m["load_dir"], latest)) or die(f"model.load_dir has no checkpoint: {m['load_dir']}")
        return ["--load", m["load_dir"]]
    ref = d["TORCH_DIST_DIR"]
    if not f["DRY_RUN"] and not os.path.isfile(os.path.join(ref, latest)):
        die(f"full fine-tune: converted base checkpoint not found at {ref}")
    # The converted "release" checkpoint loads as iteration 0; start at rollout 0 explicitly.
    return ["--load", ref, "--ref-load", ref, "--start-rollout-id", "0"]


def train_args(cfg: dict, d: dict, f: dict) -> list:
    t, h, m, lo, c, r, tr, ev = (cfg[k] for k in ("tasks", "harness", "model", "lora", "cluster", "rollout", "training", "eval"))
    run_dir = d["RUN_DIR"]
    actor_nodes, actor_gpus_per_node, rollout_gpus = gpu_layout(cfg, f)

    if r["num_steps"] is not None:
        steps = ["--num-rollout", str(r["num_steps"])]
        if r["num_steps"] == 0:
            steps += ["--lr-decay-iters", "1", "--no-load-optim", "--no-load-rng"]
    else:
        steps = ["--num-epoch", str(r["num_epoch"])]
    eval_args: list = []
    if ev["prompt_data"]:
        eval_args = ["--eval-prompt-data", *ev["prompt_data"].replace("${RUN_DIR}", run_dir).split(),
                     "--eval-interval", ev["interval"], "--n-samples-per-eval-prompt", ev["n_samples_per_prompt"]]
        if not ev["before_train"]:
            eval_args.append("--skip-eval-before-train")

    if lora_enabled(cfg):
        adapter = ["--lora-rank", lo["rank"], "--lora-alpha", lo["alpha"], "--lora-dropout", lo["dropout"],
                   "--target-modules", lo["target_modules"], "--megatron-to-hf-mode", "bridge",
                   "--sglang-max-lora-rank", lo["rank"]]
        if lo["exclude_modules"]:
            adapter += ["--exclude-modules", lo["exclude_modules"]]
    else:
        adapter = []
    # Disaggregated trainer -> engines: NCCL broadcast (full weights, or only the adapter under LoRA).
    weight_sync = ["--update-weight-transfer-mode", "broadcast", "--update-weights-interval", 1]

    if tr["qkv_format"] == "bshd":
        batching = ["--qkv-format", "bshd", "--micro-batch-size", 1, "--max-tokens-per-gpu", tr["max_tokens_per_gpu"]]
    else:
        batching = ["--use-dynamic-batch-size", "--max-tokens-per-gpu", tr["max_tokens_per_gpu"]]

    # Global batch = every trajectory of the step, one optimizer step per rollout. The bridge
    # sets rollout_id per trajectory (group_id_scope trajectory) or per prompt, so Miles's
    # rollout-side schedule counts units, not trace samples.
    units = r["batch_size"] * (1 if tr["group_id_scope"] == "prompt" else r["n_samples_per_prompt"])
    if tr["loss_aggregation"] == "token_mean":
        loss = ["--calculate-per-token-loss"]
    else:
        loss = []  # per-unit token mean (rollout_mask_sums), mean over units

    wandb_mode = os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline")
    save_hf = []
    if tr["save_hf_interval"]:
        save_hf = ["--save-hf", f"{d['SAVE_DIR']}/hf/iter_{{}}"]
    args = [
        "--actor-num-nodes", actor_nodes, "--actor-num-gpus-per-node", actor_gpus_per_node,
        "--rollout-num-gpus", rollout_gpus, "--rollout-num-gpus-per-engine", c["gpus_per_engine"],
        *model_args(d["MODEL_ARGS_FILE"]),
        "--hf-checkpoint", m["hf_checkpoint"], *checkpoint_args(cfg, d, f),
        "--save", d["SAVE_DIR"], "--save-interval", tr["save_interval"], *save_hf,
        *adapter, *weight_sync,
        # Polar bridge: rollouts, rewards and data source come from slime_bridge (Miles-compatible).
        "--rollout-function-path", "slime_bridge.rollout.generate_rollout_polar_async",
        "--custom-rm-path", "slime_bridge.reward.reward_func",
        "--custom-reward-post-process-path", "slime_bridge.reward_post_process.post_process_rewards",
        "--custom-config-path", f"{run_dir}/polar_config.yaml",
        "--data-source-path", "slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer",
        "--prompt-data", f"{run_dir}/train.jsonl", "--input-key", "prompt", "--label-key", "label",
        "--metadata-key", "metadata", "--rollout-shuffle", "--reward-key", "score",
        *steps,
        "--rollout-batch-size", r["batch_size"], "--n-samples-per-prompt", r["n_samples_per_prompt"],
        "--global-batch-size", units,
        "--rollout-max-response-len", r["max_response_len"], "--rollout-max-prompt-len", r["max_prompt_len"],
        # Parallelism / memory
        "--tensor-model-parallel-size", c["tp_size"], "--sequence-parallel", "--pipeline-model-parallel-size", 1,
        "--context-parallel-size", c["context_parallel_size"], "--expert-model-parallel-size", 1,
        "--expert-tensor-parallel-size", 1,
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", 1,
        *batching,
        "--log-probs-chunk-size", 256, "--distributed-timeout-minutes", 30,
        # Algorithm: GRPO-style group baseline (bridge: leave-one-trajectory-out), TIS, clip-higher.
        "--advantage-estimator", "grpo", "--use-tis", "--get-mismatch-metrics",
        *(["--normalize-advantages"] if tr["normalize_advantages"] else []),
        *loss,
        *(["--use-kl-loss", "--kl-loss-coef", str(tr["kl_loss_coef"]), "--kl-loss-type", "low_var_kl"] if tr["use_kl_loss"] else []),
        *([] if tr["grpo_std_normalization"] else ["--disable-grpo-std-normalization"]),
        *(["--optimizer-cpu-offload", "--optimizer-offload-fraction", "1.0", "--overlap-cpu-optimizer-d2h-h2d",
           "--use-precision-aware-optimizer"] if tr["optimizer_cpu_offload"] else []),
        *eval_args,
        "--entropy-coef", "0.0", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", tr["lr"], "--lr-decay-style", "constant", "--weight-decay", "0.1",
        "--adam-beta1", "0.9", "--adam-beta2", "0.98", "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32", "--attention-backend", "auto",
        "--no-gradient-accumulation-fusion",
        # SGLang engines + router
        "--sglang-mem-fraction-static", r["sglang_mem_fraction"], "--sglang-context-length", r["sglang_context_length"],
        "--sglang-tool-call-parser", m["sglang_tool_call_parser"], "--sglang-router-policy", c["router_policy"],
        "--sglang-router-port", f["ROUTER_PORT"],
        # Router, rollout executor (the bridge) and its callback listener on the Ray head:
        # the Polar gateways reach the router at the head IP, and the rollout server
        # (head) calls the bridge back on 127.0.0.1.
        "--pin-rollout-manager-to-head",
        "--use-wandb", "--wandb-mode", wandb_mode, "--wandb-project", cfg["wandb"]["project"],
        "--wandb-group", cfg["wandb"]["group"] or cfg["name"],
        *shlex.split(tr["extra_train_args"]),
    ]
    return args


def mode_run(cfg: dict) -> None:
    d, f = derived(cfg), runtime_facts()
    t, h, m, c, tr, j = (cfg[k] for k in ("tasks", "harness", "model", "cluster", "training", "judge"))
    run_dir = d["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)
    public_host = f["HEAD_IP"]
    judge_key = ""
    if j["api_key_env"]:
        judge_key = os.environ.get(j["api_key_env"], "")
        if not judge_key and not f["DRY_RUN"]:
            die(f"judge.api_key_env={j['api_key_env']} is not set in the environment")

    subst = {
        "ROLLOUT_URL": f"http://{public_host}:{f['ROLLOUT_PORT']}",
        "GATEWAY_URL": f"http://{public_host}:{f['GATEWAY_PORT']}",
        "CALLBACK_HOST": f["CALLBACK_HOST"],
        "BIND_HOST": f["BIND_HOST"],
        "ROLLOUT_PORT": f["ROLLOUT_PORT"],
        "GATEWAY_PORT": f["GATEWAY_PORT"],
        "ROUTER_URL": f"http://{public_host}:{f['ROUTER_PORT']}",
        "RUN_DIR": run_dir,
        "RUN_NAME": cfg["name"],
        "MODEL_SERVED": m["hf_checkpoint"],
        "IMAGE_DIR": d["APPTAINER_IMAGE_DIR"],
        "HARNESS_DIR": d["HARNESS_DIR"],
        "DATASET_DIR": t["mount_root"] or t["dir"],
        "HARNESS": h["name"],
        "HARNESS_MODEL_NAME": h["model_name"],
        "SESSION_TIMEOUT": h["session_timeout"],
        "REQUEST_TIMEOUT": h["request_timeout"],
        "MAX_ASYNC_LEVEL": h["max_async_level"],
        "MAX_RUN_WORKERS": h["max_run_workers"],
        "PATH_PREPEND": (h["path_prepend"].rstrip(":") + ":") if h["path_prepend"] else "",
        "LD_LIBRARY_PATH": h["ld_library_path"],
        "RUBRIC_MODEL": j["model"],
        "RUBRIC_MODEL_API_BASE": j["api_base"],
        "RUBRIC_MODEL_API_KEY": judge_key,
        "TIMEOUT_REWARD_ZERO": str(tr["timeout_reward_zero"]).lower(),
        "DROP_ZERO_VARIANCE_GROUPS": str(tr["drop_zero_variance_groups"]).lower(),
        "GROUP_ID_SCOPE": tr["group_id_scope"],
        "OVERLONG_POLICY": tr["overlong_policy"],
    }
    try:
        subst["EOT_TOKEN_ID"] = end_of_turn_token_id(m["hf_checkpoint"])
    except Exception as exc:
        if not f["DRY_RUN"]:
            raise
        print(f"end-of-turn token: not derived ({exc}); left unset for the dry run", file=sys.stderr)
        subst["EOT_TOKEN_ID"] = "null"

    def render(name: str) -> dict:
        with open(os.path.join(SHARED, name)) as fh:
            return yaml.safe_load(string.Template(fh.read()).substitute(subst))

    polar = render("polar_config.yaml")
    polar["polar_task_template"]["agent"]["settings"] = h["settings"]
    if subst["EOT_TOKEN_ID"] == "null":
        del polar["polar_task_template"]["builder"]["config"]["end_of_turn_token_id"]
    # The bridge verifies (at worker start and eval) that every gateway in this
    # topology routes rollouts to the trainer's live adapter.
    polar["polar_topology_path"] = f"{run_dir}/topology.yaml"

    topo = render("topology.yaml")
    proto = topo["gateway"]["nodes"][0]
    if h["thinking"] is None:
        del proto["inference"]["enable_thinking"]
    else:
        proto["inference"]["enable_thinking"] = bool(h["thinking"])
    proto["inference"]["extra_body"] = {"lora_path": LORA_ADAPTER_NAME} if lora_enabled(cfg) else {}
    if c["router_policy"] in SESSION_AFFINE:
        proto["inference"]["routing_key_header"] = ROUTING_KEY_HEADER
    hosts = [public_host] + (f["WORKER_IPS"] if c["sandbox_nodes"] == "all" else [])
    topo["gateway"]["nodes"] = [
        {**copy.deepcopy(proto), "id": f"node-{i:02d}", "public_url": f"http://{ip}:{f['GATEWAY_PORT']}"} for i, ip in enumerate(hosts, 1)
    ]
    for name, doc in (("polar_config.yaml", polar), ("topology.yaml", topo)):
        with open(os.path.join(run_dir, name), "w") as fh:
            fh.write(f"# rendered by harbor_miles_grpo/internal/render.py from harbor_slime_grpo/internal/{name}\n")
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, width=200)
    os.chmod(os.path.join(run_dir, "polar_config.yaml"), 0o600)  # holds the judge key

    args = train_args(cfg, d, f)
    actor_nodes, actor_gpus_per_node, rollout_gpus = gpu_layout(cfg, f)
    with open(os.path.join(run_dir, "train_args.sh"), "w") as fh:
        fh.write("# rendered by harbor_miles_grpo/internal/render.py; sourced by run.sh\n")
        fh.write(f"TRAIN_SCRIPT={d['TRAIN_SCRIPT']}\n")
        fh.write("TRAIN_ARGS=(\n" + "".join(f"    {shlex.quote(str(a))}\n" for a in args) + ")\n")
        fh.write("SANDBOX_IPS=(" + " ".join(shlex.quote(ip) for ip in hosts) + ")\n")
    print(f"rendered {run_dir}/{{polar_config.yaml,topology.yaml,train_args.sh}}: "
          f"train {actor_nodes}x{actor_gpus_per_node} GPUs, {rollout_gpus} engine GPUs "
          f"({rollout_gpus // c['gpus_per_engine']} engines x TP{c['gpus_per_engine']}), {len(hosts)} sandbox host(s)",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("env", "run"):
        sys.exit(__doc__)
    config = load(sys.argv[2])
    (mode_env if sys.argv[1] == "env" else mode_run)(config)
