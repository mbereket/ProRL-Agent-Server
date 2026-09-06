#!/usr/bin/env python3
"""Run config -> everything the shell scripts need. The only place defaults live.

    render.py env <run.yaml>   print `export KEY=VALUE` lines (launch.sh evals them)
    render.py run <run.yaml>   write <RUN_DIR>/{polar_config.yaml,topology.yaml,train_args.sh}

`env` needs no environment beyond WORKROOT (and optional RUN_ID). `run` is called
by launch.sh after setup, when the head/worker addresses, the tokenizer and the
checkpoint state are known; it reads them from the environment variables listed
in runtime_facts(). Unknown config keys are an error.

Config schema (defaults in SCHEMA below):

  name: run-id                       # run dir and save dir are named after it (RUN_ID env overrides)
  tasks:    dir (required), mount_root, n, seed, task_ids_file, exclude_ids_file
  harness:  name, model_name, dir, settings, session_timeout, request_timeout, max_run_workers,
            max_async_level, thinking, keep_sessions, path_prepend, ld_library_path, cli_version
  model:    hf_checkpoint, model_args_file, torch_dist_dir, load_dir, sglang_tool_call_parser
  cluster:  num_nodes, actor_num_gpus, tp_size, context_parallel_size, sandbox_nodes (head|all)
  rollout:  batch_size, n_samples_per_prompt, num_steps | num_epoch, max_prompt_len,
            max_response_len, sglang_context_length
  training: sync, max_tokens_per_gpu, lr, use_kl_loss, kl_loss_coef, grpo_std_normalization,
            loss_denominator, optimizer_cpu_offload, group_id_scope, timeout_reward_zero,
            overlong_policy, drop_zero_variance_groups, save_interval, checkpoint_keep_every,
            extra_train_args
  eval:     prompt_data ("<name> <path>", ${RUN_DIR} allowed), interval, n_samples_per_prompt
  judge:    model, api_base, api_key_env  (LLM judge for rubric-graded tasks)
  wandb:    project, group
"""
from __future__ import annotations

import copy
import json
import os
import shlex
import string
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = object()  # marker: a filesystem path (expands ${ENV} and ~, relative to the config file)

SCHEMA = {
    "tasks": {
        "dir": PATH,
        "mount_root": PATH,          # host dir mounted as /harbor_data (default: dir); for tasks that reference shared data outside dir
        "n": None,                   # random subset of tasks (None = all)
        "seed": 0,
        "task_ids_file": PATH,       # one dir name / source_id per line
        "exclude_ids_file": PATH,
    },
    "harness": {
        "name": "codex",             # codex | opencode | mini_swe_agent
        "model_name": "openai/gpt-5.4",
        "dir": PATH,                 # shared harness dir (default: ${WORKROOT}/harbor_harness)
        "settings": {},              # passed to the Polar preset as-is
        "session_timeout": 3000,     # agent + verifier + margin (s)
        "request_timeout": 3600,     # per LLM request at the gateway (s)
        "max_run_workers": 16,       # concurrent sandboxes per sandbox node
        "max_async_level": 1,        # rollout steps the sampler may run ahead (>1 needs training.sync false)
        "thinking": None,            # Qwen3: force chat-template thinking on/off; None = template default (off)
        "keep_sessions": False,      # keep per-session dirs (agent logs, verifier output)
        "path_prepend": "",          # first on the agent PATH in every sandbox, after per-task agent_path_prepend
        "ld_library_path": "",       # LD_LIBRARY_PATH inside the sandbox
        "cli_version": "",           # pin the harness CLI version (default: prepare_harness.sh's pin)
    },
    "model": {
        "hf_checkpoint": "Qwen/Qwen3.5-9B",
        "model_args_file": "qwen3_5_9b.sh",  # file in internal/model_args/ or an absolute path
        "torch_dist_dir": PATH,      # converted reference checkpoint (default: ${WORKROOT}/checkpoints/<name>_torch_dist)
        "load_dir": PATH,            # start from this checkpoint dir instead of own save dir / reference
        "sglang_tool_call_parser": "qwen3_coder",  # must match the model's tool-call format (qwen3_coder: Qwen3.5, qwen: Qwen3 dense)
    },
    "cluster": {
        "num_nodes": 1,
        "actor_num_gpus": 4,         # trainer GPUs; whole nodes when num_nodes > 1. Every other GPU serves an engine
        "tp_size": 4,
        "context_parallel_size": 1,  # trace cap = max_tokens_per_gpu x CP
        "sandbox_nodes": "all",      # head | all: nodes whose CPUs run sandboxes (one Polar gateway each)
    },
    "rollout": {
        "batch_size": 8,             # tasks per step
        "n_samples_per_prompt": 16,  # attempts per task
        "num_steps": None,           # training steps; 0 = eval only (needs eval.prompt_data)
        "num_epoch": None,           # alternative: passes over the task set
        "max_prompt_len": 8000,      # initial task prompts longer than this are skipped
        "max_response_len": 24000,   # feeds slime's response-length metrics only
        "sglang_context_length": 32768,
    },
    "training": {
        "sync": True,                # true: train.py; false: train_async.py (generation overlaps training)
        "max_tokens_per_gpu": 16384, # 16384 fits H100-80GB for 9B TP4
        "lr": "1e-6",
        "use_kl_loss": False,
        "kl_loss_coef": 0.001,
        "grpo_std_normalization": False,  # false = mean-only advantages
        "loss_denominator": "trainable_units",  # trainable_units | global_batch
        "optimizer_cpu_offload": False,   # Adam states on host (needed for 32768 tok/GPU)
        "group_id_scope": "trajectory",   # trajectory | prompt
        "timeout_reward_zero": True,
        "overlong_policy": "zero_reward_train",  # zero_reward_train | drop
        "drop_zero_variance_groups": True,
        "save_interval": 5,
        "checkpoint_keep_every": 0,  # >0: delete saved iterations that are not multiples of this (latest kept)
        "extra_train_args": "",
    },
    "eval": {
        "prompt_data": "",           # "<name> <jsonl>"; ${RUN_DIR} expands to the run dir
        "interval": 10,
        "n_samples_per_prompt": 1,
    },
    "judge": {
        "model": "",
        "api_base": "",
        "api_key_env": "",           # host env var holding the key
    },
    "wandb": {
        "project": "harbor-slime-grpo",
        "group": "",                 # default: run name
    },
}
HARNESSES = ("codex", "opencode", "mini_swe_agent")


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
            v = block.get(key, None if default is PATH else default)
            if isinstance(v, str):
                v = os.path.expandvars(v)  # ${WORKROOT}, ${HOME}, ... from the machine environment
            if default is PATH and v:
                v = os.path.expanduser(str(v))
                v = v if os.path.isabs(v) else os.path.normpath(os.path.join(cfg_dir, v))
            out[key] = v
        cfg[section] = out
    if not cfg["tasks"]["dir"]:
        die(f"{path}: tasks.dir is required")
    if (cfg["rollout"]["num_steps"] is None) == (cfg["rollout"]["num_epoch"] is None):
        die(f"{path}: set exactly one of rollout.num_steps and rollout.num_epoch")
    if cfg["cluster"]["sandbox_nodes"] not in ("head", "all"):
        die(f"{path}: cluster.sandbox_nodes must be head or all")
    if cfg["harness"]["name"] not in HARNESSES:
        die(f"{path}: harness.name must be one of {HARNESSES}")
    if cfg["rollout"]["num_steps"] == 0 and not cfg["eval"]["prompt_data"]:
        die(f"{path}: rollout.num_steps: 0 (eval only) needs eval.prompt_data")
    if cfg["harness"]["max_async_level"] > 1 and cfg["training"]["sync"]:
        die(f"{path}: harness.max_async_level > 1 needs training.sync: false")
    return cfg


def derived(cfg: dict) -> dict:
    """Values shared by both modes: run placement and file locations."""
    workroot = os.environ.get("WORKROOT") or die("WORKROOT is not set")
    run_id = os.environ.get("RUN_ID") or cfg["name"]
    m = cfg["model"]
    args_file = m["model_args_file"]
    if not os.path.isabs(args_file):
        args_file = os.path.join(HERE, "model_args", args_file)
    return {
        "RUN_NAME": cfg["name"],
        "RUN_ID": run_id,
        "RUN_DIR": f"{workroot}/harbor_slime_grpo/{run_id}",
        "SAVE_DIR": f"{workroot}/ckpt/harbor_slime_grpo/{run_id}",
        "HARNESS_DIR": cfg["harness"]["dir"] or f"{workroot}/harbor_harness",
        "APPTAINER_IMAGE_DIR": os.environ.get("APPTAINER_IMAGE_DIR") or f"{workroot}/harbor_sif_images",
        "TORCH_DIST_DIR": m["torch_dist_dir"] or f"{workroot}/checkpoints/{m['hf_checkpoint'].rstrip('/').rsplit('/', 1)[-1]}_torch_dist",
        "MODEL_ARGS_FILE": args_file,
        "TRAIN_SCRIPT": "train.py" if cfg["training"]["sync"] else "train_async.py",
    }


def mode_env(cfg: dict) -> None:
    """Export what launch.sh, head_entry.sh and run.sh need before the run is rendered."""
    d = derived(cfg)
    t, h, m, c, tr = cfg["tasks"], cfg["harness"], cfg["model"], cfg["cluster"], cfg["training"]
    subset = f" (n={t['n']}, seed {t['seed']})" if t["n"] is not None else ""
    steps = f"steps {cfg['rollout']['num_steps']}" if cfg["rollout"]["num_steps"] is not None else f"epochs {cfg['rollout']['num_epoch']}"
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
        "CHECKPOINT_KEEP_EVERY": tr["checkpoint_keep_every"],
        "POLAR_KEEP_SESSION_DIRS": "1" if h["keep_sessions"] else "",
        "JUDGE_API_KEY_ENV": cfg["judge"]["api_key_env"],
        "SUMMARY": (
            f"{cfg['name']} (RUN_ID {d['RUN_ID']}): {h['name']} on {t['dir']}{subset}; {m['hf_checkpoint']}; "
            f"{c['num_nodes']} node(s), trainer {c['actor_num_gpus']} GPUs TP{c['tp_size']} x CP{c['context_parallel_size']}, "
            f"sandboxes on {c['sandbox_nodes']}; {cfg['rollout']['batch_size']} x {cfg['rollout']['n_samples_per_prompt']} "
            f"per step, {steps}; trace cap {tr['max_tokens_per_gpu'] * c['context_parallel_size']} tok; {d['TRAIN_SCRIPT']}"
        ),
    }
    for k, v in env.items():
        print(f"export {k}={shlex.quote(str(v))}")


def runtime_facts() -> dict:
    """Environment run.sh/head_entry.sh established: addresses, ports, tool locations."""
    e = os.environ
    facts = {
        "HEAD_IP": e.get("RAY_HEAD_IP", "127.0.0.1"),
        "BIND_HOST": e.get("POLAR_BIND_HOST", "127.0.0.1"),
        "WORKER_IPS": [ip for ip in e.get("WORKER_IPS", "").split(",") if ip],
        "ROLLOUT_PORT": int(e.get("POLAR_ROLLOUT_PORT", 8080)),
        "GATEWAY_PORT": int(e.get("POLAR_GATEWAY_PORT", 8100)),
        "ROUTER_PORT": int(e.get("SGLANG_ROUTER_PORT", 9000)),
        "GPUS_PER_NODE": int(e.get("GPUS_PER_NODE", 8)),
        "DRY_RUN": e.get("DRY_RUN") == "1",
    }
    facts["CALLBACK_HOST"] = e.get("POLAR_CALLBACK_HOST", "127.0.0.1")
    return facts


def end_of_turn_token_id(hf_checkpoint: str) -> int:
    """Id of the token that closes an assistant turn (<|im_end|> on ChatML models).
    Polar's prefix-merging builder splits each completion's prompt at this token,
    so it must match the served model. Render a two-turn chat and take the last
    special token."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    special = set(tok.all_special_ids)
    eot = next((t for t in reversed(ids) if t in special), tok.eos_token_id)
    print(f"end-of-turn token: {tok.convert_ids_to_tokens(eot)!r} = {eot}", file=sys.stderr)
    return int(eot)


def model_args(args_file: str) -> list[str]:
    """MODEL_ARGS=(...) from a model_args/*.sh file (shared with convert_weights.sh)."""
    out = subprocess.run(["bash", "-c", f'source {shlex.quote(args_file)} && printf "%s\\n" "${{MODEL_ARGS[@]}}"'],
                         check=True, capture_output=True, text=True)
    return out.stdout.splitlines()


def mode_run(cfg: dict) -> None:
    d, f = derived(cfg), runtime_facts()
    t, h, m, c, r, tr, ev, j = (cfg[k] for k in ("tasks", "harness", "model", "cluster", "rollout", "training", "eval", "judge"))
    run_dir = d["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)
    public_host = f["HEAD_IP"]
    judge_key = ""
    if j["api_key_env"]:
        judge_key = os.environ.get(j["api_key_env"], "")
        if not judge_key and not f["DRY_RUN"]:
            die(f"judge.api_key_env={j['api_key_env']} is not set in the environment")

    # ── Polar bridge config + topology from the templates next to this file ──
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
    except Exception as exc:  # dry runs may not have the tokenizer yet; Polar then auto-detects
        if not f["DRY_RUN"]:
            raise
        print(f"end-of-turn token: not derived ({exc}); left unset for the dry run", file=sys.stderr)
        subst["EOT_TOKEN_ID"] = "null"

    def render(name: str) -> dict:
        with open(os.path.join(HERE, name)) as fh:
            return yaml.safe_load(string.Template(fh.read()).substitute(subst))

    polar = render("polar_config.yaml")
    polar["polar_task_template"]["agent"]["settings"] = h["settings"]
    if subst["EOT_TOKEN_ID"] == "null":
        del polar["polar_task_template"]["builder"]["config"]["end_of_turn_token_id"]

    topo = render("topology.yaml")
    proto = topo["gateway"]["nodes"][0]
    if h["thinking"] is None:
        del proto["inference"]["enable_thinking"]
    else:
        proto["inference"]["enable_thinking"] = bool(h["thinking"])
    # One gateway node per sandbox host: node-01 = head, node-02.. = workers.
    hosts = [public_host] + (f["WORKER_IPS"] if c["sandbox_nodes"] == "all" else [])
    topo["gateway"]["nodes"] = [
        {**copy.deepcopy(proto), "id": f"node-{i:02d}", "public_url": f"http://{ip}:{f['GATEWAY_PORT']}"} for i, ip in enumerate(hosts, 1)
    ]
    for name, doc in (("polar_config.yaml", polar), ("topology.yaml", topo)):
        with open(os.path.join(run_dir, name), "w") as fh:
            fh.write(f"# rendered by render.py from internal/{name}\n")
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, width=200)
    os.chmod(os.path.join(run_dir, "polar_config.yaml"), 0o600)  # holds the judge key

    # ── GPU layout ─────────────────────────────────────────────────────────
    # Slime places engines on whole nodes in rank order: on more than one node the
    # trainer must take whole nodes; every remaining GPU serves an SGLang engine.
    gpn, actor, nodes = f["GPUS_PER_NODE"], c["actor_num_gpus"], c["num_nodes"]
    if nodes > 1 and actor % gpn:
        die(f"cluster.actor_num_gpus={actor} must be a multiple of the {gpn} GPUs per node when num_nodes > 1")
    actor_nodes, actor_gpus_per_node = (actor // gpn, gpn) if actor >= gpn else (1, actor)
    rollout_gpus = nodes * gpn - actor_nodes * actor_gpus_per_node
    if rollout_gpus < 1:
        die(f"no GPUs left for rollout ({nodes} x {gpn} total, {actor} train)")
    if actor % (c["tp_size"] * c["context_parallel_size"]):
        die(f"tp_size x context_parallel_size must divide actor_num_gpus ({actor})")

    # ── Checkpoint to load ─────────────────────────────────────────────────
    latest = "latest_checkpointed_iteration.txt"
    ref = d["TORCH_DIST_DIR"]
    if not f["DRY_RUN"] and not os.path.isfile(os.path.join(ref, latest)):
        die(f"converted reference checkpoint not found at {ref}")
    if m["load_dir"]:
        os.path.isfile(os.path.join(m["load_dir"], latest)) or die(f"model.load_dir has no checkpoint: {m['load_dir']}")
        load, start = m["load_dir"], []
    elif os.path.isfile(os.path.join(d["SAVE_DIR"], latest)):
        load, start = d["SAVE_DIR"], []  # resume: slime derives start_rollout_id from the checkpoint
    else:
        # Fresh run. The converted "release" checkpoint loads as iteration 0, which slime
        # would turn into start_rollout_id=1 and silently skip one rollout.
        load, start = ref, ["--start-rollout-id", "0"]

    # ── Slime arguments ────────────────────────────────────────────────────
    steps: list[str]
    if r["num_steps"] is not None:
        steps = ["--num-rollout", str(r["num_steps"])]
        if r["num_steps"] == 0:
            # Eval only: no optimizer step runs, but slime sizes the LR schedule from the
            # step count (asserts > 0) and the optimizer/RNG state may come from another
            # GPU layout.
            steps += ["--lr-decay-iters", "1", "--no-load-optim", "--no-load-rng"]
    else:
        steps = ["--num-epoch", str(r["num_epoch"])]
    eval_args: list[str] = []
    if ev["prompt_data"]:
        eval_args = ["--eval-prompt-data", *ev["prompt_data"].replace("${RUN_DIR}", run_dir).split(),
                     "--eval-interval", str(ev["interval"]), "--n-samples-per-eval-prompt", str(ev["n_samples_per_prompt"])]
    wandb_mode = os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline")
    args = [
        "--actor-num-nodes", actor_nodes, "--actor-num-gpus-per-node", actor_gpus_per_node,
        "--rollout-num-gpus", rollout_gpus, "--rollout-num-gpus-per-engine", 1,
        *model_args(d["MODEL_ARGS_FILE"]),
        "--hf-checkpoint", m["hf_checkpoint"], "--ref-load", ref, "--load", load, *start,
        "--save", d["SAVE_DIR"], "--save-interval", tr["save_interval"], "--update-weights-interval", 1,
        # Polar bridge: rollouts, rewards and data source come from slime_bridge.
        "--rollout-function-path", "slime_bridge.rollout.generate_rollout_polar_async",
        "--custom-rm-path", "slime_bridge.reward.reward_func",
        "--custom-reward-post-process-path", "slime_bridge.reward_post_process.post_process_rewards",
        "--custom-config-path", f"{run_dir}/polar_config.yaml",
        "--data-source-path", "slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer",
        "--prompt-data", f"{run_dir}/train.jsonl", "--input-key", "prompt", "--label-key", "label",
        "--metadata-key", "metadata", "--rollout-shuffle", "--reward-key", "score",
        *steps,
        "--rollout-batch-size", r["batch_size"], "--n-samples-per-prompt", r["n_samples_per_prompt"],
        "--rollout-max-response-len", r["max_response_len"], "--rollout-max-prompt-len", r["max_prompt_len"],
        "--dynamic-history", "--num-steps-per-rollout", 1,
        # Parallelism / memory
        "--tensor-model-parallel-size", c["tp_size"], "--sequence-parallel", "--pipeline-model-parallel-size", 1,
        "--context-parallel-size", c["context_parallel_size"], "--expert-model-parallel-size", 1,
        "--expert-tensor-parallel-size", 1,
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", 1,
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", tr["max_tokens_per_gpu"],
        "--log-probs-chunk-size", 256, "--distributed-timeout-minutes", 30,
        # Algorithm: GRPO + TIS, clip-higher; KL and std-normalization are config knobs.
        "--advantage-estimator", "grpo", "--normalize-advantages", "--use-tis",
        *(["--use-kl-loss", "--kl-loss-coef", str(tr["kl_loss_coef"]), "--kl-loss-type", "low_var_kl"] if tr["use_kl_loss"] else []),
        *([] if tr["grpo_std_normalization"] else ["--disable-grpo-std-normalization"]),
        "--loss-denominator", tr["loss_denominator"],
        # Megatron asserts the hybrid device optimizer runs on the precision-aware path.
        *(["--optimizer-cpu-offload", "--optimizer-offload-fraction", "1.0", "--overlap-cpu-optimizer-d2h-h2d",
           "--use-precision-aware-optimizer"] if tr["optimizer_cpu_offload"] else []),
        *eval_args,
        *shlex.split(tr["extra_train_args"]),
        "--entropy-coef", "0.0", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", tr["lr"], "--lr-decay-style", "constant", "--weight-decay", "0.1",
        "--adam-beta1", "0.9", "--adam-beta2", "0.98", "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32", "--attention-backend", "auto",
        "--no-gradient-accumulation-fusion",
        # SGLang engines
        "--sglang-mem-fraction-static", "0.8", "--sglang-context-length", r["sglang_context_length"],
        "--sglang-tool-call-parser", m["sglang_tool_call_parser"], "--router-policy", "round_robin",
        "--sglang-router-port", f["ROUTER_PORT"],
        "--use-wandb", "--wandb-mode", wandb_mode, "--wandb-project", cfg["wandb"]["project"],
        "--wandb-group", cfg["wandb"]["group"] or cfg["name"],
    ]
    with open(os.path.join(run_dir, "train_args.sh"), "w") as fh:
        fh.write("# rendered by render.py; sourced by run.sh\n")
        fh.write(f"TRAIN_SCRIPT={d['TRAIN_SCRIPT']}\n")
        fh.write("TRAIN_ARGS=(\n" + "".join(f"    {shlex.quote(str(a))}\n" for a in args) + ")\n")
        fh.write("SANDBOX_IPS=(" + " ".join(shlex.quote(ip) for ip in hosts) + ")\n")
    print(f"rendered {run_dir}/{{polar_config.yaml,topology.yaml,train_args.sh}}: "
          f"train {actor_nodes}x{actor_gpus_per_node} GPUs, {rollout_gpus} engine GPUs, "
          f"{len(hosts)} sandbox host(s), load {load}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("env", "run"):
        sys.exit(__doc__)
    config = load(sys.argv[2])
    (mode_env if sys.argv[1] == "env" else mode_run)(config)
