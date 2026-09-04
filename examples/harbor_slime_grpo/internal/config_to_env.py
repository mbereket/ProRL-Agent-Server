"""Turn a run config (configs/*.yaml) into `export KEY=VALUE` lines for the shell.

    eval "$(python internal/config_to_env.py configs/<run>.yaml)"

Every key has a default; `name` and `tasks.dir` are required. Unknown keys are
an error so typos cannot silently fall back to defaults. Paths (`tasks.dir`,
`harness.dir`, ...) expand ${ENV} and ~, and relative ones resolve against the
config file's directory. `harness.settings` is passed through as JSON
(HARNESS_SETTINGS_JSON) for the Polar template render.
"""
from __future__ import annotations

import json
import os
import shlex
import sys

import yaml

PATH = object()  # marker: value is a filesystem path

# (yaml key, env var, default). Booleans become 1/0; PATH-typed keys resolve as paths.
SCHEMA = {
    "tasks": [
        ("dir", "TASKS_DIR", PATH),
        ("dataset", "TASKS_DATASET", ""),          # datasets/<name>.py materializes dir if missing
        ("dataset_args", "TASKS_DATASET_ARGS", ""),
        ("mount_root", "TASKS_MOUNT_ROOT", PATH),
        ("n", "TASKS_N", ""),
        ("seed", "TASKS_SEED", 0),
        ("task_ids_file", "TASK_IDS_FILE", PATH),
        ("exclude_ids_file", "EXCLUDE_IDS_FILE", PATH),
    ],
    "harness": [
        ("name", "HARNESS", "mini_swe_agent"),
        ("model_name", "HARNESS_MODEL_NAME", "openai/gpt-5.4"),
        ("dir", "HARNESS_DIR_CFG", PATH),
        ("settings", "HARNESS_SETTINGS_JSON", {}),
        ("session_timeout", "SESSION_TIMEOUT", 3000),
        ("request_timeout", "REQUEST_TIMEOUT", 3600),
        ("max_run_workers", "MAX_RUN_WORKERS", 16),
        ("max_async_level", "MAX_ASYNC_LEVEL", 1),
        ("thinking", "ENABLE_THINKING", None),
        ("keep_sessions", "POLAR_KEEP_SESSION_DIRS", False),   # keep per-session dirs (agent logs, verifier output) for debugging      # Qwen3.5: true/false force chat-template thinking; null = built-in rule (off)
        ("path_prepend", "HARNESS_PATH_PREPEND", ""),       # put first on the agent PATH for every task (after per-task agent_path_prepend)
        ("ld_library_path", "HARNESS_LD_LIBRARY_PATH", ""), # LD_LIBRARY_PATH inside the sandbox (e.g. the image's env libs)
        ("cli_version", "HARNESS_CLI_VERSION", ""),         # pin the harness CLI version installed by prepare_harness.sh
    ],
    "model": [
        ("hf_checkpoint", "HF_CHECKPOINT", "Qwen/Qwen3.5-9B"),
        ("model_args_file", "MODEL_ARGS_FILE", "model_args_9b.sh"),
        ("end_of_turn_token_id", "EOT_TOKEN_ID", 248046),
        ("torch_dist_dir", "TORCH_DIST_DIR_CFG", PATH),
        ("load_dir", "MODEL_LOAD_DIR", PATH),               # explicit checkpoint dir to load (e.g. another run's save dir); default: own save dir, else the reference
    ],
    "cluster": [
        ("num_nodes", "NUM_NODES", 1),
        ("actor_num_gpus", "ACTOR_NUM_GPUS", 4),
        ("tp_size", "TP_SIZE", 4),
        ("context_parallel_size", "CONTEXT_PARALLEL_SIZE", 1),
        ("sandbox_nodes", "SANDBOX_NODES", "head"),   # head | all: hosts that run agent sandboxes
    ],
    "rollout": [
        ("batch_size", "ROLLOUT_BATCH_SIZE", 8),
        ("n_samples_per_prompt", "N_SAMPLES_PER_PROMPT", 16),
        ("num_epoch", "NUM_EPOCH", 50),
        ("num_rollout", "NUM_ROLLOUT", ""),                 # overrides the epoch-derived step count; 0 = eval only (needs eval.prompt_data)
        ("max_prompt_len", "ROLLOUT_MAX_PROMPT_LEN", 24000),
        ("max_response_len", "ROLLOUT_MAX_RESPONSE_LEN", 8000),
        ("sglang_context_length", "SGLANG_CONTEXT_LENGTH", 32768),
    ],
    "training": [
        ("sync", "TRAIN_SYNC", True),
        ("max_tokens_per_gpu", "MAX_TOKENS_PER_GPU", 16384),
        ("lr", "LR", "1e-6"),
        ("use_kl_loss", "USE_KL_LOSS", False),
        ("kl_loss_coef", "KL_LOSS_COEF", 0.001),
        ("grpo_std_normalization", "GRPO_STD_NORMALIZATION", False),
        ("optimizer_cpu_offload", "OPTIMIZER_CPU_OFFLOAD", False),   # Megatron hybrid optimizer: Adam states + fp32 master params on host
        ("group_id_scope", "GROUP_ID_SCOPE", "trajectory"),
        ("timeout_reward_zero", "TIMEOUT_REWARD_ZERO", True),
        ("drop_zero_variance_groups", "DROP_ZERO_VARIANCE_GROUPS", True),
        ("save_interval", "SAVE_INTERVAL", 5),
        ("extra_train_args", "EXTRA_TRAIN_ARGS", ""),
    ],
    "eval": [
        ("prompt_data", "EVAL_PROMPT_DATA", ""),
        ("interval", "EVAL_INTERVAL", 10),
        ("n_samples_per_prompt", "N_SAMPLES_PER_EVAL_PROMPT", 1),
    ],
    "judge": [                                             # LLM judge for rubric-graded tasks (BixBench-Hypothesis)
        ("model", "RUBRIC_MODEL", ""),
        ("api_base", "RUBRIC_MODEL_API_BASE", ""),
        ("api_key_env", "RUBRIC_MODEL_API_KEY_ENV", ""),     # host env var holding the key; read when templates are rendered
    ],
    "wandb": [
        ("project", "WANDB_PROJECT", "harbor-slime-grpo"),
        ("group", "WANDB_GROUP", ""),
    ],
}
REQUIRED = {("tasks", "dir")}


class ConfigError(Exception):
    pass


def resolve_path(v, cfg_dir: str) -> str:
    if not v:
        return ""
    v = os.path.expanduser(os.path.expandvars(str(v)))
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(cfg_dir, v))


def resolve(path: str) -> dict[str, str]:
    """Read a run config and return the environment variables it defines (all strings)."""
    cfg_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "name" not in cfg:
        raise ConfigError(f"{path}: 'name' is required")
    unknown = set(cfg) - set(SCHEMA) - {"name"}
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys {sorted(unknown)}; known: {sorted(SCHEMA)}")
    out = {"RUN_NAME": str(cfg["name"])}
    for section, fields in SCHEMA.items():
        block = cfg.get(section) or {}
        known = {k for k, _, _ in fields}
        if set(block) - known:
            raise ConfigError(
                f"{path}: unknown keys in '{section}': {sorted(set(block) - known)}; known: {sorted(known)}"
            )
        for key, env, default in fields:
            if (section, key) in REQUIRED and key not in block:
                raise ConfigError(f"{path}: {section}.{key} is required")
            if default is PATH:
                out[env] = resolve_path(block.get(key), cfg_dir)
                continue
            v = block.get(key, default)
            if isinstance(v, dict):
                v = json.dumps(v)
            elif isinstance(v, bool):
                v = int(v)
            out[env] = "" if v is None else str(v)
    if out["SANDBOX_NODES"] not in ("head", "all"):
        raise ConfigError(f"{path}: cluster.sandbox_nodes must be head or all")
    out["TRAIN_SCRIPT"] = "train.py" if out["TRAIN_SYNC"] == "1" else "train_async.py"
    if not out["WANDB_GROUP"]:
        out["WANDB_GROUP"] = out["RUN_NAME"]
    return out


def main(path: str) -> None:
    try:
        env = resolve(path)
    except ConfigError as e:
        sys.exit(str(e))
    for k, v in env.items():
        print(f"export {k}={shlex.quote(v)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: config_to_env.py <run-config.yaml>")
    main(sys.argv[1])
