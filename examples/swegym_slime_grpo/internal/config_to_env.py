"""Turn a run config (configs/*.yaml) into `export KEY=VALUE` lines for the shell.

    eval "$(python internal/config_to_env.py configs/qwen35-4b-1node.yaml)"

Every key has a default (the single-node recipe); only `name` is required.
Unknown keys are an error so typos cannot silently fall back to defaults.
The environment variables produced are the ones run.sh / pipeline.sh read.
"""
from __future__ import annotations

import shlex
import sys

import yaml

# (yaml key, env var, default). Booleans become 1/0.
SCHEMA = {
    "model": [
        ("hf_checkpoint", "HF_CHECKPOINT", "Qwen/Qwen3.5-4B"),
        ("model_args_file", "MODEL_ARGS_FILE", "model_args.sh"),
    ],
    "cluster": [
        ("num_nodes", "NUM_NODES", 1),
        ("actor_num_gpus", "ACTOR_NUM_GPUS", 4),
        ("tp_size", "TP_SIZE", 2),
        ("context_parallel_size", "CONTEXT_PARALLEL_SIZE", 1),
    ],
    "rollout": [
        ("batch_size", "ROLLOUT_BATCH_SIZE", 4),
        ("n_samples_per_prompt", "N_SAMPLES_PER_PROMPT", 16),
        ("num_epoch", "NUM_EPOCH", 1),
        ("max_prompt_len", "ROLLOUT_MAX_PROMPT_LEN", 32000),
        ("max_response_len", "ROLLOUT_MAX_RESPONSE_LEN", 16000),
        ("sglang_context_length", "SGLANG_CONTEXT_LENGTH", 50000),
    ],
    "training": [
        ("max_tokens_per_gpu", "MAX_TOKENS_PER_GPU", 30000),
        ("lr", "LR", "1e-6"),
        ("use_kl_loss", "USE_KL_LOSS", True),
        ("kl_loss_coef", "KL_LOSS_COEF", 0.001),
        ("grpo_std_normalization", "GRPO_STD_NORMALIZATION", False),
        ("save_interval", "SAVE_INTERVAL", 10),
        ("extra_train_args", "EXTRA_TRAIN_ARGS", ""),
    ],
    "eval": [
        ("prompt_data", "EVAL_PROMPT_DATA", ""),
        ("interval", "EVAL_INTERVAL", 10),
        ("n_samples_per_prompt", "N_SAMPLES_PER_EVAL_PROMPT", 1),
    ],
    "wandb": [
        ("project", "WANDB_PROJECT", "polar-swegym-grpo"),
        ("group", "WANDB_GROUP", ""),
    ],
}


class ConfigError(Exception):
    pass


def resolve(path: str) -> dict[str, str]:
    """Read a run config and return the environment variables it defines (all strings)."""
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
            v = block.get(key, default)
            if isinstance(v, bool):
                v = int(v)
            out[env] = "" if v is None else str(v)
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
