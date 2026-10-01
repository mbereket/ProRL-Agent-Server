"""Miles agent function: one Harbor trial per rollout, on the least-loaded node.

Wired as ``--custom-agent-function-path hm_agent.run`` under
``miles.rollout.generate_hub.agentic_tool_call.generate`` (TITO session
server). The agent inside the sandbox talks to the session URL; Miles records
the exact tokens/logprobs there, so this function only has to run the trial
and report its outcome.

Per-sample metadata (from the prompt JSONL, see tools/prepare_data.py):
  instance_id        Harbor task dir name under the servers' HARBOR_TASKS_DIR (required)
  agent_name         Harbor agent whose connection config to use (default: opencode)
  agent_import_path  optional custom agent class, e.g. harbor_miles_agents.opencode_agents:BbhOpenCode
  agent_kwargs       optional kwargs merged into the agent's (e.g. opencode_config)

Environment (rollout worker):
  HARBOR_AGENT_SERVERS / HARBOR_AGENT_SERVERS_FILE   see hm_dispatch.py
  AGENT_MODEL_NAME       served model id the agent requests (default: model)
  AGENT_TRIAL_TIMEOUT    client-side ceiling per trial, s (default 7200; keep above the
                         servers' --agent-timeout so the server ends trials first)
  HARBOR_INFRA_RETRIES   re-dispatches of a trial that failed before the agent ran (default 2)
  HM_TRIAL_LOG           optional JSONL path: one line per finished trial (reward, exit status,
                         timings, server, wall clock). Works the same under train.py and
                         train_async.py --fully-async (where the sync RolloutFn metrics do not run).

Failure attribution. A trial whose agent never started (sandbox start, agent
setup, unreachable server) made no model calls, so its session is still empty
and the trial is re-dispatched (to the least-loaded server, usually another
node). Failures after the agent started are reported with ``infra_failure``
set when the cause is the platform rather than the policy; the reward
post-process (rollout.py) then removes them from the loss instead of training
them as reward 0.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlsplit

from miles.rollout.agentic.session import openai_session_url

from hm_dispatch import POOL

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 7200.0

# Harbor exception types that end a trial through the platform, not the policy.
INFRA_EXCEPTIONS = {
    "EnvironmentStartTimeoutError",
    "RemoteProtocolError",
    "ConnectError",
    "ReadError",
    "NoResult",
}
_OVERFLOW_MARKERS = ("remote compact task", "/responses/compact", "context_length_exceeded",
                     "maximum context length", "ContextOverflowError", "context window")

# Exit statuses produced by the agent server itself (never a policy outcome).
INFRA_STATUSES = {"DispatchError", "Error: NoResult", "Flushed", "ImportError", "TaskNotFound", "InvalidInstanceId"}


def _timeout_s() -> float:
    return float(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_TIMEOUT_S))


def _is_infra(resp: dict[str, Any] | None) -> tuple[bool, bool]:
    """(infra_failure, agent_started) for an agent-server response."""
    if resp is None:
        return True, False
    status = resp.get("exit_status", "") or ""
    metrics = resp.get("agent_metrics") or {}
    started = bool(metrics.get("agent_started", status == "Submitted"))
    if status in INFRA_STATUSES or status.startswith("Error:"):
        return True, started
    if status == "Submitted":
        return False, True
    if not started:
        return True, False
    return metrics.get("exception_type", "") in INFRA_EXCEPTIONS, True


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    metadata = metadata or {}
    session_url = openai_session_url(base_url)
    request: dict[str, Any] = {
        "base_url": session_url,
        "model": f"openai/{os.environ.get('AGENT_MODEL_NAME', 'model')}",
        "sampling_params": dict(request_kwargs or {}),
        "instance_id": metadata["instance_id"],
        "agent_name": metadata.get("agent_name", "opencode"),
    }
    if (max_seq_len := metadata.get("max_seq_len")) is not None:
        request["max_seq_len"] = int(max_seq_len)
    if metadata.get("session_server_id") is not None:
        # the agent server heartbeats the address base_url names
        request["session_server_id"] = urlsplit(session_url).netloc
    if (instance_id := metadata.get("session_server_instance_id")) is not None:
        request["session_server_instance_id"] = instance_id
    for key in ("agent_import_path", "agent_kwargs", "agent_env"):
        if metadata.get(key):
            request[key] = metadata[key]

    t_start = time.time()
    retries = int(os.environ.get("HARBOR_INFRA_RETRIES", "2"))
    servers: list[str] = []
    resp: dict[str, Any] | None = None
    infra, started = True, False
    for attempt in range(retries + 1):
        resp, server = await POOL.run(request, timeout_s=_timeout_s())
        servers.append(server)
        infra, started = _is_infra(resp)
        if not infra or started or (resp or {}).get("exit_status") == "Flushed":
            break
        logger.warning(
            "trial %s failed before the agent ran on %s (%s); re-dispatching (%d/%d)",
            request["instance_id"], server, (resp or {}).get("exit_status", "no response"), attempt + 1, retries,
        )

    resp = resp or {"exit_status": "DispatchError"}
    agent_metrics = dict(resp.get("agent_metrics") or {})
    # Context exhaustion reported only in the agent's error text (e.g. codex's remote-compaction attempt)
    # is an overlong trajectory, not an agent error.
    msg = str(agent_metrics.get("exception_message") or "")
    if resp.get("exit_status") == "AgentError" and any(m in msg for m in _OVERFLOW_MARKERS):
        resp = {**resp, "exit_status": "SequenceLengthLimitExceeded"}
    agent_metrics.update(agent_server=servers[-1], dispatch_attempts=len(servers))
    out = {
        "reward": float(resp.get("reward", 0.0) or 0.0),
        "exit_status": resp.get("exit_status", ""),
        "eval_report": resp.get("eval_report", {}),
        "agent_metrics": agent_metrics,
        "infra_failure": bool(infra),
    }
    _log_trial(request["instance_id"], out, t_start)
    return out


def _log_trial(instance_id: str, out: dict[str, Any], t_start: float) -> None:
    path = os.environ.get("HM_TRIAL_LOG")
    if not path:
        return
    m = out["agent_metrics"]
    rec = {
        "t_end": time.time(),
        "wall_s": round(time.time() - t_start, 1),
        "instance_id": instance_id,
        "reward": out["reward"],
        "exit_status": out["exit_status"],
        "infra_failure": out["infra_failure"],
        **{k: m.get(k) for k in ("agent_server", "dispatch_attempts", "env_setup_time", "agent_setup_time",
                                 "agent_run_time", "eval_time", "total_time", "turns", "n_input_tokens",
                                 "n_output_tokens", "exception_type")},
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as exc:
        logger.warning("could not append to HM_TRIAL_LOG %s: %r", path, exc)


async def abort(args) -> None:
    """Oversampling abort: cancel this Miles instance's in-flight trials on every server."""
    instances = getattr(args, "session_server_instances", None) or []
    ids = {i.instance_id for i in instances if getattr(i, "instance_id", None)}
    if single := getattr(args, "session_server_instance_id", None):
        ids.add(single)
    for instance_id in ids:
        result = await POOL.broadcast("/flush", {"session_server_instance_id": instance_id})
        logger.info("flushed %s on agent servers: %s", instance_id, result)
