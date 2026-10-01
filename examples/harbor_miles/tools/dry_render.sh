#!/usr/bin/env bash
# Dry-render a harbor_miles config: run the REAL launcher (launch/node_entry.sh -> node_peer.sh -> start_agent_server.sh ->
# ray_node.sh -> train_driver.sh) with Slurm, GPUs, Ray, the Miles SIF and Harbor stubbed, and capture exactly what a job would
# launch: the Miles args (= RUN_DIR/args-<job>.txt), the train command, the runtime patch sets, the agent servers' settings per
# node, the task list handed to prepare_data.py and the job environment. No cluster, no GPU; runs on a laptop (macOS bash 3.2)
# or a login node. Use it before every submission, and to prove two configs equivalent (tools/config_equiv.py).
#
#   tools/dry_render.sh [--ref GIT_REF] [--cluster dfw] [--nodes N] [--hm-root DIR] [--out DIR] CONFIG [K=V ...]
#
#   CONFIG          experiment/config file, absolute or relative to examples/harbor_miles (as node_entry.sh resolves it)
#   --ref REF       render with the launcher of another commit (git archive REF:examples/harbor_miles), e.g. an old config on
#                   its own launcher; CONFIG is then resolved inside that tree unless it is an existing path
#   --cluster C     hel | dfw (default) | aws-iad | ord | draco  (paths from cluster/clusters.sh)
#   --nodes N       job nodes (default: the layout's NODES, else 1)
#   --hm-root DIR   HM_ROOT as submit.py would export it (default: clusters.sh's)
#   --eval-src NAME[:ITER[:SHARDS]]  for EVAL_FROM_RUN configs: create a stub source run NAME (run name or absolute dir) with a
#                   complete checkpoint iter ITER (default 3) of SHARDS trainer-rank shards (default 4)
#   K=V             extra job env, like submit.py --env (e.g. DE4_HALF=B)
# Output (default ${TMPDIR:-/tmp}/hm-dry/<config>-<cluster>[-<ref>]): miles-args.txt, train-cmd.txt, patch-sets.txt,
# agent-servers.txt, prepare-data.txt, job-env.txt (secrets masked), config.txt (layers + derived knobs), render.log.
# Cluster paths appear verbatim (the stubs write under a scratch prefix that is stripped from the outputs).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EX_SELF="$(cd "${HERE}/.." && pwd)"
REPO="$(git -C "${EX_SELF}" rev-parse --show-toplevel)"
REF=""; CL=dfw; NN=""; HMR=""; OUT=""; EVSRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --cluster) CL="$2"; shift 2 ;;
        --nodes) NN="$2"; shift 2 ;;
        --hm-root) HMR="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --eval-src) EVSRC="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) break ;;
    esac
done
CFG_ARG="${1:?usage: dry_render.sh [opts] CONFIG [K=V ...]}"; shift
REAL_PY="$(command -v python3)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hm-dry.XXXXXX")"
PFX="${WORK}/fs"                     # every cluster path the stubs write to lives under this prefix
DRY_OUT="${WORK}/out"; mkdir -p "${DRY_OUT}" "${PFX}" "${WORK}/bin" "${WORK}/home" "${WORK}/miles"

# ---- 1. the launcher tree (this checkout or another commit)
mkdir -p "${WORK}/src/examples"
if [ -n "${REF}" ]; then
    git -C "${REPO}" archive "${REF}" examples/harbor_miles | tar -x -C "${WORK}/src"
else
    cp -R "${EX_SELF}" "${WORK}/src/examples/harbor_miles"
fi
EX="${WORK}/src/examples/harbor_miles"
if [ -f "${EX}/${CFG_ARG}" ]; then CFG="${EX}/${CFG_ARG}"
elif [ -f "${CFG_ARG}" ]; then CFG="$(cd "$(dirname "${CFG_ARG}")" && pwd)/$(basename "${CFG_ARG}")"
else echo "dry_render: config not found: ${CFG_ARG}" >&2; exit 2; fi
# The driver changes into the Miles source tree before launching; point it at a scratch dir.
sed -i.orig -e 's#^cd /root/miles$#cd "${DRY_MILES_DIR}"#' -e 's#^cd "${MILES_DIR:-/root/miles}"$#cd "${DRY_MILES_DIR}"#' \
    "${EX}/launch/train_driver.sh"
# Harbor/agent setup scripts -> stubs that print the dirs the real ones would build (pre-created here: head and peers race).
mkdir -p "${WORK}/harbor/.venv/bin" "${WORK}/sbrt/bin" "${WORK}/tools/codex" "${WORK}/tools/mini-swe-agent" "${WORK}/tools/opencode"
ln -s "${WORK}/bin/harbor-python" "${WORK}/harbor/.venv/bin/python"
for t in codex mini-swe-agent opencode; do echo dry > "${WORK}/tools/${t}/VERSION"; done
printf '#!/usr/bin/env bash\necho "${DRY_WORK}/harbor"\n' > "${EX}/setup/ensure_harbor.sh"
printf '#!/usr/bin/env bash\necho "${DRY_WORK}/sbrt/bin/python"\n' > "${EX}/setup/ensure_sandbox_runtime.sh"
printf '#!/usr/bin/env bash\nmkdir -p "${DRY_WORK}/tools/$1"; echo "${DRY_WORK}/tools/$1"\n' > "${EX}/setup/ensure_agent_tools.sh"

# ---- 2. stubs on PATH
B="${WORK}/bin"
cat > "${B}/python3" <<EOF
#!${REAL_PY}
import hashlib, json, os, sys
REAL = "${REAL_PY}"
argv = sys.argv[1:]
out = os.environ.get("DRY_OUT", ".")
base = os.path.basename(argv[0]) if argv and not argv[0].startswith("-") else ""
def rec(name, obj):
    with open(os.path.join(out, name), "a") as f: f.write(json.dumps(obj) + "\n")
if base == "prepare_data.py":
    a = argv[1:]; ids = a[a.index("--ids-file") + 1] if "--ids-file" in a else ""
    h = hashlib.sha256(open(ids, "rb").read()).hexdigest()[:16] if ids and os.path.exists(ids) else "-"
    n = sum(1 for l in open(ids) if l.strip()) if ids and os.path.exists(ids) else 0
    rec("prepare-data.jsonl", {"args": a, "ids_sha256": h, "ids_count": n})
    o = a[a.index("--out") + 1]
    open(o, "w").write('{"prompt": "dry", "metadata": {}}\n'); sys.exit(0)
if base == "model_args_utils.py":
    print("--DRY-MODEL-ARGS-OF " + argv[1]); sys.exit(0)        # the pinned Miles prints the real model args in the job
if base in ("train.py", "train_async.py"):
    rec("train-cmd.jsonl", [base] + argv[1:])
    a = argv[1:]
    if "--lora-adapter-path" in a:   # what Miles logs when the adapter loads (the driver's eval mode checks for it)
        print("[dry] Successfully loaded LoRA adapter from " + a[a.index("--lora-adapter-path") + 1], flush=True)
    if "--eval-interval" in a and ("--skip-eval-before-train" not in a):
        print("[2026-01-01 00:00:00 dry] log_utils.py:1 - eval 0: {}", flush=True)   # eval-only modes stop on this line
    sys.exit(0)
if base == "node_monitor.py":
    sys.exit(0)
if argv[:1] == ["-c"] and "sched_getaffinity" in argv[1] and not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(int(os.environ.get("DRY_CPUS", "96"))))   # macOS has none
    sys.argv = ["-c"] + argv[2:]; exec(compile(argv[1], "<string>", "exec")); sys.exit(0)
os.execv(REAL, [REAL] + argv)
EOF
cat > "${B}/harbor-python" <<'EOF'
#!/usr/bin/env bash
# the Harbor agent server: record how it would start (per node), then exit
echo "node=${DRY_NODE:-0} pin=${DRY_PIN:-none} $* | env HARBOR_SANDBOX_DATA_LIMIT_KB=${HARBOR_SANDBOX_DATA_LIMIT_KB:-unset} HARBOR_MEM_WATCHDOG_FRAC=${HARBOR_MEM_WATCHDOG_FRAC:-unset} HARBOR_CANCEL_DIR=${HARBOR_CANCEL_DIR:+set}" >> "${DRY_OUT}/agent-servers.txt"
EOF
cat > "${B}/taskset" <<'EOF'
#!/usr/bin/env bash
# taskset -c LIST CMD...: record how many CPUs the sandboxes would get
[ "$1" = -c ] || exec "$@"
n="$(echo "$2" | tr ',' '\n' | grep -c .)"; shift 2
DRY_PIN="${n}cpus" exec "$@"
EOF
cat > "${B}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in *--list-gpus*) for i in $(seq 0 $(( ${DRY_NODE_GPUS:-8} - 1 ))); do echo "GPU $i: NVIDIA H100 80GB HBM3 (UUID: GPU-dry-$i)"; done ;; esac
exit 0
EOF
cat > "${B}/scontrol" <<'EOF'
#!/usr/bin/env bash
case "$1 $2" in
  "show hostnames") for i in $(seq 0 $(( ${SLURM_JOB_NUM_NODES:-1} - 1 ))); do echo "drynode${i}"; done ;;
  "show job") echo "   StdOut=<joblogs>/${SLURM_JOB_ID}-hm-${JOB_NAME:-dry}.log" ;;
esac
exit 0
EOF
cat > "${B}/srun" <<'EOF'
#!/usr/bin/env bash
# srun --overlap ... -w NODE cmd...: run cmd as a 1-node, 1-task step (the env a real step gets)
node=1
while [ $# -gt 0 ]; do case "$1" in -w) node="${2#drynode}"; shift 2 ;; --*=*|--overlap|--kill-on-bad-exit*) shift ;; --*) shift 2 ;; *) break ;; esac; done
SLURM_NNODES=1 SLURM_NODEID="${node}" SLURM_STEP_NUM_TASKS=1 DRY_NODE="${node}" exec "$@"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "${B}/flock"; cp "${B}/flock" "${B}/curl"; cp "${B}/flock" "${B}/pkill"
printf '#!/usr/bin/env bash\necho "127.0.0.1       STREAM drynode"\n' > "${B}/getent"
printf '#!/usr/bin/env bash\nexec /bin/sleep 0.05\n' > "${B}/sleep"   # the launcher's poll loops, at speed
cat > "${B}/stat" <<EOF
#!/usr/bin/env bash
# GNU "stat -c %Y FILE" (mtime) on any platform; anything else -> the system stat
if [ "\$1" = -c ] && [ "\$2" = %Y ]; then exec "${REAL_PY}" -c 'import os, sys; print(int(os.path.getmtime(sys.argv[1])))' "\$3"; fi
exec /usr/bin/stat "\$@"
EOF
chmod +x "${B}"/*
# Runtime package: ray_node.sh stub records the patch sets + the job env, then runs the driver like the Ray head would.
mkdir -p "${WORK}/pkgs/miles_runtime"
cat > "${WORK}/pkgs/miles_runtime/ray_node.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
pp=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do pp+=("$1"); shift; done; shift
printf '%s\n' "${pp[@]}" > "${DRY_OUT}/ray-node-args.txt"
env | LC_ALL=C sort > "${DRY_OUT}/job-env.raw"
export MILES_NUM_NODES="${SLURM_JOB_NUM_NODES:-1}" RAY_ADDRESS=dry MASTER_ADDR=127.0.0.1
exec "$@"
EOF
chmod +x "${WORK}/pkgs/miles_runtime/ray_node.sh"

# ---- 3. job environment (clean: nothing from this shell leaks in)
roots="$(env -i PATH="/usr/bin:/bin" CLUSTER="${CL}" ${HMR:+HM_ROOT="${HMR}"} bash -c \
    'source "$1/cluster/clusters.sh" && printf "%s|%s" "${HM_ROOT}" "${HM_RUNS_ROOT:-}"' _ "${EX}")"
HM_ROOT_REAL="${roots%%|*}"; RUNS_REAL="${roots#*|}"
if [ -z "${NN}" ]; then
    NN="$(env -i PATH="/usr/bin:/bin" HM_EXAMPLE_DIR="${EX}" bash -c 'set -a; . "$1" >/dev/null 2>&1; l="${LAYOUT_PRESET:-}"
        if [ -n "$l" ] && [ -f "$2/configs/layouts/$l.env" ]; then . "$2/configs/layouts/$l.env" >/dev/null 2>&1; . "$1" >/dev/null 2>&1; fi
        echo "${NODES:-1}"' _ "${CFG}" "${EX}" 2>/dev/null || echo 1)"
fi
JOBENV=(PATH="${B}:/usr/bin:/bin:/usr/sbin:/sbin" HOME="${WORK}/home" USER="${USER:-dry}" LOGNAME="${USER:-dry}"
    CLUSTER="${CL}" JOB_NAME=dry HM_ROOT="${PFX}${HM_ROOT_REAL}"
    SLURM_JOB_ID=dry SLURM_NODEID=0 SLURM_NNODES="${NN}" SLURM_JOB_NUM_NODES="${NN}" SLURM_JOB_NODELIST="drynode[0-$(( NN - 1 ))]"
    SLURM_CPUS_PER_TASK=96 SCOMPOSE_PKGS="${WORK}/pkgs" MILES_RUNTIME_DIR="${WORK}/pkgs/miles_runtime"
    SESSION_PORT=65460 NVINF_API_KEY=dry-render-placeholder
    DRY_OUT="${DRY_OUT}" DRY_WORK="${WORK}" DRY_MILES_DIR="${WORK}/miles" DRY_NODE_GPUS=8 DRY_CPUS=96)
[ -n "${RUNS_REAL}" ] && JOBENV+=(HM_RUNS_ROOT="${PFX}${RUNS_REAL}")
# Stub source run for EVAL_FROM_RUN (decoupled eval): a complete checkpoint, written 10 min ago, in every root the driver looks.
if [ -n "${EVSRC}" ]; then
    IFS=: read -r ev_name ev_iter ev_shards <<< "${EVSRC}"
    case "${ev_name}" in /*) ev_dirs=("${PFX}${ev_name}") ;; *) ev_dirs=("${PFX}${HM_ROOT_REAL}/runs/${ev_name}"); [ -n "${RUNS_REAL}" ] && ev_dirs+=("${PFX}${RUNS_REAL}/${ev_name}") ;; esac
    for d in "${ev_dirs[@]}"; do
        a="${d}/ckpt/iter_$(printf %07d "${ev_iter:-3}")/adapter"; mkdir -p "${a}"
        for r in $(seq 0 $(( ${ev_shards:-4} - 1 ))); do echo stub > "${a}/adapter_megatron_rank${r}.pt"; echo stub > "${a}/training_state_rank${r}.pt"; done
        echo '{}' > "${a}/adapter_config.json"
        "${REAL_PY}" -c 'import os, sys, time
t = time.time() - 600
for dp, _, fs in os.walk(sys.argv[1]):
    for f in fs: os.utime(os.path.join(dp, f), (t, t))' "${d}"
    done
fi
for kv in "$@"; do JOBENV+=("${kv}"); done

# ---- 4. run the real launcher
set +e
( cd "${WORK}" && env -i "${JOBENV[@]}" bash "${EX}/launch/node_entry.sh" "${CFG}" ) > "${DRY_OUT}/render.log" 2>&1
rc=$?
set -e
wait 2>/dev/null || true
sleep 1   # let a background peer step finish writing

# ---- 5. collect (strip the scratch prefix, mask secrets)
strip() { sed -e "s#${PFX}##g" -e "s#${WORK}/src/examples/harbor_miles#<EXAMPLE>#g" -e "s#${WORK}/pkgs/miles_runtime#<RUNTIME>#g" -e "s#${WORK}#<DRY>#g"; }
tag="$(basename "${CFG}" .env)-${CL}${REF:+-$(echo "${REF}" | tr '/' '_')}"
OUT="${OUT:-${TMPDIR:-/tmp}/hm-dry/${tag}}"
mkdir -p "${OUT}"
args_file="$(find "${PFX}" -name 'args-dry.txt' 2>/dev/null | head -1)"
if [ -n "${args_file}" ]; then strip < "${args_file}" > "${OUT}/miles-args.txt"; else : > "${OUT}/miles-args.txt"; fi
"${REAL_PY}" - "${DRY_OUT}" "${OUT}" "${PFX}" "${WORK}" <<'EOF'
import json, os, re, sys
src, out, pfx, work = sys.argv[1:5]
def strip(s):
    s = s.replace(pfx, "").replace(work + "/src/examples/harbor_miles", "<EXAMPLE>").replace(work + "/pkgs/miles_runtime", "<RUNTIME>")
    return s.replace(work, "<DRY>")
def read(name):
    p = os.path.join(src, name); return open(p).read() if os.path.exists(p) else ""
tc = [json.loads(l) for l in read("train-cmd.jsonl").splitlines() if l.strip()]
open(os.path.join(out, "train-cmd.txt"), "w").write(" ".join(tc[0][:2]) + "\n" if tc else "(driver did not reach the train command)\n")
pd = [json.loads(l) for l in read("prepare-data.jsonl").splitlines() if l.strip()]
with open(os.path.join(out, "prepare-data.txt"), "w") as f:
    for r in pd:
        f.write(strip(" ".join(r["args"])) + "\n" + f"ids_sha256={r['ids_sha256']} ids_count={r['ids_count']}\n")
open(os.path.join(out, "patch-sets.txt"), "w").write(strip(read("ray-node-args.txt")))
lines = sorted(set(strip(l) for l in read("agent-servers.txt").splitlines() if l.strip()))
lines = [re.sub(r"--port \d+", "--port <port>", l) for l in lines]
open(os.path.join(out, "agent-servers.txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
noise = {"PWD", "OLDPWD", "SHLVL", "_", "PATH", "HOME", "TMPDIR", "APPTAINER_TMPDIR", "DRY_OUT", "DRY_WORK", "DRY_MILES_DIR",
         "DRY_NODE_GPUS", "DRY_CPUS", "SESSION_PORT"}
env = []
for l in read("job-env.raw").splitlines():
    if "=" not in l: continue
    k, v = l.split("=", 1)
    if k in noise or k.startswith("BASH_FUNC_"): continue
    if re.search(r"API_KEY|_KEY$|TOKEN|SECRET|PASSWORD", k): v = "<masked>"
    env.append(f"{k}={strip(v)}")
open(os.path.join(out, "job-env.txt"), "w").write("\n".join(env) + "\n")
EOF
strip < "${DRY_OUT}/render.log" > "${OUT}/render.log"
echo "${EX}" > "${OUT}/example-dir.txt"     # the launcher tree used (config_equiv.py scans it for consumed variables)
# Code the job would run besides the launcher: content hashes (path-independent: renamed patches hash the same).
if [ -n "${REF}" ]; then git -C "${REPO}" archive "${REF}" examples/miles_runtime/patches | tar -x -C "${WORK}/src"; RTP="${WORK}/src/examples/miles_runtime/patches"
else RTP="${REPO}/examples/miles_runtime/patches"; fi
"${REAL_PY}" - "${EX}" "${RTP}" > "${OUT}/code-hashes.txt" <<'EOF'
import hashlib, os, sys
def h(paths):
    return hashlib.sha256("".join(sorted(hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths)).encode()).hexdigest()[:16]
def files(root, ext):
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(ext)] if os.path.isdir(root) else []
ex, rtp = sys.argv[1:3]
print("harbor (PIN + patches)", h([os.path.join(ex, "harbor/PIN")] + files(os.path.join(ex, "harbor/patches"), ".patch")))
print("miles_patches", h(files(os.path.join(ex, "miles_patches"), ".patch")))
print("runtime patches (all sets)", h(files(rtp, ".patch") + files(rtp, ".txt")))
EOF
grep -h "config: " "${OUT}/render.log" | sed 's/^.*config: /config: /' | head -1 > "${OUT}/config.txt" || true
grep -h "\[driver\]" "${OUT}/render.log" >> "${OUT}/config.txt" || true
n_args="$(grep -c . "${OUT}/miles-args.txt" || true)"
echo "dry_render: ${CFG_ARG}${REF:+ @ ${REF}} on ${CL} (${NN} node(s)): rc=${rc}, ${n_args} Miles args -> ${OUT}"
[ "${rc}" = 0 ] && [ "${n_args}" -gt 0 ] || { echo "dry_render: FAILED, see ${OUT}/render.log" >&2; tail -20 "${OUT}/render.log" >&2; exit 1; }
cat "${OUT}/config.txt"
