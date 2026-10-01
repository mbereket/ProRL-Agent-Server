#!/usr/bin/env python3
"""Are two dry-rendered configs the same run? Compares two tools/dry_render.sh output dirs.

usage: config_equiv.py OLD_RENDER_DIR NEW_RENDER_DIR [--old-example DIR] [--new-example DIR]

Compared, after normalizing each side's run dir to <RUN_DIR> and run name to <RUN_NAME> (cosmetic by definition):
  1. Miles args (miles-args.txt = args-<job>.txt): flags and values as a multiset, plus whether the order is identical;
  2. train command (train.py vs train_async.py --fully-async);
  3. runtime patch sets handed to ray_node.sh (miles_runtime patch dirs + harbor_miles miles_patches);
  4. agent servers per node: --max-concurrent, --agent-timeout, CPU pinning (HM_SANDBOX_RESERVE_CPUS);
  5. prepare_data.py inputs: tasks dir, task list CONTENT (sha256 + count; the list's path may differ), harness/agent args;
  6. job environment, restricted to variables some launcher/agent code consumes (scanned from both example trees) plus
     PYTORCH_CUDA_ALLOC_CONF. Variables only one side sets and no code reads are listed as dead knobs (no effect);
     derivation inputs of the layered config (setup/config.sh: CAP, LAYOUT_PRESET, ...) are reported as such.
Exit 0 when every difference is cosmetic (paths, run name), 1 otherwise.
"""
import argparse
import hashlib
import os
import re
import sys

COSMETIC_ENV = {"RUN_NAME", "RUN_DIR", "HM_ROOT", "HM_RUNS_ROOT", "HM_TRIAL_LOG", "HARBOR_AGENT_SERVERS_FILE", "HM_CONFIG_FILE",
                "HM_CONFIG_LAYERS", "TASK_IDS_FILE", "JOB_NAME", "HM_EXAMPLE_DIR", "MILES_PATCH_DIR", "MILES_RUNTIME_DIR",
                "SCOMPOSE_PKGS", "XDG_CACHE_HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "APPTAINER_CONFIGDIR",
                "APPTAINER_CACHEDIR", "HF_HOME"}
EXTERNAL_CONSUMED = {"PYTORCH_CUDA_ALLOC_CONF", "NVINF_API_KEY", "WANDB_API_KEY", "WANDB_PROJECT"}
VAR_RE = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)|environ(?:\.get)?\(\s*[\"']([A-Z_][A-Z0-9_]*)|environ\[[\"']([A-Z_][A-Z0-9_]*)")


def read(d, name):
    p = os.path.join(d, name)
    return open(p).read() if os.path.exists(p) else ""


def env_of(d):
    out = {}
    for line in read(d, "job-env.txt").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def normalizer(env):
    run_dir, run_name = env.get("RUN_DIR", ""), env.get("RUN_NAME", "")

    def norm(s):
        if run_dir:
            s = s.replace(run_dir, "<RUN_DIR>")
        if run_name:
            s = re.sub(r"(?<![\w-])" + re.escape(run_name) + r"(?![\w-])", "<RUN_NAME>", s)
        s = re.sub(r"\.tmp\.[\w.-]+\.\d+", ".tmp.<host>.<pid>", s)
        return s
    return norm


def flags(lines):
    """['--a', 'x', '--b', ...] -> [('--a', ('x',)), ('--b', ()), ...]"""
    out, cur, vals = [], None, []
    for tok in lines:
        if tok.startswith("--"):
            if cur is not None:
                out.append((cur, tuple(vals)))
            cur, vals = tok, []
        else:
            vals.append(tok)
    if cur is not None:
        out.append((cur, tuple(vals)))
    return out


def consumed_vars(example_dir):
    """Variables referenced by launcher / agent / rollout code (not by the layered-config derivation), and which files read
    each (DRIVER_ONLY[var] = True when only launch/train_driver.sh reads it: its effect is entirely in the Miles args and the
    train command, compared above)."""
    names, derive = set(), set()
    if not example_dir or not os.path.isdir(example_dir):
        return None, None
    for sub in ("launch", "setup", "miles_side", "harbor_miles_agents", "cluster", "tools/prepare_data.py"):
        root = os.path.join(example_dir, sub)
        paths = [root] if os.path.isfile(root) else [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
                                                     if f.endswith((".sh", ".py"))]
        for p in paths:
            text = open(p, errors="replace").read()
            found = {a or b or c for a, b, c in VAR_RE.findall(text)}
            (derive if p.endswith("setup/config.sh") else names).update(found)
            if not p.endswith("setup/config.sh"):
                for v in found:
                    READERS.setdefault(v, set()).add(os.path.relpath(p, example_dir))
    for patch_dir in ("harbor/patches", "miles_patches"):
        root = os.path.join(example_dir, patch_dir)
        for dp, _, fs in os.walk(root):
            for f in fs:
                text = open(os.path.join(dp, f), errors="replace").read()
                found = {a or b or c for a, b, c in VAR_RE.findall(text)}
                names.update(found)
                for v in found:
                    READERS.setdefault(v, set()).add(os.path.relpath(os.path.join(dp, f), example_dir))
    for v in EXTERNAL_CONSUMED:
        READERS.setdefault(v, set()).add("<external>")
    return names | EXTERNAL_CONSUMED, derive - names


READERS = {}


def driver_only(var):
    return READERS.get(var, set()) == {"launch/train_driver.sh"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--old-example", help="example tree the old render used (default: from OLD/example-dir.txt)")
    ap.add_argument("--new-example")
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eo, en = env_of(a.old), env_of(a.new)
    no, nn = normalizer(eo), normalizer(en)
    real, cosmetic = [], []

    # 1. Miles args
    ao = [no(x) for x in read(a.old, "miles-args.txt").splitlines()]
    an = [nn(x) for x in read(a.new, "miles-args.txt").splitlines()]
    ro = [x for x in read(a.old, "miles-args.txt").splitlines()]
    rn = [x for x in read(a.new, "miles-args.txt").splitlines()]
    fo, fn = flags(ao), flags(an)
    so, sn = sorted(fo), sorted(fn)
    print(f"1. Miles args: old {len(ao)} tokens / {len(fo)} flags, new {len(an)} tokens / {len(fn)} flags")
    if so == sn:
        n_path = sum(1 for x, y in zip(ro, rn) if x != y) if len(ro) == len(rn) else None
        print(f"   SAME flags and values{' and order' if fo == fn else ' (order differs; argparse: no effect)'}"
              + (f"; {n_path} token(s) differ only by run dir / run name" if n_path else ""))
        if n_path:
            cosmetic.append(f"Miles args: {n_path} path token(s) (run dir)")
        for x, y in zip(ro, rn):
            if x != y:
                print(f"     cosmetic: {x}  ->  {y}")
    else:
        import collections
        co, cn = collections.Counter(fo), collections.Counter(fn)
        for k in sorted(set(co) | set(cn)):
            if co[k] != cn[k]:
                side = "old only" if co[k] > cn[k] else "new only"
                print(f"   DIFF {side}: {k[0]} {' '.join(k[1])}")
                real.append(f"Miles arg {side}: {k[0]} {' '.join(k[1])}")
    dup = [f for f, _ in fn if [g for g, _ in fn].count(f) > 1]
    if dup:
        print(f"   note: repeated flags in new (last wins): {sorted(set(dup))}")

    # 2. train command
    to, tn = read(a.old, "train-cmd.txt").strip(), read(a.new, "train-cmd.txt").strip()
    print(f"2. train command: {'SAME' if to == tn else 'DIFF'} ({tn})")
    if to != tn:
        real.append(f"train command {to!r} vs {tn!r}")

    # 3. patch sets
    po, pn = read(a.old, "patch-sets.txt").split(), read(a.new, "patch-sets.txt").split()
    print(f"3. runtime patch sets: {'SAME' if po == pn else 'DIFF'}: {' '.join(pn)}")
    if po != pn:
        real.append(f"patch sets {po} vs {pn}")

    # 4. agent servers
    def servers(d, norm):
        out = []
        for line in read(d, "agent-servers.txt").splitlines():
            line = norm(line)
            line = re.sub(r"--trials-dir \S+", "--trials-dir <RUN_DIR>/trials/<host>", line)
            line = re.sub(r"--dashboard-log-path \S+", "--dashboard-log-path <RUN_DIR>/agent_servers/<host>.requests.jsonl", line)
            out.append(line)
        return sorted(out)
    sv_o, sv_n = servers(a.old, no), servers(a.new, nn)
    print(f"4. agent servers ({len(sv_n)} node(s)): {'SAME' if sv_o == sv_n else 'DIFF'}")
    for line in sv_n:
        m = re.search(r"node=(\d+) pin=(\S+).*--max-concurrent (\d+)", line)
        t = re.search(r"--agent-timeout (\d+)", line)
        if m:
            print(f"     node {m.group(1)}: sandboxes {m.group(3)}, pinned {m.group(2)}" + (f", agent timeout {t.group(1)} s" if t else ""))
    if sv_o != sv_n:
        for x in sorted(set(sv_o) ^ set(sv_n)):
            print(f"   {'old' if x in sv_o else 'new'}: {x}")
        real.append("agent servers differ")

    # 5. prepare_data
    def prep(d, norm):
        t = read(d, "prepare-data.txt").splitlines()
        if not t:
            return "", ""
        args = re.sub(r"--ids-file \S+", "--ids-file <LIST>", norm(t[0]))
        return args, (t[1] if len(t) > 1 else "")
    (pa_o, ids_o), (pa_n, ids_n) = prep(a.old, no), prep(a.new, nn)
    same_ids = ids_o == ids_n
    print(f"5. prepare_data: args {'SAME' if pa_o == pa_n else 'DIFF'}; task list content {'SAME' if same_ids else 'DIFF'} ({ids_n})")
    if pa_o != pa_n:
        print(f"   old: {pa_o}\n   new: {pa_n}")
        real.append("prepare_data args differ")
    if not same_ids:
        real.append(f"task list content {ids_o} vs {ids_n}")

    # 6. env
    ex_o = a.old_example or read(a.old, "example-dir.txt").strip()
    ex_n = a.new_example or read(a.new, "example-dir.txt").strip() or here
    cons_n, derive_n = consumed_vars(ex_n)
    cons_o, _ = consumed_vars(ex_o) if ex_o else (None, None)
    consumed = (cons_n or set()) | (cons_o or set())
    print("6. job environment (variables the launcher/agent/rollout code consumes):")
    env_real, env_cos, dead, inputs, captured = [], [], [], [], []
    args_same = so == sn and to == tn
    for k in sorted(set(eo) | set(en)):
        vo, vn = eo.get(k), en.get(k)
        if vo == vn:
            continue
        if k in COSMETIC_ENV:
            env_cos.append(k)
            continue
        nvo, nvn = (no(vo) if vo is not None else None), (nn(vn) if vn is not None else None)
        if nvo == nvn:
            env_cos.append(k)
            continue
        if k in consumed and args_same and driver_only(k):
            captured.append((k, vo, vn))
        elif k in consumed:
            env_real.append((k, vo, vn))
        elif derive_n and k in derive_n:
            inputs.append((k, vn))
        else:
            dead.append((k, vo, vn))
    for k, vo, vn in env_real:
        print(f"   DIFF {k}: {vo!r} -> {vn!r}")
        real.append(f"env {k}: {vo!r} -> {vn!r}")
    if captured:
        print("   read only by train_driver.sh, whose output (Miles args + train command) is identical: " + ", ".join(
            f"{k} {vo!r}->{vn!r}" for k, vo, vn in captured))
    if env_cos:
        print(f"   cosmetic (paths / run name): {', '.join(env_cos)}")
        cosmetic.append("env: " + ", ".join(env_cos))
    if inputs:
        print("   derivation inputs (new layered config; their effect is in the args/env above): "
              + ", ".join(f"{k}={v}" for k, v in inputs))
    if dead:
        print("   set on one side, read by no code (no effect): " + ", ".join(
            f"{k}={vo if vo is not None else vn}" + (" (old)" if vo is not None and vn is None else " (new)" if vo is None else "")
            for k, vo, vn in dead))
    print()
    if real:
        print(f"VERDICT: NOT equivalent ({len(real)} real difference(s))")
        return 1
    print("VERDICT: EQUIVALENT (only cosmetic differences: " + "; ".join(cosmetic or ["none"]) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
