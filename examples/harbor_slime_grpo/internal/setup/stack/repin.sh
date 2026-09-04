#!/usr/bin/env bash
# Move the slime / sglang pins of this stack to other commits of the polar forks
# and re-lock. Usage:
#   repin.sh [--slime SHA] [--sglang SHA] [--slime-url URL] [--sglang-url URL]
# Full 40-hex commit shas. Defaults keep the current value. After a merge into
# the fork branch: `repin.sh --slime $(git -C <slime checkout> rev-parse HEAD)`.
# Runs `uv lock` (network; restricted to linux x86_64 by pyproject) and prints
# the resulting pins. Verify with `uv lock --check` afterwards.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

pin() {   # pin NAME FIELD — current value of `NAME = { git = "...", rev = "..." }`
    sed -n "s/^${1} = { git = \"\([^\"]*\)\", rev = \"\([0-9a-f]*\)\".*/\\${2}/p" pyproject.toml
}
SLIME_URL="$(pin slime 1)"; SLIME_SHA="$(pin slime 2)"
SGLANG_URL="$(pin sglang 1)"; SGLANG_SHA="$(pin sglang 2)"
while [ $# -gt 0 ]; do
    case "$1" in
        --slime) SLIME_SHA="$2"; shift 2 ;;
        --sglang) SGLANG_SHA="$2"; shift 2 ;;
        --slime-url) SLIME_URL="$2"; shift 2 ;;
        --sglang-url) SGLANG_URL="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for sha in "${SLIME_SHA}" "${SGLANG_SHA}"; do
    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "not a full commit sha: ${sha}" >&2; exit 2; }
done

python3 - "${SLIME_URL}" "${SLIME_SHA}" "${SGLANG_URL}" "${SGLANG_SHA}" <<'PY'
import re, sys
slime_url, slime_sha, sglang_url, sglang_sha = sys.argv[1:]
p = "pyproject.toml"; t = open(p).read()
t, n1 = re.subn(r'^slime = \{ git = "[^"]*", rev = "[0-9a-f]*"( *\})',
                f'slime = {{ git = "{slime_url}", rev = "{slime_sha}"\\1', t, flags=re.M)
t, n2 = re.subn(r'^sglang = \{ git = "[^"]*", rev = "[0-9a-f]*"(.*\})',
                f'sglang = {{ git = "{sglang_url}", rev = "{sglang_sha}"\\1', t, flags=re.M)
assert n1 == 1 and n2 == 1, (n1, n2)
open(p, "w").write(t)
PY
uv lock
echo "pins:"; grep -n '^slime = \|^sglang = ' pyproject.toml
grep -n -A2 '^name = "slime"$\|^name = "sglang"$' uv.lock | grep source
