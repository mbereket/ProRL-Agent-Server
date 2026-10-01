#!/usr/bin/env bash
# compose_check.sh NAME SRC_DIR STEPS : build replay steps from real dumps and verify Miles can load them.
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
OUT="${MILES_STACK_ROOT}/bench/q35-9b-real/${1:?name}/rollout_data"
"${MR}/mrun" --no-nv -- bash -c "python3 '${MR}/bench/compose_replay.py' '${OUT}' ${2:?src}/*.pt --steps ${3:-3} && python3 - <<'PY'
import glob, torch
from miles.utils.types import Sample
for f in sorted(glob.glob('${OUT}/*.pt')):
    S = [Sample.from_dict(x) for x in torch.load(f, weights_only=False)['samples']]
    L = [len(s.tokens) for s in S]
    print('LOAD_OK', f.rsplit('/', 1)[-1], len(S), sum(L), max(L), sorted({s.status.value for s in S}), sorted({s.group_index for s in S})[:3])
PY"
