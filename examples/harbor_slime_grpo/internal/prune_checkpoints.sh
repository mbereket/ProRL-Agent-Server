#!/usr/bin/env bash
# Delete saved iterations that are neither a multiple of KEEP_EVERY nor the
# latest checkpoint. Run in a loop by run.sh (CHECKPOINT_KEEP_EVERY > 0) so a
# save_interval used for periodic eval does not accumulate ~180 GB per step.
#   prune_checkpoints.sh SAVE_DIR KEEP_EVERY
set -euo pipefail
save_dir="$1"; keep_every="$2"
latest_file="${save_dir}/latest_checkpointed_iteration.txt"
[ -f "${latest_file}" ] || exit 0
latest="$(tr -dc '0-9' < "${latest_file}")"
[ -n "${latest}" ] || exit 0          # "release" or unreadable: nothing to prune
for d in "${save_dir}"/iter_*/; do
    [ -d "${d}" ] || continue
    it="$(basename "${d}" | sed 's/iter_0*//')"; it="${it:-0}"
    if [ "${it}" -lt "${latest}" ] && [ $(( it % keep_every )) -ne 0 ]; then
        echo "[prune] $(date -u +%FT%TZ) removing iter_$(printf '%07d' "${it}") (latest ${latest}, keep every ${keep_every})"
        rm -rf "${d}"
    fi
done
