# shellcheck shell=bash
# Layered run configuration (sourced via setup/common.sh by launch/node_entry.sh, launch/node_peer.sh, tools/dry_render.sh).
#
#   hm_load_config <experiment.env>        load the layers below into the (exported) environment, then derive
#
# Layers, later wins (configs/README.md):
#   1. configs/recipe.env                          recipe defaults: every run starts from these (FINDINGS.md)
#   2. configs/layouts/${LAYOUT_PRESET}.env        hardware layout: model, nodes, trainer/engine split, in-flight per engine
#   3. configs/datasets/${DATASET}.env             task packages + what their sandboxes/verifiers need
#   4. the experiment file                         what makes this run different: run name, tasks, cap, batch, LR, ...
# LAYOUT_PRESET and DATASET are read from the experiment file itself. Files are KEY=VALUE assignments sourced with set -a
# (they may reference variables of earlier layers and of the job: HM_USER_ROOT, HM_SHARED, HM_EXAMPLE_DIR).
# A flat config without LAYOUT_PRESET/DATASET still works: recipe defaults, then the file.
#
# hm_derive_config then fills every DERIVED knob that no layer set (an explicit value always wins):
#   MAX_SEQ_LEN    = CAP in tokens (64k = 65536, 96k = 98304, 128k = 131072)
#   MTPG           = MAX_SEQ_LEN / CP (max tokens per trainer GPU = one full-length sequence per CP shard)
#   LOGPROB_CHUNK  = 1024 when one sequence puts >= 96k tokens on a trainer GPU (MAX_SEQ_LEN / CP >= 98304), else 4096
#   TRAIN_ALLOC_CONF = expandable_segments:True when MAX_SEQ_LEN >= 98304 (trainer workers only, Miles --train-env-vars);
#                    empty when PYTORCH_CUDA_ALLOC_CONF is exported job-wide (then trainer and engines both inherit it)
#                    (STACK §13: REAL 96k on a 4-GPU TP4 trainer fits at full speed only with chunk 1024 + expandable
#                    segments; 8-GPU TP4xCP2 128k: 79.0 GB without, 68.3 GB with expandable segments)
#   N_ENGINES      = engine GPUs / ENGINE_TP (split: NODES x NODE_GPUS - TRAIN_GPUS; colocate / rollout-only: all GPUs)
#   ASYNC_CONCURRENCY = N_ENGINES x INFLIGHT_PER_ENGINE (fully-async in-flight sessions; the engine KV knee caps
#                    INFLIGHT_PER_ENGINE: 27B TP4 on de4 <= 35, qwen27b STATUS "de4 ENGINE OPERATING POINT")
#   HM_SANDBOXES_PER_NODE = ceil(in-flight sessions / NODES)
# and checks the layout against the job: NODES == the job's node count, cap <= the layout's MAX_CAP (measured trainer fit),
# TP <= MAX_TP, headwise CP needs TP*CP | GDN_KEY_HEADS.

# hm_tokens 96k -> 98304 ; 131072 -> 131072
hm_tokens() {
    case "$1" in
        *[kK]) echo $(( ${1%[kK]} * 1024 )) ;;
        ''|*[!0-9]*) echo "hm_tokens: bad token count '$1' (use e.g. 96k or 98304)" >&2; return 2 ;;
        *) echo "$1" ;;
    esac
}

hm_load_config() {
    local _hm_cfg="${1:?config file}" _hm_presets _hm_layout _hm_dataset _hm_f
    [ -f "${_hm_cfg}" ] || _hm_cfg="${HM_EXAMPLE_DIR}/${_hm_cfg}"
    [ -f "${_hm_cfg}" ] || { echo "hm_load_config: config not found: $1" >&2; return 2; }
    # Which presets does the experiment ask for? (dry pass in a subshell: nothing it sets survives)
    _hm_presets="$( (set +eu; set -a; . "${_hm_cfg}"; printf '%s|%s' "${LAYOUT_PRESET:-}" "${DATASET:-}") 2>/dev/null )"
    _hm_layout="${_hm_presets%%|*}"; _hm_dataset="${_hm_presets#*|}"
    HM_CONFIG_LAYERS="recipe"
    set -a
    # shellcheck disable=SC1091
    . "${HM_EXAMPLE_DIR}/configs/recipe.env"
    if [ -n "${_hm_layout}" ]; then
        _hm_f="${HM_EXAMPLE_DIR}/configs/layouts/${_hm_layout}.env"
        [ -f "${_hm_f}" ] || { set +a; echo "hm_load_config: unknown LAYOUT_PRESET ${_hm_layout} (configs/layouts/)" >&2; return 2; }
        # shellcheck disable=SC1090
        . "${_hm_f}"; HM_CONFIG_LAYERS="${HM_CONFIG_LAYERS} > layouts/${_hm_layout}"
    fi
    if [ -n "${_hm_dataset}" ]; then
        _hm_f="${HM_EXAMPLE_DIR}/configs/datasets/${_hm_dataset}.env"
        [ -f "${_hm_f}" ] || { set +a; echo "hm_load_config: unknown DATASET ${_hm_dataset} (configs/datasets/)" >&2; return 2; }
        # shellcheck disable=SC1090
        . "${_hm_f}"; HM_CONFIG_LAYERS="${HM_CONFIG_LAYERS} > datasets/${_hm_dataset}"
    fi
    # shellcheck disable=SC1090
    . "${_hm_cfg}"
    HM_CONFIG_FILE="${_hm_cfg}"; HM_CONFIG_LAYERS="${HM_CONFIG_LAYERS} > ${_hm_cfg#"${HM_EXAMPLE_DIR}"/}"
    set +a
    hm_derive_config
}

hm_derive_config() {
    local _hm_jobnodes _hm_nodes _hm_gpn _hm_total _hm_engine_gpus _hm_inflight _hm_per_seq
    set -a
    # ---- context cap and trainer memory knobs
    if [ -z "${MAX_SEQ_LEN:-}" ]; then MAX_SEQ_LEN="$(hm_tokens "${CAP:-64k}")" || return 2; fi
    CP="${CP:-1}"; TP="${TP:-2}"
    MTPG="${MTPG:-$(( MAX_SEQ_LEN / CP ))}"
    _hm_per_seq=$(( MAX_SEQ_LEN / CP ))
    if [ -z "${LOGPROB_CHUNK:-}" ]; then
        if [ "${_hm_per_seq}" -ge 98304 ]; then LOGPROB_CHUNK=1024; else LOGPROB_CHUNK=4096; fi
    fi
    if [ -z "${TRAIN_ALLOC_CONF+x}" ]; then
        if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ "${MAX_SEQ_LEN}" -ge 98304 ] && [ "${ROLLOUT_ONLY:-0}" != 1 ]; then
            TRAIN_ALLOC_CONF=expandable_segments:True
        else
            TRAIN_ALLOC_CONF=
        fi
    fi
    # ---- engines and in-flight sessions
    # Job node count: SLURM_JOB_NUM_NODES (srun steps such as node_peer.sh see SLURM_NNODES = 1).
    _hm_jobnodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"
    _hm_nodes="${NODES:-${_hm_jobnodes:-1}}"; _hm_gpn="${NODE_GPUS:-8}"; _hm_total=$(( _hm_nodes * _hm_gpn ))
    ENGINE_TP="${ENGINE_TP:-2}"
    if [ "${ROLLOUT_ONLY:-0}" = 1 ] || [ "${LAYOUT:-disagg}" = colocate ]; then _hm_engine_gpus="${_hm_total}"
    else _hm_engine_gpus=$(( _hm_total - ${TRAIN_GPUS:-4} )); fi
    N_ENGINES=$(( _hm_engine_gpus / ENGINE_TP ))
    if [ -n "${INFLIGHT_PER_ENGINE:-}" ]; then
        _hm_inflight=$(( N_ENGINES * INFLIGHT_PER_ENGINE ))
        if [ "${ASYNC:-1}" = 1 ] && [ "${LAYOUT:-disagg}" != colocate ] && [ "${ROLLOUT_ONLY:-0}" != 1 ]; then
            ASYNC_CONCURRENCY="${ASYNC_CONCURRENCY:-${_hm_inflight}}"
            _hm_inflight="${ASYNC_CONCURRENCY}"
        fi
        HM_SANDBOXES_PER_NODE="${HM_SANDBOXES_PER_NODE:-$(( (_hm_inflight + _hm_nodes - 1) / _hm_nodes ))}"
    fi
    set +a
    # ---- layout sanity (fail before any GPU work)
    if [ -n "${NODES:-}" ] && [ -n "${_hm_jobnodes}" ] && [ "${NODES}" != "${_hm_jobnodes}" ]; then
        echo "hm_derive_config: layout ${LAYOUT_PRESET:-?} needs NODES=${NODES}, the job has ${_hm_jobnodes} (submit with --nodes ${NODES})" >&2
        return 2
    fi
    if [ -n "${HM_CLUSTER_MAX_NODES:-}" ] && [ "${_hm_nodes}" -gt "${HM_CLUSTER_MAX_NODES}" ]; then
        echo "hm_derive_config: ${CLUSTER:-this cluster} runs at most ${HM_CLUSTER_MAX_NODES} node(s); layout ${LAYOUT_PRESET:-?} needs ${_hm_nodes}" >&2
        return 2
    fi
    if [ -n "${HM_CLUSTER_MAX_SANDBOXES:-}" ] && [ "${HM_SANDBOXES_PER_NODE:-32}" -gt "${HM_CLUSTER_MAX_SANDBOXES}" ]; then
        echo "hm_derive_config: WARNING ${HM_SANDBOXES_PER_NODE} sandboxes/node > ${HM_CLUSTER_MAX_SANDBOXES} validated on ${CLUSTER:-this cluster} (host memory)" >&2
    fi
    if [ -n "${MAX_CAP:-}" ] && [ "${MAX_SEQ_LEN}" -gt "${MAX_CAP}" ]; then
        echo "hm_derive_config: cap ${MAX_SEQ_LEN} exceeds layout ${LAYOUT_PRESET:-?} MAX_CAP ${MAX_CAP} (trainer does not fit: STACK.md §13)" >&2
        return 2
    fi
    if [ -n "${MAX_TP:-}" ] && [ "${TP}" -gt "${MAX_TP}" ]; then
        echo "hm_derive_config: TP ${TP} > MAX_TP ${MAX_TP} for ${MODEL_TYPE:-this model}" >&2; return 2
    fi
    if [ "${CP}" -gt 1 ] && [ -n "${GDN_KEY_HEADS:-}" ] && [ $(( GDN_KEY_HEADS % (TP * CP) )) -ne 0 ]; then
        echo "hm_derive_config: headwise CP needs TP*CP (${TP}*${CP}) to divide GDN_KEY_HEADS ${GDN_KEY_HEADS}" >&2; return 2
    fi
    if [ -n "${KNEE_INFLIGHT_PER_ENGINE:-}" ] && [ "${N_ENGINES}" -gt 0 ] && [ -n "${ASYNC_CONCURRENCY:-}" ] \
        && [ $(( ASYNC_CONCURRENCY )) -gt $(( N_ENGINES * KNEE_INFLIGHT_PER_ENGINE )) ]; then
        echo "hm_derive_config: WARNING in-flight ${ASYNC_CONCURRENCY} > ${N_ENGINES} engines x knee ${KNEE_INFLIGHT_PER_ENGINE} (KV thrash: decode /2-3)" >&2
    fi
    return 0
}

# One line for the job log: the layers and the derived knobs a reader needs to check a run.
hm_config_summary() {
    echo "config: ${HM_CONFIG_LAYERS:-?} | cap ${MAX_SEQ_LEN} mtpg ${MTPG} TP${TP} CP${CP} logprob-chunk ${LOGPROB_CHUNK}" \
         "train-alloc '${TRAIN_ALLOC_CONF:-}' job-alloc '${PYTORCH_CUDA_ALLOC_CONF:-}' | engines ${N_ENGINES:-?} x TP${ENGINE_TP}" \
         "in-flight ${ASYNC_CONCURRENCY:--} sandboxes/node ${HM_SANDBOXES_PER_NODE:-32} | B ${RBS:-?}x${NS:-?} lr ${LR:-?}"
}
