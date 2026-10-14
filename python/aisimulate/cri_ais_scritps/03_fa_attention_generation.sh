#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=0
AIS_GPU_HELP="GPU id this collector runs on"
ais_parse_args "$@"

AIS_CASE_FILTERS=()
_ais_rest=()
for _ais_a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    case "${_ais_a}" in
        --disable-fp8kv)
            AIS_CASE_FILTERS=(--case-filter "'fp8'" --case-filter ", False, True, ")
            echo ">>> --disable-fp8kv: skipping bf16-compute + fp8-KV cases" >&2 ;;
        *) _ais_rest+=("${_ais_a}") ;;
    esac
done
EXTRA_ARGS=(${_ais_rest[@]+"${_ais_rest[@]}"})

ais_run COLLECTOR_ATTN_YAML_ONLY=1 VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    "${PYTHON}" "${COLLECT_PY}" \
    --backend vllm --ops attention_generation --gpu cri --keep-csv \
    ${AIS_CASE_FILTERS[@]+"${AIS_CASE_FILTERS[@]}"} \
    "${AIS_SMOKE_ARGS[@]}" "${EXTRA_ARGS[@]}"
