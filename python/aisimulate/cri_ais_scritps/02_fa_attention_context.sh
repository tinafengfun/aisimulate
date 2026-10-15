#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=0
AIS_GPU_HELP="GPU id this collector runs on"
ais_parse_args "$@"

# fp8-kv cases are skipped by default; pass --enable-fp8kv to run the full
# sweep (--disable-fp8kv is kept as a no-op for backward compatibility).
AIS_CASE_FILTERS=(--case-filter "'fp8'" --case-filter ", False, True, ")
_ais_rest=()
for _ais_a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    case "${_ais_a}" in
        --disable-fp8kv)
            echo ">>> --disable-fp8kv: skipping bf16-compute + fp8-KV cases (already the default)" ;;
        --enable-fp8kv)
            AIS_CASE_FILTERS=()
            echo ">>> --enable-fp8kv: running the full sweep, including fp8-KV cases" ;;
        *) _ais_rest+=("${_ais_a}") ;;
    esac
done
EXTRA_ARGS=(${_ais_rest[@]+"${_ais_rest[@]}"})

ais_run COLLECTOR_ATTN_YAML_ONLY=1 \
    "${PYTHON}" "${COLLECT_PY}" \
    --backend vllm --ops attention_context --gpu cri --keep-csv \
    ${AIS_CASE_FILTERS[@]+"${AIS_CASE_FILTERS[@]}"} \
    "${AIS_SMOKE_ARGS[@]}" "${EXTRA_ARGS[@]}"
