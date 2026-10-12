#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=0
AIS_GPU_HELP="GPU id this collector runs on"
ais_parse_args "$@"

ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK}" COLLECTOR_ATTN_YAML_ONLY=1 VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    "${PYTHON}" "${COLLECT_PY}" \
    --backend vllm --ops attention_generation --gpu cri --keep-csv "${EXTRA_ARGS[@]}"
