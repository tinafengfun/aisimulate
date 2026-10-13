#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=0
AIS_GPU_HELP="GPU id this collector runs on"
ais_parse_args "$@"

ais_run COLLECTOR_ATTN_YAML_ONLY=1 \
    "${PYTHON}" "${COLLECT_PY}" \
    --backend vllm --ops attention_context --gpu cri --keep-csv "${AIS_SMOKE_ARGS[@]}" "${EXTRA_ARGS[@]}"
