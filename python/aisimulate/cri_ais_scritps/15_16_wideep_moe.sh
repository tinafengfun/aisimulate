#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=0
AIS_GPU_HELP="GPU id this collector runs on"
ais_parse_args "$@"

# moe_ep is collected on a single physical rank: the logical EP world size
# comes from the case definition, not from the number of visible GPUs
# (collector/vllm/collect_moe.py:595-601).
ais_run VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    "${PYTHON}" "${COLLECT_PY}" \
    --backend vllm --ops moe_ep --gpu cri --keep-csv "${AIS_SMOKE_ARGS[@]}" "${EXTRA_ARGS[@]}"
