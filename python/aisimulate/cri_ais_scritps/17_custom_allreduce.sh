#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=
AIS_GPU_HELP="comma-separated GPU ids to run the collectives on, e.g. 0,1,2,3"
ais_parse_args "$@"

VLLM_XPU_ENABLE_XPU_GRAPH=1 bash "${COLLECT_COMM}" --device xpu --skip-nccl "${EXTRA_ARGS[@]}"
