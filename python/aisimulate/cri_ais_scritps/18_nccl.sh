#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

AIS_GPU_DEFAULT=
AIS_GPU_HELP="comma-separated GPU ids to run the collectives on, e.g. 0,1,2,3"
ais_parse_args "$@"

# collect_comm.sh has no case limit: it derives the world sizes from
# torch.xpu.device_count(). Smoke = expose two GPUs so only tp=2 runs.
if [[ "${AIS_SMOKE}" = 1 && -z "${ZE_AFFINITY_MASK:-}" ]]; then
    export ZE_AFFINITY_MASK=0,1
    echo ">>> smoke: ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} (tp=2 only)"
fi

ais_run VLLM_XPU_ENABLE_XPU_GRAPH=1 bash "${COLLECT_COMM}" --device xpu "${EXTRA_ARGS[@]}"
