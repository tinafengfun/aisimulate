#!/usr/bin/env bash
# vLLM custom allreduce (graph/eager, per world size) with per-group retry.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/comm_group_lib.sh"

AIS_GPU_DEFAULT=
export CCL_SELECTION_POLICY="$(pwd)/cri_ce+chunking_policy.json"

AIS_GPU_HELP="comma-separated GPU ids to run the collectives on, e.g. 0,1,2,3"
ais_parse_args "$@"

# unset ZE_AFFINITY_MASK
echo ">>> execution: ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}"
retry_init custom_allreduce_perf.txt

run_all_groups() {
    gpu_sizes || return $?
    run_custom_groups
    finish_groups
}

echo ">>> cmd: cd $(printf '%q' "${ROOT}") && run_all_groups (retries=${RETRIES}, timeout=${ATTEMPT_TIMEOUT}s)" | tee -a "${AIS_LOG}"
run_all_groups 2>&1 | tee -a "${AIS_LOG}"
exit "${PIPESTATUS[0]}"
