#!/usr/bin/env bash
# oneCCL (op x dtype x np) and custom-allreduce (np) collection with per-group retry.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/comm_group_lib.sh"
export CCL_SELECTION_POLICY="${AIS_SCRIPT_DIR}/cri_ce+chunking_policy.json"

AIS_GPU_DEFAULT=
AIS_GPU_HELP="comma-separated GPU ids to run the collectives on, e.g. 0,1,2,3"
ais_parse_args "$@"

# unset ZE_AFFINITY_MASK
echo ">>> execution: ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}"
retry_init oneccl_perf.txt custom_allreduce_perf.txt

run_all_groups() {
    local n op dt
    gpu_sizes || return $?
    for n in "${SIZES[@]}"; do
        for op in all_gather reduce_scatter all_reduce alltoall; do
            for dt in half int8; do
                run_group "oneccl_${op}_np${n}_${dt}" oneccl_perf.txt oneccl "${n}" "${op}" "${dt}" -- \
                    "${PYTHON}" -u "${NET}/collect_oneccl_xpu.py" -n "${n}" -O "${op}" --dtype "${dt}" ||
                    FAILURES+=("oneccl_${op}_np${n}_${dt}")
            done
        done
    done
    run_custom_groups
    finish_groups
}

echo ">>> cmd: cd $(printf '%q' "${ROOT}") && run_all_groups (retries=${RETRIES}, timeout=${ATTEMPT_TIMEOUT}s)" | tee -a "${AIS_LOG}"
run_all_groups 2>&1 | tee -a "${AIS_LOG}"
exit "${PIPESTATUS[0]}"
