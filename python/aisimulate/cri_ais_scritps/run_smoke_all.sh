#!/usr/bin/env bash
# run_smoke_all.sh — smoke every collector in this directory.
#
# Each collector samples a handful of cases instead of running its full sweep,
# so this answers "does the whole pipeline still run end to end" in minutes.
# It is a health check, not a measurement: the numbers it produces are noise.
#
# Usage:
#   ./run_smoke_all.sh                 # single-GPU collectors on GPU 0, comm on 0,1
#   ./run_smoke_all.sh -d 3            # single-GPU collectors on GPU 3
#   ./run_smoke_all.sh -o /tmp/smoke   # results and logs elsewhere
#   ./run_smoke_all.sh --skip-comm     # skip 17/18 (they need >= 2 free GPUs)
#   ./run_smoke_all.sh 01_gemm.sh 07_moe.sh   # only these
#
# Exit code is the number of failed collectors, so CI can gate on it.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

GPU=0
OUTDIR=""
SKIP_COMM=0
SELECTED=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--device) GPU="$2"; shift 2 ;;
        -o|--outdir) OUTDIR="$2"; shift 2 ;;
        --skip-comm) SKIP_COMM=1; shift ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
        *) SELECTED+=("$1"); shift ;;
    esac
done

ts="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/smoke_${ts}}"
mkdir -p "${OUTDIR}"
LOG_DIR="${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"

# Single-GPU collectors run on one card; 17/18 need at least two.
SINGLE_GPU=(01_gemm.sh 02_fa_attention_context.sh 03_fa_attention_generation.sh
            06_gdn.sh 07_moe.sh 14_mhc.sh 15_16_wideep_moe.sh
            12_13_hca_context_generation.sh 26_27_swa_context_generation.sh
            04_05_mla_attention.sh 10_11_csa_attention.sh)
COMM=(17_custom_allreduce.sh 18_nccl.sh)

if [[ ${#SELECTED[@]} -gt 0 ]]; then
    TARGETS=("${SELECTED[@]}")
else
    TARGETS=("${SINGLE_GPU[@]}")
    [[ "${SKIP_COMM}" = 1 ]] || TARGETS+=("${COMM[@]}")
fi

is_comm() {
    local s
    for s in "${COMM[@]}"; do [[ "$1" = "$s" ]] && return 0; done
    return 1
}

echo "=============================================="
echo " smoke run ${ts}"
echo " results : ${OUTDIR}"
echo " logs    : ${LOG_DIR}"
echo " gpu     : ${GPU} (comm collectors: 0,1)"
echo " targets : ${#TARGETS[@]}"
echo "=============================================="

PASS=(); FAIL=(); SKIP=()
for s in "${TARGETS[@]}"; do
    path="${SCRIPT_DIR}/${s}"
    if [[ ! -x "${path}" ]]; then
        echo "--- ${s}: SKIP (not found)"; SKIP+=("${s}"); continue
    fi
    log="${LOG_DIR}/${s%.sh}.log"
    # Comm collectors pick their world sizes from the visible GPU count, so let
    # their own smoke path choose the cards instead of pinning one.
    if is_comm "${s}"; then
        args=(-s -o "${OUTDIR}")
    else
        args=(-s -d "${GPU}" -o "${OUTDIR}")
    fi

    printf -- "--- %-46s " "${s}"
    start=$SECONDS
    "${path}" "${args[@]}" >"${log}" 2>&1
    rc=$?
    dur=$((SECONDS - start))
    if [[ ${rc} -ne 0 ]]; then
        echo "FAIL (${dur}s, exit ${rc})"; FAIL+=("${s}")
        continue
    fi
    # A zero exit still hides per-case failures, so check the collector's own
    # error tally when it reports one.
    errs="$(grep -oE 'Total errors: [0-9]+' "${log}" | grep -oE '[0-9]+$' |
            awk '{t+=$1} END{print t+0}')"
    if [[ "${errs}" != 0 ]]; then
        echo "FAIL (${dur}s, ${errs} case errors)"; FAIL+=("${s}")
    else
        echo "PASS (${dur}s)"; PASS+=("${s}")
    fi
done

echo "=============================================="
echo " PASS ${#PASS[@]}  FAIL ${#FAIL[@]}  SKIP ${#SKIP[@]}"
[[ ${#FAIL[@]} -gt 0 ]] && printf ' failed: %s\n' "${FAIL[@]}"
[[ ${#SKIP[@]} -gt 0 ]] && printf ' skipped: %s\n' "${SKIP[@]}"
echo " logs: ${LOG_DIR}"
echo "=============================================="
exit "${#FAIL[@]}"
