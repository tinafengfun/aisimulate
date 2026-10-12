#!/usr/bin/env bash
# Shared setup for the cri_ais_scritps collectors. Source it, don't run it.
#
# Provides:
#   COLLECT_PY / COLLECT_COMM  absolute paths, so a script works from any cwd
#   PYTHON                     the gfx-deps venv interpreter
#   ais_parse_args "$@"        handles -d/--device, -o/--outdir, -s/--smoke and
#                              -h, leaves the rest in EXTRA_ARGS, exports
#                              ZE_AFFINITY_MASK and chdirs into the results
#                              directory. Sets AIS_SMOKE (0/1), AIS_SMOKE_ARGS
#                              (collect.py flags for smoke runs) and AIS_LOG
#   ais_run VAR=v cmd args...  runs a command, recording a copy-pasteable
#                              version of it and all of its output in AIS_LOG

AIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AIS_ROOT="$(cd -- "${AIS_SCRIPT_DIR}/.." && pwd)"

COLLECT_PY="${AIS_ROOT}/collector/collect.py"
COLLECT_COMM="${AIS_ROOT}/collector/network/collect_comm.sh"

AIS_RESULTS_DIR="${AIS_RESULTS_DIR:-${AIS_SCRIPT_DIR}}"

PYTHON="/opt/gfx-deps/venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python3"

# Optional DPC++ runtime override. Disabled by default; uncomment and point
# DPCPP_FIX_LIB at a runtime directory if a build needs one.
#DPCPP_FIX_LIB="${DPCPP_FIX_LIB:-}"
#if [[ -n "${DPCPP_FIX_LIB}" && -d "${DPCPP_FIX_LIB}" ]]; then
#    export LD_LIBRARY_PATH="${DPCPP_FIX_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
#fi

# collect_comm.sh shells out to the oneCCL benchmark binaries by bare name, so
# they have to be on PATH.
ONECCL_BENCH_DIR="${ONECCL_BENCH_DIR:-/workspace/intel_gpu_cri.ai.customer.poc/CRI/Alibaba/micro_benchmark/oneccl/benchmark}"
if [[ -x "${ONECCL_BENCH_DIR}/allreduce_perf" ]]; then
    export PATH="${ONECCL_BENCH_DIR}:${PATH}"
fi

# Set by the caller before ais_parse_args:
#   AIS_GPU_DEFAULT  default value for -d; empty means "leave every GPU visible"
#   AIS_GPU_HELP     one line describing what -d accepts for this script
ais_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [-d <gpu>] [-o <dir>] [-s] [extra args passed to the collector]

  -d, --device <gpu>  ${AIS_GPU_HELP:-GPU id to run on} (default: ${AIS_GPU_DEFAULT:-all})
  -o, --outdir <dir>  where the *_perf.txt land (default: ${AIS_RESULTS_DIR})
  -s, --smoke         sample a few cases per op instead of the full sweep
  -h, --help          show this help

Any other argument is forwarded to the collector unchanged.
EOF
}

ais_parse_args() {
    AIS_GPU="${AIS_GPU_DEFAULT:-}"
    AIS_SMOKE=0
    AIS_SMOKE_ARGS=()
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--device)
                [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
                AIS_GPU="$2"; shift 2 ;;
            -d=*|--device=*)
                AIS_GPU="${1#*=}"; shift ;;
            -o|--outdir)
                [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
                AIS_RESULTS_DIR="$2"; shift 2 ;;
            -o=*|--outdir=*)
                AIS_RESULTS_DIR="${1#*=}"; shift ;;
            -s|--smoke)
                AIS_SMOKE=1; AIS_SMOKE_ARGS=(--smoke); shift ;;
            -h|--help)
                ais_usage; exit 0 ;;
            *)
                EXTRA_ARGS+=("$1"); shift ;;
        esac
    done

    # ZE_AFFINITY_MASK renumbers the selected GPUs to 0..n-1, so the collector
    # always sees device 0.
    if [[ -n "${AIS_GPU}" ]]; then
        [[ "${AIS_GPU}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
            echo "ERROR: -d expects a GPU id or comma-separated ids, got '${AIS_GPU}'" >&2
            exit 2
        }
        export ZE_AFFINITY_MASK="${AIS_GPU}"
    fi
    local smoke_note=""
    [[ "${AIS_SMOKE}" = 1 ]] && smoke_note=" smoke=on"
    echo ">>> $(basename "$0"): ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}${smoke_note}"

    # collect.py / collect_comm.sh write *_perf.txt into the cwd and offer no
    # output-directory flag, so pin the cwd instead.
    mkdir -p "${AIS_RESULTS_DIR}" || {
        echo "ERROR: cannot create results dir '${AIS_RESULTS_DIR}'" >&2; exit 2
    }
    cd "${AIS_RESULTS_DIR}" || exit 2
    echo ">>> results -> $(pwd)"

    if [[ -z "${AIS_LOG:-}" ]]; then
        AIS_LOG="$(pwd)/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
    fi
    echo ">>> log -> ${AIS_LOG}"
}

# ais_run VAR=value ... /abs/cmd args...
# Logs the command as one paste-ready line (cwd + the exported ZE_AFFINITY_MASK,
# which is not part of argv + every argument, shell-quoted), then runs it.
ais_run() {
    local envs=()
    [[ -n "${ZE_AFFINITY_MASK:-}" ]] && envs+=("ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}")
    echo ">>> cmd: cd $(printf '%q' "$(pwd)") && env $(printf '%q ' "${envs[@]}" "$@")" |
        tee -a "${AIS_LOG}"
    env "${envs[@]}" "$@" 2>&1 | tee -a "${AIS_LOG}"
    return "${PIPESTATUS[0]}"
}
