#!/usr/bin/env bash
# Shared setup for the cri_ais_scritps collectors. Source it, don't run it.
#
# Provides:
#   COLLECT_PY / COLLECT_COMM  absolute paths, so a script works from any cwd
#   PYTHON                     the gfx-deps venv interpreter
#   ais_parse_args "$@"        handles -d/--device, -o/--outdir and -h, leaves
#                              the rest in EXTRA_ARGS, exports ZE_AFFINITY_MASK
#                              and chdirs into the results directory

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
Usage: $(basename "$0") [-d <gpu>] [-o <dir>] [extra args passed to the collector]

  -d, --device <gpu>  ${AIS_GPU_HELP:-GPU id to run on} (default: ${AIS_GPU_DEFAULT:-all})
  -o, --outdir <dir>  where the *_perf.txt land (default: ${AIS_RESULTS_DIR})
  -h, --help          show this help

Any other argument is forwarded to the collector unchanged.
EOF
}

ais_parse_args() {
    AIS_GPU="${AIS_GPU_DEFAULT:-}"
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
    echo ">>> $(basename "$0"): ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}"

    # collect.py / collect_comm.sh write *_perf.txt into the cwd and offer no
    # output-directory flag, so pin the cwd instead.
    mkdir -p "${AIS_RESULTS_DIR}" || {
        echo "ERROR: cannot create results dir '${AIS_RESULTS_DIR}'" >&2; exit 2
    }
    cd "${AIS_RESULTS_DIR}" || exit 2
    echo ">>> results -> $(pwd)"
}
