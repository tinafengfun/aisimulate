#!/usr/bin/env bash
# Shared setup for the cri_ais_scritps collectors. Source it, don't run it.
#
# Provides:
#   COLLECT_PY / COLLECT_COMM  absolute paths, so a script works from any cwd
#   PYTHON                     the gfx-deps venv interpreter
#   ais_parse_args "$@"        handles -d/--device, -o/--outdir and -h, leaves
#                              the rest in EXTRA_ARGS, exports ZE_AFFINITY_MASK
#                              and chdirs into the results directory
#
# Why the chdir:
#   collect.py and collect_comm.sh have no output-directory option; they write
#   their *_perf.txt next to the current working directory. Results therefore
#   used to land wherever the script happened to be invoked from and were easy
#   to lose. Pinning the cwd keeps every run's output in one place regardless
#   of the caller's directory.
#
# Why the DPC++ override below:
#   vllm_xpu_kernels is built against a newer DPC++ than the one in
#   /opt/gfx-deps/dpcpp, whose libsycl.so.9 is missing the
#   sycl::handler::setKernelLaunchProperties overload that carries the
#   super_cluster_partition property. Importing vllm_xpu_kernels therefore
#   fails with "undefined symbol", vLLM's XPU platform probe swallows that
#   error, and every task then dies with "Failed to infer device type".
#
#   /opt/gfx-deps/dpcpp-20260929-2225 is a copy of the matching runtime, taken
#   from the containerd layer
#   /mnt/hf_models/disk3/containerd/io.containerd.snapshotter.v1.overlayfs/
#       snapshots/52/fs/opt/gfx-deps/dpcpp/lib
#   It is not persistent infrastructure: if the host is reset, copy it again.
#   The real fix is to upgrade /opt/gfx-deps/dpcpp to match vllm_xpu_kernels.

AIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AIS_ROOT="$(cd -- "${AIS_SCRIPT_DIR}/.." && pwd)"

COLLECT_PY="${AIS_ROOT}/collector/collect.py"
COLLECT_COMM="${AIS_ROOT}/collector/network/collect_comm.sh"

AIS_RESULTS_DIR="${AIS_RESULTS_DIR:-${AIS_SCRIPT_DIR}}"

PYTHON="/opt/gfx-deps/venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python3"

DPCPP_FIX_LIB="${DPCPP_FIX_LIB:-/opt/gfx-deps/dpcpp-20260929-2225/lib}"
if [[ -d "${DPCPP_FIX_LIB}" ]]; then
    export LD_LIBRARY_PATH="${DPCPP_FIX_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "WARNING: ${DPCPP_FIX_LIB} not found; vLLM may fall back to" \
         "UnspecifiedPlatform and every task will fail." >&2
fi

# collect_comm.sh shells out to the oneCCL benchmark binaries by bare name, so
# they have to be on PATH. They are built in the customer POC repo rather than
# installed to /usr/local/bin on this host.
ONECCL_BENCH_DIR="${ONECCL_BENCH_DIR:-/mnt/host/Git/intel_gpu_cri.ai.customer.poc/CRI/Alibaba/micro_benchmark/oneccl/benchmark}"
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
    # always sees device 0 and multi-GPU runs see exactly the listed cards.
    if [[ -n "${AIS_GPU}" ]]; then
        [[ "${AIS_GPU}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
            echo "ERROR: -d expects a GPU id or comma-separated ids, got '${AIS_GPU}'" >&2
            exit 2
        }
        export ZE_AFFINITY_MASK="${AIS_GPU}"
    fi
    echo ">>> $(basename "$0"): ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}"

    # collect.py / collect_comm.sh write *_perf.txt into the cwd and offer no
    # output-directory flag, so pin the cwd instead. COLLECT_PY and
    # COLLECT_COMM are absolute, so they are unaffected.
    mkdir -p "${AIS_RESULTS_DIR}" || {
        echo "ERROR: cannot create results dir '${AIS_RESULTS_DIR}'" >&2; exit 2
    }
    cd "${AIS_RESULTS_DIR}" || exit 2
    echo ">>> results -> $(pwd)"
}
