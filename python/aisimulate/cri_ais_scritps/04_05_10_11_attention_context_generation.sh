#!/usr/bin/env bash
# 04_05_10_11_attention_context_generation.sh — DSV4 attention module collection on CRI (XPU).
# Single self-contained entry: runs the csa/swa context+generation module ops.
# dsa ops are temporarily disabled (they need the DSA fastpath patch, which this
# script does not apply; uncomment the dsa line once the patch is in place).
#
# Usage (inside the GPU container):
#   bash 04_05_10_11_attention_context_generation.sh                      # full grids
#   bash 04_05_10_11_attention_context_generation.sh smoke                # smoke (sampled cases)
#   bash 04_05_10_11_attention_context_generation.sh smoke dsv4_swa_generation_module
#   OUT=<existing dir> bash 04_05_10_11_attention_context_generation.sh   # resume
#
# Results land in ./<outdir> under the current directory (perf.txt per op + logs).
# Test case config: test_yaml/dsv4_attention.yaml — declared snapshot of the
# active case yaml; the script fails loudly if it drifts from the repo copy.
set -euo pipefail

# capture caller args BEFORE any sourcing: the oneAPI setvars chain leaves
# positional params behind (observed: $@ gains /opt/gfx-deps/dpcpp/lib/libintelocl.so)
USER_ARGS=("$@")

AIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AIS_ROOT="$(cd -- "${AIS_SCRIPT_DIR}/.." && pwd)"

# --- environment: 20261012 campaign contract (pack off / SYCLTLA off / graph on) ---
source /opt/gfx-deps/env.sh
unset NEOReadDebugKeys OverrideL1CachePolicyInSurfaceStateAndStateless \
      LimitBlitterMaxHeight PostBlitCommand
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export AIC_DSV4_PACK_TARGET_MS=0        # pack off = CUDA baseline method
export VLLM_XPU_MXFP8_USE_SYCLTLA=0
unset DEEPKLOX_SPARSE_PREFILL_DISABLE_FUSED AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS \
      AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE \
      AIC_DSV4_PREFILL_TIMING AIC_DSV4_FLUSH_CACHE_MB AIC_VLLM_DSV4_MAX_MODEL_LEN

export ZE_AFFINITY_MASK="${GPU:-0}"     # campaign pin; never use 4 (known segfault)

# --- test case yaml: declared snapshot must match the repo's active config ---
TEST_YAML="${AIS_SCRIPT_DIR}/test_yaml/dsv4_attention.yaml"
REPO_YAML="${AIS_ROOT}/collector/cases/base_ops/dsv4_attention.yaml"
if [[ ! -f "${TEST_YAML}" ]]; then
    echo "ERROR: missing ${TEST_YAML}" >&2
    exit 2
fi
if [[ "$(sha256sum < "${TEST_YAML}" | cut -d' ' -f1)" != \
      "$(sha256sum < "${REPO_YAML}" | cut -d' ' -f1)" ]]; then
    echo "ERROR: test_yaml/dsv4_attention.yaml differs from ${REPO_YAML};" \
         "refresh the snapshot from the repo copy" >&2
    exit 2
fi

export PYTHONPATH="${AIS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
C="${AIS_ROOT}/collector/collect.py"

MODE="full"
if [[ "${USER_ARGS[0]:-}" = "smoke" ]]; then
    MODE="smoke"
    USER_ARGS=("${USER_ARGS[@]:1}")
fi

OPS_DEFAULT="dsv4_csa_context_module dsv4_csa_generation_module \
dsv4_swa_context_module dsv4_swa_generation_module"
# dsa 暂停（需 DSA fastpath patch；本脚本不打 patch）：
# OPS_DEFAULT="$OPS_DEFAULT dsa_context_module dsa_generation_module"
OPS="${USER_ARGS[*]:-${OPS_DEFAULT}}"

ts=$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
OUT="${OUT:-dsv4_attention_xpu_${MODE}_${ts}}"
mkdir -p "${OUT}" && cd "${OUT}"
{
    env | grep -E '^(ZE_AFFINITY_MASK|VLLM_XPU_ENABLE_XPU_GRAPH|AIC_DSV4_PACK_TARGET_MS|VLLM_XPU_MXFP8_USE_SYCLTLA)=' | sort
    git -C "${AIS_ROOT}" rev-parse HEAD 2>/dev/null || true
} > "run_env_${ts}.log"
echo ">>> mode=${MODE} ops: ${OPS}"
echo ">>> results -> $(pwd)"

FLAGS="--backend vllm --gpu cri --keep-csv --resume --resume-retry-failed"
if [[ "${MODE}" = "smoke" ]]; then
    FLAGS="--backend vllm --gpu cri --smoke --keep-csv"
fi

for op in ${OPS}; do
    echo "===== RUN ${op} (${MODE}) ====="
    python3 "${C}" ${FLAGS} --ops "${op}" 2>&1 | tee -a "run_${op}_${ts}.log"
done
echo "ALL_OPS_DONE mode=${MODE} run_dir=$(pwd)"
