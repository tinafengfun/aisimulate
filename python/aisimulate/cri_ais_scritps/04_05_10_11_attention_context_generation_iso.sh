#!/usr/bin/env bash
# 04_05_10_11_attention_context_generation_iso.sh — testcase-YAML-isolated DSV4
# attention collection (delivery pilot for the run-local case-set mechanism).
#
# Difference vs 04_05_10_11_attention_context_generation_tina.sh: this script
# NEVER deploys over the shared repo yaml. Each run gets its own workspace:
#
#   runs/run_<ts>/cases_snapshot/   run-local case set (SNAPSHOT_PROVENANCE.yaml)
#   runs/run_<ts>/raw/              OUT: perf txt / csv / .collector_checkpoint
#   runs/run_<ts>/logs/             runner + per-op logs + run_env contract log
#   runs/run_<ts>/reports/          isolation-equivalence gate output
#
# The collector reads the run-local snapshot via AIC_COLLECTOR_CASES_ROOT
# (case_generator / capabilities / model_cases / provenance all honor it).
# Shared state (repo yaml, global checkpoints) is never touched; two runs of
# the same op in different run dirs cannot contaminate each other.
#
# Usage (inside the GPU container):
#   bash 04_05_10_11_attention_context_generation_iso.sh                 # smoke, 6 default ops
#   bash 04_05_10_11_attention_context_generation_iso.sh full            # full grids
#   bash 04_05_10_11_attention_context_generation_iso.sh dsv4_csa_context_module
#   GPU=0 bash 04_05_10_11_attention_context_generation_iso.sh smoke dsa_context_module
set -euo pipefail

# capture caller args BEFORE any sourcing: the oneAPI setvars chain leaves
# positional params behind (observed: $@ gains /opt/gfx-deps/dpcpp/lib/libintelocl.so)
USER_ARGS=("$@")

AIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AIS_ROOT="$(cd -- "${AIS_SCRIPT_DIR}/.." && pwd)"

# --- environment first: gates below need the collector deps importable ---
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

# --- fail-loud: the collector tree must understand the cases-root override ---
# (an old tree silently ignores the env var -> fake isolation -> wrong case set)
if ! grep -q "AIC_COLLECTOR_CASES_ROOT" "${AIS_ROOT}/collector/case_generator.py"; then
    echo "ERROR: ${AIS_ROOT}/collector/case_generator.py has no AIC_COLLECTOR_CASES_ROOT" \
         "support; isolated runs need a collector that honors the override" >&2
    exit 2
fi

MODE="smoke"
if [[ "${USER_ARGS[0]:-}" = "smoke" || "${USER_ARGS[0]:-}" = "full" ]]; then
    MODE="${USER_ARGS[0]}"
    USER_ARGS=("${USER_ARGS[@]:1}")
fi

OPS_DEFAULT="dsv4_csa_context_module dsv4_csa_generation_module \
dsv4_swa_context_module dsv4_swa_generation_module \
dsa_context_module dsa_generation_module"
OPS="${USER_ARGS[*]:-${OPS_DEFAULT}}"

# --- fail-loud: dsa needs the fastpath patch; refuse instead of failing mid-GPU ---
if [[ "${OPS}" = *"dsa_"* ]]; then
    VLLM_DIR="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)' 2>/dev/null || true)"
    SPARSE_PY="${VLLM_DIR}/v1/attention/backends/mla/xpu_mla_sparse.py"
    if [[ -z "${VLLM_DIR}" || ! -f "${SPARSE_PY}" ]] \
       || ! grep -q "xpu_ops.sparse_mla_prefill" "${SPARSE_PY}"; then
        echo "ERROR: dsa ops requested but the DSA fastpath patch is not in" \
             "(${SPARSE_PY:-vllm not importable}); apply the patch first" >&2
        exit 2
    fi
fi

ts=$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
RUN_DIR="${AIS_SCRIPT_DIR}/runs/run_${ts}"
mkdir -p "${RUN_DIR}/raw" "${RUN_DIR}/logs" "${RUN_DIR}/reports"

# --- run-local case snapshot: the ONLY case source for this run ---
python3 "${AIS_SCRIPT_DIR}/scripts/snapshot_cases.py" \
    --repo "${AIS_ROOT}" --out "${RUN_DIR}/cases_snapshot" --tag "iso-pilot-${ts}"
export AIC_COLLECTOR_CASES_ROOT="${RUN_DIR}/cases_snapshot"

# --- isolation-equivalence gate (P4 case assertion, run-local edition) ---
# case keys from the run-local snapshot must equal the repo-cases expansion.
python3 "${AIS_SCRIPT_DIR}/scripts/verify_isolation_equivalence.py" \
    --repo "${AIS_ROOT}" --snapshot "${RUN_DIR}/cases_snapshot" \
    --out "${RUN_DIR}/reports/equivalence_pre_run.txt"

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

export PYTHONPATH="${AIS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
C="${AIS_ROOT}/collector/collect.py"

{
    env | grep -E '^(ZE_AFFINITY_MASK|VLLM_XPU_ENABLE_XPU_GRAPH|AIC_DSV4_PACK_TARGET_MS|VLLM_XPU_MXFP8_USE_SYCLTLA|AIC_COLLECTOR_CASES_ROOT)=' | sort
    git -C "${AIS_ROOT}" rev-parse HEAD 2>/dev/null || true
    head -6 "${RUN_DIR}/cases_snapshot/SNAPSHOT_PROVENANCE.yaml"
} > "${RUN_DIR}/logs/run_env_${ts}.log"

echo ">>> mode=${MODE} ops: ${OPS}"
echo ">>> run dir: ${RUN_DIR} (case set isolated; repo yaml untouched)"

FLAGS="--backend vllm --gpu cri --keep-csv --resume --resume-retry-failed"
if [[ "${MODE}" = "smoke" ]]; then
    FLAGS="--backend vllm --gpu cri --smoke --keep-csv"
fi

for op in ${OPS}; do
    echo "===== RUN ${op} (${MODE}) ====="
    (cd "${RUN_DIR}/raw" && python3 "${C}" ${FLAGS} --ops "${op}") \
        2>&1 | tee -a "${RUN_DIR}/logs/run_${op}_${ts}.log"
done
echo "ALL_OPS_DONE mode=${MODE} run_dir=${RUN_DIR}"
