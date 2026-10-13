#!/bin/bash
# run_dsv4.sh — DSV4 attention module collection entry (CRI/XPU).
# Lives inside the repo clone at python/aisimulate/cri_ais_scritps/.
#
# Usage (inside a GPU container):
#   bash run_dsv4.sh                              # full grid, all 6 ops
#   bash run_dsv4.sh dsv4_swa_generation_module   # full grid, selected op(s)
#   bash run_dsv4.sh smoke                        # smoke: sampled cases per op
#   bash run_dsv4.sh smoke dsv4_csa_context_module
#   OUT=<existing dir> bash run_dsv4.sh           # resume an interrupted run dir
#
# Env contract = 20261012 campaign pins (pack off / SYCLTLA off / graph on /
# ZE_AFFINITY_MASK=0); case set = AIC_COLLECTOR_CASES_ROOT if exported, else
# the repo's cases dir as-is.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/.." && pwd)"   # python/aisimulate (package root: collector/ lives here)
USER_ARGS=("$@")

MODE="full"
if [ "${USER_ARGS[0]:-}" = "smoke" ]; then
  MODE="smoke"
  USER_ARGS=("${USER_ARGS[@]:1}")
fi

source /opt/gfx-deps/env.sh
export ZE_AFFINITY_MASK=0   # campaign pin; never use 4 (segfault on some devices)
unset NEOReadDebugKeys OverrideL1CachePolicyInSurfaceStateAndStateless \
      LimitBlitterMaxHeight PostBlitCommand
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export AIC_DSV4_PACK_TARGET_MS=0          # pack off = CUDA baseline method
export VLLM_XPU_MXFP8_USE_SYCLTLA=0
unset AIC_DSV4_PREFILL_TIMING AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS \
      AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE \
      AIC_DSV4_FLUSH_CACHE_MB AIC_VLLM_DSV4_MAX_MODEL_LEN

OPS_DEFAULT="dsv4_swa_generation_module dsv4_swa_context_module \
dsa_context_module dsa_generation_module \
dsv4_csa_generation_module dsv4_csa_context_module"
OPS="${USER_ARGS[*]:-$OPS_DEFAULT}"

# dsa ops need the fastpath patch in the installed vllm
case " $OPS " in *" dsa_"*)
  TARGET="$(python3 -c 'import vllm, os; print(os.path.join(os.path.dirname(os.path.dirname(vllm.__file__)), "vllm", "v1", "attention", "backends", "mla", "xpu_mla_sparse.py"))' 2>/dev/null)" || TARGET=""
  if [ -z "$TARGET" ] || ! grep -q "sparse_mla_prefill" "$TARGET" 2>/dev/null; then
    echo "[run_dsv4] FATAL: dsa ops need the DSA fastpath patch in $TARGET" >&2
    echo "[run_dsv4] apply the fastpath patch to the installed vllm first" >&2
    exit 1
  fi ;;
esac

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ts=$(date +%F-%H-%M-%S)
OUT="${OUT:-$PWD/dsv4_run_${MODE}_${ts}}"
mkdir -p "$OUT" && cd "$OUT" || exit 1
{
  env | grep -E '^(ZE_AFFINITY_MASK|VLLM_XPU_ENABLE_XPU_GRAPH|AIC_DSV4_PACK_TARGET_MS|VLLM_XPU_MXFP8_USE_SYCLTLA|AIC_COLLECTOR_CASES_ROOT)=' | sort
  git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
} > "run_env_${ts}.log"
cat "run_env_${ts}.log"

FLAGS="--backend vllm --gpu cri --keep-csv --resume --resume-retry-failed"
if [ "$MODE" = "smoke" ]; then
  FLAGS="--backend vllm --gpu cri --smoke --keep-csv"
fi

for op in $OPS; do
  echo "===== RUN $op ($MODE) ====="
  python3 "$REPO_ROOT/collector/collect.py" $FLAGS --ops "$op" 2>&1 | tee -a "run_${op}_${ts}.log"
done
echo "ALL_OPS_DONE mode=$MODE run_dir=$OUT"
