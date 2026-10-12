#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
 
AIS_GPU_DEFAULT=0
AIS_GPU_HELP="GPU id this collector runs on"
ais_parse_args "$@"

# DSV4 attention modules on CRI
# Resume an interrupted run: OUT=<existing dir> ./12_13_hca_context_generation.sh

source /opt/gfx-deps/env.sh
unset NEOReadDebugKeys OverrideL1CachePolicyInSurfaceStateAndStateless LimitBlitterMaxHeight PostBlitCommand
export VLLM_XPU_ENABLE_XPU_GRAPH=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_XPU_MXFP8_USE_SYCLTLA=1 DEEPKLOX_SPARSE_MLA_SPLITD_KERNEL_ENABLED=1
unset DEEPKLOX_SPARSE_PREFILL_DISABLE_FUSED AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE AIC_DSV4_PREFILL_TIMING

export AIC_DSV4_GRAPH_POLICY=full AIC_DSV4_PACK_TARGET_MS=15 AIC_DSV4_DUMMY_DATA=random
export AIC_VLLM_DSV4_MAX_MODEL_LEN=${MML:-0} AIC_DSV4_FLUSH_CACHE_MB=${FLUSH:-0}
export PYTHONPATH=/workspace/intel_gpu_cri.ai.customer.poc/CRI/Alibaba/ops_benchmark/ais/intel-aisimulate/python/aisimulate

C=$PYTHONPATH/collector/collect.py

OPS=${OPS:-"dsv4_hca_generation_module dsv4_hca_context_module"}
ts=$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
OUT=${OUT:-dsv4_hca_attn_context_generation_xpu_$ts}
mkdir -p "$OUT" && cd "$OUT"
MODEL="deepseek-ai/DeepSeek-V4-Flash"

for op in $OPS; do
  python $C --backend vllm --gpu cri --model-path $MODEL --keep-csv --resume --resume-retry-failed --ops $op
done

MODEL="deepseek-ai/DeepSeek-V4-Pro"
for op in $OPS; do
  python $C --backend vllm --gpu cri --model-path $MODEL --keep-csv --resume --resume-retry-failed --ops $op
done
