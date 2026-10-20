#!/usr/bin/env bash
# 08_09_dsa_context_gen.sh — GLM-5.3 DSA (glm_moe_dsa) context+generation
# collection on CRI (XPU), v2 collector (collect_mla_module_xpu_v2).
#
# Modeled on 04_05_mla_attention.sh; case labels keep the ops_benchmark
# numbering and are DISPLAY-ONLY (08 dsa_context / 09 dsa_generation map to
# the dsa module ops; collect.py always receives the real op names).
#
# Usage (inside the GPU container):
#   bash 08_09_dsa_context_gen.sh                    # both cases (08 + 09), full grids
#   bash 08_09_dsa_context_gen.sh --case 8           # one case
#   bash 08_09_dsa_context_gen.sh --case 8,9         # both, explicit
#   bash 08_09_dsa_context_gen.sh -d 0               # pin GPU (never 4)
#   bash 08_09_dsa_context_gen.sh -s                 # smoke grids (skips count gate)
#   bash 08_09_dsa_context_gen.sh -h                 # help with the case table
#
# Each case runs in its own results dir <label>_<YYYYMMDD_HHMMSS>/ (run_env
# log + perf.txt + parquet + collector logs, checkpoint isolated per case).
# Case yaml ISOLATION (run-local snapshot, iso mode): the repo yaml is never
# touched. Each run generates <run_dir>/cases_snapshot/ via
# scripts/snapshot_cases.py (with SNAPSHOT_PROVENANCE.yaml), injects
# test_yaml/mla_module.yaml over the snapshot's mla_module.yaml
# (--set-yaml; provenance records the injected sha), and points
# AIC_COLLECTOR_CASES_ROOT at the snapshot — all four consumers
# (case_generator / capabilities / model_cases / provenance) honor the
# override. A fail-loud guard aborts if the collector tree does not read
# the override (old tree would silently ignore it = fake isolation).
# Concurrent runs of this and other scripts (04/05 mla) sharing
# mla_module.yaml cannot pollute each other; SIGKILL leaves no residue.
#
# Startup gates: full mode asserts the expanded case grid equals the
# campaign-pinned counts (08: 28 / 09: 32) before any GPU work; GPU 4 is
# rejected (known segfault). 08 pins the post-filter count (32 generated -
# 4 dropped by the v2 collector's 160GiB context memory-feasibility filter);
# on a device with different memory the context gate fails loudly — re-pin
# for that device, do not widen the filter.
set -euo pipefail

# capture caller args BEFORE any sourcing: the oneAPI setvars chain leaves
# positional params behind (observed: $@ gains /opt/gfx-deps/dpcpp/lib/libintelocl.so)
USER_ARGS=("$@")

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/attention_common.sh"

AIS_GPU_DEFAULT="${GPU:-0}"
AIS_GPU_HELP="GPU id this campaign runs on (never 4 - known segfault)"

ais_glm_usage() {
    cat >&2 <<EOF
Usage: bash $(basename "$0") [--case 8|9|8,9] [--all] [-d <gpu>] [-o <dir>] [-s]

Cases (labels = ops_benchmark numbering, display-only; the collector always
receives the real op name):
  08  dsa_context     -> dsa_context_module     (GLM-5.3 glm_moe_dsa, v2 collector)
  09  dsa_generation  -> dsa_generation_module
  --all               both cases (default when neither --case nor --all given)

  -d, --device <gpu>  ${AIS_GPU_HELP:-GPU id} (default: ${AIS_GPU_DEFAULT:-0})
  -o, --outdir <dir>  parent results dir; per-case subdirs land here
                      (default: script dir)
  -s, --smoke         sampled smoke grids (case-count gate skipped)
  -h, --help          this help

Each case gets its own <label>_<YYYYMMDD_HHMMSS>/ results dir. Case yaml runs
from an isolated run-local snapshot (AIC_COLLECTOR_CASES_ROOT); the repo yaml
is never touched.
EOF
}

for a in "${USER_ARGS[@]}"; do
    if [[ "$a" == "-h" || "$a" == "--help" ]]; then
        ais_glm_usage
        exit 0
    fi
done

ais_parse_args "${USER_ARGS[@]}"
USER_ARGS=("${EXTRA_ARGS[@]}")

ais_att_parse_cases "08 09" "${USER_ARGS[@]}"

# GLM-5.3 case registry (overrides the DSV4 maps from attention_common.sh;
# the ais_att_* helpers below read these maps).
declare -A AIS_ATT_CASE_OP=(
    [08]=dsa_context_module
    [09]=dsa_generation_module
)
declare -A AIS_ATT_CASE_LABEL=(
    [08]=08_dsa_context
    [09]=09_dsa_generation
)
# Campaign-pinned full-grid counts (glm53-v2-mxfp8 campaign yaml, verified in
# the 2026-10-18 grid runs: 28/32 parquet rows on the 160GiB device).
declare -A AIS_ATT_EXPECTED_COUNTS=(
    [dsa_context_module]=28
    [dsa_generation_module]=32
)

TEST_YAML="${AIS_SCRIPT_DIR}/test_yaml/mla_module.yaml"
# iso mode: no repo-yaml deploy/restore, so the EXIT yaml cleanup is not
# wired; INT/TERM traps stay (they stop the collector child precisely).
trap 'ais_att_on_signal 130' INT
trap 'ais_att_on_signal 143' TERM

# GLM-5.3 campaign env contract (run_mla_module_xpu_v2 knobs; DSV4-era knobs
# are unset so they cannot leak into this run).
ais_glm_env() {
    [[ "${ZE_AFFINITY_MASK:-}" != "4" ]] || {
        echo "ERROR: GPU 4 is forbidden (known interpreter segfault)" >&2; exit 2; }
    source /opt/gfx-deps/env.sh
    unset NEOReadDebugKeys OverrideL1CachePolicyInSurfaceStateAndStateless \
          LimitBlitterMaxHeight PostBlitCommand
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    export AIC_MLA_MODULE_PACK_TARGET_MS="${AIC_MLA_MODULE_PACK_TARGET_MS:-15}"
    export AIC_MLA_MODULE_FLUSH_CACHE_MB="${AIC_MLA_MODULE_FLUSH_CACHE_MB:-0}"
    export AIC_MLA_MODULE_DUMMY_DATA="${AIC_MLA_MODULE_DUMMY_DATA:-random}"
    unset AIC_DSV4_PREFILL_TIMING AIC_DSV4_GRAPH_POLICY AIC_DSV4_PACK_TARGET_MS \
          VLLM_XPU_MXFP8_USE_SYCLTLA DEEPKLOX_SPARSE_PREFILL_DISABLE_FUSED \
          AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW \
          AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE AIC_DSV4_FLUSH_CACHE_MB \
          AIC_VLLM_DSV4_MAX_MODEL_LEN
    export PYTHONPATH="${AIS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
}
ais_glm_env

# iso isolation: run-local case snapshot + AIC_COLLECTOR_CASES_ROOT override
# (shared helper; test_yaml/mla_module.yaml is this script's campaign yaml).
ais_att_iso_snapshot "${TEST_YAML}" "collector/cases/base_ops/mla_module.yaml" "glm53-08_09"

# Per-case results dir + run_env log + run (modeled on ais_att_run_case; the
# run_env log captures the GLM-5.3 env contract instead of the DSV4 one).
ais_glm_run_case() {
    local n="$1" label op ts dir
    [[ "${AIS_ATT_FLAGS[@]+set}" = "set" && ${#AIS_ATT_FLAGS[@]} -gt 0 ]] || {
        echo "ERROR: AIS_ATT_FLAGS not set (script bug)" >&2; exit 2; }
    label="${AIS_ATT_CASE_LABEL[$n]}"
    op="${AIS_ATT_CASE_OP[$n]}"
    ts="$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)"
    dir="${label}_${ts}"
    mkdir -p "$dir"
    cd "$dir"
    {
        env | grep -E '^(ZE_AFFINITY_MASK|VLLM_XPU_ENABLE_XPU_GRAPH|AIC_MLA_MODULE_)=' | sort
        echo "AIC_COLLECTOR_CASES_ROOT=${AIC_COLLECTOR_CASES_ROOT:-<unset, repo cases>}"
        git -C "${AIS_ROOT}" rev-parse HEAD 2>/dev/null || true
    } > "run_env_${ts}.log"
    echo "===== CASE ${n} ${label} op=${op} dir=${dir} ====="
    ais_att_run_cmd "${PYTHON}" "${COLLECT_PY}" ${AIS_ATT_FLAGS[@]} --ops "${op}"
    cd "${AIS_RESULTS_DIR}"
}


MODE="full"
[[ "${AIS_SMOKE}" = 1 ]] && MODE="smoke"
if [[ "${MODE}" = "full" ]]; then
    # P0 gate: expanded grid must equal the campaign-pinned case counts
    local_ops=()
    for n in "${AIS_ATT_CASES[@]}"; do local_ops+=("${AIS_ATT_CASE_OP[$n]}"); done
    ais_att_assert_case_counts "${local_ops[@]}"
    # No --keep-csv: the v2 collector's product is the finalized parquet
    # (perf.txt -> parquet + collection_meta.yaml at the output root);
    # --keep-csv keeps the staging CSVs and SKIPS parquet finalization.
    AIS_ATT_FLAGS=(--backend vllm --gpu cri --resume --resume-retry-failed)
else
    AIS_ATT_FLAGS=(--backend vllm --gpu cri --smoke)
fi

echo ">>> mode=${MODE} cases: ${AIS_ATT_CASES[*]}"
for n in "${AIS_ATT_CASES[@]}"; do
    ais_glm_run_case "$n"
done

echo "GLM_CASES_DONE mode=${MODE} cases=${AIS_ATT_CASES[*]} run_dir=$(pwd)"
