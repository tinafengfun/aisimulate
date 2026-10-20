#!/usr/bin/env bash
# attention_common.sh — shared machinery for the split DSV4 attention scripts
# (04_05_attention.sh / 10_11_attention.sh). Source AFTER common.sh.
#
# Why a separate file instead of extending common.sh: common.sh is shared
# delivery infrastructure used by the whole ops_benchmark suite (01~26_27) and
# the delivery rule is to not modify other people's files — the case machinery
# lives here additively and is only sourced by the split scripts.
#
# Provides:
#   ais_att_parse_cases ALLOWED args...  -> AIS_ATT_CASES (case ids, ordered,
#                                           deduped; no args = ALLOWED order)
#   ais_att_env                          -> campaign env contract + GPU!=4 guard
#   ais_att_iso_snapshot TEST REL TAG    -> run-local cases snapshot + yaml
#                                           injection + AIC_COLLECTOR_CASES_ROOT
#                                           (iso mode; repo yaml never touched)
#   ais_att_yaml_deploy TEST REPO        -> backup-once + overwrite + sha verify
#   ais_att_yaml_restore REPO            -> restore + sha evidence (idempotent)
#   ais_att_assert_case_counts ops...    -> full-grid expansion == campaign pins
#   ais_att_run_cmd cmd...               -> collector exec, pid recorded for signals
#   ais_att_run_case N                   -> per-case results dir + run_env + run
#   ais_att_on_signal RC                 -> INT/TERM: stop collector, exit RC
#   ais_att_cleanup                      -> EXIT trap: yaml restore, rc preserved
#
# Case registry — labels keep the ops_benchmark numbering and are DISPLAY-ONLY
# (results-dir names and logs); collect.py always receives the real op names.
# 04/05 are the mla-labeled cases and map to the swa module ops.
declare -A AIS_ATT_CASE_OP=(
    [04]=dsv4_swa_context_module
    [05]=dsv4_swa_generation_module
    [10]=dsv4_csa_context_module
    [11]=dsv4_csa_generation_module
)
declare -A AIS_ATT_CASE_LABEL=(
    [04]=04_mla_context
    [05]=05_mla_generation
    [10]=10_csa_context
    [11]=11_csa_generation
)
# Campaign-pinned full-grid counts (20261012 case yaml; re-verified 2026-10-13
# in the ww42 container under the contract env). Mismatch = wrong case set =
# abort before any GPU work (P0 gate). Only valid under the contract env and
# with the campaign yaml deployed: this tree's SYCLTLA knob defaults to 1 when
# unset, which inflates the memory estimate and silently shrinks the grid.
declare -A AIS_ATT_EXPECTED_COUNTS=(
    [dsv4_swa_context_module]=337
    [dsv4_swa_generation_module]=44
    [dsv4_csa_context_module]=1274
    [dsv4_csa_generation_module]=1648
)

ais_att_parse_cases() {
    local allowed="$1"; shift
    AIS_ATT_CASES=()
    if [[ $# -eq 0 ]]; then
        local c
        for c in $allowed; do AIS_ATT_CASES+=("$c"); done
        return 0
    fi
    local seen="" c raw
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)
                AIS_ATT_CASES=()
                for c in $allowed; do AIS_ATT_CASES+=("$c"); done ;;
            --case)
                [[ $# -ge 2 ]] || {
                    echo "ERROR: --case needs a value (e.g. --case 4 or --case 4,5)" >&2; exit 2; }
                raw="$2"
                IFS=',' read -ra _cs <<< "$raw"
                for c in "${_cs[@]}"; do
                    c="${c//[[:space:]]/}"
                    [[ "$c" =~ ^[0-9]+$ ]] || {
                        echo "ERROR: invalid case number '$c' (valid: ${allowed// /, })" >&2; exit 2; }
                    printf -v c '%02d' "$((10#$c))"
                    [[ " $allowed " == *" $c "* ]] || {
                        echo "ERROR: case $c is not in this script (valid: ${allowed// /, });" \
                             "check the other split script" >&2; exit 2; }
                    [[ "$seen" != *"|$c|"* ]] || continue
                    seen+="|$c|"
                    AIS_ATT_CASES+=("$c")
                done
                shift ;;
            *)
                echo "ERROR: unknown argument '$1' (valid: --case N[,M] / --all)" >&2
                exit 2 ;;
        esac
        shift
    done
    [[ ${#AIS_ATT_CASES[@]} -gt 0 ]] || {
        echo "ERROR: no case selected (valid: ${allowed// /, })" >&2; exit 2; }
}

ais_att_env() {
    [[ "${ZE_AFFINITY_MASK:-}" != "4" ]] || {
        echo "ERROR: GPU 4 is forbidden (known interpreter segfault)" >&2; exit 2; }
    source /opt/gfx-deps/env.sh
    unset NEOReadDebugKeys OverrideL1CachePolicyInSurfaceStateAndStateless \
          LimitBlitterMaxHeight PostBlitCommand
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    export AIC_DSV4_PACK_TARGET_MS=0        # pack off = CUDA baseline method
    # MUST be explicit: this delivery tree reads SYCLTLA as ON when the var is
    # unset, which inflates the context memory estimate and pre-skips shapes
    # at case expansion (silently runs fewer cases than the campaign grid).
    export VLLM_XPU_MXFP8_USE_SYCLTLA=0
    unset DEEPKLOX_SPARSE_PREFILL_DISABLE_FUSED AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS \
          AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE \
          AIC_DSV4_PREFILL_TIMING AIC_DSV4_FLUSH_CACHE_MB AIC_VLLM_DSV4_MAX_MODEL_LEN
    export PYTHONPATH="${AIS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
}

# iso isolation (preferred): materialize a run-local cases snapshot and point
# AIC_COLLECTOR_CASES_ROOT at it, injecting the script's own campaign yaml
# over the snapshot copy. The collector's case-yaml contract is a full cases
# TREE (base_ops + capabilities.yaml + denylist.yaml + models), so the env
# var selects a root, not a single file — the snapshot (~1s) is how a per-
# script test_yaml participates in it. Requires scripts/snapshot_cases.py
# in this delivery package. Call AFTER ais_parse_args (run dir exists) and
# the env contract; the count gate under the override then validates the
# snapshot itself.
ais_att_iso_snapshot() {
    local test_yaml="$1" rel="$2" tag="$3" snap t r
    snap="${AIS_RESULTS_DIR}/cases_snapshot"
    # fail-loud: the override must be honored by the collector tree, else an
    # old tree silently reads the repo cases = fake isolation.
    grep -rq "AIC_COLLECTOR_CASES_ROOT" "${AIS_ROOT}/collector/case_generator.py" \
        "${AIS_ROOT}/collector/capabilities.py" || {
        echo "ERROR: collector tree does not honor AIC_COLLECTOR_CASES_ROOT " \
             "(old tree; refusing to fake isolation)" >&2; exit 2; }
    "${PYTHON}" "${AIS_SCRIPT_DIR}/scripts/snapshot_cases.py" \
        --repo "${AIS_ROOT}" --out "${snap}" \
        --tag "${tag}-$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)" \
        --set-yaml "${rel}=${test_yaml}" || exit 2
    t="$(sha256sum < "${test_yaml}" | cut -d' ' -f1)"
    r="$(sha256sum < "${snap}/${rel}" | cut -d' ' -f1)"
    [[ "$t" == "$r" ]] || { echo "ERROR: snapshot yaml injection mismatch" >&2; exit 2; }
    export AIC_COLLECTOR_CASES_ROOT="${snap}"
    echo ">>> case yaml isolated: AIC_COLLECTOR_CASES_ROOT=${snap}"
}

ais_att_yaml_deploy() {
    local test_yaml="$1" repo_yaml="$2"
    [[ -f "$test_yaml" ]] || { echo "ERROR: missing $test_yaml" >&2; exit 2; }
    [[ -f "$repo_yaml" ]] || { echo "ERROR: missing $repo_yaml" >&2; exit 2; }
    local t r
    t="$(sha256sum < "$test_yaml" | cut -d' ' -f1)"
    r="$(sha256sum < "$repo_yaml" | cut -d' ' -f1)"
    if [[ "$t" != "$r" ]]; then
        [[ -f "${repo_yaml}.pristine-bak" ]] || {
            cp -p "$repo_yaml" "${repo_yaml}.pristine-bak"
            echo ">>> backed up repo yaml -> ${repo_yaml}.pristine-bak"; }
        cp "$test_yaml" "$repo_yaml"
        echo ">>> deployed $(basename "$test_yaml") over $repo_yaml"
    else
        echo ">>> repo yaml already matches test yaml (no deploy needed)"
    fi
    r="$(sha256sum < "$repo_yaml" | cut -d' ' -f1)"
    [[ "$t" == "$r" ]] || { echo "ERROR: yaml deploy verification failed" >&2; exit 2; }
}

ais_att_yaml_restore() {
    local repo_yaml="$1"
    local bak="${repo_yaml}.pristine-bak"
    if [[ ! -f "$bak" ]]; then
        echo ">>> yaml restore: no backup present, repo yaml untouched"
        return 0
    fi
    local b r
    b="$(sha256sum < "$bak" | cut -d' ' -f1)"
    r="$(sha256sum < "$repo_yaml" | cut -d' ' -f1)"
    if [[ "$r" == "$b" ]]; then
        echo ">>> yaml restore OK: repo yaml already matches pristine-bak (sha256 ${b:0:12}...)"
        return 0
    fi
    cp -p "$bak" "$repo_yaml"
    r="$(sha256sum < "$repo_yaml" | cut -d' ' -f1)"
    if [[ "$r" == "$b" ]]; then
        echo ">>> yaml restore OK: $repo_yaml restored from pristine-bak (sha256 ${b:0:12}...)"
    else
        echo "ERROR: yaml restore verification FAILED (repo yaml != pristine-bak)" >&2
        return 1
    fi
}

ais_att_assert_case_counts() {
    # Full-grid expansion must equal the campaign-pinned counts (P0 gate).
    # Runs under the contract env with the campaign yaml already deployed.
    local exp="{" first=1 op
    for op in "$@"; do
        [[ -n "${AIS_ATT_EXPECTED_COUNTS[$op]:-}" ]] || {
            echo "ERROR: no pinned case count for $op" >&2; exit 2; }
        [[ $first -eq 1 ]] || exp+=","
        exp+="\"${op}\":${AIS_ATT_EXPECTED_COUNTS[$op]}"
        first=0
    done
    exp+="}"
    "${PYTHON}" - "${AIS_ROOT}" "$exp" <<'AIC_PROBE'
import importlib, json, sys
expected = json.loads(sys.argv[2])
sys.path.insert(0, sys.argv[1])
from collector.vllm.registry import REGISTRY_XPU
entries = {e.op: e for e in REGISTRY_XPU}
bad = []
for op, want in expected.items():
    e = entries[op]
    get = getattr(importlib.import_module(e.module), e.get_func)
    try:
        cases = get()
    except TypeError:
        cases = get(model_path=None)
    got = len(cases)
    print(f"[case-count] {op}: expanded={got} expected={want} "
          f"{'OK' if got == want else 'MISMATCH'}", flush=True)
    if got != want:
        bad.append(f"{op}: expanded {got} != campaign {want}")
if bad:
    print("ERROR: case-count assertion FAILED: " + "; ".join(bad),
          file=sys.stderr)
    sys.exit(2)
print("[case-count] PASS")
AIC_PROBE
}

# Run the collector with its pid recorded so INT/TERM handlers can stop it
# precisely (signal semantics identical to Ctrl+C on the process). Output
# goes to the run log; the paste-ready command line is logged first.
ais_att_run_cmd() {
    local envs=()
    [[ -n "${ZE_AFFINITY_MASK:-}" ]] && envs+=("ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}")
    echo ">>> cmd: cd $(printf '%q' "$(pwd)") && env $(printf '%q ' "${envs[@]}" "$@")" |
        tee -a "${AIS_LOG}"
    env "${envs[@]}" "$@" >> "${AIS_LOG}" 2>&1 &
    AIS_ATT_CHILD_PID=$!
    local rc=0
    wait "${AIS_ATT_CHILD_PID}" || rc=$?
    AIS_ATT_CHILD_PID=""
    return "$rc"
}

ais_att_run_case() {
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
        env | grep -E '^(ZE_AFFINITY_MASK|VLLM_XPU_ENABLE_XPU_GRAPH|AIC_DSV4_PACK_TARGET_MS|VLLM_XPU_MXFP8_USE_SYCLTLA)=' | sort
        git -C "${AIS_ROOT}" rev-parse HEAD 2>/dev/null || true
    } > "run_env_${ts}.log"
    echo "===== CASE ${n} ${label} op=${op} dir=${dir} ====="
    ais_att_run_cmd "${PYTHON}" "${COLLECT_PY}" ${AIS_ATT_FLAGS[@]} --ops "${op}"
    cd "${AIS_RESULTS_DIR}"
}

ais_att_on_signal() {
    echo ">>> signal received (rc=$1), stopping collector pid=${AIS_ATT_CHILD_PID:-none}" |
        tee -a "${AIS_LOG}"
    if [[ -n "${AIS_ATT_CHILD_PID:-}" ]] && kill -0 "${AIS_ATT_CHILD_PID}" 2>/dev/null; then
        kill -INT "${AIS_ATT_CHILD_PID}" 2>/dev/null || true
        local i
        for i in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "${AIS_ATT_CHILD_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${AIS_ATT_CHILD_PID}" 2>/dev/null || true
    fi
    exit "$1"
}

ais_att_cleanup() {
    local rc=$?
    ais_att_yaml_restore "${AIS_ATT_REPO_YAML}" ||
        echo "WARNING: yaml restore failed on exit; manual fix: cp ${AIS_ATT_REPO_YAML}.pristine-bak ${AIS_ATT_REPO_YAML}" >&2
    exit "$rc"
}
