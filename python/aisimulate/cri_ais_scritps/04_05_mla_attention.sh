#!/usr/bin/env bash
# 04_05_mla_attention.sh — DSV4 swa (mla-labeled) context+generation collection on CRI (XPU).
# Split from 04_05_10_11_attention_context_generation.sh; case labels keep the
# ops_benchmark numbering and are DISPLAY-ONLY (04 mla_context / 05
# mla_generation map to the swa module ops; collect.py always receives the
# real op names).
#
# Usage (inside the GPU container):
#   bash 04_05_mla_attention.sh                    # both cases (04 + 05), full grids
#   bash 04_05_mla_attention.sh --case 4           # one case
#   bash 04_05_mla_attention.sh --case 4,5         # both, explicit
#   bash 04_05_mla_attention.sh -d 2               # pin GPU (never 4)
#   bash 04_05_mla_attention.sh -s                 # smoke grids (skips count gate)
#   bash 04_05_mla_attention.sh -h                 # help with the case table
#
# Each case runs in its own results dir <label>_<YYYYMMDD_HHMMSS>/ (run_env
# log + perf.txt + csv staging + collector logs, checkpoint isolated per
# case). Case yaml ISOLATION (run-local snapshot, iso mode): the repo yaml is
# never touched. Each run generates <run_dir>/cases_snapshot/ via
# scripts/snapshot_cases.py and injects test_yaml/dsv4_attention.yaml over
# the snapshot's dsv4_attention.yaml (--set-yaml; provenance records the
# injected sha), then points AIC_COLLECTOR_CASES_ROOT at the snapshot —
# all four consumers honor the override. Concurrent runs of the split
# scripts sharing the repo yaml cannot pollute each other; SIGKILL leaves
# no residue.
#
# Startup gates: full mode asserts the expanded case grid equals the
# campaign-pinned counts (1274/1648/337/44) before any GPU work; GPU 4 is
# rejected (known segfault).
set -euo pipefail

# capture caller args BEFORE any sourcing: the oneAPI setvars chain leaves
# positional params behind (observed: $@ gains /opt/gfx-deps/dpcpp/lib/libintelocl.so)
USER_ARGS=("$@")

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/attention_common.sh"

AIS_GPU_DEFAULT="${GPU:-0}"
AIS_GPU_HELP="GPU id this campaign runs on (never 4 - known segfault)"

ais_att_usage() {
    cat >&2 <<EOF
Usage: bash $(basename "$0") [--case 4|5|4,5] [--all] [-d <gpu>] [-o <dir>] [-s]

Cases (labels = ops_benchmark numbering, display-only; the collector always
receives the real op name):
  04  mla_context     -> dsv4_swa_context_module
  05  mla_generation  -> dsv4_swa_generation_module
  --all               both cases (default when neither --case nor --all given)

  -d, --device <gpu>  ${AIS_GPU_HELP:-GPU id} (default: ${AIS_GPU_DEFAULT:-all})
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
        ais_att_usage
        exit 0
    fi
done

ais_parse_args "${USER_ARGS[@]}"
USER_ARGS=("${EXTRA_ARGS[@]}")

ais_att_parse_cases "04 05" "${USER_ARGS[@]}"

TEST_YAML="${AIS_SCRIPT_DIR}/test_yaml/dsv4_attention.yaml"
# iso mode: no repo-yaml deploy/restore, so the EXIT yaml cleanup is not
# wired; INT/TERM traps stay (they stop the collector child precisely).
trap 'ais_att_on_signal 130' INT
trap 'ais_att_on_signal 143' TERM

ais_att_env

# iso isolation: run-local case snapshot + AIC_COLLECTOR_CASES_ROOT override
# (shared helper; test_yaml/dsv4_attention.yaml is this script's campaign yaml).
ais_att_iso_snapshot "${TEST_YAML}" "collector/cases/base_ops/dsv4_attention.yaml" "dsv4-$(basename "$0" .sh)"

MODE="full"
[[ "${AIS_SMOKE}" = 1 ]] && MODE="smoke"
if [[ "${MODE}" = "full" ]]; then
    # P0 gate: expanded grid must equal the campaign-pinned case counts
    local_ops=()
    for n in "${AIS_ATT_CASES[@]}"; do local_ops+=("${AIS_ATT_CASE_OP[$n]}"); done
    ais_att_assert_case_counts "${local_ops[@]}"
    AIS_ATT_FLAGS=(--backend vllm --gpu cri --keep-csv --resume --resume-retry-failed)
else
    AIS_ATT_FLAGS=(--backend vllm --gpu cri --smoke --keep-csv)
fi

echo ">>> mode=${MODE} cases: ${AIS_ATT_CASES[*]}"
for n in "${AIS_ATT_CASES[@]}"; do
    ais_att_run_case "$n"
done

echo "ATT_CASES_DONE mode=${MODE} cases=${AIS_ATT_CASES[*]} run_dir=$(pwd)"
