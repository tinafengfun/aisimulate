#!/usr/bin/env bash
# Shared per-group retry helpers for 17_custom_allreduce.sh and 18_oneccl.sh.
# A group's rows are published only after a fully valid attempt; failed
# attempts stay under attempts/<group>/attempt_N (run.log, status.txt, CSV).
# Needs from the caller: common.sh sourced, AIS_ROOT, PYTHON.

RETRIES="${AIS_RETRIES:-3}"        # extra attempts per failed group, 0..3
ATTEMPT_TIMEOUT="${AIS_ATTEMPT_TIMEOUT:-660}"
NET="${AIS_ROOT}/collector/network"
FAILURES=()

retry_init() {  # retry_init OUTPUT_FILE...  (refuse a non-fresh results dir)
    [[ "${RETRIES}" =~ ^[0-3]$ ]] || { echo "AIS_RETRIES must be 0..3" >&2; exit 2; }
    [[ "${ATTEMPT_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo "AIS_ATTEMPT_TIMEOUT must be a positive integer" >&2; exit 2; }
    ROOT="$(pwd)"   # after ais_parse_args has entered the results directory
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    local f
    for f in attempts "$@"; do
        [[ ! -e "${ROOT}/${f}" ]] || { echo "Output is not fresh (${f}); use a new results directory" >&2; exit 2; }
    done
}

# validate_csv FILE KIND NP OP DTYPE: header, finite latency > 0, no duplicates, exact size coverage.
validate_csv() {
    awk -F, -v kind="$2" -v np="$3" -v op="$4" -v dt="$5" '
    function fail(m) { print m; bad = 1; exit 1 }
    { sub(/\r$/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    {
        if (NF != ncol && NR > 2) fail("Malformed row " NR)
        ncol = NF
        if (!("latency" in col) || !("message_size" in col)) fail("Missing columns")
        lat = $(col["latency"]) + 0
        if ($(col["latency"]) !~ /^[0-9.eE+-]+$/ || lat <= 0 || lat != lat || lat > 1e300) fail("Invalid latency row " NR)
        if ($(col["num_gpus"]) != np || $(col["op_name"]) != op) fail("Unexpected op or GPU count row " NR)
        key = (kind == "oneccl" ? $(col["nccl_dtype"]) : $(col["backend"])) SUBSEP $(col["message_size"])
        if (key in seen) fail("Duplicate row " NR)
        seen[key] = 1; rows++
    }
    END {
        if (bad) exit 1
        if (rows == 0) { print "Missing header or perf rows"; exit 1 }
        want = 0
        if (kind == "oneccl") {
            div = (dt == "half") ? 2 : 1
            for (i = 9; i <= 29; i++) { want++; if (!((dt SUBSEP (2^i / div)) in seen)) miss++ }
        } else {
            split("vllm_graph vllm_eager", b, " ")
            for (j = 1; j <= 2; j++) for (i = 7; i <= 29; i++) { want++; if (!((b[j] SUBSEP 2^i) in seen)) miss++ }
        }
        if (miss || rows != want) { print "Incomplete or unexpected coverage: " rows "/" want " rows"; exit 1 }
    }' "$1"
}

publish_csv() {  # publish_csv SRC DEST
    local tmp="$2.tmp"
    if [[ -f "$2" ]]; then
        [[ "$(head -n1 "$1")" == "$(head -n1 "$2")" ]] || { echo "Header mismatch: $2" >&2; return 1; }
        { cat "$2"; tail -n +2 "$1"; } > "${tmp}"
    else
        cp "$1" "${tmp}"
    fi
    mv "${tmp}" "$2"
}

# run_group NAME FILE KIND NP OP DTYPE -- COMMAND...
run_group() {
    local name="$1" file="$2" kind="$3" np="$4" op="$5" dt="$6"
    shift 7
    local attempt dir rc pid msg
    for ((attempt = 1; attempt <= RETRIES + 1; attempt++)); do
        dir="${ROOT}/attempts/${name}/attempt_${attempt}"
        mkdir -p "${dir}"
        echo ">>> ${name}: attempt ${attempt}/$((RETRIES + 1))"
        rc=0
        (cd "${dir}" && exec setsid timeout -k 10 "${ATTEMPT_TIMEOUT}" "$@" < /dev/null > run.log 2>&1) &
        pid=$!
        wait "${pid}" || rc=$?
        # Remove MPI ranks or workers orphaned by a timeout/failure.
        kill -TERM -- "-${pid}" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-${pid}" 2>/dev/null || true
        msg=
        if ((rc == 124 || rc == 137)); then
            msg="Outer timeout after ${ATTEMPT_TIMEOUT}s"
        elif ((rc != 0)); then
            msg="Collector exited with ${rc}"
        elif [[ ! -f "${dir}/${file}" ]]; then
            msg="Missing ${file}"
        else
            msg="$(validate_csv "${dir}/${file}" "${kind}" "${np}" "${op}" "${dt}")" || true
        fi
        if [[ -z "${msg}" ]]; then
            printf 'success\n' > "${dir}/status.txt"
            publish_csv "${dir}/${file}" "${ROOT}/${file}"
            echo ">>> ${name}: accepted attempt ${attempt}"
            return 0
        fi
        printf 'failed: %s\n' "${msg}" > "${dir}/status.txt"
        echo "ERROR: ${name} attempt ${attempt}: ${msg}; see ${dir}/run.log" >&2
        ((attempt > RETRIES)) || sleep 5
    done
    return 1
}


gpu_sizes() {  # sets SIZES to the world sizes (2/4/8) the visible GPUs allow
    local count n
    count="$("${PYTHON}" -c 'import torch; print(torch.xpu.device_count())')"
    SIZES=()
    for n in 2 4 8; do if ((n <= count)); then SIZES+=("${n}"); fi; done
    ((${#SIZES[@]})) || { echo "Need at least two visible GPUs, found ${count}" >&2; return 2; }
}

run_custom_groups() {  # needs SIZES
    local n
    for n in "${SIZES[@]}"; do
        run_group "custom_allreduce_np${n}" custom_allreduce_perf.txt custom "${n}" all_reduce "" -- \
            "${PYTHON}" -u -m torch.distributed.run --nproc_per_node="${n}" "${NET}/collect_all_reduce.py" \
            --backend vllm --perf-filename custom_allreduce_perf.txt ||
            FAILURES+=("custom_allreduce_np${n}")
    done
}

finish_groups() {
    printf 'failed_groups: %s\n' "${FAILURES[*]:-none}" > "${ROOT}/retry_summary.txt"
    if ((${#FAILURES[@]})); then
        echo "FAILED groups: ${FAILURES[*]}" >&2
        return 1
    fi
    echo "All groups completed with validated perf data."
}
