#!/usr/bin/env python3
"""Isolation-equivalence gate: expand each op's case set twice — repo cases
(no override) vs run-local snapshot (AIC_COLLECTOR_CASES_ROOT) — and diff the
per-op case keys. Zero diff = isolation changed nothing but yaml ownership.

Expansion runs in a fresh subprocess per variant so the env override cannot
leak between them. Must run where the collector's deps are importable
(container / gfx-deps venv).

Also supports ``--check-header f1.perf.txt f2.perf.txt ...``: verifies every
perf txt carries the identical 19-column schema header.

Usage (container):
    python3 scripts/verify_isolation_equivalence.py \
        --repo /workspace \
        --snapshot <run_dir>/cases_snapshot \
        --out <run_dir>/reports/equivalence.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

OVERRIDES = "AIC_COLLECTOR_CASES_ROOT"

DEFAULT_OPS = (
    "dsv4_csa_context_module",
    "dsv4_csa_generation_module",
    "dsv4_swa_context_module",
    "dsv4_swa_generation_module",
    "dsa_context_module",
    "dsa_generation_module",
)

EXPECTED_HEADER = (
    "framework,version,device,op_name,kernel_source,model,architecture,mla_dtype,"
    "kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,compress_ratio,"
    "latency,used_cuda_graph,timing_mode"
)
# DSA module rows carry no compress_ratio by design (collect_mla_module_xpu
# item_list has no such key) -> 18 columns is the declared dsa schema.
EXPECTED_HEADER_DSA = ",".join(
    c for c in EXPECTED_HEADER.split(",") if c != "compress_ratio"
)

_CHILD = r"""
import importlib, json, sys
repo, ops = sys.argv[1], sys.argv[2].split(",")
sys.path.insert(0, repo)
from collector.vllm.registry import REGISTRY_XPU
entries = {e.op: e for e in REGISTRY_XPU}
out = {}
for op in ops:
    e = entries[op]
    get = getattr(importlib.import_module(e.module), e.get_func)
    try:
        cases = get()
    except TypeError:
        cases = get(model_path=None)
    out[op] = sorted("|".join(str(x) for x in c) for c in cases)
print("AIC_JSON:" + json.dumps(out))  # sentinel: vllm logs also go to stdout
"""


def expand(repo: Path, ops: list[str], snapshot: Path | None) -> dict[str, list[str]]:
    env = dict(os.environ)
    if snapshot is None:
        env.pop(OVERRIDES, None)
    else:
        env[OVERRIDES] = str(snapshot)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(repo), ",".join(ops)],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"expansion failed ({'snapshot' if snapshot else 'repo'}):\n{proc.stderr}")
    payload = [ln for ln in proc.stdout.splitlines() if ln.startswith("AIC_JSON:")]
    if len(payload) != 1:
        raise RuntimeError(f"expansion produced no unique JSON sentinel ({'snapshot' if snapshot else 'repo'}):\n{proc.stderr[-500:]}")
    return json.loads(payload[0][len("AIC_JSON:"):])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True,
                        help="tree containing collector/ (e.g. /workspace)")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="run-local snapshot root (AIC_COLLECTOR_CASES_ROOT layout)")
    parser.add_argument("--ops", type=str, default=",".join(DEFAULT_OPS))
    parser.add_argument("--check-header", nargs="+", type=Path, default=[],
                        help="perf txt files whose header lines must be identical and 19-col")
    parser.add_argument("--out", type=Path, default=None, help="report file (default: stdout)")
    args = parser.parse_args()

    lines: list[str] = []
    fail = False

    for perf in args.check_header:
        header = perf.read_text().splitlines()[0] if perf.exists() else ""
        expected = EXPECTED_HEADER_DSA if perf.name.startswith("dsa_") else EXPECTED_HEADER
        ok = header == expected
        fail |= not ok
        lines.append(f"[header] {perf.name}: {'PASS' if ok else 'FAIL'} ({len(header.split(','))} cols)")
        if not ok:
            lines.append(f"  expected: {expected}")
            lines.append(f"  actual:   {header}")

    if args.snapshot is not None:
        base = expand(args.repo, args.ops.split(","), None)
        iso = expand(args.repo, args.ops.split(","), args.snapshot)
        for op in args.ops.split(","):
            b, i = base.get(op, []), iso.get(op, [])
            diff_b, diff_i = set(b) - set(i), set(i) - set(b)
            ok = not diff_b and not diff_i
            fail |= not ok
            lines.append(
                f"[expand] {op}: {'PASS' if ok else 'FAIL'} "
                f"(repo={len(b)} snapshot={len(i)} removed={len(diff_b)} added={len(diff_i)})"
            )
            for k in sorted(diff_b)[:3]:
                lines.append(f"  only-in-repo: {k}")
            for k in sorted(diff_i)[:3]:
                lines.append(f"  only-in-snapshot: {k}")
    elif args.check_header:
        lines.append("[expand] skipped (--snapshot not given)")
    else:
        base = expand(args.repo, args.ops.split(","), None)
        for op, cases in base.items():
            lines.append(f"[expand] {op}: {len(cases)} cases (repo baseline, no snapshot comparison)")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
