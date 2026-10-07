# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) mHC module collector for DeepSeek-V4.

XPU port of ``collect_mhc_module.py``. vLLM-XPU ships only the NVIDIA tilelang
mHC kernels, so this collector routes pre/post through the deepklox XE3P
kernels (``deepklox.mhc_pre`` / ``deepklox.mhc_post``), mirroring the DSA
module port. The tensor layout, site fan-out and sinkhorn settings match the
NVIDIA collector so the two produce comparable rows.

Test cases are the curated DeepSeek-V4 family sweep declared under
``common_case_values.mhc.xpu`` (hidden_size 4096/7168/5120, hc_mult 4).
"""

from __future__ import annotations

__compat__ = "vllm==0.28.0"

import argparse
import gc
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_xpu_mhc_module_test_cases
from collector.helper import (
    EXIT_CODE_RESTART,
    benchmark_with_power,
    get_device_module,
    log_perf,
    xpu_graph_measure_enabled,
)
from collector.registry_types import PerfFile
from collector.vllm.utils_xpu import setup_distributed

DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_HC_MULT = 4
ARCHITECTURE = "DeepseekV4ForCausalLM"
MHC_NUM_SITES = 2
MHC_SINKHORN_ITERS = 20
MHC_EPS = 1.0e-6

# Pack pack_n op calls per capture to amortize the replay
# launch on small shapes; large shapes run pack_n=1. pack_n is sized from a
# working-set byte window.
_PACK_WINDOW_BYTES = 340_000_000
_PACK_N_MAX = 64

# Recycle the worker after a case whose residual working set exceeds this, so
# small cases share one persistent worker and only big cases pay the respawn.
_RECYCLE_BYTES = 1_000_000_000

_DEEPKLOX_KERNELS: tuple | None = None

def _deepklox_kernels():
    global _DEEPKLOX_KERNELS
    if _DEEPKLOX_KERNELS is None:
        from deepklox import mhc_post, mhc_pre

        _DEEPKLOX_KERNELS = (mhc_pre, mhc_post)
    return _DEEPKLOX_KERNELS


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    if filename is None:
        raise ValueError("filename is required")
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _init_xpu(device: str) -> None:
    setup_distributed(device)
    get_device_module().set_device(device)


def get_mhc_module_test_cases() -> list[dict]:
    cases = get_xpu_mhc_module_test_cases()
    if "--smoke" in sys.argv:
        smoke: list[dict] = []
        seen: set[tuple[str, int, int]] = set()
        for case in cases:
            phase, _num_tokens, hidden_size, hc_mult = case["params"]
            key = (phase, hidden_size, hc_mult)
            if key in seen:
                continue
            seen.add(key)
            smoke.append(
                {
                    "id": f"mhc_{phase}_hs{hidden_size}_hcm{hc_mult}_16",
                    "params": [phase, 16, hidden_size, hc_mult],
                }
            )
        return smoke
    return cases


def _default_num_tokens() -> list[int]:
    cases = get_xpu_mhc_module_test_cases()
    if not cases:
        raise RuntimeError("get_xpu_mhc_module_test_cases() returned no cases")
    return [case["params"][1] for case in cases if case["params"][0] == "pre"][: len(cases)]


def _make_mhc_tensors(num_tokens: int, hidden_size: int, hc_mult: int, *, device: str):
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(mix_hc, hc_dim, dtype=torch.float32, device=device)
    base = torch.randn(mix_hc, dtype=torch.float32, device=device)
    scale = torch.ones(3, dtype=torch.float32, device=device)
    return residual, fn, base, scale


def _mhc_pack_n(num_tokens: int, hidden_size: int, hc_mult: int) -> int:
    """Op calls to pack per graph capture: working-set bytes vs the window, pow2 1..64."""
    window = int(os.getenv("aic_mhc_launch_amortize_bytes", str(_PACK_WINDOW_BYTES)))
    per_call_bytes = MHC_NUM_SITES * num_tokens * hc_mult * hidden_size * 2  # bf16 residual
    raw = min(float(_PACK_N_MAX), max(1.0, window / max(per_call_bytes, 1)))
    return min(_PACK_N_MAX, max(1, 1 << round(math.log2(raw))))


def _mhc_pre(residual, fn, base, scale):
    mhc_pre, _ = _deepklox_kernels()
    post, comb, layer_input = mhc_pre(
        residual,
        fn,
        scale,
        base,
        MHC_EPS,
        MHC_EPS,
        MHC_EPS,
        2.0,
        MHC_SINKHORN_ITERS,
    )
    return layer_input, post, comb


def run_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    hc_mult: int = DEFAULT_HC_MULT,
    device: str = "xpu:0",
    output_path: str | None = None,
    perf_filename: str | None = None,
    num_warmup: int = 5,
    num_iterations: int = 10,
) -> list[dict]:
    _init_xpu(device)
    hidden_size = int(hidden_size)
    hc_mult = int(hc_mult)
    token_cases = list(num_tokens_cases or _default_num_tokens())
    if "--smoke" in sys.argv and num_tokens_cases is None:
        token_cases = [16]

    dev_mod = get_device_module()
    # Graph gated on VLLM_XPU_ENABLE_XPU_GRAPH (same as every XPU op). When on,
    # every shape is captured (matching the NVIDIA mHC collector).
    use_graph = xpu_graph_measure_enabled()

    results = []
    for op in ops:
        if op not in {"pre", "post"}:
            raise ValueError(f"unsupported mHC op: {op}")
        for num_tokens in token_cases:
            # Free the prior case's memory (incl. its graph pool).
            gc.collect()
            dev_mod.empty_cache()

            site_inputs = [
                _make_mhc_tensors(num_tokens, hidden_size, hc_mult, device=device) for _ in range(MHC_NUM_SITES)
            ]
            post_inputs = None
            if op == "pre":

                def kernel_func(site_inputs=site_inputs):
                    with torch.no_grad():
                        return [_mhc_pre(residual, fn, base, scale) for residual, fn, base, scale in site_inputs]

            else:
                _, mhc_post = _deepklox_kernels()
                with torch.no_grad():
                    post_inputs = [
                        (_mhc_pre(residual, fn, base, scale), residual) for residual, fn, base, scale in site_inputs
                    ]
                dev_mod.synchronize()

                def kernel_func(post_inputs=post_inputs, mhc_post=mhc_post):
                    with torch.no_grad():
                        return [mhc_post(x, residual, post, comb) for (x, post, comb), residual in post_inputs]

            # Pack pack_n calls per capture to amortize the launch; small shapes
            # pack >1, large shapes resolve to pack_n=1. benchmark_with_power
            # divides elapsed by repeat_n so the logged latency is per call.
            pack_n = _mhc_pack_n(num_tokens, hidden_size, hc_mult) if use_graph else 1
            case_graph = use_graph
            try:
                with benchmark_with_power(
                    device=torch.device(device),
                    kernel_func=kernel_func,
                    num_warmups=num_warmup,
                    num_runs=num_iterations,
                    repeat_n=pack_n,
                    use_cuda_graph=case_graph,
                    allow_graph_fail=False,
                ) as result:
                    pass
                latency = float(result["latency_ms"])
            finally:
                # Drop refs so the next reclaim can free the memory.
                site_inputs = None
                post_inputs = None
                kernel_func = None

            log_perf(
                item_list=[
                    {
                        "architecture": ARCHITECTURE,
                        "num_tokens": num_tokens,
                        "num_sites": MHC_NUM_SITES,
                        "hc_mult": hc_mult,
                        "hidden_size": hidden_size,
                        "sinkhorn_iters": MHC_SINKHORN_ITERS,
                        "latency": f"{latency:.4f}",
                        "used_cuda_graph": result["used_cuda_graph"],
                    }
                ],
                framework="VLLM",
                version=vllm_version,
                device_name=dev_mod.get_device_name(device),
                op_name=op,
                kernel_source=f"deepklox.mhc_{op}",
                perf_filename=_resolve_perf_path(output_path, perf_filename or PerfFile.MHC_MODULE.value),
                power_stats=result.get("power_stats"),
            )
            print(
                f"[vllm-xpu-mhc] op={op} tokens={num_tokens} hs={hidden_size} "
                f"pack_n={pack_n} graph={result['used_cuda_graph']} latency={latency:.4f} ms"
            )
            results.append({"op": op, "num_tokens": num_tokens, "latency": latency})
    gc.collect()
    dev_mod.empty_cache()
    return results


def run_mhc_module_worker(
    op: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    *,
    perf_filename: str,
    device: str = "xpu:0",
) -> None:
    """Worker-compatible positional wrapper used by collector/collect.py."""
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    run_mhc_module(
        ops=[op],
        num_tokens_cases=[num_tokens],
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
        num_warmup=3 if "--smoke" in sys.argv else 5,
        num_iterations=3 if "--smoke" in sys.argv else 10,
    )
    # Big cases retain/fragment device memory, so recycle the worker after them
    # (exit with the restart code; collect.py respawns a fresh one). Small cases
    # return normally and keep the same worker to avoid respawn cost.
    residual_bytes = MHC_NUM_SITES * num_tokens * hc_mult * hidden_size * 2
    if residual_bytes >= _RECYCLE_BYTES:
        sys.exit(EXIT_CODE_RESTART)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM-XPU DeepSeek-V4 mHC module latency.")
    parser.add_argument("--op", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--num-tokens", default="16")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--hc-mult", type=int, default=DEFAULT_HC_MULT)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    run_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens),
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
