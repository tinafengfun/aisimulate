# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) Qwen3.5 Gated DeltaNet collector.

Times the two production core ops separately, matching the NVIDIA collector:
the packed-QKV depthwise convolution and the gated-delta-rule scan. Context
uses ``causal_conv1d_fn`` + ``chunk_gated_delta_rule``; generation uses
``causal_conv1d_update`` + ``gated_delta_rule_decode``. Input/output projection
GEMMs stay with the GEMM collector.

The per-op GDN primitives are imported from vLLM-XPU's
``mamba.gdn.deepklox_interface``, which dispatches each op to DeepKLOX for now
"""

__compat__ = "vllm==0.28.0"

import gc
import math
import os

import torch
from vllm.model_executor.layers.mamba.gdn.deepklox_interface import (
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_gated_delta_rule,
    gated_delta_rule_decode,
    gdn_prefill_backend,
)
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_xpu_gdn_test_cases
from collector.helper import (
    benchmark_with_power,
    get_device_module,
    log_perf,
    xpu_graph_measure_enabled,
)

aic_debug = int(os.getenv("aic_gdn_debug", "0"))  # noqa: SIM112

# Kernel grid-y = (context) num_tokens/128 * num_v_heads, (decode) batch * num_v_heads.
# Hardware caps grid-y at 65535 (NV collect_gdn.py raises; XPU overflow = uncatchable
# DEVICE_LOST that kills the worker), so pre-skip offending shapes.
_GRID_Y_LIMIT = int(os.getenv("aic_gdn_grid_y_limit", str(65535)))  # noqa: SIM112
_CTX_CHUNK = 128  # chunk length of the context (prefill) scan kernel
_CTX_SEQ_MULTIPLE = 64  # Xe3p prefill conv requires each sequence length divisible by 64
_CONV_ELEM_LIMIT = 2**31  # packed conv (num_tokens*conv_dim) element ceiling; NV int32-offset parity

# Per COLLECTOR_SYSTEM: CRI needs the larger window, b60 keeps the smaller default.
_LAUNCH_AMORTIZE_BYTES = {"cri": 340_000_000}
_LAUNCH_AMORTIZE_BYTES_DEFAULT = 64_000_000
_PACK_N_MAX = 64


def get_gdn_test_cases():
    """CRI GDN cases (curated model set) as positional per-case lists."""
    return [
        [
            case.phase,
            case.d_model,
            case.d_conv,
            case.num_k_heads,
            case.head_k_dim,
            case.num_v_heads,
            case.head_v_dim,
            case.batch_size_list,
            case.seq_len_list,
            case.model_name,
        ]
        for case in get_xpu_gdn_test_cases()
    ]


def _conv_dim(num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    """Packed-QKV convolution width: q and k each span key_dim, v spans value_dim."""
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    return 2 * key_dim + value_dim


def _grid_y(phase, batch_size, num_tokens, num_v_heads):
    """Kernel grid-y launch extent for this shape (see _GRID_Y_LIMIT note)."""
    if phase == "context":
        # chunk scan: one grid-y row per (chunk, v-head)
        return (num_tokens // _CTX_CHUNK) * num_v_heads
    # packed recurrent decode: one grid-y row per (request, v-head)
    return batch_size * num_v_heads


def _decode_pack_n(common_log_data):
    """Kernels to pack per decode graph: fp32 ssm-state bytes vs amortize threshold, pow2 1..64."""
    system = os.environ.get("COLLECTOR_SYSTEM") or None
    amortize = _LAUNCH_AMORTIZE_BYTES.get(system, _LAUNCH_AMORTIZE_BYTES_DEFAULT)
    state_bytes = (
        common_log_data["batch_size"]
        * common_log_data["num_v_heads"]
        * common_log_data["head_v_dim"]
        * common_log_data["head_k_dim"]
        * 4  # ssm_state is fp32
    )
    raw = min(float(_PACK_N_MAX), max(1.0, amortize / max(state_bytes, 1)))
    return min(_PACK_N_MAX, max(1, 1 << round(math.log2(raw))))


def _benchmark(kernel_func, device, common_log_data, kernel_source, perf_filename, *, phase):
    """Time one op and log a row. Context: eager, repeat_n=10 (NV parity). Generation: graph, pack_n per replay."""
    use_graph = xpu_graph_measure_enabled() and phase != "context"

    if use_graph:
        pack_n = _decode_pack_n(common_log_data)
        repeat_n = 1

        def run():
            for _ in range(pack_n):
                kernel_func()
    else:
        # eager context: NV-style back-to-back amortization, no graph packing
        pack_n = 1
        repeat_n = 10
        run = kernel_func

    kernel_func()
    get_device_module().synchronize()
    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=3,
        num_runs=10,
        repeat_n=repeat_n,
        use_cuda_graph=use_graph,
        allow_graph_fail=False,  # when graphing (generation), capture failure fails the case
    ) as results:
        log_perf(
            item_list=[
                {
                    **common_log_data,
                    "latency": results["latency_ms"] / pack_n,
                    "used_cuda_graph": results["used_cuda_graph"],
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=get_device_module().get_device_name(),
            op_name="gdn",
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )


def _run_phase(
    *,
    phase,
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    seq_len_list,
    model_name,
    perf_filename,
    device,
):
    """Shared driver: time conv then scan separately for each shape."""
    device = torch.device(device)
    get_device_module().set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    conv_dim = _conv_dim(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    taps = d_conv - 1  # conv-state history length
    scale = head_k_dim**-0.5
    backend = gdn_prefill_backend(device)  # prefill scan specialization (xe3p/xe2)
    conv_weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)

    # context: seq_len sub-sweep; generation: single token per request.
    seq_iter = seq_len_list if phase == "context" else [1]

    if aic_debug:
        print(
            f"GDN {phase}: d_model={d_model}, conv_dim={conv_dim}, backend={backend}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}"
        )

    for batch_size in batch_size_list:
        for seq_len in seq_iter:
            num_tokens = batch_size * seq_len
            if phase == "context" and seq_len % _CTX_SEQ_MULTIPLE:
                # Xe3p prefill conv rejects sequence lengths not divisible by 64;
                # skip the whole cell (conv + scan) rather than measure half of it.
                print(
                    f"GDN {phase} skip {model_name} bs={batch_size} seq={seq_len} (seq_len % {_CTX_SEQ_MULTIPLE} != 0)"
                )
                continue
            if phase == "context" and num_tokens * conv_dim >= _CONV_ELEM_LIMIT:
                # packed conv input has num_tokens*conv_dim elements; past 2**31 the
                # allocation balloons to tens of GB and stalls the device (NV raises
                # the same cell on its int32 kernel-offset overflow). Skip the cell.
                print(
                    f"GDN {phase} skip {model_name} bs={batch_size} seq={seq_len} "
                    f"(num_tokens*conv_dim={num_tokens * conv_dim} >= {_CONV_ELEM_LIMIT})"
                )
                continue
            common_log_data = {
                "phase": phase,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_tokens": num_tokens,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }
            # slot 0 is vLLM's null block; one live slot per request.
            state_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)

            if phase == "context":
                # Conv: packed-QKV depthwise conv over the prompt. vLLM layout:
                # feature-contiguous [conv_dim, tokens] input, paged conv-state
                # cache with unit channel stride.
                conv_x = torch.randn(num_tokens, conv_dim, dtype=dtype, device=device).transpose(0, 1)
                conv_states = torch.zeros(batch_size + 1, taps, conv_dim, dtype=dtype, device=device).transpose(-1, -2)
                has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
                query_start_loc = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32, device=device)

                # q and k occupy key_dim, so conv q-heads mirror the k-heads;
                # l2norm is left to the scan to avoid double-normalizing.
                def run_conv(_x=conv_x, _st=conv_states, _qsl=query_start_loc, _has=has_initial_state):
                    causal_conv1d_fn(
                        _x,
                        conv_weight,
                        None,
                        "silu",
                        _st,
                        state_indices,
                        _has,
                        _qsl,
                        apply_qkv_split=True,
                        num_q_heads=num_k_heads,
                        num_k_heads=num_k_heads,
                        num_v_heads=num_v_heads,
                        head_q_dim=head_k_dim,
                        head_k_dim=head_k_dim,
                        head_v_dim=head_v_dim,
                        apply_l2norm=False,
                    )

                _benchmark(run_conv, device, common_log_data, "causal_conv1d_fn", perf_filename, phase=phase)
                del conv_x, conv_states, has_initial_state, query_start_loc
                gc.collect()
                get_device_module().empty_cache()

                grid_y = _grid_y(phase, batch_size, num_tokens, num_v_heads)
                if grid_y > _GRID_Y_LIMIT:
                    # scan grid-y exceeds the hardware limit -> uncatchable
                    # DEVICE_LOST (see _GRID_Y_LIMIT note). Conv is still logged.
                    print(
                        f"GDN {phase} scan skip {model_name} bs={batch_size} "
                        f"seq={seq_len} (grid_y={grid_y} > {_GRID_Y_LIMIT})"
                    )
                    continue

                # Scan: chunked gated-delta-rule. q/k span head_k_dim, v spans
                # head_v_dim; g/beta are already exp/sigmoid-space per v-head.
                q = torch.randn(num_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(num_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(num_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device)
                g = (0.8 + 0.2 * torch.rand(num_tokens, num_v_heads, device=device)).float()
                beta = torch.sigmoid(torch.randn(num_tokens, num_v_heads, device=device)).float()
                cu_seqlens = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int64, device=device)

                def run_scan(_q=q, _k=k, _v=v, _g=g, _beta=beta, _cu=cu_seqlens):
                    chunk_gated_delta_rule(
                        _q,
                        _k,
                        _v,
                        _g,
                        _beta,
                        scale=scale,
                        cu_seqlens=_cu,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=backend == "xe3p",
                    )

                _benchmark(
                    run_scan,
                    device,
                    common_log_data,
                    f"chunk_gated_delta_rule_{backend}",
                    perf_filename,
                    phase=phase,
                )
                del q, k, v, g, beta, cu_seqlens
                gc.collect()
                get_device_module().empty_cache()
            else:
                # Conv: single-token packed-QKV conv update against cached state.
                conv_x = torch.randn(batch_size, conv_dim, dtype=dtype, device=device)
                conv_state = torch.randn(batch_size + 1, taps, conv_dim, dtype=dtype, device=device).transpose(-1, -2)

                # Xe3p decode conv runs only no-split (split update is rejected);
                # the kernel processes the packed conv_dim channels directly.
                def run_conv(_x=conv_x, _st=conv_state):
                    causal_conv1d_update(
                        _x,
                        _st,
                        conv_weight,
                        None,
                        "silu",
                        state_indices,
                        apply_qkv_split=False,
                        num_q_heads=num_k_heads,
                        num_k_heads=num_k_heads,
                        num_v_heads=num_v_heads,
                        head_q_dim=head_k_dim,
                        head_k_dim=head_k_dim,
                        head_v_dim=head_v_dim,
                        apply_l2norm=False,
                    )

                _benchmark(run_conv, device, common_log_data, "causal_conv1d_update", perf_filename, phase=phase)
                del conv_x, conv_state
                gc.collect()
                get_device_module().empty_cache()

                grid_y = _grid_y(phase, batch_size, num_tokens, num_v_heads)
                if grid_y > _GRID_Y_LIMIT:
                    # scan grid-y exceeds the hardware limit -> uncatchable
                    # DEVICE_LOST (see _GRID_Y_LIMIT note). Conv is still logged.
                    print(f"GDN {phase} scan skip {model_name} bs={batch_size} (grid_y={grid_y} > {_GRID_Y_LIMIT})")
                    continue

                # Scan: recurrent decode. gating (a/b/A_log/dt_bias) is fused in
                # the kernel; the fp32 ssm-state pool is V-major [HV, V, K].
                q = torch.randn(batch_size, 1, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(batch_size, 1, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(batch_size, 1, num_v_heads, head_v_dim, dtype=dtype, device=device)
                a_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
                dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
                a = torch.randn(batch_size, 1, num_v_heads, dtype=dtype, device=device)
                b = torch.randn(batch_size, 1, num_v_heads, dtype=dtype, device=device)
                ssm_state = torch.randn(
                    batch_size + 1, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
                )

                def run_scan(_q=q, _k=k, _v=v, _a=a, _b=b, _st=ssm_state):
                    gated_delta_rule_decode(
                        q=_q,
                        k=_k,
                        v=_v,
                        state=None,
                        A_log=a_log,
                        a=_a,
                        dt_bias=dt_bias,
                        b=_b,
                        scale=scale,
                        use_qk_l2norm=True,
                        initial_state=_st,
                        initial_state_indices=state_indices,
                    )

                _benchmark(run_scan, device, common_log_data, "gated_delta_rule_decode", perf_filename, phase=phase)
                del q, k, v, a_log, dt_bias, a, b, ssm_state
                gc.collect()
                get_device_module().empty_cache()


def run_gdn_torch(
    phase: str,
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    seq_len_list: list[int] | None,
    model_name: str,
    *,
    perf_filename: str,
    device: str = "xpu:0",
):
    """Route one collector-v2 GDN case to the shared XPU driver."""
    if phase == "context":
        if seq_len_list is None:
            raise ValueError("context GDN cases require seq_len_list")
    elif phase != "generation":
        raise ValueError(f"Unknown phase: {phase}")

    _run_phase(
        phase=phase,
        d_model=d_model,
        d_conv=d_conv,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        batch_size_list=batch_size_list,
        seq_len_list=seq_len_list,
        model_name=model_name,
        perf_filename=perf_filename,
        device=device,
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    print(f"GDN XPU Collector - vLLM {vllm_version}")
    print(f"Device: {get_device_module().get_device_name()}")

    test_cases = get_gdn_test_cases()
    print(f"Total test cases: {len(test_cases)}")
    for i, test_case in enumerate(test_cases):
        (
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            batch_size_list,
            seq_len_list,
            model_name,
        ) = test_case
        print(f"\n[{i + 1}/{len(test_cases)}] {model_name} - {phase}")
        run_gdn_torch(
            phase=phase,
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=PerfFile.GDN,
        )
