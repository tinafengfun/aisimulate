# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) Qwen3.5 Gated DeltaNet collector.

Times the two production core ops separately, matching the NVIDIA collector:
the packed-QKV depthwise convolution and the gated-delta-rule scan. Context
uses ``causal_conv1d_varlen`` + ``chunk_gated_delta_rule``; generation uses
``causal_conv1d_varlen_update`` + ``gated_delta_rule_decode``. Input/output
projection GEMMs stay with the GEMM collector.

The conv rows use DeepKLOX's ``*_varlen*`` pair rather than its GDN-fused
``causal_conv1d_fn``/``causal_conv1d_update``, because the varlen pair computes
exactly what the CUDA collector times -- depthwise causal conv + SiLU over a
packed varlen batch, no Q/K/V split and no Q/K L2 norm
(``vllm/model_executor/layers/mamba/ops/causal_conv1d.py``). The fused variants
do strictly more work (verified: 2.09 ms vs 0.80 ms at bs=4 x seq=2048,
conv_dim=10240), so timing them against the CUDA row would overstate the gap by
work the CUDA row never does. Their Xe3p support surfaces also disagree -- the
fused prefill kernel requires ``apply_qkv_split=True`` and sequence lengths
divisible by 64, the fused decode kernel rejects both the split and the norm --
whereas the varlen pair takes every shape in the sweep. The fused path is what
vLLM-XPU serving calls (``vllm/_xpu_ops.py``); its extra split+norm cost is a
model-level concern, not part of this op's cross-vendor row.

The prefill scan row L2-normalizes q/k before the timed closure and always
calls ``chunk_gated_delta_rule`` with ``use_qk_l2norm_in_kernel=False``,
mirroring the NVIDIA collector exactly (it normalizes q/k outside its own
timed region and passes the same ``False``). DeepKLOX's Xe2 backend only
implements ``False`` for this flag; its Xe3p backend implements both and
defaults production serving to ``True``, timing the norm in-kernel. Forcing
``False`` here on Xe3p too drops that in-kernel norm cost from the row, the
same production-fidelity-for-comparability tradeoff as the conv rows above.
Decode's ``gated_delta_rule_decode`` is left at ``use_qk_l2norm=True`` on both
collectors (no tradeoff there: both already time the norm in-kernel).

The primitives are imported from vLLM-XPU's ``mamba.gdn.deepklox_interface``,
which dispatches each op to DeepKLOX for now.
"""

__compat__ = "vllm==0.28.0"

import gc
import math
import os

import torch
from vllm.model_executor.layers.mamba.gdn.deepklox_interface import (
    causal_conv1d_prepare_weight,
    causal_conv1d_varlen,
    causal_conv1d_varlen_update,
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


def _decode_pack_n(state_bytes):
    """Kernels to pack per decode graph: this op's own per-replay state bytes vs amortize threshold, pow2 1..64.

    state_bytes must be the byte size of *this* kernel's per-request recurrent
    state (conv_state for conv, ssm_state for the scan) -- they differ by
    ~40x, so a shared estimate collapses pack_n for the smaller-state kernel
    far earlier than its own memory budget would allow, under-amortizing the
    XPU-graph replay overhead for that op.
    """
    system = os.environ.get("COLLECTOR_SYSTEM") or None
    amortize = _LAUNCH_AMORTIZE_BYTES.get(system, _LAUNCH_AMORTIZE_BYTES_DEFAULT)
    raw = min(float(_PACK_N_MAX), max(1.0, amortize / max(state_bytes, 1)))
    return min(_PACK_N_MAX, max(1, 1 << round(math.log2(raw))))


def _benchmark(kernel_func, device, common_log_data, kernel_source, perf_filename, *, phase, state_bytes=0):
    """Time one op and log a row. Context: eager, repeat_n=10 (NV parity). Generation: graph, pack_n per replay."""
    use_graph = xpu_graph_measure_enabled() and phase != "context"

    if use_graph:
        pack_n = _decode_pack_n(state_bytes)
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
    # The kernels read the weight tap-major. Serving converts it once when the
    # layer is built, so the conversion stays outside the timed region.
    conv_weight_tap_major = causal_conv1d_prepare_weight(conv_weight)

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
                # Conv: packed-QKV depthwise conv over the prompt. Token-major
                # [tokens, conv_dim] activations (the projection output feeds
                # the kernel without a transpose) and a paged conv-state cache
                # in vLLM's SD layout [slots, taps, conv_dim].
                conv_x = torch.randn(num_tokens, conv_dim, dtype=dtype, device=device)
                conv_states = torch.zeros(batch_size + 1, taps, conv_dim, dtype=dtype, device=device)
                has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
                query_start_loc = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32, device=device)

                # Plain depthwise conv + SiLU, as the CUDA collector times it:
                # the Q/K/V split and the Q/K L2 norm are not part of this op.
                def run_conv(_x=conv_x, _st=conv_states, _qsl=query_start_loc, _has=has_initial_state):
                    causal_conv1d_varlen(
                        _x,
                        conv_weight_tap_major,
                        _qsl,
                        conv_states=_st,
                        cache_indices=state_indices,
                        has_initial_state=_has,
                        bias=None,
                        activation="silu",
                    )

                # Row label stays the CUDA kernel_source name: the SDK and the
                # perf-reuse manifest key GDN conv rows on it, and this is the
                # same op on the other vendor.
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
                # q/k are L2-normalized here, outside the timed region, and
                # the kernel is always called with use_qk_l2norm_in_kernel=False
                # -- matching the CUDA collector, which normalizes q/k before
                # entering its own timed closure and also passes False
                # (NV collect_gdn.py's run_gdn_context_benchmark). DeepKLOX's
                # Xe2 backend only implements False (raises on True); Xe3p
                # implements both and defaults to True in production, but its
                # in-kernel norm cost would otherwise be timed on XPU and not
                # on CUDA, overstating the cross-vendor gap by work the CUDA
                # row never does. Forcing False here trades Xe3p's production
                # fidelity for CUDA-comparable accounting, same tradeoff as
                # the conv rows' varlen-vs-fused choice above.
                q = torch.nn.functional.normalize(
                    torch.randn(num_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device), dim=-1
                )
                k = torch.nn.functional.normalize(
                    torch.randn(num_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device), dim=-1
                )
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
                        use_qk_l2norm_in_kernel=False,
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
                # Conv: single-token packed-QKV conv update against cached
                # state, SD layout [slots, taps, conv_dim] as in serving.
                conv_x = torch.randn(batch_size, conv_dim, dtype=dtype, device=device)
                conv_state = torch.randn(batch_size + 1, taps, conv_dim, dtype=dtype, device=device)

                # Plain depthwise conv + SiLU over the packed conv_dim channels,
                # matching the CUDA causal_conv1d_update row (no split, no norm).
                def run_conv(_x=conv_x, _st=conv_state):
                    causal_conv1d_varlen_update(
                        _x,
                        _st,
                        conv_weight_tap_major,
                        conv_state_indices=state_indices,
                        bias=None,
                        activation="silu",
                    )

                # conv_state bytes per request (bf16) -- ~40x smaller than the
                # ssm_state used below, so it earns a much larger pack_n at
                # the same batch size (see _decode_pack_n).
                conv_state_bytes = batch_size * taps * conv_dim * conv_state.element_size()
                _benchmark(
                    run_conv,
                    device,
                    common_log_data,
                    "causal_conv1d_update",
                    perf_filename,
                    phase=phase,
                    state_bytes=conv_state_bytes,
                )
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

                # fp32 ssm_state bytes per request (V-major [HV, V, K]).
                ssm_state_bytes = batch_size * num_v_heads * head_v_dim * head_k_dim * ssm_state.element_size()
                _benchmark(
                    run_scan,
                    device,
                    common_log_data,
                    "gated_delta_rule_decode",
                    perf_filename,
                    phase=phase,
                    state_bytes=ssm_state_bytes,
                )
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
