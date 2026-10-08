# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 module-level attention collector for vLLM on Intel XPU (CRI).

XPU port of ``collect_dsv4_attn.py`` (the CUDA baseline is authoritative for
op semantics and row shape and is deliberately not modified). Covers the
CSA / HCA / SWA attention kinds: csa=4, hca=128, swa=0 (pure sliding window —
the first two V4-Flash / V4.1-Flash layers, whose ``compress_ratios`` entry is
0 and which serving clamps via ``max(1, ratio)`` into the ``swa_only`` branch
of ``DeepseekV4XPUAttention``, xpu_sparse.py:145/169). vLLM-XPU has no
DeepSeek-V4.1 model path; a V4.1 SWA layer is structurally a V4 SWA layer, so
it is built as ``DeepseekV4XPUAttention`` from V4.1's ``text_config``.

vLLM-XPU builds every DSV4 decoder layer's attention as
``DeepseekV4XPUAttention`` (vllm/models/deepseek_v4/xpu/model.py), so the
collector constructs that class for one layer, binds the DSV4
main/SWA/indexer/compressor-state metadata and KV caches through the
production builders, and benchmarks the full attention wrapper forward
``attn(positions, hidden_states, None)`` (input projections, compressor +
indexer, sparse MLA, inverse-RoPE + fp8 ``wo_a`` bmm + ``wo_b``). Sparse HCA
is also isolated from that module. Paged MQA logits is collected as a
kernel-level benchmark with directly constructed deepklox inputs, matching the
sparse correction model.

The audited deltas from the CUDA collector are:

* F1 — the XPU worker's process-level ``torch.cuda → torch.xpu`` aliasing
  (``_torch_cuda_wrapper``, xpu_model_runner.py:315-337) is replicated from
  the wheel's own implementation; DSV4's shared forward crashes without it
  (multi_stream_utils.py:121 ``torch.cuda.stream``, attention.py:1103
  ``is_current_stream_capturing``).
* F2 — KV-cache allocation goes through the engine's spec-based path
  (``compute_layer_kv_cache_shape_bytes`` → flat int8 buffer →
  ``create_kv_cache_views``) with the layout resolved the way the worker
  merges per-backend supported layouts; the NV-only
  ``backend.get_kv_cache_shape`` API does not exist on XPU backends.
* F3 — metadata builders are created through
  ``AttentionGroup.create_metadata_builders`` (a hand-built
  ``get_builder_cls()(...)`` call TypeErrors on the indexer builder, which
  needs ``block_table_width``); binding goes through the layer's own
  ``bind_kv_cache`` hook, which squeezes the view to the 3D
  ``[num_blocks, block_size, bytes]`` layout the SYCL fp8mix insert kernel
  requires (attention.py:867 / compressor.py:178).
* F4 — the three aux streams are kept (``torch.xpu.Stream()``), so the
  multi-stream GEMM path (<=1024 tokens, on by default) matches serving.
* paged_mqa_logits runs the deepklox bare core (``context_lens`` is ``[B]``
  there; ``[B,1]`` raises) with the scheduler metadata sized by the device's
  EU count instead of the NV SM-count translation.

gemm_type: ``mxfp8`` = online MXFP8 linears (``--quantization mxfp8`` ->
``XPUMxFp8LinearKernel``), the 1x32/E8M0 granularity the XPU ``_o_proj``
bmm expects (``deepseek_inv_rope_fp8_quant(use_mxfp8=True)`` +
``xpu_fp8_bmm``); the checkpoint's 128x128 ``fp8_block`` scales send ``wo_a``
to oneDNN's reference bmm, which is not a serving path — the XPU grids
therefore collect mxfp8 only (NV↔XPU alignment on the fp8_block axis is a
declared XPU-side coverage gap).

Timing follows the V1 model runner's own dispatch (``VLLM_USE_V2_MODEL_RUNNER=0``)
with ``VLLM_XPU_ENABLE_XPU_GRAPH=1``: batches within
``AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE`` tokens (default 256 = vLLM's
``2 * --max-num-seqs`` for the CRI recipe's 128) replay a graph, larger ones
run eagerly. Decode replays a FULL graph whose metadata is built like the
runner's capture (``max_seq_len = max_model_len``, which pins the C128A top-k
width and the indexer top-k bound to the serving context limit for every
replay), packed ``_decode_pack_n`` forwards per replay. Prefill replays a
breakable PIECEWISE graph (graphed input/output projections around the eager
``_prepare_and_attn_eager`` region) once per forward — serving pays one
piecewise replay per layer, so no further packing. ``AIC_DSV4_PREFILL_TIMING``
retains the eager investigation modes from the four-quadrant experiment.

The attention path uses dummy weights. With dummy FP8 block weights, vLLM's
DeepGEMM path may require real checkpoint layouts/scales to be representative;
the collector does not override vLLM's backend selection at import time.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import order is load-bearing on this stack (container-verified 2026-10-06):
# torch first (deepklox's _C extension needs libc10 from torch), then
# deepklox, then the vllm_xpu_kernels extension that registers
# torch.ops._xpu_C. Reversing the last two (torch → _xpu_C → deepklox)
# segfaults the interpreter.
#
# VLLM_USE_BREAKABLE_CUDAGRAPH must be set before vLLM's DeepseekV4 attention
# is imported: the piecewise prefill capture (_capture_breakable_graph) needs
# the breakable-graph machinery registered at layer init, and serving runs
# with it enabled (the eager-break region _prepare_and_attn_eager exists only
# under it). setdefault keeps an explicit external setting authoritative.
import os as _os_pre

_os_pre.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

import torch
import vllm.envs as vllm_envs
from deepklox import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm._xpu_ops import xpu_ops
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.models.deepseek_v4.xpu.xpu_sparse import DeepseekV4XPUAttention
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

from collector.case_generator import (
    DSV4_SPARSE_KERNELS,
    _DSV4_DEFAULT_MODELS,
    _DSV4_MODULE_TP_SIZES,
    _DSV4_SPARSE_MAX_FULL_S,
    get_xpu_dsv4_attn_test_cases,
    get_xpu_dsv4_hca_attn_test_cases,
    get_xpu_dsv4_paged_mqa_logits_test_cases,
)
from collector.helper import (
    benchmark_with_power,
    get_device_module,
    log_perf,
    xpu_graph_measure_enabled,
)
from collector.registry_types import PerfFile
from collector.vllm.utils_xpu import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    setup_distributed,
)

__compat__ = "vllm==0.28.0"


DEFAULT_MODEL = _DSV4_DEFAULT_MODELS[0]
ARCHITECTURE = "DeepseekV4ForCausalLM"
# ``swa`` keeps the config's raw entry (0): serving clamps it with
# max(1, ratio) internally, and the perf-row compress_ratio records the
# config value the layer was built from.
ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128, "swa": 0}
# (attn_kind, mode) -> canonical perf filename (one OpEntry per pair).
PERF_FILES = {
    ("csa", "context"): PerfFile.DSV4_CSA_CONTEXT_MODULE,
    ("hca", "context"): PerfFile.DSV4_HCA_CONTEXT_MODULE,
    ("swa", "context"): PerfFile.DSV4_SWA_CONTEXT_MODULE,
    ("csa", "generation"): PerfFile.DSV4_CSA_GENERATION_MODULE,
    ("hca", "generation"): PerfFile.DSV4_HCA_GENERATION_MODULE,
    ("swa", "generation"): PerfFile.DSV4_SWA_GENERATION_MODULE,
}
SPARSE_KERNEL_TO_ATTN_KIND = {"paged_mqa_logits": "csa", "hca_attn": "hca"}
SPARSE_KERNEL_TO_OP_NAME = {
    "paged_mqa_logits": "dsv4_paged_mqa_logits_module",
    "hca_attn": "dsv4_hca_attn_module",
}
SPARSE_KERNEL_TO_PERF_FILE = {
    "paged_mqa_logits": PerfFile.DSV4_PAGED_MQA_LOGITS_MODULE,
    "hca_attn": PerfFile.DSV4_HCA_ATTN_MODULE,
}
SPARSE_KERNEL_TO_KERNEL_SOURCE = {
    "paged_mqa_logits": "deepklox.fp8_fp4_paged_mqa_logits",
}
MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"
SUPPORTED_GEMM_TYPES = {"fp8_block", "mxfp8"}
SWA_BLOCK_SIZE = 64  # DeepseekV4SWACache page size
COMPRESSED_BLOCK_SIZE = 256  # cache_config.block_size of the compressed (C4A/C128A) caches

# 0 = one sparse_mla_prefill call per vLLM prefill chunk (PREFILL_CHUNK_SIZE
# requests), as on CUDA. >0 splits each call into <=N rows (rows are
# independent, output unchanged; such rows are tagged +prefill_chunkN in
# kernel_source).
PREFILL_CHUNK_TOKENS = int(os.environ.get("AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS", "0"))
# TODO: default to "0" (collect them) once deepklox fixes the fused-prefill
# int32 overflow (repro_sparse_prefill_int32_overflow.py).
SKIP_PREFILL_OVERFLOW = os.environ.get("AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW", "1") != "0"
# vLLM max_cudagraph_capture_size: min(2 * --max-num-seqs, 512,
# --max-num-batched-tokens) — the CRI recipe's 128 max-num-seqs gives 256.
MAX_CUDAGRAPH_CAPTURE_SIZE = int(os.environ.get("AIC_VLLM_DSV4_MAX_CUDAGRAPH_CAPTURE_SIZE", "256"))
# Serving --max-model-len for decode; 0 = the model's max_position_embeddings
# (vLLM's default).
SERVING_MAX_MODEL_LEN = int(os.environ.get("AIC_VLLM_DSV4_MAX_MODEL_LEN", "0"))

# Context-forward peak fit on CRI: hidden in + q/output (2 KiB per local head)
# + this per token, + static.
CONTEXT_EXTRA_KIB_PER_TOKEN = {"csa": 48, "hca": 24, "swa": 24}
CONTEXT_STATIC_BYTES = 8 * 2**30
# VLLM_XPU_MXFP8_USE_SYCLTLA=1: deepklox's GEMM takes an fp32 [M, N] workspace,
# at wq_b (N = 512 per local head) 2 KiB per local head and token; 2.5 measured,
# with allocator fragmentation.
SYCLTLA_KIB_PER_LOCAL_HEAD = 2.5
# An XPU OOM surfaces as UR_RESULT_ERROR_DEVICE_LOST (not OutOfMemoryError), so
# stay below it.
MAX_MEM_FRACTION = 0.9

# Per COLLECTOR_SYSTEM: CRI needs the larger window, other systems the smaller
# default (gdn precedent, collect_gdn_xpu.py).
_LAUNCH_AMORTIZE_BYTES = {"cri": 340_000_000}
_LAUNCH_AMORTIZE_BYTES_DEFAULT = 64_000_000
_PACK_N_MAX = 64


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    if filename is None:
        raise ValueError("filename is required")
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _read_model_config(model_id: str) -> dict:
    if os.path.isdir(model_id):
        config_file = Path(model_id) / "config.json"
    else:
        config_file = MODEL_CONFIGS_DIR / f"{model_id.replace('/', '--')}_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: {config_file}")
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        # DeepseekV41ForCausalLM nests the decoder geometry under text_config.
        config = {
            **text_config,
            "architectures": config.get("architectures"),
            "quantization_config": config.get("quantization_config"),
        }
    return config


@contextmanager
def _patched_config_dir(model_id: str, *, compress_ratio: int, gemm_type: str):
    config = dict(_read_model_config(model_id))
    config.pop("auto_map", None)
    if gemm_type == "mxfp8":
        # BF16 checkpoint + online MXFP8 (--quantization mxfp8): the XPUMxFp8LinearKernel
        # path the XPU o_proj bmm expects.
        config.pop("quantization_config", None)
    elif not config.get("quantization_config"):
        raise ValueError(f"{model_id} has no quantization_config; fp8_block needs the checkpoint's block-fp8 recipe")
    config["model_type"] = "deepseek_v4"
    config["architectures"] = [ARCHITECTURE]
    config["num_hidden_layers"] = 1
    config["num_key_value_heads"] = 1
    config["compress_ratios"] = [compress_ratio]
    config.setdefault("rms_norm_eps", 1e-6)

    with tempfile.TemporaryDirectory(prefix=f"aic_vllm_dsv4_xpu_{compress_ratio}_{os.getpid()}_") as tmp_dir:
        with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        yield tmp_dir


def _torch_cuda_alias_wrapper():
    """F1: the serving-faithful process-level ``torch.cuda → torch.xpu`` aliasing.

    The XPU model runner installs these aliases in its ``__init__`` and never
    restores them (xpu_model_runner.py:315-337 — leaky by design; the worker
    process relies on them for its whole lifetime). DSV4's shared forward
    crashes without them: multi_stream_utils.py:121 calls ``torch.cuda.stream``
    (multi-stream GEMM, enabled by default at <=1024 tokens) and
    attention.py:1103 calls ``torch.cuda.is_current_stream_capturing`` from the
    indexer. Use the wheel's own implementation as the single source — a
    hand-copied symbol list would silently drift on wheel upgrades — and fail
    closed if the wheel stops exporting it.
    """
    try:
        from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper
    except ImportError as exc:
        raise RuntimeError(
            "vLLM XPU wheel does not export "
            "vllm.v1.worker.xpu_model_runner._torch_cuda_wrapper; the collector "
            "refuses to guess the torch.cuda→torch.xpu alias set (fail-closed). "
            "Re-audit xpu_model_runner.py on this wheel and pin the alias list "
            "explicitly if it was renamed."
        ) from exc
    return _torch_cuda_wrapper()


def _init_xpu(device: str) -> None:
    # The alias wrapper is leaky (serving installs it once per process and
    # never restores), so entering it here covers the worker's whole lifetime.
    with _torch_cuda_alias_wrapper():
        setup_distributed(device)
        get_device_module(device).set_device(device)
        # forward_mqa's prefill path asserts on the workspace manager (F5).
        init_workspace_manager(torch.device(device))
    _install_prefill_chunking()
    # No enable_engine_fused_ops(): the CUDA helper pokes the NV IR registry,
    # while the XPU platform installs its own IR priorities
    # (rms_norm/fused_add_rms_norm -> ['vllm_c', 'native']) at module
    # construction — audited as the framework default on this wheel.


def _install_prefill_chunking() -> None:
    """Optionally split each sparse_mla_prefill call into <=PREFILL_CHUNK_TOKENS rows.

    Rows are independent (per-query), so the output is unchanged; the split
    only bounds deepklox's int32 q/out offsets. Idempotent via the
    ``chunk_tokens`` marker so repeated _init_xpu calls never double-wrap.
    """
    original = xpu_ops.sparse_mla_prefill
    if PREFILL_CHUNK_TOKENS <= 0 or getattr(original, "chunk_tokens", None) == PREFILL_CHUNK_TOKENS:
        return

    def chunked_sparse_mla_prefill(q, kv, indices, sm_scale, d_v, topk_length=None, output=None):
        if output is None:
            output = q.new_empty((*q.shape[:-1], d_v))
        for start in range(0, q.shape[0], PREFILL_CHUNK_TOKENS):
            rows = slice(start, start + PREFILL_CHUNK_TOKENS)
            original(
                q=q[rows],
                kv=kv,
                indices=indices[rows],
                sm_scale=sm_scale,
                d_v=d_v,
                topk_length=None if topk_length is None else topk_length[rows],
                output=output[rows],
            )
        return output

    chunked_sparse_mla_prefill.chunk_tokens = PREFILL_CHUNK_TOKENS
    xpu_ops.sparse_mla_prefill = staticmethod(chunked_sparse_mla_prefill)


def _local_num_heads(model_path: str, tp_size: int) -> int:
    return int(_read_model_config(model_path)["num_attention_heads"]) // tp_size


def _prefill_rows_per_call(batch_size: int, query_len: int) -> int:
    # _forward_prefill hands PREFILL_CHUNK_SIZE requests to each kernel call.
    rows = min(batch_size, DeepseekV4XPUAttention.PREFILL_CHUNK_SIZE) * query_len
    return min(rows, PREFILL_CHUNK_TOKENS) if PREFILL_CHUNK_TOKENS > 0 else rows


def _fused_prefill_overflows(model_path: str, batch_size: int, query_len: int, tp_size: int) -> bool:
    """deepklox fused sparse_mla_prefill indexes q/out with int32 ``token * h_q * d``.

    A call with ``(s_q - 1) * h_q * head_dim >= 2**31`` silently corrupts memory
    or loses the device (deepklox bug; repro script ships on the node). Only
    reachable when PREFILL_CHUNK_TOKENS is 0 or huge.
    """
    if query_len <= 1:
        return False
    head_dim = int(_read_model_config(model_path)["head_dim"])
    s_q = _prefill_rows_per_call(batch_size, query_len)
    return (s_q - 1) * _local_num_heads(model_path, tp_size) * head_dim >= 2**31


def _context_exceeds_memory(model_path: str, attn_kind: str, batch_size: int, query_len: int, tp_size: int) -> bool:
    """Generation-time memory-feasibility filter (the one sanctioned in-collector
    filter: size vs capacity, counted + logged — see _get_test_cases)."""
    hidden_bytes = 2 * int(_read_model_config(model_path)["hidden_size"])
    kib_per_head = 2 + (SYCLTLA_KIB_PER_LOCAL_HEAD if vllm_envs.VLLM_XPU_MXFP8_USE_SYCLTLA else 0)
    per_token = (
        hidden_bytes
        + (kib_per_head * _local_num_heads(model_path, tp_size) + CONTEXT_EXTRA_KIB_PER_TOKEN[attn_kind]) * 1024
    )
    peak = CONTEXT_STATIC_BYTES + batch_size * query_len * per_token
    return peak > MAX_MEM_FRACTION * get_device_module().get_device_properties(0).total_memory


@contextmanager
def _tp_simulation(tp_size: int):
    if tp_size == 1:
        yield
        return

    import vllm.model_executor.layers.linear as linear_mod
    import vllm.models.deepseek_v4.attention as dsv4_attn_mod

    def world_size() -> int:
        return tp_size

    def rank() -> int:
        return 0

    def identity_collective(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    patches = [
        (linear_mod, "get_tensor_model_parallel_world_size", world_size),
        (linear_mod, "get_tensor_model_parallel_rank", rank),
        (linear_mod, "tensor_model_parallel_all_reduce", identity_collective),
        (linear_mod, "tensor_model_parallel_all_gather", identity_collective),
        (dsv4_attn_mod, "get_tensor_model_parallel_world_size", world_size),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, replacement in patches:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def _init_dummy_module_tensors(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor.is_meta:
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.zero_()
            elif tensor.dtype == torch.float8_e8m0fnu:
                tensor.view(torch.uint8).fill_(127)  # e8m0 exponent 127 == 1.0
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.fill_(1.0)
            else:
                tensor.fill_(0.01)


def _process_quantized_weights(module: torch.nn.Module, vllm_config) -> None:
    with set_current_vllm_config(vllm_config):
        for _, child in module.named_modules():
            quant_method = getattr(child, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(child)


@contextmanager
def _create_dsv4_attention_module(
    *,
    model_path: str,
    attn_kind: str,
    gemm_type: str,
    batch_size: int,
    seq_len: int,
    tp_size: int,
    is_context: bool,
    device: str,
    query_len: int | None = None,
    max_model_len: int | None = None,
    multi_stream: bool = True,
):
    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    with _patched_config_dir(model_path, compress_ratio=compress_ratio, gemm_type=gemm_type) as local_model:
        if max_model_len is None:
            max_model_len = max(seq_len, 4096)
        if query_len is None:
            query_len = seq_len if is_context else 1
        query_tokens = batch_size * query_len
        max_num_batched_tokens = max(query_tokens, 2048)
        block_size = COMPRESSED_BLOCK_SIZE
        cache_blocks = _cache_blocks(batch_size, seq_len)

        vllm_config = create_vllm_config(
            model_name=local_model,
            tensor_parallel_size=tp_size,
            # The collector simulates TP shard shapes in one process instead of
            # launching one vLLM executor rank per shard.
            distributed_executor_backend="mp",
            max_model_len=max_model_len,
            block_size=block_size,
            num_gpu_blocks=cache_blocks,
            max_num_seqs=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            use_fp8_kv_cache=True,
            trust_remote_code=True,
            # mxfp8: point ModelConfig at the wheel's mxfp8 quantization method
            # (the patched config dir already popped the checkpoint's fp8_block
            # quantization_config). fp8_block: None — the config's own
            # quantization_config drives the oneDNN reference path.
            quantization="mxfp8" if gemm_type == "mxfp8" else None,
        )
        hf_config = vllm_config.model_config.hf_config
        hf_config.num_hidden_layers = 1
        hf_config.compress_ratios = [compress_ratio]
        hf_config.num_key_value_heads = 1

        # Serving truth: the XPU model runner constructs DeepseekV4XPUAttention
        # for DSV4 layers on this wheel (vllm/models/deepseek_v4/xpu/ — the
        # platform split is the wheel's own), so the collector pins the same
        # class instead of re-deriving a selection the framework already makes.
        # No capability gate: current_platform.get_device_capability() returns
        # None on XPU, so the CUDA collector's SM89/SMxx checks are no-ops here.
        attn_cls = DeepseekV4XPUAttention

        # DeepSeek rotary construction creates CPU tensors internally; setting
        # the default device only for module construction keeps those tensors
        # aligned.
        torch.set_default_device(device)
        try:
            with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
                topk_indices_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    hf_config.index_topk,
                    dtype=torch.int32,
                    device=device,
                )
                # Aux streams (F4): with the torch.cuda aliases live, the
                # multi-stream GEMM path (<=1024 tokens, on by default) and the
                # indexer/compressor overlap run exactly as in XPU serving —
                # for eager execution. When the benchmark will capture an XPU
                # graph, construct single-stream instead: serving itself
                # disables the aux-stream path during capture
                # (multi_stream_utils.maybe_execute_in_parallel forces
                # aux_stream=None whenever BreakableCUDAGraphCapture.is_active(),
                # so captured serving forwards are sequential); cross-stream
                # ops inside a raw torch.xpu.graph capture deadlock the replay
                # on this stack (container-verified 2026-10-06).
                aux_streams = [torch.xpu.Stream() for _ in range(3)] if multi_stream else None
                attn_module = attn_cls(
                    vllm_config,
                    prefix="model.layers.0.attn",
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream_list=aux_streams,
                )
        finally:
            torch.set_default_device("cpu")

        if any(p.is_meta for p in attn_module.parameters()):
            attn_module = attn_module.to_empty(device=torch.device(device))
        else:
            attn_module = attn_module.to(device)
        attn_module.eval()
        attn_module.requires_grad_(False)
        _init_dummy_module_tensors(attn_module)
        _process_quantized_weights(attn_module, vllm_config)
        yield attn_module, vllm_config


def _cache_blocks_for_block_size(batch_size: int, seq_len: int, block_size: int) -> int:
    logical_blocks = batch_size * max(1, math.ceil(seq_len / block_size))
    return max(256, logical_blocks + 64)


def _cache_blocks(batch_size: int, seq_len: int) -> int:
    return _cache_blocks_for_block_size(batch_size, seq_len, SWA_BLOCK_SIZE)


def _common_seq_lens_cpu(common):
    """The CPU seq-len tensor's attribute name varies across vLLM versions
    (utils_xpu sets ``_seq_lens_cpu`` / ``seq_lens_cpu`` / the upper-bound
    alias depending on which the CommonAttentionMetadata dataclass declares);
    probe them in a fixed order instead of assuming one."""
    for attr in ("_seq_lens_cpu", "seq_lens_cpu_upper_bound", "seq_lens_cpu"):
        value = getattr(common, attr, None)
        if value is not None:
            return value
    raise RuntimeError(f"common metadata carries no CPU seq_lens tensor (type={type(common).__name__})")


def _make_common_metadata(
    *,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    device: str,
    query_len: int | None = None,
):
    if query_len is None:
        query_len = seq_len if is_context else 1
    batch_spec = BatchSpec(
        seq_lens=[seq_len] * batch_size,
        query_lens=[query_len] * batch_size,
    )
    common = create_common_attn_metadata(
        batch_spec,
        block_size=SWA_BLOCK_SIZE,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if getattr(common, "seq_lens_cpu_upper_bound", None) is None:
        common.seq_lens_cpu_upper_bound = _common_seq_lens_cpu(common)
    common.positions = _positions(batch_size, seq_len, is_context, query_len=query_len, device=device)

    context_len = seq_len - query_len
    # Vectorized slot computation — the CUDA baseline's per-element
    # ``block_table[req, pos // 64].item()`` loop costs a device sync per
    # element on XPU (~25ms each; 256×64 slots ≈ 8 min), which is collector
    # overhead, not measured work. Same arithmetic, one gather.
    pos = context_len + torch.arange(query_len, device=device, dtype=torch.int64)
    block_col = pos // 64
    within_block = pos % 64
    block_ids = common.block_table_tensor[:, block_col].to(torch.int64)  # [B, Q]
    slots = block_ids * 64 + within_block  # [B, Q]
    common.slot_mapping.copy_(slots.reshape(-1))
    return common


def _remap_common_metadata(common, *, block_size: int, device: str):
    seq_lens = [int(x) for x in _common_seq_lens_cpu(common).tolist()]
    query_start_loc_cpu = common.query_start_loc_cpu
    query_lens = [int((query_start_loc_cpu[i + 1] - query_start_loc_cpu[i]).item()) for i in range(len(seq_lens))]
    remapped = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=block_size,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if getattr(remapped, "seq_lens_cpu_upper_bound", None) is None:
        remapped.seq_lens_cpu_upper_bound = _common_seq_lens_cpu(remapped)
    remapped.positions = common.positions
    return remapped


def _resolve_kv_cache_layout(vllm_config, cache_layers, *, device: str):
    """F2: resolve the KV layout the engine does.

    The worker first merges the per-backend supported-layout lists once per
    process (the XPU sparse backend declares None — any layout — and the
    indexer declares [BLHNC, BLNHC]; audited resolution = BLHNC), then the
    engine resolves that merged list against the spec set
    (vllm/v1/attention/backends/utils.py:resolve_kv_cache_layout).
    """
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )

    static_ctx = vllm_config.compilation_config.static_forward_context
    backends = []
    for layer in cache_layers:
        backend = static_ctx[layer.prefix].get_attn_backend()
        backends.append(backend if isinstance(backend, type) else type(backend))
    combined = list(get_supported_kv_cache_layouts(backends))
    specs = []
    for layer in cache_layers:
        spec = static_ctx[layer.prefix].get_kv_cache_spec(vllm_config)
        if spec is not None:
            specs.append(spec)
    return resolve_kv_cache_layout(vllm_config, [[layout.name for layout in combined]], specs)


def _alloc_and_build_layer_cache(
    layer_obj,
    registered,
    common,
    *,
    vllm_config,
    device: str,
    layout,
    num_blocks: int,
    remap_block_size: int | None = None,
):
    """F2/F3: metadata + KV cache for one layer via the framework's own paths.

    Metadata: ``AttentionGroup.create_metadata_builders`` (a hand-built
    ``get_builder_cls()(...)`` call TypeErrors on the indexer builder, which
    needs ``block_table_width``). Allocation:
    ``compute_layer_kv_cache_shape_bytes`` → flat int8 buffer →
    ``create_kv_cache_views`` (the engine's per-layer view path,
    vllm/v1/worker/utils.py:408). Binding: the layer's own ``bind_kv_cache``
    hook, which squeezes the view to the 3D [num_blocks, block_size, bytes]
    layout the SYCL fp8mix insert kernel requires (attention.py:867 /
    compressor.py:178) — a raw 4D view assignment ValueErrors there.
    """
    from vllm.v1.kv_cache_interface import (
        KVCacheTensor,
        compute_layer_kv_cache_shape_bytes,
        create_kv_cache_views,
    )
    from vllm.v1.worker.utils import AttentionGroup

    spec = registered.get_kv_cache_spec(vllm_config)
    if spec is None:
        print(f"  [{layer_obj.prefix}] spec=None (no KV cache)", flush=True)
        return None
    kernel_block_size = (
        remap_block_size if remap_block_size is not None else getattr(spec, "storage_block_size", None)
    )
    group = AttentionGroup(
        backend=registered.get_attn_backend(),
        layer_names=[layer_obj.prefix],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(vllm_config, torch.device(device), kernel_block_size=kernel_block_size)
    builder = group.get_metadata_builder()
    if remap_block_size is not None:
        # Compressor state caches run their own (smaller) logical block size;
        # remap the common metadata to it before building (F3).
        compressor_common = _remap_common_metadata(common, block_size=remap_block_size, device=device)
        metadata = builder.build(0, compressor_common)
    else:
        metadata = builder.build(0, common)

    alloc_block_size = getattr(spec, "storage_block_size", None)
    if alloc_block_size == spec.block_size:
        alloc_block_size = None  # the helper then uses spec.block_size internally
    shape_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks, alloc_block_size)
    page_bytes = int(math.prod(shape_bytes[1:]))
    cache_tensor = KVCacheTensor(
        size=num_blocks * page_bytes,
        layers=[layer_obj.prefix],
        layer_stride=page_bytes,
        block_stride=page_bytes,
        offset=0,
        host_resident=False,
    )
    buf = torch.zeros(cache_tensor.size, dtype=torch.int8, device=device)
    views = create_kv_cache_views(
        buf, spec, num_blocks, layout, cache_tensor, kernel_block_size=alloc_block_size
    )
    registered.bind_kv_cache(views[0])
    print(
        f"  [{layer_obj.prefix}] backend={registered.get_attn_backend().get_name()} "
        f"spec={type(spec).__name__} view={tuple(views[0].shape)} {views[0].dtype}",
        flush=True,
    )
    return metadata


def _build_metadata_and_bind_caches(attn_module: DeepseekV4XPUAttention, vllm_config, common, *, device: str):
    static_ctx = vllm_config.compilation_config.static_forward_context

    cache_layers = [attn_module, attn_module.swa_cache_layer]
    if attn_module.indexer is not None:
        cache_layers.append(attn_module.indexer.k_cache)
    layout = _resolve_kv_cache_layout(vllm_config, cache_layers, device=device)
    print(f"  resolved KV cache layout: {layout}", flush=True)

    metadata = {}
    # Allocate for the REAL context, not the (capture-pinned) common.max_seq_len:
    # the pin exists so capture-time metadata widths match serving; the KV cache
    # only needs to cover the actual seq lens (same basis as the runner's block
    # pool sizing for this benchmark).
    real_max_seq = int(_common_seq_lens_cpu(common).max())
    num_blocks = _cache_blocks(int(common.num_reqs), real_max_seq)
    for layer in cache_layers:
        layer_metadata = _alloc_and_build_layer_cache(
            layer,
            static_ctx[layer.prefix],
            common,
            vllm_config=vllm_config,
            device=device,
            layout=layout,
            num_blocks=num_blocks,
        )
        if layer_metadata is not None:
            metadata[layer.prefix] = layer_metadata

    compressors = [attn_module.compressor]
    if attn_module.indexer is not None:
        compressors.append(attn_module.indexer.compressor)
    for compressor in filter(None, compressors):
        state_cache = compressor.state_cache
        registered = static_ctx[state_cache.prefix]
        spec = registered.get_kv_cache_spec(vllm_config)
        if spec is None:
            continue
        state_blocks = _cache_blocks_for_block_size(
            int(common.num_reqs), real_max_seq, spec.block_size
        )
        state_metadata = _alloc_and_build_layer_cache(
            state_cache,
            registered,
            common,
            vllm_config=vllm_config,
            device=device,
            layout=layout,
            num_blocks=state_blocks,
            remap_block_size=spec.block_size,
        )
        if state_metadata is not None:
            metadata[state_cache.prefix] = state_metadata

    if not metadata:
        raise RuntimeError("DSV4 XPU collector built no attention metadata")
    return metadata


def _positions(
    batch_size: int,
    seq_len: int,
    is_context: bool,
    *,
    device: str,
    query_len: int | None = None,
) -> torch.Tensor:
    if query_len is None:
        query_len = seq_len if is_context else 1
    start_pos = seq_len - query_len
    return (start_pos + torch.arange(query_len, device=device, dtype=torch.long)).repeat(batch_size)


def _decode_pack_n(*, state_bytes: int) -> int:
    """Graph-replay packing for decode: pow2 1..64 (report §4.5).

    Amortizes the per-replay launch overhead against the measured op's own
    per-call traffic; the window follows COLLECTOR_SYSTEM (CRI gets the larger
    one, gdn precedent), overridable via ``aic_dsv4_launch_amortize_bytes``.
    """
    system = os.environ.get("COLLECTOR_SYSTEM") or None
    amortize = int(
        os.getenv(
            "aic_dsv4_launch_amortize_bytes",
            str(_LAUNCH_AMORTIZE_BYTES.get(system, _LAUNCH_AMORTIZE_BYTES_DEFAULT)),
        )
    )
    raw = min(float(_PACK_N_MAX), max(1.0, amortize / max(state_bytes, 1)))
    return min(_PACK_N_MAX, max(1, 1 << round(math.log2(raw))))


# Context timing modes (AIC_DSV4_PREFILL_TIMING), converged by the
# 2026-10-07 four-quadrant experiment (eager/graph x pack_n orthogonality,
# csa+hca x seq 64-1024 x 3 independent processes; report:
# dsv4_xpu/reports/20261007_fourquadrant_prefill_timing.md). Measured facts:
# module prefill is launch-hidden down to the smallest grid shape (b=1 s=64,
# eager/graph = 1.002x / 0.996x), graph (pack_n=1) is the true-value method
# with cross-process CV <= 0.96%, and neither eager_pack nor graph_pack
# improves on it — so the bare-eager and pack-inside-capture quadrants were
# pruned from the switch after the experiment.
_CONTEXT_TIMING_MODES = ("graph", "eager_rep", "eager_pack")


def _context_timing_mode() -> str:
    """Read + validate AIC_DSV4_PREFILL_TIMING (fail-closed).

    Production decision: context is benchmarked under XPU graph with pack_n=1
    — the CUDA baseline does exactly that (collect_dsv4_attn.py passes
    ``use_cuda_graph=True`` on every benchmark_with_power call, context
    included), and for launch-bound <0.1ms cases eager measures host
    launch/sync overhead instead of the op (probed: 0.065ms eager vs 0.003ms
    graph-replay on a ~3us kernel). So the default is "graph", locked to the
    production verdict.

    The switch is an INVESTIGATION/rollback tool — it can explicitly select
    the eager methods (eager_rep = legacy repeat_n=10; eager_pack =
    pack_n-amortized eager) for comparison and debugging; it must never make
    the production method config-dependent (unset env ⇒ "graph"). "pack" is
    kept as an alias of "eager_pack" for the early investigation runs.
    Unknown values raise (fail-closed).
    """
    mode = os.getenv("AIC_DSV4_PREFILL_TIMING", "graph")
    if mode == "pack":
        return "eager_pack"
    if mode not in _CONTEXT_TIMING_MODES:
        raise RuntimeError(
            f"unknown AIC_DSV4_PREFILL_TIMING={mode!r}; "
            f"expected {' | '.join(_CONTEXT_TIMING_MODES)}"
        )
    return mode


def _graph_mode(batch_size: int, query_len: int) -> str:
    """Graph mode the vLLM-XPU V1 runner dispatches this batch to: full, piecewise or eager."""
    if not xpu_graph_measure_enabled() or batch_size * query_len > MAX_CUDAGRAPH_CAPTURE_SIZE:
        return "eager"
    return "full" if query_len == 1 else "piecewise"  # uniform decode -> FULL


def _capture_breakable_graph(kernel_func) -> BreakableCUDAGraphCapture:
    """Capture ``kernel_func`` as vLLM's BreakableCUDAGraphWrapper does for a PIECEWISE batch."""
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream), BreakableCUDAGraphCapture(pool=torch.xpu.graph_pool_handle()) as capture:
        kernel_func()
    torch.xpu.current_stream().wait_stream(stream)
    torch.xpu.synchronize()
    if capture.num_eager_breaks == 0:
        raise RuntimeError(
            "PIECEWISE capture has no eager break: VLLM_USE_BREAKABLE_CUDAGRAPH must be 1 before vLLM's "
            "DeepseekV4 attention is imported"
        )
    return capture


def _bench_attention_shape(
    *,
    model_path: str,
    attn_kind: str,
    mode: str,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    tp_size: int,
    gemm_type: str,
    device: str,
    perf_filename: str,
    warming_up: int,
    test_ite: int,
) -> float | None:
    is_context = mode == "context"
    # Module-generation ``seq_len`` is the decode step / past KV length used
    # as the perf DB key.  vLLM metadata expects the sequence length including
    # the current decode token, so construct with +1 while logging the original
    # step.  This matches the SGLang collector's generation convention.
    metadata_seq_len = prefix_len + seq_len if is_context else seq_len + 1
    query_len = seq_len if is_context else 1
    # Single-token batches (decode, isl=1 prefill) take the decode path, whose
    # capture-time metadata widths follow max_model_len (serving parity).
    serving_max_model_len = SERVING_MAX_MODEL_LEN or int(
        _read_model_config(model_path)["max_position_embeddings"]
    )
    max_model_len = max(metadata_seq_len, serving_max_model_len if query_len == 1 else 4096)
    # Context resolves its timing via the AIC_DSV4_PREFILL_TIMING switch
    # (default "graph" = production verdict — see _context_timing_mode). With
    # the switch at "graph", "graph" means the vLLM-XPU V1 runner's OWN
    # dispatch (see _graph_mode): piecewise capture for context, full capture
    # for decode. The eager/pack variants exist only for investigation;
    # unknown values fail closed before any GPU work.
    prefill_timing_mode = _context_timing_mode()
    graph_mode = _graph_mode(batch_size, query_len)
    if is_context and prefill_timing_mode != "graph":
        graph_mode = "eager"  # investigation switch selects the eager quadrants
    # Capture intent must be known before construction: graph-mode forwards run
    # single-stream (serving parity under capture — multi_stream_utils forces
    # aux_stream=None while a BreakableCUDAGraphCapture is active; cross-stream
    # ops inside a raw torch.xpu.graph capture deadlock the replay on this
    # stack, container-verified 2026-10-06).
    capture_planned = graph_mode != "eager"
    with (
        _tp_simulation(tp_size),
        _create_dsv4_attention_module(
            model_path=model_path,
            attn_kind=attn_kind,
            gemm_type=gemm_type,
            batch_size=batch_size,
            seq_len=metadata_seq_len,
            tp_size=tp_size,
            is_context=is_context,
            device=device,
            query_len=query_len,
            max_model_len=max_model_len,
            multi_stream=not capture_planned,
        ) as (attn_module, vllm_config),
    ):
        common = _make_common_metadata(
            batch_size=batch_size,
            seq_len=metadata_seq_len,
            is_context=is_context,
            device=device,
            query_len=query_len,
        )
        if graph_mode == "full":
            # GPUModelRunner._build_attention_metadata(for_cudagraph_capture=True):
            # capture-time metadata is built against max_model_len so replayed
            # shapes match serving (pins the C128A top-k width). The KV cache
            # itself is still allocated for the real context (see
            # _build_metadata_and_bind_caches).
            common.max_seq_len = max_model_len
        metadata = _build_metadata_and_bind_caches(attn_module, vllm_config, common, device=device)

        hf_config = vllm_config.model_config.hf_config
        concrete_layer = vllm_config.compilation_config.static_forward_context[attn_module.prefix]
        backend_name = concrete_layer.get_attn_backend().get_name()
        cache_spec = concrete_layer.get_kv_cache_spec(vllm_config)
        if cache_spec is None and attn_kind == "swa":
            # swa-only layer (compress_ratios == 0): the main prefix registers
            # no KV cache — its KV lives in swa_cache_layer (serving's
            # swa_only branch of DeepseekV4XPUAttention). Resolve the spec and
            # bytes/token from the SWA cache so the row still records the
            # framework's own page geometry.
            swa_layer = attn_module.swa_cache_layer
            swa_registered = vllm_config.compilation_config.static_forward_context[swa_layer.prefix]
            cache_spec = swa_registered.get_kv_cache_spec(vllm_config)
            if cache_spec is None:
                raise RuntimeError("DSV4 swa layer has neither a main nor an swa KV-cache spec")
            backend_name += "+swa_cache"
        if cache_spec is None:
            raise RuntimeError(f"DSV4 {attn_kind} layer did not register a KV-cache spec")
        if is_context and 0 < PREFILL_CHUNK_TOKENS < min(batch_size, attn_module.PREFILL_CHUNK_SIZE) * query_len:
            # _install_prefill_chunking splits the fused prefill call; say so in
            # kernel_source so rows stay attributable (deepklox int32 guard).
            backend_name += f"+prefill_chunk{PREFILL_CHUNK_TOKENS}"
        architecture = hf_config.architectures[0] if hf_config.architectures else ARCHITECTURE
        # Persisted ``num_heads`` is rank-LOCAL (unified #1429 convention);
        # consumers derive native as ``num_heads * tp_size``. vllm's own
        # ``n_local_heads`` is that count — cross-check it against the config
        # so a TP-simulation regression cannot mislabel rows.
        local_num_heads = int(attn_module.n_local_heads)
        expected_local = max(1, int(hf_config.num_attention_heads) // tp_size)
        if local_num_heads != expected_local:
            raise RuntimeError(
                f"DSV4 attention head geometry mismatch: module n_local_heads={local_num_heads} != "
                f"config num_attention_heads // tp_size = {expected_local} (tp_size={tp_size})"
            )
        num_tokens = batch_size * seq_len if is_context else batch_size
        hidden_states = torch.full(
            (num_tokens, hf_config.hidden_size),
            0.01,
            dtype=torch.bfloat16,
            device=device,
        )
        positions = common.positions

        capture = None
        with set_current_vllm_config(vllm_config), set_forward_context(metadata, vllm_config), torch.inference_mode():
            attn_module(positions, hidden_states, None)
            get_device_module(device).synchronize()

            def kernel_func(
                attn_module=attn_module,
                positions=positions,
                hidden_states=hidden_states,
            ):
                attn_module(positions, hidden_states, None)

            # Context resolves its timing via the AIC_DSV4_PREFILL_TIMING
            # switch (default "graph" = production verdict, NV parity — see
            # _context_timing_mode / graph_mode above). With the switch at
            # "graph", context runs PIECEWISE (the serving dispatch) with
            # pack_n=1: serving pays one piecewise replay per layer per step,
            # so packing would understate it. repeat_n=1 (kernel-bound; the
            # CUDA baseline also uses repeat_n=1). Generation is NOT controlled
            # by the switch: it always takes the runner's full-graph decode
            # path and packs pack_n replays into one capture when graph
            # measurement is enabled — decode launch overhead dominates small
            # shapes (report §4.5). A capture failure fails the case
            # (allow_graph_fail=False).
            timing_label = None
            kv_bytes_per_token = int(cache_spec.page_size_bytes) // int(cache_spec.block_size)
            if is_context:
                if graph_mode == "piecewise":
                    capture = _capture_breakable_graph(kernel_func)
                    pack_n = 1
                    repeat_n = 1
                    timing_label = "graph_piecewise"
                    run = capture.replay
                elif prefill_timing_mode == "eager_rep":
                    pack_n = 1
                    repeat_n = 10
                    timing_label = "eager_repeat10"
                    run = kernel_func
                elif prefill_timing_mode == "eager_pack":
                    pack_n = _decode_pack_n(
                        state_bytes=batch_size * seq_len * kv_bytes_per_token
                    )
                    repeat_n = 1
                    timing_label = f"eager_pack{pack_n}"

                    def run():
                        for _ in range(pack_n):
                            kernel_func()
                else:
                    # graph requested but the runner dispatch says eager
                    # (batch*query above MAX_CUDAGRAPH_CAPTURE_SIZE, or the
                    # platform has no graph support) — degrade and record what
                    # actually ran (used_cuda_graph / timing_mode).
                    pack_n = 1
                    repeat_n = 10
                    timing_label = "eager_repeat10"
                    run = kernel_func
            elif graph_mode == "full":
                # Per-call KV read: index_topk tokens from the main cache plus
                # the sliding window from the SWA cache, both in the same
                # per-token bytes. Bytes/token come from the framework spec at
                # runtime (584 = 448B NoPE fp8 + 128B RoPE bf16 + 8B scale for
                # the Flash geometry) — never hardcoded.
                state_bytes = (
                    batch_size
                    * (int(hf_config.index_topk) + int(hf_config.sliding_window))
                    * kv_bytes_per_token
                )
                pack_n = _decode_pack_n(state_bytes=state_bytes)
                repeat_n = 1

                def run():
                    for _ in range(pack_n):
                        kernel_func()
            else:
                pack_n = 1
                repeat_n = 10
                run = kernel_func

            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=run,
                num_warmups=warming_up,
                num_runs=test_ite,
                repeat_n=repeat_n,
                use_cuda_graph=capture is None and graph_mode == "full",
                allow_graph_fail=False,
            ) as result:
                pass

    used_graph = capture is not None or bool(result.get("used_cuda_graph", False))
    timing_mode = timing_label or ("graph_pack" if used_graph else "eager_repeat10")
    latency = float(result["latency_ms"]) / pack_n
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gemm_type": gemm_type,
                "num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": seq_len if is_context else 1,
                "tp_size": tp_size,
                "step": prefix_len if is_context else seq_len,
                "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                "latency": f"{latency:.4f}",
                "used_cuda_graph": used_graph,
                # Measurement-method self-description (AIC_DSV4_PREFILL_TIMING
                # design): rows must be attributable to a timing method without
                # the run log. Context rows carry the switch-resolved method
                # (graph_piecewise / eager_repeat10 / eager_packN); generation
                # rows carry their effective method (graph_pack / eager).
                "timing_mode": timing_mode,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=get_device_module(device).get_device_name(),
        op_name=f"dsv4_{attn_kind}_{mode}_module",
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )
    print(
        f"[vllm-dsv4-xpu] {attn_kind} {mode} b={batch_size} s={seq_len} prefix={prefix_len} "
        f"heads={local_num_heads} backend={backend_name} pack_n={pack_n} "
        f"graph={graph_mode} timing={timing_mode} graph_exec={used_graph} "
        f"latency={latency:.4f} ms",
        flush=True,
    )
    del attn_module, vllm_config, metadata, common, hidden_states, positions, capture
    get_device_module(device).empty_cache()
    gc.collect()
    return latency


def _kv_cache_cast_to_fp8_indexer(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = x_amax / 448.0
    x_scaled = (x * (1.0 / scale)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    out[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    out[:, block_size * head_dim :] = scale.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def _bench_paged_mqa_logits_kernel(
    *,
    hf_config,
    num_query_rows: int,
    past_kv: int,
    device: str,
    warming_up: int,
    test_ite: int,
) -> tuple[dict, int]:
    """Benchmark paged MQA logits with packed M rows across the batch.

    deepklox bare core (audited deltas vs the NV DeepGEMM call): the import
    source is deepklox, ``context_lens`` must be ``[B]`` (a ``[B,1]`` view
    raises RuntimeError: "context_lens must be [batch_size]"), and the
    schedule metadata is sized by the device's own EU count (=256 on this
    part), not the NV SM-count translation.
    """
    m = num_query_rows
    full_s = m + past_kv
    full_c4 = max(1, full_s // 4)
    block_kv = 64
    n_heads = int(hf_config.index_n_heads)
    head_dim = int(hf_config.index_head_dim)

    q_bf16 = torch.randn(m, 1, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_quant = q_bf16.to(torch.float8_e4m3fn)

    blocks_per_req = (full_c4 + block_kv - 1) // block_kv
    kv_bf16 = torch.randn(
        blocks_per_req,
        block_kv,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = _kv_cache_cast_to_fp8_indexer(kv_bf16)

    weights = torch.randn(m, n_heads, dtype=torch.float32, device=device)
    causal_seq = torch.arange(
        past_kv + 1,
        past_kv + m + 1,
        dtype=torch.int32,
        device=device,
    )
    context_lens = (causal_seq // 4).clamp(min=1).view(m)

    block_table = torch.arange(blocks_per_req, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(m, blocks_per_req).contiguous()
    num_sms = torch.xpu.get_device_properties(torch.device(device)).gpu_eu_count
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_sms,
    )

    def kernel_func():
        return fp8_fp4_paged_mqa_logits(
            (q_quant, None),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_metadata,
            max_model_len=full_c4,
            clean_logits=False,
        )

    with torch.inference_mode():
        kernel_func()
        get_device_module(device).synchronize()
        use_graph = xpu_graph_measure_enabled()
        if use_graph:
            # Own traffic term (report §4.5): logical KV read — block-table
            # rows are shared across queries and not deduplicated — plus the
            # fp32 logits write. 132B/token = index_head_dim 128 fp8 + 4B
            # dequant scale (deepklox packed layout).
            avg_ctx = (past_kv + m / 2) / 4
            total_bytes = m * avg_ctx * (head_dim + 4) + m * full_c4 * 4
            pack_n = _decode_pack_n(state_bytes=int(total_bytes))
            repeat_n = 1

            def run():
                for _ in range(pack_n):
                    kernel_func()
        else:
            pack_n = 1
            repeat_n = 10
            run = kernel_func
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=run,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=repeat_n,
            use_cuda_graph=use_graph,
            allow_graph_fail=False,
        ) as result:
            pass
    if use_graph and not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use XPU graph")
    return result, pack_n


def _bench_mla_sparse_op(
    *,
    attn_module: DeepseekV4XPUAttention,
    vllm_config,
    metadata,
    positions: torch.Tensor,
    num_tokens: int,
    device: str,
    warming_up: int,
    test_ite: int,
    is_decode: bool,
) -> tuple[dict, int]:
    sparse_attn = attn_module
    q = torch.full(
        (num_tokens, sparse_attn.padded_heads, sparse_attn.head_dim),
        0.01,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.zeros((num_tokens, sparse_attn.head_dim), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)

    with set_current_vllm_config(vllm_config), set_forward_context(metadata, vllm_config), torch.inference_mode():
        sparse_attn.forward_mqa(q, kv, positions, output)
        get_device_module(device).synchronize()

        def kernel_func():
            sparse_attn.forward_mqa(q, kv, positions, output)

        # decode (isl=1) runs under XPU graph; the prefill chunk path
        # allocates workspace inside the call, which is illegal during capture
        # — prefill stays eager (report §4.5). Identical to the caller's
        # capture_planned (which also drives the single-stream construction).
        use_graph = xpu_graph_measure_enabled() and is_decode
        if use_graph:
            # forward_mqa reads only the already-gathered q/kv/output tensors
            # passed in; per-call traffic is those tensors' bytes.
            state_bytes = (q.numel() + kv.numel() + output.numel()) * q.element_size()
            pack_n = _decode_pack_n(state_bytes=state_bytes)
            repeat_n = 1

            def run():
                for _ in range(pack_n):
                    kernel_func()
        else:
            pack_n = 1
            repeat_n = 10
            run = kernel_func

        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=run,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=repeat_n,
            use_cuda_graph=use_graph,
            allow_graph_fail=False,
        ) as result:
            pass
    return result, pack_n


def _bench_sparse_kernel_shape(
    *,
    model_path: str,
    kernel: str,
    batch_size: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    device: str,
    perf_filename: str,
    warming_up: int,
    test_ite: int,
) -> float | None:
    if kernel not in SPARSE_KERNEL_TO_ATTN_KIND:
        raise ValueError(f"unknown sparse kernel={kernel}")
    full_seq_len = past_kv + isl
    if full_seq_len <= 0:
        raise ValueError(f"invalid sparse sequence length: isl={isl}, past_kv={past_kv}")

    attn_kind = SPARSE_KERNEL_TO_ATTN_KIND[kernel]
    is_context = isl > 1
    query_len = isl
    num_tokens = batch_size * query_len

    if kernel == "paged_mqa_logits":
        hf_config = SimpleNamespace(**_read_model_config(model_path))
        architecture = hf_config.architectures[0] if hf_config.architectures else ARCHITECTURE
        native_num_heads = int(hf_config.num_attention_heads)
        local_num_heads = native_num_heads // tp_size
        backend_name = SPARSE_KERNEL_TO_KERNEL_SOURCE[kernel]
        result, pack_n = _bench_paged_mqa_logits_kernel(
            hf_config=hf_config,
            num_query_rows=num_tokens,
            past_kv=past_kv,
            device=device,
            warming_up=warming_up,
            test_ite=test_ite,
        )
    else:
        # isl==1 (decode) captures an XPU graph -> single-stream construction
        # (serving parity under capture); isl>1 (prefill chunk) stays eager
        # with the serving aux streams.
        capture_planned = xpu_graph_measure_enabled() and isl == 1
        with (
            _tp_simulation(tp_size),
            _create_dsv4_attention_module(
                model_path=model_path,
                attn_kind=attn_kind,
                # Sparse-kernel benches measure the deepklox kernel directly;
                # the module is only scaffolding for cache/metadata, so pin
                # the serving mxfp8 quant path for its construction (the
                # checkpoint's fp8_block config dir is not needed here).
                gemm_type="mxfp8",
                batch_size=batch_size,
                seq_len=full_seq_len,
                tp_size=tp_size,
                is_context=is_context,
                device=device,
                query_len=query_len,
                multi_stream=not capture_planned,
            ) as (attn_module, vllm_config),
        ):
            common = _make_common_metadata(
                batch_size=batch_size,
                seq_len=full_seq_len,
                is_context=is_context,
                query_len=query_len,
                device=device,
            )
            metadata = _build_metadata_and_bind_caches(attn_module, vllm_config, common, device=device)
            positions = common.positions
            hf_config = vllm_config.model_config.hf_config
            architecture = hf_config.architectures[0] if hf_config.architectures else ARCHITECTURE
            native_num_heads = int(hf_config.num_attention_heads)
            local_num_heads = int(attn_module.n_local_heads)
            concrete_layer = vllm_config.compilation_config.static_forward_context[attn_module.prefix]
            backend_name = concrete_layer.get_attn_backend().get_name()
            result, pack_n = _bench_mla_sparse_op(
                attn_module=attn_module,
                vllm_config=vllm_config,
                metadata=metadata,
                positions=positions,
                num_tokens=num_tokens,
                device=device,
                warming_up=warming_up,
                test_ite=test_ite,
                is_decode=isl == 1,
            )
            del attn_module, vllm_config, metadata, common, positions

    latency = float(result["latency_ms"]) / pack_n

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "fp8_e4m3" if kernel == "paged_mqa_logits" else "bfloat16",
                "kv_cache_dtype": "fp8",
                "gemm_type": "fp8_block",
                # Sparse-op consumers key ``num_heads`` by the native model
                # count. Keep that contract and record the TP-local count too.
                "num_heads": native_num_heads,
                "local_num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": past_kv,
                "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                "latency": f"{latency:.4f}",
                "used_cuda_graph": result.get("used_cuda_graph", False),
                "timing_mode": (
                    "graph_pack" if result.get("used_cuda_graph", False) else "eager_repeat10"
                ),
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=get_device_module(device).get_device_name(),
        op_name=SPARSE_KERNEL_TO_OP_NAME[kernel],
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )
    print(
        f"[vllm-dsv4-xpu] {kernel} b={batch_size} isl={isl} past_kv={past_kv} "
        f"tp={tp_size} local_heads={local_num_heads} backend={backend_name} "
        f"pack_n={pack_n} timing={'graph_pack' if result.get('used_cuda_graph', False) else 'eager_repeat10'} "
        f"graph={result.get('used_cuda_graph', False)} latency={latency:.4f} ms",
        flush=True,
    )
    get_device_module(device).empty_cache()
    gc.collect()
    return latency


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    attention_backend: str | None = None,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "xpu:0",
) -> None:
    if attn_kind not in ATTN_KIND_TO_COMPRESS_RATIO:
        raise ValueError(f"unknown attn_kind={attn_kind}")
    if tp_size not in _DSV4_MODULE_TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}")
    if kv_cache_dtype != "fp8":
        raise ValueError(f"unsupported vLLM DSV4 kv_cache_dtype={kv_cache_dtype}; expected fp8")
    if compute_dtype != "bfloat16":
        raise ValueError(f"unsupported vLLM DSV4 compute_dtype={compute_dtype}; expected bfloat16")
    if gemm_type not in SUPPORTED_GEMM_TYPES:
        raise ValueError(f"unsupported vLLM DSV4 gemm_type={gemm_type}; supported={sorted(SUPPORTED_GEMM_TYPES)}")
    if attention_backend is not None:
        raise ValueError(
            f"vLLM DSV4 attention_backend must be unset because vLLM selects it internally; got {attention_backend!r}"
        )

    mode = "context" if "context" in os.path.basename(perf_filename) else "generation"
    _init_xpu(device)
    try:
        _bench_attention_shape(
            model_path=model_path,
            attn_kind=attn_kind,
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            prefix_len=prefix_len,
            tp_size=tp_size,
            gemm_type=gemm_type,
            device=device,
            perf_filename=perf_filename,
            warming_up=3 if "--smoke" in sys.argv else 5,
            test_ite=3 if "--smoke" in sys.argv else 10,
        )
    except torch.OutOfMemoryError:
        # torch.xpu.OutOfMemoryError does not exist on this torch; the unified
        # torch.OutOfMemoryError covers the XPU path.
        print(f"[vllm-dsv4-xpu] OOM: {attn_kind} {mode} b={batch_size} s={seq_len} prefix={prefix_len}", flush=True)
        get_device_module(device).empty_cache()
        raise
    except Exception:
        traceback.print_exc()
        raise


def run_dsv4_sparse_kernel_worker(
    batch_size: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    kernel: str,
    model_path: str,
    *,
    perf_filename: str,
    device: str = "xpu:0",
) -> None:
    if kernel not in SPARSE_KERNEL_TO_ATTN_KIND:
        raise ValueError(f"unknown sparse kernel={kernel}")
    if tp_size not in _DSV4_MODULE_TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}")
    full_s = isl + past_kv
    if full_s > _DSV4_SPARSE_MAX_FULL_S:
        raise ValueError(
            f"{kernel} b={batch_size} isl={isl} past_kv={past_kv} has "
            f"full_s={full_s} > max_position_embeddings={_DSV4_SPARSE_MAX_FULL_S}"
        )
    _init_xpu(device)
    try:
        _bench_sparse_kernel_shape(
            model_path=model_path,
            kernel=kernel,
            batch_size=batch_size,
            isl=isl,
            past_kv=past_kv,
            tp_size=tp_size,
            device=device,
            perf_filename=perf_filename,
            warming_up=3 if "--smoke" in sys.argv else 5,
            test_ite=3 if "--smoke" in sys.argv else 10,
        )
    except torch.OutOfMemoryError:
        print(f"[vllm-dsv4-xpu] OOM: {kernel} b={batch_size} isl={isl} past_kv={past_kv} tp={tp_size}", flush=True)
        get_device_module(device).empty_cache()
        raise
    except Exception:
        traceback.print_exc()
        raise


# collect.py resolves get_func AND run_func from the OpEntry's own module, so
# the XPU module exposes the case builders under the same names as the CUDA
# registry (gdn precedent). These wrap the yaml-driven grids with the two
# XPU-only pre-queue filters: deepklox's fused-prefill int32 overflow and the
# generation-time memory-feasibility filter (drops are counted + logged, per
# layer_permissions.md's sanctioned filter).


def _get_test_cases(attn_kind: str, mode: str) -> list[list]:
    cases = []
    skipped: dict[str, list] = {}
    for case in get_xpu_dsv4_attn_test_cases(attn_kind, mode):
        seq_len, batch_size, tp_size, model_path = case[0], case[1], case[2], case[6]
        query_len = seq_len if mode == "context" else 1
        if SKIP_PREFILL_OVERFLOW and _fused_prefill_overflows(model_path, batch_size, query_len, tp_size):
            reason = (
                "fused sparse_mla_prefill int32 overflow (deepklox bug; run with "
                "AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS=8192, or AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW=0 once fixed)"
            )
        elif mode == "context" and _context_exceeds_memory(model_path, attn_kind, batch_size, query_len, tp_size):
            reason = f"estimated forward peak > {MAX_MEM_FRACTION:.0%} of device memory (XPU OOM = DEVICE_LOST)"
        else:
            cases.append(case)
            continue
        skipped.setdefault(reason, []).append((model_path.split("/")[-1], batch_size, seq_len, tp_size))
    for reason, shapes in skipped.items():
        print(
            f"[vllm-xpu-dsv4] dsv4_{attn_kind}_{mode}_module: skipping {len(shapes)} (model, batch, isl, tp) "
            f"shapes, {reason}: {shapes}",
            flush=True,
        )
    return cases


def get_dsv4_csa_context_test_cases():
    return _get_test_cases("csa", "context")


def get_dsv4_swa_context_test_cases():
    return _get_test_cases("swa", "context")


def get_dsv4_hca_context_test_cases():
    return _get_test_cases("hca", "context")


def get_dsv4_csa_generation_test_cases():
    return _get_test_cases("csa", "generation")


def get_dsv4_swa_generation_test_cases():
    return _get_test_cases("swa", "generation")


def get_dsv4_hca_generation_test_cases():
    return _get_test_cases("hca", "generation")


get_dsv4_paged_mqa_logits_test_cases = get_xpu_dsv4_paged_mqa_logits_test_cases
get_dsv4_hca_attn_test_cases = get_xpu_dsv4_hca_attn_test_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one vLLM-XPU DeepSeek-V4 attention-module shape.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--attn-kind", choices=list(ATTN_KIND_TO_COMPRESS_RATIO), default="csa")
    parser.add_argument("--sparse-kernel", choices=list(DSV4_SPARSE_KERNELS), default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--past-kv", type=int, default=0, help="context prefix length")
    parser.add_argument("--seq-len", type=int, default=64, help="context: isl; generation: decode step (past KV)")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gemm-type", choices=sorted(SUPPORTED_GEMM_TYPES), default="mxfp8")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--output-path", default=None, help="directory (or *.txt file) for the perf rows")

    args = parser.parse_args()
    if args.gemm_type not in SUPPORTED_GEMM_TYPES:
        raise ValueError(f"unsupported vLLM DSV4 gemm_type={args.gemm_type}; supported={sorted(SUPPORTED_GEMM_TYPES)}")
    perf_filename = PERF_FILES[(args.attn_kind, args.mode)].value
    if args.output_path:
        if args.output_path.endswith(".txt"):
            perf_filename = args.output_path
        else:
            os.makedirs(args.output_path, exist_ok=True)
            perf_filename = os.path.join(args.output_path, perf_filename)
    query_len = args.seq_len if args.mode == "context" else 1
    if SKIP_PREFILL_OVERFLOW and _fused_prefill_overflows(args.model_path, args.batch_size, query_len, args.tp_size):
        raise SystemExit(
            "shape overflows the fused sparse_mla_prefill int32 offsets (deepklox bug); set "
            "AIC_VLLM_DSV4_PREFILL_CHUNK_TOKENS=8192, or AIC_VLLM_DSV4_SKIP_PREFILL_OVERFLOW=0 once fixed"
        )
    if args.mode == "context" and _context_exceeds_memory(
        args.model_path, args.attn_kind, args.batch_size, query_len, args.tp_size
    ):
        raise SystemExit(f"estimated forward peak > {MAX_MEM_FRACTION:.0%} of device memory (XPU OOM = DEVICE_LOST)")
    _init_xpu(args.device)
    if args.sparse_kernel is not None:
        sparse_filename = SPARSE_KERNEL_TO_PERF_FILE[args.sparse_kernel].value
        perf_filename = _resolve_perf_path(args.output_path, sparse_filename)
        _bench_sparse_kernel_shape(
            model_path=args.model_path,
            kernel=args.sparse_kernel,
            batch_size=args.batch_size,
            isl=args.seq_len,
            past_kv=args.past_kv,
            tp_size=args.tp_size,
            device=args.device,
            perf_filename=perf_filename,
            warming_up=3,
            test_ite=3,
        )
        return

    _bench_attention_shape(
        model_path=args.model_path,
        attn_kind=args.attn_kind,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        prefix_len=args.past_kv if args.mode == "context" else 0,
        tp_size=args.tp_size,
        gemm_type=args.gemm_type,
        device=args.device,
        perf_filename=perf_filename,
        warming_up=3,
        test_ite=3,
    )


if __name__ == "__main__":
    main()
