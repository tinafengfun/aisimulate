# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) DSA module-level collector, v2 (GLM-5.3 glm_moe_dsa family).

Redesigned against vllm-xpu main 0.28.0.dev0+g03bf988 (the prior
``collect_mla_module_xpu.py`` targeted the pre-upgrade image wheel and is left
untouched). Mirrors the CUDA ``collect_mla_module.py`` op semantics and row
shape: builds one ``DeepseekV2MLAAttention`` from vLLM's own modeling code and
times the complete forward (projections + indexer + sparse attention + output).
XPU deltas vs the CUDA collector:

* KV cache is bf16-only — XPU_MLA_SPARSE raises on fp8 KV
  (xpu_mla_sparse.py supported_dtypes). fp8 KV is a planned backend
  improvement (2026-10-18); widen the kv grid here when it lands.
* GEMM grid: bfloat16 + mxfp8 (online MXFP8 via the ``--quantization mxfp8``
  shorthand); the CUDA fp8_block recipe has no XPU serving path (DSV4
  precedent).
* Data construction is seeded-random — weights seed 0, hidden states seed 1,
  KV contexts seed 2 (``AIC_MLA_MODULE_DUMMY_DATA=const`` restores the CUDA
  collector's constant fill). Constant inputs risk idle-compute artifacts on
  Xe (constant-page memory compression); the CUDA collector cannot seed
  (vLLM #39371 graph-RNG crash) — declared NV↔XPU caliber difference.
* Timing: full XPU graph for both phases (XPU_MLA_SPARSE is UNIFORM_BATCH; the
  NV DSA-context eager carve-out has no XPU equivalent). Forwards are packed
  per replay until ``AIC_MLA_MODULE_PACK_TARGET_MS``; 0 restores the CUDA
  collector's one-forward-per-replay method.
* Context shapes with rows*heads*576 >= 2^31 (s >= 76800) stress the deepklox
  fused-prefill int32 offsets. The collector does not pre-disable the fused
  path (no serving-equivalent dispatch rule); if the image's deepklox predates
  the int64-offset fix, those cases fault at runtime — classified failure +
  JIRA material, per the 2026-10-18 design decision 6.

Fail-loud: the DSA backend must resolve to XPU_MLA_SPARSE. A config that
loses ``index_topk`` silently routes to TRITON_MLA (dense MLA) and would
record wrong-kernel rows.
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import order is load-bearing: torch first (deepklox's _C extension needs
# libc10), then vllm_xpu_kernels (registers torch.ops._xpu_C); reversing the
# two segfaults the interpreter.
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401
from collector.case_generator import get_xpu_mla_module_test_cases
from collector.helper import (
    benchmark_with_power,
    get_device_module,
    log_perf,
    xpu_graph_measure_enabled,
)
from collector.registry_types import PerfFile
from collector.vllm.utils_xpu import (
    BatchSpec,
    create_and_prepopulate_kv_cache_mla,
    create_common_attn_metadata,
    create_vllm_config,
    setup_distributed,
    with_exit_stack,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

__compat__ = "vllm==0.28.0.dev0+g03bf988.xpu"

SUPPORTED_GEMM_TYPES = ("bfloat16", "mxfp8")

# Forwards per graph replay grow until one replay lasts >= PACK_TARGET_MS
# (default 15, CRI production: amortizes the ~20us XPU replay launch cost that
# inflates sub-ms cases; 0 = one forward per replay, the CUDA collector's
# measurement method).
PACK_TARGET_MS = float(os.environ.get("AIC_MLA_MODULE_PACK_TARGET_MS", "15"))
PACK_MAX = 64
# >0: read an N MiB buffer before every forward to evict the last-level cache;
# its cost is subtracted. Investigation knob for cold-cache pairing (G3).
FLUSH_CACHE_MB = int(os.environ.get("AIC_MLA_MODULE_FLUSH_CACHE_MB", "0"))
DUMMY_DATA = os.environ.get("AIC_MLA_MODULE_DUMMY_DATA", "random")
if DUMMY_DATA not in ("random", "const"):
    raise ValueError(f"AIC_MLA_MODULE_DUMMY_DATA must be 'random' or 'const', got {DUMMY_DATA!r}")

# vLLM registers GlmMoeDsaForCausalLM but omits the "glm_moe_dsa" config-type
# mapping; the layout matches DeepSeek-V3, so reuse DeepseekV3Config.
if "glm_moe_dsa" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"


_MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"
_local_config_cache: dict[str, str] = {}


def _resolve_model_path(model_name: str) -> str:
    """Return a local dir with config.json for *model_name* if cached, else the name."""
    if model_name in _local_config_cache:
        return _local_config_cache[model_name]
    config_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_config.json"
    if not config_file.exists():
        return model_name
    tmp_dir = tempfile.mkdtemp(prefix=f"aic_model_{model_name.replace('/', '_')}_")
    os.symlink(config_file, os.path.join(tmp_dir, "config.json"))
    # Strip auto_map: vLLM natively supports these architectures and only needs
    # the JSON fields; the auto_map import target is absent from the temp dir.
    with open(config_file) as f:
        config_data = json.load(f)
    if "auto_map" in config_data:
        config_data.pop("auto_map")
        os.remove(os.path.join(tmp_dir, "config.json"))
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_data, f)
    _local_config_cache[model_name] = tmp_dir
    return tmp_dir


def _context_activation_footprint_bytes(total_tokens: int, num_heads: int) -> int:
    """Lower bound of a context case's peak transient activation set.

    The module forward coexists (at least partially): concatenated q
    [tokens, heads, 576], q_b_proj out [.., 256], kv_b_proj out [.., 448],
    sparse-attn out [.., 512] and the o_proj input [.., 256] — 2048 bf16
    elements per token per head. Weights/KV cache/workspace are excluded so
    the estimate is a lower bound on the forward's transient demand.
    Validated on the 160 GB CRI part: the boundary matches measurement
    exactly — b*s = 614,400 (b=8, s=76800) DEVICE_LOSTs, b*s = 409,600
    (b=4, s=102400) passes (2026-10-18 grid run).
    """
    return total_tokens * num_heads * (576 + 256 + 448 + 512 + 256) * 2


def _filter_context_cases_by_memory(cases: list) -> list:
    """Drop context cases whose activation lower bound exceeds the device budget.

    The one sanctioned in-collector filter (layer_permissions.md): generation
    time only, size-vs-capacity predicate, drops counted in the log — never a
    runtime skip. Capacity is queried live; no device (case-expansion dry run
    off-GPU) means no filter.
    """
    try:
        total_memory = get_device_module().get_device_properties(0).total_memory
    except Exception:
        return cases
    budget = int(total_memory * 0.9)
    kept, dropped = [], 0
    for case in cases:
        seq_len, batch_size, num_heads = case[0], case[1], case[2]
        if _context_activation_footprint_bytes(seq_len * batch_size, num_heads) > budget:
            dropped += 1
            continue
        kept.append(case)
    if dropped:
        print(
            f"dsa_context_module: dropped {dropped}/{len(cases)} cases "
            f"(memory budget, device={total_memory / 2**30:.0f}GiB)"
        )
    return kept


def get_dsa_context_module_test_cases():
    """collect.py entrypoint for DSA context module collection (XPU)."""
    return _filter_context_cases_by_memory(get_xpu_mla_module_test_cases("context"))


def get_dsa_generation_module_test_cases():
    """collect.py entrypoint for DSA generation module collection (XPU)."""
    return get_xpu_mla_module_test_cases("generation")


def _init_dummy_module_tensors(module: torch.nn.Module) -> None:
    # Constant/zero data lowers switching power, so matrices are seeded-random
    # by default (seed 0); scales and 1D tensors keep the CUDA collector's
    # constant values. The CUDA collector fills everything because its DSA
    # modules crash on any RNG call (vLLM #39371); seeding here happens at
    # construction, before graph capture, so it is safe on XPU.
    randomize = DUMMY_DATA == "random"
    generators = {}
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor.is_meta:
                continue
            matrix = randomize and tensor.dim() >= 2 and "scale" not in name
            if matrix and tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float16, torch.float32):
                gen = generators.setdefault(tensor.device, torch.Generator(device=tensor.device).manual_seed(0))
                std = 0.05 if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0.02
                tensor.copy_((torch.randn(tensor.shape, generator=gen, device=tensor.device) * std).to(tensor.dtype))
            elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.zero_()
            elif tensor.dtype == torch.float8_e8m0fnu:
                tensor.view(torch.uint8).fill_(127)  # e8m0 exponent 127 == 1.0
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.fill_(1.0)
            else:
                tensor.fill_(0.01)


def _dummy_hidden_states(num_tokens: int, hidden_size: int, device: str) -> torch.Tensor:
    if DUMMY_DATA == "const":
        return torch.full((num_tokens, hidden_size), 0.01, dtype=torch.bfloat16, device=device)
    gen = torch.Generator(device=device).manual_seed(1)
    return torch.randn((num_tokens, hidden_size), generator=gen, dtype=torch.bfloat16, device=device)


def _create_attention_module(
    model_path: str,
    num_heads: int,
    gemm_type: str,
    max_seq_len: int,
    max_batch_size: int,
    device: str,
):
    """Build a DeepseekV2MLAAttention module with dummy weights from vLLM modeling code."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype

    local_model_path = _resolve_model_path(model_path)
    block_size = 64
    max_model_len = max(max_seq_len, 4096)
    num_kv_cache_blocks = max(1 + math.ceil((max_seq_len + 1) / block_size) * max_batch_size, 8192)

    # mxfp8: the online-shorthand path (dynamic MXFP8 of bf16 weights, oneDNN
    # fp8_gemm) — the XPU serving recipe; bfloat16: unquantized.
    vllm_config = create_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=max(max_batch_size * max_seq_len, 131072),
        use_fp8_kv_cache=False,
        trust_remote_code=True,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        quantization="mxfp8" if gemm_type == "mxfp8" else None,
    )

    hf_config = vllm_config.model_config.hf_text_config
    hf_config.num_hidden_layers = 1
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_heads

    topk_indices_buffer = None
    if hasattr(hf_config, "index_topk"):
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = torch.empty(max_tokens, hf_config.index_topk, dtype=torch.int32, device=device)

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
        attn_module = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=hf_config,
            hidden_size=hf_config.hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            q_lora_rank=hf_config.q_lora_rank if hasattr(hf_config, "q_lora_rank") else None,
            kv_lora_rank=hf_config.kv_lora_rank,
            max_position_embeddings=hf_config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

    # Serialized block-scaled FP8 creates params on meta device; to() can't copy them.
    if any(p.is_meta for p in attn_module.parameters()):
        attn_module = attn_module.to_empty(device=torch.device(device))
    else:
        attn_module = attn_module.to(device)
    attn_module.eval()
    attn_module.requires_grad_(False)
    _init_dummy_module_tensors(attn_module)
    return attn_module, vllm_config


def _process_module_weights(attn_module, vllm_config):
    """Run process_weights_after_loading (quant + MLAAttention W_UK_T / W_UV)."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        for _, module in attn_module.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)
        for _, module in attn_module.named_modules():
            if isinstance(module, MLAAttention):
                module.process_weights_after_loading(vllm_config.model_config.dtype)


def _create_context_kv_inputs(batch_spec: BatchSpec, kv_lora_rank: int, qk_rope_head_dim: int, device: str):
    """Cached KV tensors for tokens already in the paged cache before this forward.

    Context phase: empty (prefix_len=0 grid) — all tokens are processed in the
    same prefill step. Generation phase: the cache holds seq_len - 1 historical
    tokens and the current forward processes exactly 1 new token per request.
    Seeded random (seed 2): the sparse MQA reads these rows, so constant fill
    risks the idle-compute artifacts the weights seeding avoids.
    """
    kv_c_contexts, k_pe_contexts = [], []
    gen = None if DUMMY_DATA == "const" else torch.Generator(device=device).manual_seed(2)
    for seq_len, query_len in zip(batch_spec.seq_lens, batch_spec.query_lens, strict=True):
        context_len = max(0, int(seq_len) - int(query_len))
        if DUMMY_DATA == "const":
            kv_c_contexts.append(torch.full((context_len, kv_lora_rank), 0.01, dtype=torch.bfloat16, device=device))
            k_pe_contexts.append(torch.full((context_len, 1, qk_rope_head_dim), 0.01, dtype=torch.bfloat16, device=device))
        else:
            kv_c_contexts.append(
                torch.randn((context_len, kv_lora_rank), generator=gen, dtype=torch.bfloat16, device=device)
            )
            k_pe_contexts.append(
                torch.randn((context_len, 1, qk_rope_head_dim), generator=gen, dtype=torch.bfloat16, device=device)
            )
    return kv_c_contexts, k_pe_contexts


def _populate_indexer_kv_cache(indexer_kv_cache, common_attn_metadata, context_lens):
    """Fill the DSA indexer cache so decode sees historical K, not zeros.

    Constant 42, matching the CUDA collector (both sides see the same
    degenerate-history top-k, so the caliber stays paired; the indexer's
    value sensitivity is covered by the DUMMY_DATA const/random spot check).
    """
    block_table = common_attn_metadata.block_table_tensor
    block_size = indexer_kv_cache.shape[1]
    entry_dim = indexer_kv_cache.shape[2]
    device = indexer_kv_cache.device
    for i, context_len in enumerate(context_lens):
        if context_len <= 0:
            continue
        token_offsets = torch.arange(context_len, dtype=torch.long, device=device)
        block_indices = token_offsets // block_size
        intra = token_offsets % block_size
        block_ids = block_table[i, block_indices]
        indexer_kv_cache[block_ids, intra, :] = torch.full(
            (context_len, entry_dim), 42, dtype=torch.uint8, device=device
        )


def _create_kv_cache_and_metadata(vllm_config, batch_size, seq_len, is_context, prefix_len, device):
    """Build the paged KV cache + attention/indexer metadata for one case."""
    hf_config = vllm_config.model_config.hf_text_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    block_size = vllm_config.cache_config.block_size
    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        batch_spec = BatchSpec(seq_lens=[prefix_len + seq_len] * batch_size, query_lens=[seq_len] * batch_size)
    else:
        batch_spec = BatchSpec(seq_lens=[seq_len] * batch_size, query_lens=[1] * batch_size)

    num_kv_cache_blocks = max(1 + math.ceil((prefix_len + seq_len + 1) / block_size) * batch_size, 8192)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, torch.device(device), arange_block_indices=True
    )

    # Pad the page table's block dim to a multiple of 128/block_size (kernels
    # that require it otherwise raise on single-block short sequences).
    required_divisor = max(1, 128 // block_size)
    current_block_num = common_attn_metadata.block_table_tensor.shape[1]
    if current_block_num % required_divisor != 0:
        padded = ((current_block_num + required_divisor - 1) // required_divisor) * required_divisor
        padding = torch.zeros(
            (common_attn_metadata.block_table_tensor.shape[0], padded - current_block_num),
            dtype=common_attn_metadata.block_table_tensor.dtype,
            device=common_attn_metadata.block_table_tensor.device,
        )
        common_attn_metadata.block_table_tensor = torch.cat([common_attn_metadata.block_table_tensor, padding], dim=1)

    attn_layer_name = "model.layers.0.self_attn.attn"
    attn_layer = vllm_config.compilation_config.static_forward_context[attn_layer_name]
    backend_cls = attn_layer.get_attn_backend()
    kv_cache_spec = attn_layer.get_kv_cache_spec(vllm_config)
    cache_dtype = kv_cache_spec.dtype
    kv_cache_dtype_str = kv_cache_spec.cache_dtype_str

    kv_c_contexts, k_pe_contexts = _create_context_kv_inputs(
        batch_spec, kv_lora_rank, qk_rope_head_dim, device
    )
    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=kv_cache_spec.head_size,
        dtype=cache_dtype,
        device=torch.device(device),
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype_str,
        scale=attn_layer._k_scale,
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, [attn_layer_name], vllm_config, torch.device(device))
    attn_metadata = builder.build(common_prefix_len=prefix_len, common_attn_metadata=common_attn_metadata)

    indexer_kv_cache = None
    indexer_metadata = None
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    if indexer_layer_name in vllm_config.compilation_config.static_forward_context:
        indexer_layer = vllm_config.compilation_config.static_forward_context[indexer_layer_name]
        indexer_spec = indexer_layer.get_kv_cache_spec(vllm_config)
        indexer_kv_cache = torch.zeros(
            num_kv_cache_blocks, block_size, indexer_spec.head_size, dtype=indexer_spec.dtype, device=device
        )
        indexer_builder_cls = indexer_layer.get_attn_backend().get_builder_cls()
        builder_kwargs = {}
        if getattr(indexer_builder_cls, "requires_block_table_width", False):
            builder_kwargs["block_table_width"] = common_attn_metadata.block_table_tensor.shape[1]
        indexer_builder = indexer_builder_cls(
            indexer_spec, [indexer_layer_name], vllm_config, torch.device(device), **builder_kwargs
        )
        indexer_metadata = indexer_builder.build(
            common_prefix_len=prefix_len, common_attn_metadata=common_attn_metadata
        )
        _populate_indexer_kv_cache(indexer_kv_cache, common_attn_metadata, [t.shape[0] for t in kv_c_contexts])

    return kv_cache, attn_metadata, common_attn_metadata, indexer_kv_cache, indexer_metadata


def _mla_backend_name(mla_layer):
    """Ground-truth backend for the perf row (DSA routes to attn_backend)."""
    return mla_layer.attn_backend.get_name()


def _bench_forward_graph(kernel_func, *, device: str, warming_up: int, test_ite: int) -> tuple[float, int, dict]:
    """Per-forward latency of ``kernel_func`` replayed from one full device graph.

    Forwards are packed into the graph until a replay lasts >= PACK_TARGET_MS, so the
    platform's per-replay launch cost (XPU ~20 us) does not leak into the
    per-layer number. With FLUSH_CACHE_MB, each forward starts from a cold last-level cache.
    """
    flush = None
    if FLUSH_CACHE_MB > 0:
        flush_buf = torch.ones(FLUSH_CACHE_MB << 19, dtype=torch.bfloat16, device=device)

        def flush():
            flush_buf.amax()

    def step():
        if flush is not None:
            flush()
        kernel_func()

    def measure(fn, pack: int) -> dict:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=fn,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=pack,
            use_cuda_graph=True,
            allow_graph_fail=False,
        ) as result:
            pass
        return result

    pack = 1
    result = measure(step, pack)
    if float(result["latency_ms"]) < PACK_TARGET_MS:
        pack = min(PACK_MAX, math.ceil(PACK_TARGET_MS / max(float(result["latency_ms"]), 1e-3)))
        result = measure(step, pack)
    latency = float(result["latency_ms"])
    if flush is not None:
        latency -= float(measure(flush, pack)["latency_ms"])
    return latency, pack, result


def _timing_mode(pack: int) -> str:
    return f"graph_full_pack{pack}" + (f"_flush{FLUSH_CACHE_MB}mb" if FLUSH_CACHE_MB > 0 else "")


@with_exit_stack
def run_mla_module(
    exit_stack,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "xpu:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run a single vLLM-XPU DSA module-level benchmark point."""
    if attn_type != "dsa":
        raise ValueError(f"v2 collector covers the glm_moe_dsa (dsa) family only; got {attn_type!r}")
    if kv_cache_dtype != "bfloat16":
        # XPU_MLA_SPARSE is bf16/fp16 KV only (xpu_mla_sparse.py); fp8 KV is a
        # planned backend improvement — fail here rather than at kernel dispatch.
        raise ValueError(f"XPU MLA Sparse KV cache is bfloat16-only; got {kv_cache_dtype!r}")
    if compute_dtype != "bfloat16":
        raise ValueError(f"XPU MLA query compute is bfloat16-only; got {compute_dtype!r}")
    if gemm_type not in SUPPORTED_GEMM_TYPES:
        raise ValueError(f"unsupported vLLM XPU gemm_type={gemm_type!r}; supported={SUPPORTED_GEMM_TYPES}")

    setup_distributed(device)
    get_device_module().set_device(device)
    # DSA's sparse_attn_indexer requires a WorkspaceManager.
    init_workspace_manager(torch.device(device))

    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"

    print(
        f"\n[DSA module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    attn_module, vllm_config = _create_attention_module(
        model_path=model_path,
        num_heads=num_heads,
        gemm_type=gemm_type,
        max_seq_len=prefix_len + seq_len,
        max_batch_size=batch_size,
        device=device,
    )
    _process_module_weights(attn_module, vllm_config)

    # Fail-loud backend guard: a config that loses index_topk routes to
    # TRITON_MLA (dense MLA, no indexer) and would record wrong-kernel rows.
    with set_current_vllm_config(vllm_config):
        attn_layer_name = "model.layers.0.self_attn.attn"
        attn_layer = vllm_config.compilation_config.static_forward_context[attn_layer_name]
        backend_name = _mla_backend_name(attn_layer)
        if backend_name != "XPU_MLA_SPARSE":
            raise RuntimeError(
                f"resolved attention backend {backend_name!r} is not XPU_MLA_SPARSE; "
                "the config lost index_topk (glm_moe_dsa parse failure) — refusing to "
                "record dense-MLA rows as DSA (fail-closed)"
            )
        if attn_module.indexer is None:
            raise RuntimeError(
                "layer 0 built without an indexer (skip_topk resolved True); the module "
                "caliber diverges from the CUDA collector's layer-0 module"
            )

        kv_cache, attn_metadata, common_attn_metadata, indexer_kv_cache, indexer_metadata = _create_kv_cache_and_metadata(
            vllm_config, batch_size, seq_len, is_context, prefix_len, device
        )

    forward_ctx = vllm_config.compilation_config.static_forward_context
    forward_ctx["model.layers.0.self_attn.attn"].kv_cache = kv_cache
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    if indexer_kv_cache is not None and indexer_layer_name in forward_ctx:
        forward_ctx[indexer_layer_name].kv_cache = indexer_kv_cache

    hidden_size = vllm_config.model_config.hf_text_config.hidden_size
    if is_context:
        num_tokens = seq_len * batch_size
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        positions = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
    hidden_states = _dummy_hidden_states(num_tokens, hidden_size, device)

    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    attn_metadata_dict = {"model.layers.0.self_attn.attn": attn_metadata}
    if indexer_metadata is not None:
        attn_metadata_dict[indexer_layer_name] = indexer_metadata
    exit_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))

    try:
        with torch.inference_mode():
            attn_module.forward(positions, hidden_states, None)
    except torch.OutOfMemoryError as e:
        print(f"  Dry run OOM: {e}")
        _cleanup()
        raise
    except Exception as e:
        print(f"  Dry run failed: {e}")
        traceback.print_exc()
        _cleanup()
        raise

    def kernel_func():
        attn_module.forward(positions, hidden_states, None)

    # Full XPU graph both phases: XPU_MLA_SPARSE is UNIFORM_BATCH and the
    # indexer's workspace is sized by the dry run, before capture.
    if xpu_graph_measure_enabled():
        latency, pack_n, result = _bench_forward_graph(
            kernel_func, device=device, warming_up=warming_up, test_ite=test_ite
        )
        timing_mode = _timing_mode(pack_n)
    else:
        pack_n = 1
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            use_cuda_graph=False,
            allow_graph_fail=False,
        ) as result:
            pass
        latency = float(result["latency_ms"])
        timing_mode = "eager_repeat1"
    used_graph = bool(result.get("used_cuda_graph", False))
    if "XPU graph" in timing_mode and not used_graph:
        raise RuntimeError("graph timing requested but benchmark_with_power fell back to eager")

    if is_context:
        isl, step = seq_len, prefix_len
    else:
        isl, step = 1, seq_len

    hf_cfg = vllm_config.model_config.hf_config
    architecture = getattr(hf_cfg, "architectures", [getattr(hf_cfg, "model_type", "unknown")])[0]
    actual_kv_cache_dtype = "fp8" if attn_layer.kv_cache_dtype.startswith("fp8") else "bfloat16"

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "bfloat16" if compute_dtype == "bfloat16" else compute_dtype,
                "kv_cache_dtype": actual_kv_cache_dtype,
                "gemm_type": "bfloat16" if gemm_type == "bfloat16" else gemm_type,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": 1,
                "step": step,
                "latency": f"{latency:.4f}",
                "used_cuda_graph": used_graph,
                "timing_mode": timing_mode,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=get_device_module().get_device_name(device),
        op_name=f"{attn_type}_{phase}_module",
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, prefix={prefix_len}, "
        f"gemm={gemm_type}, kv={kv_cache_dtype}, backend={backend_name}, "
        f"timing={timing_mode}, graph={used_graph}: {latency:.4f} ms"
    )
    _cleanup()
    return latency


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "xpu:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    # Serving runs forward under inference mode; keep init, dry run, warmup and
    # timing in the SAME mode for consistent workspace/kernel caching.
    with torch.inference_mode():
        return run_mla_module(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            model_path=model_path,
            attn_type=attn_type,
            prefix_len=prefix_len,
            perf_filename=perf_filename,
            device=device,
        )


def _cleanup():
    # Drop vLLM's WorkspaceManager singleton so its scratch is freed; it only
    # grows, so a large task would otherwise pin memory for the worker's life.
    import vllm.v1.worker.workspace as _ws_mod

    _ws_mod._manager = None
    gc.collect()
    get_device_module().empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one vLLM-XPU GLM-5.3 DSA module shape.")
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=3072, help="context: isl; generation: decode step (past KV)")
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--gemm-type", choices=SUPPORTED_GEMM_TYPES, default="bfloat16")
    parser.add_argument("--model-path", default="zai-org/GLM-5.3")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--output-path", default=None, help="directory (or *.txt file) for the perf rows")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "context":
        perf_filename = PerfFile.DSA_CONTEXT_MODULE.value
    else:
        perf_filename = PerfFile.DSA_GENERATION_MODULE.value
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        perf_filename = os.path.join(args.output_path, perf_filename)

    run_mla_module(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        kv_cache_dtype="bfloat16",
        compute_dtype="bfloat16",
        gemm_type=args.gemm_type,
        model_path=args.model_path,
        attn_type="dsa",
        prefix_len=args.prefix_len if args.mode == "context" else 0,
        perf_filename=perf_filename,
        device=args.device,
        warming_up=args.warmup,
        test_ite=args.iters,
    )


if __name__ == "__main__":
    main()
