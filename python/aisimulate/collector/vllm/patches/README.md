# vLLM-XPU patches

Patches this collector needs applied to the vLLM XPU fork before it can run.
They live here rather than in the fork so the collector and the change it
depends on stay reviewable together.

## `vllm-xpu-gdn-deepklox-interface.patch`

| | |
|---|---|
| target repo | `https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu` |
| base commit | `e3e075508d506a721093dcd0aa94ac2e32811fda` (`tpsp: prepare moe gemm weight (#343)`) |
| branch | `main` |
| needed by | `collect_gdn_xpu.py` |

Adds `vllm/model_executor/layers/mamba/gdn/deepklox_interface.py`, which
exposes the DeepKLOX XPU GDN kernels under stable names: both conv families
(the plain `causal_conv1d_varlen`/`causal_conv1d_varlen_update` pair the
collector times for CUDA parity, and the XPU-only split+L2-norm-fused
`causal_conv1d_fn`/`causal_conv1d_update` serving variants), the two scan
primitives, `causal_conv1d_prepare_weight` and `gdn_prefill_backend()`.
`_resolve_gdn_prefill_backend` returns `"triton"` on every non-CUDA platform,
so without this module `collect_gdn_xpu.py` fails with `ModuleNotFoundError`
and collects nothing.

Apply and rebuild the wheel:

```sh
cd /path/to/applications.ai.gpu.vllm-xpu
git am /path/to/vllm-xpu-gdn-deepklox-interface.patch
./build.sh --skip-deps
```

The rebuild is required: the collector imports vLLM from site-packages, not
from the source tree.
