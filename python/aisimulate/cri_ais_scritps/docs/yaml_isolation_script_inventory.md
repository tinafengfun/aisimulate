# YAML Isolation Script Inventory

日期：2026-10-13 ｜ 复用优先序核查结论：**无需新建任何 collector 侧机制**，
新代码只有 delivery 层 1 个 runner + 1 个验证脚本。

## 组件清单

| 组件 | 路径 | 用途 | 复用策略 | 所需改动 |
|---|---|---|---|---|
| cases-root override | `collector/case_generator.py` + `capabilities.py` + `model_cases.py` + `provenance.py` | 隔离核心钩子（`AIC_COLLECTOR_CASES_ROOT`，commit `3c381e2a`） | **原样复用** | 零 |
| collect.py | `collector/collect.py` | 执行入口；checkpoint 相对 output root（:1708）→ per-run OUT 天然隔离 | **原样复用** | 零 |
| snapshot_cases.py | `dsv4_xpu/scripts/` → 拷入 `cri_ais_scritps/scripts/` | 生成 run-local 快照 + provenance | **原样复用**（已支持 `--repo/--out/--tag/--commit`） | 零（sha256 `2e40a8e9…` 逐字节拷贝核对） |
| container_env.sh | `dsv4_xpu/scripts/container_env.sh` | env 契约 + cases-root 默认导出（`${AIC_COLLECTOR_CASES_ROOT:-…}` 尊重预设值） | 复用（iso 脚本内联同款契约，保持 delivery 包自包含） | 零 |
| verify_isolation_equivalence.py | `cri_ais_scritps/scripts/`（新建） | 展开等价性 diff + perf header 校验 | 新建（无既有脚本覆盖此门禁） | — |
| iso runner | `cri_ais_scritps/04_05_10_11_attention_context_generation_iso.sh`（新建） | run-local 工作区 + 门禁 + 执行 | 新建（gen-1 tina 脚本的隔离版；骨架沿用其 env 契约块） | — |
| tina 脚本（gen-1） | `cri_ais_scritps/04_05_10_11_attention_context_generation_tina.sh` | ww42 交付（deploy-over-repo 模式） | **冻结不动**——ww42 旧 collector 无 override，部署式是其唯一可行模式 | 零 |
| 12_13_hca / run_smoke_all.sh / common.sh | 同目录（Zhang, Yanli L 所有） | 他人交付物 | **零触碰**；列为未来 rollout 适配项 | 零 |
| precheck.py | `dsv4_xpu/scripts/precheck.py` | 测试前静态快检（pinning / dsa-patch / picks / runtime） | 验证阶段复用 | 零 |
| audit_cases_vs_upstream.py | `dsv4_xpu/scripts/audit_cases_vs_upstream.py` | 上游同步后 case 集 audit | 复用（与隔离互补：audit 管"源头变没变"，isolation 管"消费位置"） | 零 |
| reset_checkpoint.sh | `dsv4_xpu/scripts/reset_checkpoint.sh` | checkpoint 归档复位 | 复用（iso run 的 checkpoint 在 run 目录内，一般无需） | 零 |

## 新建组件的最小性说明

- runner 的 env 契约块与 gen-1 tina 脚本逐行相同（四项契约 + unset 链），唯一
  新逻辑 = 快照生成 + 两道 fail-loud 门禁 + 等价性前置
- 未改动任何测试选择/过滤/分组代码；`--smoke` 语义沿用 case_generator 既有实现
