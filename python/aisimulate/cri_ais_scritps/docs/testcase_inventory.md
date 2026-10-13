# Testcase Inventory（test 分支，隔离机制覆盖对象）

日期：2026-10-13 ｜ 展开数量以 236 容器实测为准（见
`yaml_isolation_validation.md`），本表记录定义来源与隔离策略。

## op 清单（REGISTRY_XPU 全部 dsv4/dsa module op + sparse op）

| op | kernel 类别 | case 定义来源 | 历史"runtime 修改" | 隔离策略 | run-local 位置 |
|---|---|---|---|---|---|
| `dsv4_csa_context_module` | csa | `cases/base_ops/dsv4_attention.yaml` → `common_case_values.dsv4_xpu_attention.grids`（attn_kinds=[csa] 块） | 无（collector 只读 yaml） | run-local 快照 | `runs/run_<ts>/cases_snapshot/` |
| `dsv4_csa_generation_module` | csa | 同上（gen 轴取自 csa 块 generation 字段） | 无 | 同上 | 同上 |
| `dsv4_swa_context_module` | swa | 同上（attn_kinds=[swa] 块，V4-Flash 稠密网格） | 无 | 同上 | 同上 |
| `dsv4_swa_generation_module` | swa | 同上 | 无 | 同上 | 同上 |
| `dsv4_hca_context_module` | hca | 同上——test 分支暂无 hca 网格块（deferred） | 无 | **不在 scope（人工定案 20261013），仅列出** | — |
| `dsv4_hca_generation_module` | hca | 同上（deferred） | 无 | **不在 scope** | — |
| `dsa_context_module` | dsa | `cases/base_ops/mla_module.yaml` → `common_case_values.mla_module.xpu*`（GLM-5.3 DSA） | 无 | 同上 | 同上 |
| `dsa_generation_module` | dsa | 同上 | 无 | 同上 | 同上 |
| `dsv4_paged_mqa_logits_module` | sparse | `dsv4_attention.yaml` + `case_generator._xpu_dsv4_sparse_rows` | 无 | 同上 | 同上 |
| `dsv4_hca_attn_module` | sparse | 同上 | 无 | 同上 | 同上 |

## 网格来源与两代 case 集

| 代 | yaml | 内容 | 用途 |
|---|---|---|---|
| gen-1（ww42 交付） | `cri_ais_scritps/test_yaml/dsv4_attention_tina.yaml`（campaign 对齐：csa_ctx 1274 / csa_gen 1648 / swa_ctx 337 / swa_gen 44 / hca 1312/1648） | campaign 剪枝网格 + ww42 hca 块合并 | ww42 旧 collector（无 override），tina 脚本部署式消费 |
| gen-2（test 分支正本） | `collector/cases/base_ops/dsv4_attention.yaml`（NV 对齐：csa+swa，hca deferred；`7ac1d8c9`/`33be28c9`） | NV test-case specs 对齐网格 | iso 脚本经 run-local 快照消费 |

注意：gen-2 的基线数量 ≠ campaign 数量（网格重定案），隔离验证的判据是
**before/after 完全一致**（`verify_isolation_equivalence.py` diff=0），
不是对齐某个历史数量。

## 覆盖保全承诺（prompt mandatory 项映射）

- case 数量 / 选择逻辑 / 过滤规则 / 类别分组：全部由快照树的逐字节拷贝保证
  （provenance per-file sha256 对账）
- CSV schema（19 列）与聚合逻辑：collector 零改动，schema 由
  `verify_isolation_equivalence.py --check-header` 复核
- 唯一允许的变化：case 集的**读取位置**（共享正本 → run-local 快照）
