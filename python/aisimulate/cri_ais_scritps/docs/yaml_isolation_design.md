# Testcase YAML 隔离设计（delivery pilot）

日期：2026-10-13 ｜ 范围：delivery workspace（`cri_ais_scritps/`）试点；
collector 生产代码零改动，机制复用既有 `AIC_COLLECTOR_CASES_ROOT`。

## 1. 根因分析（以真实发生为准）

Prompt 假设的污染向量之一是"runtime modification writing back into shared
YAMLs"——**实际代码中 collector 运行期从不回写 yaml**。真实污染向量是三个，
全部发生在部署/共享层：

| # | 向量 | 实际案例 |
|---|---|---|
| 1 | **部署期覆盖共享正本** | tina 脚本启动时把 campaign yaml 覆盖到 `collector/cases/base_ops/dsv4_attention.yaml`（ww42 旧 collector 无 override，覆盖是唯一生效途径） |
| 2 | **协作者 git 操作还原正本** | 2026-10-12 Zhang 还原 repo yaml 为 fabiao 网格、清除 pristine-bak——我方 case 集静默失效 |
| 3 | **多 run 共享 checkpoint** | checkpoint 若在全局目录，同 op 并发 run 会互相跳 case（已核实本分支 checkpoint 相对 output root，per-run OUT 即隔离） |

## 2. 隔离架构

```
之前（gen-1，tina 脚本）：                     之后（gen-2，iso 脚本）：
共享 repo yaml ← 覆盖部署（写共享文件）        共享 repo yaml（只读，零触碰）
    ↓                                             ↓
execute                                       run-local 快照（snapshot_cases.py --out）
                                                  ↓ AIC_COLLECTOR_CASES_ROOT=<run>/cases_snapshot
                                              execute（case 集只来自 run-local 快照）
```

**复用既有机制**：`AIC_COLLECTOR_CASES_ROOT`（commit `3c381e2a`）已被
case_generator / capabilities / model_cases / provenance 四消费方遵守，
layout `<root>/collector/cases`。隔离 = 每个 run 用 `snapshot_cases.py`
生成自己的快照根并把 env 指过去——**没有新增 collector 侧机制**。

## 3. Run 工作区结构

```
cri_ais_scritps/runs/run_<ts>/          ← gitignore（运行数据不进源码树）
├── cases_snapshot/                     ← run-local case 集（唯一 case 来源）
│   ├── collector/cases/...             ← 与 repo cases 同 layout
│   └── SNAPSHOT_PROVENANCE.yaml        ← 源 commit / tag / per-file sha256
├── raw/                                ← collect.py OUT：perf txt / csv staging /
│                                          .collector_checkpoint（per-run 天然隔离）
├── logs/                               ← run_env 契约日志 + 每 op 运行日志
└── reports/                            ← 等价性门禁输出
```

## 4. 执行流与 fail-loud 门禁

iso 脚本（`04_05_10_11_attention_context_generation_iso.sh`）顺序：

1. `USER_ARGS=("$@")` 先于一切 source（P0：oneAPI setvars 污染 `$@`）
2. **override 支持门禁**：collector 的 case_generator.py 无
   `AIC_COLLECTOR_CASES_ROOT` → exit 2（旧树静默忽略 override = 假隔离）
3. **dsa patch 门禁**：选跑 dsa op 而 site-packages xpu_mla_sparse.py 无
   fastpath → exit 2
4. 生成 run-local 快照（`snapshot_cases.py --out`，带 provenance）
5. **等价性门禁**：`verify_isolation_equivalence.py` 对比 repo-cases 展开 vs
   快照展开，per-op case key diff ≠ 0 → exit 1，拒绝烧 GPU
6. env 契约内联（graph=1 / pack=0 / SYCLTLA=0 / ZE_MASK=${GPU:-0}，禁 4）
7. `OUT=raw/` 执行；checkpoint/perf/日志全落 run 目录

## 5. 设计理由

- **快照树而非单 yaml 文件**：四个消费方读的是 cases 目录树且 provenance 对
  全树取哈希；单文件语义等价于快照里的一个成员，树才是隔离的完整单位
- **等价性门禁前置于 GPU**：隔离的全部风险就是"case 集变了"；展开层 diff=0
  即可证明行为不变，成本秒级
- **gen-1 脚本冻结不删**：ww42 旧 collector 无 override 支持，交付环境继续用
  tina 脚本；两代并存，design 上 gen-2 是 gen-1 的替代而非补丁

## 6. 后续 rollout（框架级 adoption）

1. collector 侧：零改动（override 已在）——其他 workflow（如 12_13_hca、
   run_smoke_all.sh）逐个改读 run-local 快照即可
2. 共享 yaml 消费方迁移时保持向后兼容：无 override 时行为与今天完全一致
3. 多用户约定：每 run 独立 `runs/run_<ts>/`；共享正本从"被部署对象"降级为
   "只读声明源"
