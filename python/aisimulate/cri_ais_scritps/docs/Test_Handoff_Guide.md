# Test Handoff Guide — DSV4 attention 交付（cri_ais_scritps）

日期：2026-10-13 ｜ 交付包：本目录（脚本 + test_yaml + scripts + docs）。
本地 git 是唯一版本源；测试机只执行、不同步版本。

## 1. 两代交付脚本

| 脚本 | 目标环境 | case 集模式 |
|---|---|---|
| `04_05_10_11_attention_context_generation_tina.sh` | ww42 等旧 collector（无 cases-root override） | 部署式：启动时把 `test_yaml/dsv4_attention_tina.yaml` 覆盖到 repo 正本（首次备份 `.pristine-bak`） |
| `04_05_10_11_attention_context_generation_iso.sh` | 支持 `AIC_COLLECTOR_CASES_ROOT` 的 collector 树（test 分支起） | **隔离式（推荐）**：每 run 独立快照，共享正本零触碰 |

## 2. Run-local YAML 创建（隔离式工作流）

```bash
cd <cri_ais_scritps 所在目录>
GPU=0 bash 04_05_10_11_attention_context_generation_iso.sh          # smoke，6 op
GPU=0 bash 04_05_10_11_attention_context_generation_iso.sh full     # 全量
GPU=0 bash 04_05_10_11_attention_context_generation_iso.sh dsv4_csa_context_module  # 选跑
```

脚本自动完成：`runs/run_<ts>/` 创建 → `scripts/snapshot_cases.py` 生成
run-local case 快照（带 SNAPSHOT_PROVENANCE.yaml）→
`export AIC_COLLECTOR_CASES_ROOT=<run>/cases_snapshot` →
**等价性门禁**（快照展开 vs repo 展开 case key diff 必须为 0，否则拒绝执行）→
env 契约落日志 → 执行。默认 op 集 = csa/swa/dsa × ctx/gen 共 6 op（hca 不在
scope）；其余 op（sparse kernel 等）经位置参数选跑。用户无需手工 copy yaml。

## 3. Run 工作区结构与产物位置

```
runs/run_<ts>/
├── cases_snapshot/      ← 本 run 实际消费的 case 集（含 per-file sha256 provenance）
├── raw/                 ← perf txt（每 op 一份）、csv staging（--keep-csv）、
│                           .collector_checkpoint（断点续跑，per-run）
├── logs/                ← run_env_<ts>.log（四项契约 + commit + 快照头）、run_<op>_<ts>.log
└── reports/             ← equivalence_pre_run.txt（门禁输出）
```

- 断点续跑：`OUT=runs/run_<ts>/raw` 重跑同一 run 目录（checkpoint 自动跳过已完成）
- 清理：run 目录整体删除即可（快照/数据/日志同源）；**共享 repo yaml 无需恢复**
  （隔离式从未改过它）

## 4. 环境契约（四项，违反即数据不可信）

| 变量 | 值 |
|---|---|
| `ZE_AFFINITY_MASK` | `${GPU:-0}`；**禁 4**（确定性段错误） |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | 1（缺省静默降级 eager，数据非目标测法） |
| `AIC_DSV4_PACK_TARGET_MS` | 0（pack off = CUDA 基线同方法） |
| `VLLM_XPU_MXFP8_USE_SYCLTLA` | 0 |

dsa op 需 DSA fastpath patch（脚本有门禁，无 patch 直接 exit 2）。

## 5. 审计与验证

- 测试前：`precheck.py`（pinning / dsa-patch / picks / runtime）
- 隔离等价性：`scripts/verify_isolation_equivalence.py --repo <tree>
  --snapshot <run>/cases_snapshot --out <run>/reports/...`（可加
  `--check-header` 复核 perf txt schema）
- 数据有效性/case 合理性：沿用 `dsv4_xpu/runs/*/audit_*.py` 模式
- 长跑纪律：`docker exec -d` 脱离 ssh；测量期禁 xpu-smi；只按记录 PID 杀

## 6. 禁止事项

- 共享 repo yaml：隔离式流程下**永不写**；gen-1 部署式流程写前必须备份
- 运行数据不进源码树（`runs/` 已 gitignore）
- 他人交付文件（12_13_hca、run_smoke_all.sh 等）零触碰
