# Benchmark 兼容性分析（testcase YAML 消费面）

日期：2026-10-13 ｜ 结论先行：**隔离对 benchmark 语义零影响**——机制就是
collector 既有的 cases-root override，不新增消费逻辑。

## 1. 当前 benchmark（collector）的 YAML 使用

| 消费方 | 读取内容 | 发现逻辑 | override 支持 |
|---|---|---|---|
| `case_generator.py` | `base_ops/*.yaml` 网格 + `models/*_cases.yaml` | `BASE_OP_CASES_DIR = <collector>/cases/base_ops`（硬编码）+ `_CASES_ROOT_OVERRIDE`（env 优先） | ✓（`3c381e2a`） |
| `capabilities.py` | `capabilities.yaml` / denylist | 同一 override | ✓ |
| `model_cases.py` | models yaml | 同一 override | ✓ |
| `provenance.py` | cases 树哈希（fail-closed） | 同一 override；hash 记录 = 实际消费文件 | ✓ |

**运行期回写：不存在**——四个消费方全部只读；collector 的写入只发生在
output root（perf txt / csv / checkpoint），与 cases 目录无关。

## 2. 隔离的影响面

- collector：**零影响**。`AIC_COLLECTOR_CASES_ROOT` 指向 run-local 快照后，
  四消费方原子地整体切换 case 源；无 override 时回落 repo cases，行为与
  改动前逐字节一致（向后兼容由"override 缺省 = 现状"保证）
- 测试选择/过滤/分组：逻辑不在脚本层改，快照是逐字节拷贝 → 展开结果唯一
  决定于快照内容，`verify_isolation_equivalence.py` 门禁 diff=0 强制证明
- 结果收集 / CSV schema / 聚合：output root 独立于 cases 源，`--keep-csv`
  perf txt schema 不变（19 列 header 由验证脚本复核）

## 3. 未来 benchmark 侧 adoption（rollout 迁移步骤）

目标态（与 prompt 一致，用户不手工 copy yaml）：

```
benchmark → run-local 快照生成 → execute → collect results
```

1. **driver 脚本层**（唯一需要改的层）：任何现在的 runner（如
   `run_dsv4_full.sh`、他人的 12_13_hca）在启动前插入两步——
   `snapshot_cases.py --out <run>/cases_snapshot` +
   `export AIC_COLLECTOR_CASES_ROOT=<run>/cases_snapshot`
2. **失败模式钉死**：override 支持门禁（case_generator 无该 env 支持即
   exit 2）防止旧树假隔离
3. **并发**：每 run 独立 output root（checkpoint 相对 output root）→ 同 op
   并发天然不互跳
4. **回滚**：删 iso runner / 停用 override 即回到共享正本模式；collector
   侧无需回滚（零改动）

## 4. 已声明的边界

- ww42 旧 collector（b7d64db6）无 override 支持：gen-1 tina 脚本继续服务该
  环境（deploy-over-repo 是其唯一可行模式），rollout 只覆盖支持 override 的树
- `test_yaml/` 快照（drift 检测声明件）与 run-local 快照职责不同：前者钉
  "repo 正本应是什么"，后者是"本 run 实际消费什么"；隔离后前者只服务 gen-1
