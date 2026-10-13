# YAML Isolation Validation（执行证据）

状态：**验证完成，全部 PASS**（2026-10-13）。

## 1. 环境

| 项 | 值 |
|---|---|
| 日期 | 2026-10-13 |
| 节点 | 10.239.44.236（236 节点） |
| 容器 | `aisim-dsv4-dev`（image `aisim_devy:20261011`，vllm 0.28.0.dev0 常规安装，DSA fastpath patch 在位） |
| GPU | 0 号（人工指定；`ZE_AFFINITY_MASK=0`，禁 4） |
| repo | tinafengfun/aisimulate @ test 分支（远端树与本地 `08afcb3f` 逐字节一致，3 文件 sha256 核对） |

## 2. 验证矩阵与结果

| # | 验证项 | 方法 | 判据 | 结果 |
|---|---|---|---|---|
| 1 | 展开等价性（6 op） | `verify_isolation_equivalence.py --repo /workspace --snapshot <run>/cases_snapshot` | per-op case key diff=0 | **PASS**（csa_ctx 1275 / csa_gen 1648 / swa_ctx 337 / swa_gen 44 / dsa_ctx 32 / dsa_gen 32，removed=0 added=0） |
| 2 | swa GPU 冒烟 | iso 脚本默认集 smoke（GPU=0） | 行 key ⊆ 展开、header、契约 | **PASS**（swa_ctx 4 行 + swa_gen 4 行，19 列 header PASS，0 错误） |
| 3 | dsa GPU 冒烟 | iso 脚本默认集含 dsa（patch 门禁放行） | 同上 | **PASS**（dsa_ctx 3 行 / dsa_gen 4 行，18 列 header PASS——dsa 无 compress_ratio 列系 collector 设计；timing_mode=graph_full_pack1，0 错误） |
| 4 | csa GPU 冒烟 | iso 脚本选跑 csa 两 op | 同上 | **PASS**（RC=0，0 错误） |
| 5 | 负向①只读正本 | 共享 yaml chmod 444 后 iso run | run 成功、正本未变 | **PASS**（RC=0，ALL_OPS_DONE；权限已恢复） |
| 6 | 负向②双并发 | 同 op（swa_gen）双 iso run 同时跑、独立 RUN_DIR | 各自完整、无交叉跳 case | **PASS**（两 run 各 RC=0、各 4 行、各 0 错误；checkpoint 随 OUT 隔离在各自 raw/.collector_checkpoint） |
| 7 | 负向③运行中篡改正本 | run 启动 +40s 向共享 yaml 追加注释 | run 不受影响 + collector 零回写 | **PASS**（run RC=0、4 行照常产出；篡改注释在 run 结束后仍在 = collector 运行期不写 yaml；yaml 随后恢复原 sha `0889111c…`） |
| 8 | perf/CSV 兼容 | `--check-header` 对全部 smoke perf txt | dsv4=19 列 / dsa=18 列逐列一致 | **PASS** |

## 3. 修正记录（验证过程发现并修复）

1. **vllm INFO 日志写 stdout 污染 JSON 解析** → verify 脚本改用 `AIC_JSON:` 哨兵行
2. **dsa header 校验误用 19 列模板** → dsa 行本无 `compress_ratio`（
   `collect_mla_module_xpu` item_list 不含该键），改为 per-op-family 期望 schema
3. **smoke 采样含替换**：dsa_ctx 4 次随机采样得 3 个唯一 case（同 case 双抽，
   全键含 gemm_type 后唯一）——smoke 语义为随机采样，行数非确定性属正常；
   隔离判据以"行 ⊆ 基线展开集 + 键唯一"为准

## 4. 证据文件

- 236 原始证据 tar sha256 `6913a7b0ad530a1548d155e626e27bca0a46ba7820059ea81cde4487c76c74fc`（双端一致）
- 本地：`dsv4_xpu/runs/20261013_iso_pilot/runs/`（run_val0 等价门禁 /
  smoke2 4-op / csa smoke / neg1 neg2 neg3 系列 + iso_*.log + rc 文件）
- 等价性门禁输出：`runs/run_val0/reports/equivalence.txt` 及各 run 的
  `reports/equivalence_pre_run.txt`

## 5. 失败记录

- 236 节点验证中途失联 ~10 分钟（单节点网络中断，其他节点正常）：容器
  Exited(255)，smoke1 中断；节点恢复后重启容器重跑（smoke2），中断 run 目录
  保留作证据（其中 swa 两 op perf 已产出且 schema 正确）
- 验证期间 ww42 节点零触碰
