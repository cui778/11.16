# script2_new 当前状态与配置速查（B 路线）

供每次开工前快速确认：数据从哪来、改了什么、要对比什么。

---

## 一、当前数据与路径（config 为准）

| 用途 | config 项 | 典型默认值 |
|------|------------|------------|
| 候选缺陷节点集 C | `candidate_nodes_file` | `input_1/candidate_nodes_new.json` |
| 监测节点集 S | `monitor_nodes_file` | `input_1/monitor_nodes_degree_N25.json` |
| 缺陷矩阵 | `defect_matrix_file` | `input_1/defect_matrix_diverse.csv`（或 defect_matrix_new.csv） |
| 时序（原始） | — | 由 extract_timeseries `--output-dir` 决定 |
| 时序（残差） | `node_timeseries_file` | `training_data_dir` / `training_data_subdir` / `node_timeseries_with_residuals.parquet` |
| 时序数据子目录 | `training_data_subdir` | `""` \| `"time_gated"` \| `"full_injection"` |

- **训练/检查** 使用的 parquet 以日志中打印的「时序数据: …」为准。
- 对比「时间门控 vs 全程注入」时：两份数据放在不同子目录，通过 `training_data_subdir` 切换。

---

## 二、可调项（迭代时优先看）

- **候选集**：`candidate_nodes_file`、或 build_candidates 的 `--k` 等。
- **缺陷矩阵**：`defect_matrix_file`、场景数、时间门控参数（start_hour/duration_h）。
- **监测点**：`monitor_nodes_file`（degree / betweenness / downstream / observability）。
- **训练**：`learning_rate`、`patience`、`dataset_e_class_oversample_ratio`、`dataset_loc_ratio_clip` 等。
- **模型类型**：`model_type` = `gru_gcn`（默认）或 `hydraulic_inverse`（水力逆向注意力，需先跑 `prep/build_graph_features.py`）；命令行覆盖：`--model-type hydraulic_inverse`。
- **数据源**：`training_data_subdir`（用哪份时序）、`sequence_length`、`window_stride`、`dataset_overlap_threshold`。

每次只改一个变量，改前记 baseline，改后对比 STEP_RESULTS_LOG。

---

## 三、结果记录与流程

- **实验记录**：`script2_new/outputs/reports/STEP_RESULTS_LOG.md`
- **运行流程**：`script2_new/RUN_WORKFLOW.md`
- **预实验历史（script2）**：`script2/reports/STEP_RESULTS_LOG.md`、`MASTER_DEBUG_AND_STATUS.md`（已闭环，仅参考）

---

## 四、Split 诊断说明（create_dataloaders 输出）

- **场景数 / 窗口数**：各 split 的 scenario 数与样本（窗口）数。
- **缺陷类型(按场景)**：I/E/P 在各自 split 的分布。
- **缺陷节点数(去重)**：该 split 内涉及多少种缺陷节点。
- **缺陷节点在候选集覆盖率**：B 路线下应为 100%。
- **测试缺陷节点 ⊆ 训练集**：B 路线下为「是」；用于确认按 scenario 划分时测试节点均在训练中出现。
- **各 split 定位启用窗口数与占比**：有 `target_node_idx>=0` 的窗口数，用于确认 val/test 有足够定位信号。

不再打印「TEST 中训练未见过的缺陷节点」，因 B 路线设计上无未见过节点。

---

## 五、B 路线 scenario 划分的已知局限

**问题**：`defect_matrix_diverse.csv` 覆盖全部 50 个候选节点，按 scenario 划分时，测试集中的缺陷节点在训练集中均已出现过（即「无未见过节点」）。这使 scenario 划分下的指标偏高，不能反映模型对未知缺陷位置的真实泛化能力。

**已有应对**：
- script2_new 内已做 **node-holdout** 对比（留出 10 个节点，测试集 = 100% 未见过节点），结果见 `STEP_RESULTS_LOG.md` 中「模型对比（node-holdout 划分）」一节。
- **script3** 将 **node_holdout 作为主比较划分**，并引入更强模型（`lstm_graphsage_edge`），在 node_holdout 下达到 MRR=0.9147、Top-1=0.8764，优于 script2_new 中的 hydraulic_inverse（MRR=0.930，但 script3 使用 I/E 两类、不含 P 类，数据设定不完全一致）。

**结论**：
- 组会汇报时，scenario 划分结果可作为「有效性校验」展示，但**主结论应基于 node-holdout 划分**。
- script2_new 的 B 路线实验（A/B1/B2/B3 组）仍有效，用于支撑「监测策略选择」和「数据设计」的消融分析。

---

## 六、script3 进展摘要（截至 2026-03-18）

`script3` 是第一章重置后的独立实验线，主要改进：

| 改进点 | script2_new | script3 |
|--------|-------------|---------|
| 划分方式 | scenario（主）+ node_holdout（辅） | **node_holdout 为主** |
| 缺陷类型 | I/E/P 三类 | **仅 I/E 两类**（对齐开题） |
| 模型 | gru_gcn / gru_only / hydraulic_inverse | + **lstm_graphsage_edge** / tcn_graphsage_edge |
| 任务粒度 | 节点级定位 | 节点级 + **管段级（segment mode）** |

**节点级主结果**（node_holdout，raw_residual，λ=0.2）：

| 模型 | MRR | Top-1 | Top-3 | I Top-1 | E Top-1 |
|------|-----|-------|-------|---------|---------|
| hydraulic_inverse | 0.838 | 0.802 | 0.838 | 0.996 | 0.668 |
| tcn_graphsage_edge | 0.897 | 0.836 | 0.971 | 0.981 | 0.736 |
| **lstm_graphsage_edge** ✅ | **0.915** | **0.876** | **0.952** | **0.996** | **0.793** |

**管段级结果**（segment_ie_full_v1）：MRR=0.709，Top-1=0.625，I Top-1=0.791，E Top-1=0.438。

详细报告见 `script3/outputs/reports/CHAPTER1_NODE_FINAL_SUMMARY_CN.md`。
> Warning
>
> This file contains historical debug notes and old result interpretations.
> It is no longer a reliable source for the current clean formal conclusion unless the specific section is explicitly revalidated after the monitor-node leakage fix.
> Treat it as legacy notes, not as the current authoritative result summary.
