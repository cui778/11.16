# 第一章节点模式模型对比总结

日期：2026-03-12  
分支：`exp/script3-chapter1-reset`  
数据来源：`E:\11.16\script2_new\training_data_new\time_gated_downstream`  
缺陷集合：`E:\11.16\script3\input_1\defect_matrix_ie.csv`  
任务模式：节点级定位，`I/E` 两类

## 实验设置

- 候选节点：`candidate_nodes_new.json`
- 监测点集合：`monitor_nodes_downstream_N25.json`
- 序列长度：36
- 窗口步长：6
- 特征模式：`raw_residual`
- 对比模型：
  - `hydraulic_inverse`
  - `lstm_graphsage_edge`
  - `tcn_graphsage_edge`

## 核心结果

### Scenario 划分

| 模型 | MRR | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| hydraulic_inverse | 0.9933 | 0.9885 | 1.0000 | 1.0000 |
| lstm_graphsage_edge | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tcn_graphsage_edge | 0.9756 | 0.9618 | 0.9866 | 0.9943 |

### Node holdout 划分

| 模型 | MRR | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| hydraulic_inverse | 0.8383 | 0.8022 | 0.8377 | 0.9150 |
| lstm_graphsage_edge | 0.9147 | 0.8764 | 0.9521 | 0.9969 |
| tcn_graphsage_edge | 0.8965 | 0.8362 | 0.9706 | 0.9815 |

## 分类型 Top-1 结果

### Scenario 划分

| 模型 | I | E |
| --- | ---: | ---: |
| hydraulic_inverse | 1.0000 | 0.9712 |
| lstm_graphsage_edge | 1.0000 | 1.0000 |
| tcn_graphsage_edge | 0.9810 | 0.9327 |

### Node holdout 划分

| 模型 | I | E |
| --- | ---: | ---: |
| hydraulic_inverse | 0.9962 | 0.6675 |
| lstm_graphsage_edge | 0.9962 | 0.7932 |
| tcn_graphsage_edge | 0.9811 | 0.7356 |

## 结果解读

- 对三种模型而言，`scenario` 划分下的结果已经接近饱和，更适合作为有效性校验，而不是主对比依据。
- `node_holdout` 更能反映泛化能力，因为测试集中的缺陷节点在训练中未出现过。
- 在当前节点模式下，`lstm_graphsage_edge` 是综合表现最强的模型。
- 主要提升来自 `E` 类缺陷的泛化能力；`I` 类定位在三种模型上都已接近饱和。
- 相比 `hydraulic_inverse`，`lstm_graphsage_edge` 在 `node_holdout` 下的提升为：
  - MRR：`0.8383 -> 0.9147`
  - Top-1：`0.8022 -> 0.8764`
  - `E` 类 Top-1：`0.6675 -> 0.7932`

## 对第一章的建议

- 将 `node_holdout` 作为第一章的主比较划分。
- 将 `lstm_graphsage_edge` 作为当前节点模式下的创新主模型。
- 保留 `hydraulic_inverse` 作为具有物理建模意味的对照基线。
- `scenario` 划分结果作为补充证据使用即可。
- 下一步建议把相同的比较协议扩展到管段模式定位与 `I/E` 联合预测。

## 对应结果文件

- `CHAPTER1_MODEL_COMPARISON.json`
- `last_run_metrics_hydraulic_inverse_scenario_raw_residual.json`
- `last_run_metrics_lstm_graphsage_edge_scenario_raw_residual.json`
- `last_run_metrics_tcn_graphsage_edge_scenario_raw_residual.json`
- `last_run_metrics_hydraulic_inverse_node_holdout_raw_residual.json`
- `last_run_metrics_lstm_graphsage_edge_node_holdout_raw_residual.json`
- `last_run_metrics_tcn_graphsage_edge_node_holdout_raw_residual.json`
