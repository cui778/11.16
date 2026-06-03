# 第4章协议冻结文档

## 正式诊断协议

```text
数据集：ie420_plus_normal20_v1（441 场景 = 421 IE420 + 20 normal）
业务时序特征集：raw_plus_residual（4 raw + 4 residual + 4 relative residual = 12 维）
模型动态输入张量：12 维业务时序特征 + 2 维小时周期编码 + 1 维 observed mask = 15 维
lambda_loc：0.5
模型：hydraulic_inverse_deepattn
布局：degree_N25（25 监测节点，全网 128 节点）
划分：scenario split
诊断种子：7 / 42 / 123
教师蒸馏：disabled（lambda_kd=0, lambda_active_kd=0）
```

## 特征维度明细

| 类别 | 特征变量 | 维度 |
|---|---|---:|
| 原始水力 | depth, total_outflow | 2 |
| 原始水质 | pollut_NH4, pollut_TSSs | 2 |
| 绝对残差 | depth_residual, total_outflow_residual, pollut_NH4_residual, pollut_TSSs_residual | 4 |
| 相对残差 | depth_residual_rel, total_outflow_residual_rel, pollut_NH4_residual_rel, pollut_TSSs_residual_rel | 4 |
| **合计** | | **12** |

## 辅助输入与静态结构先验

| 类别 | 变量 | 维度 | 说明 |
|---|---|---:|---|
| 时间辅助通道 | sin(hour), cos(hour) | 2 | 数据加载阶段自动追加 |
| 稀疏观测掩码 | observed_mask | 1 | 标记当前布局中的监测节点 |
| **动态输入张量合计** | 12 维业务时序 + 3 维辅助通道 | **15** | 作为时序编码器输入 |
| 节点对静态路径先验 | shortest_dist, pipe_length_dist, flow_direction, elevation_diff | 4 | 作为液压逆向注意力的节点对关系输入，不计入节点动态特征维度 |

正式配置中 `use_trend_feature=false`、`use_propagation_delay=false`，因此不追加趋势通道和传播延迟通道。`head` 与 `volume` 保留在 SWMM 输出层，但不进入正式诊断输入。

## 正式文件路径

| 项目 | 路径 |
|---|---|
| 训练入口 | `script2_new/scripts/train_privileged_teacher_student.py` |
| 训练数据 | `script2_new/training_data_new/ie420_plus_normal20_v1` |
| 缺陷矩阵 | `script2_new/input_1/defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv` |
| 监测节点 | degree_N25 布局 |
| 候选节点 | 50 个候选缺陷节点 |
| 正式 checkpoint | `script2_new/outputs/model_checkpoints/ch1_fullgraph_degree_ie420_s{7,42,123}_fix1` |
| 正式报告 | `script2_new/outputs/reports/last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s{7,42,123}.json` |

## 主指标（seed42 单种子，正式协议）

| 指标 | 值 | 来源 |
|---|---|---|
| MRR | 0.8457 | CH4-F07 formal_mainline |
| Top-1 | 0.7728 | CH4-F07 formal_mainline |
| Top-3 | 0.8973 | CH4-F07 formal_mainline |
| Top-5 | 0.9463 | CH4-F07 formal_mainline |
| Active F1 | 0.9790 | CH4-F07 formal_mainline |
| Normal Window FPR | 0.0005 | CH4-F07 formal_mainline |
| Scene F1 | 0.9917 | CH4-F07 formal_mainline |

## 多种子稳定性（degree_N25, scenario split, 正式协议）

| seed | MRR | Top-1 | Top-3 | Top-5 | Event Top-1 | Scene F1 |
|---|---|---|---|---|---|---|
| 7 | 0.8477 | 0.7413 | 0.9443 | 0.9741 | 0.7969 | 1.0000 |
| 42 | 0.8457 | 0.7728 | 0.8973 | 0.9463 | 0.8197 | 0.9917 |
| 123 | 0.7827 | 0.6425 | 0.9161 | 0.9759 | 0.6885 | 1.0000 |

来源：`CH5-EXPT_fixed_protocol_N25_main_table.csv`（Degree 行）

## 模型输出

模型只有两个窗口级输出：
- `p_active(t)`：窗口级活跃概率
- `node_scores(t)`：候选节点得分

时间级、空间级、场景报警级和综合诊断级均为聚合结果，不是独立模型头。

## 不属于正式管线的实验

| 实验 | 状态 | 原因 |
|---|---|---|
| feature ablation（raw/residual/raw+residual） | historical | 来自 seedset10 数据集，调参依据 |
| lambda sweep（0.3/0.5/1.0/2.0） | historical | 来自 seedset10 数据集，调参依据 |
| node_holdout | supplementary | 压力测试，非主结论 |
| model comparison（gru_gcn/lstm_graphsage） | supplementary | 模型消融 |
| persistent/mixed experiments | legacy | 非正式协议 |
| process_diagnosis series | legacy | 旧口径 |
| 窗口长度实验（2h/3h/4h/6h） | historical | 旧协议数据，趋势参考 |
| I/E 分组定位分析 | historical | 旧协议数据，趋势参考 |
