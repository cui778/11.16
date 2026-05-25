# script2_new

## 1. 当前定位
`script2_new` 当前处于**第一章正式实验重启阶段**，目标不是继续沿用旧探索主线，而是基于已经确认的问题，重新建立一套可复核、可写论文、可持续扩展的正式实验流程。

当前必须先明确 4 个事实：

1. 旧的 `monitor-only` 子图设定会裁掉大量全网拓扑信息。
2. 第一章后续统一采用：
   - 保留全网 `128` 节点拓扑
   - 仅在输入层对非监测节点执行 sparse observation mask
3. 旧的 `v2e_dense_ie` 虽然曾作为探索主线，但它只激活了 `22` 个唯一缺陷节点，不符合当前正式任务定义。
4. `script3` 的旧结果不能直接作为第一章正式证据，因为它属于旧口径实验线，和当前正式协议不一致。

因此，从现在开始：

- 旧结果只保留为探索记录或附录材料
- 第一章正式实验以**重新定义的正式缺陷矩阵**为起点
- 任何会改变任务定义的修改，必须先确认再执行

---

## 2. 第一章正式实验协议

### 2.1 固定不变的条件
第一章正式实验默认固定：

- 候选节点集：`50`
- 监测布局：`degree N25`
- 图输入：全网 `128` 节点拓扑
- 输入模式：`full-graph sparse-observation`
- 主特征：`residual8`
- 主任务：`I/E` 缺陷定位

### 2.2 当前不进入正式主线的内容
以下内容不直接进入第一章正式主线：

- 旧 `monitor-only` 子图高分结果
- `script3` 的历史高 `node_holdout` 结果
- 只覆盖 `22` 个唯一缺陷节点的旧 `v2e_dense_ie` 正式化表述
- `P` 类型缺陷

### 2.3 为什么当前正式主线先不做 `P`
当前第一章正式主线先聚焦 `I/E`，原因是：

1. 当前 `P` 的注入语义还没有被严格定义为独立、原生的水质缺陷源任务。
2. 当前可执行管线中，`P` 仍然更像探索性扩展，不适合作为正式主线的一部分。
3. 第一章当前的重点是先把空间定位主线与节点泛化问题做扎实。

这不等于以后永远不做 `P`，而是：

> `P` 可以保留为后续扩展，但当前第一章正式重启先做 `I/E`。 

### 2.4 流量扰动与水质残差
当前正式主线虽然不把 `P` 纳入正式任务，但水质特征仍有保留价值。

原因是：

- 当前 `I/E` 主线主要通过流量增减注入扰动
- 流量变化会影响输移、混合、稀释和停留时间
- 因此会在水质变量上形成真实残差

当前正式数据中已确认存在非零水质残差，包括：

- `pollut_BODf_residual`
- `pollut_NH4_residual`
- `pollut_DO_residual`

因此，当前第一章可以保留水质动态特征作为观测输入，但不能把这条主线表述为“已经完成严格的水质缺陷源模拟”。

---

## 3. 正式实验运行工作流（Run Flow）

## Step 0. 先检查协议，再跑实验
在开始任何正式实验前，先阅读：

- [CHAPTER1_EXPERIMENT_RESTART_PLAN_20260402.md](/e:/11.16/script2_new/CHAPTER1_EXPERIMENT_RESTART_PLAN_20260402.md)
- [89_chapter1_frozen_protocol_audit_20260401.md](/e:/11.16/process_diagnosis_revision_20260321/89_chapter1_frozen_protocol_audit_20260401.md)

## Step 1. 检查全网图输入与基础集合

### 脚本
- [inp_parser.py](/e:/11.16/script2_new/prep/inp_parser.py)
  - 解析 SWMM 输入文件
- [build_graph.py](/e:/11.16/script2_new/prep/build_graph.py)
  - 生成邻接矩阵
- [build_graph_features.py](/e:/11.16/script2_new/prep/build_graph_features.py)
  - 生成图路径先验特征
- [validate_node_sets.py](/e:/11.16/script2_new/prep/validate_node_sets.py)
  - 检查节点集合一致性

### 输出
- [parsed_inp_data.json](/e:/11.16/script2_new/input_1/parsed_inp_data.json)
- [adj_matrix.npy](/e:/11.16/script2_new/input_1/adj_matrix.npy)
- [graph_path_features.npz](/e:/11.16/script2_new/input_1/graph_path_features.npz)
- [node_list.json](/e:/11.16/script2_new/input_1/node_list.json)

## Step 2. 固定候选节点与监测节点

### 脚本
- [build_candidates.py](/e:/11.16/script2_new/prep/build_candidates.py)
  - 生成候选节点集
- [build_monitors.py](/e:/11.16/script2_new/prep/build_monitors.py)
  - 生成 `degree N25` 监测布局

### 输出
- [candidate_nodes_new.json](/e:/11.16/script2_new/input_1/candidate_nodes_new.json)
- [candidate_nodes_50_report.csv](/e:/11.16/script2_new/input_1/candidate_nodes_50_report.csv)
- [monitor_nodes_degree_N25.json](/e:/11.16/script2_new/input_1/monitor_nodes_degree_N25.json)

### 必须额外报告
从现在开始，候选节点集与缺陷矩阵覆盖必须分开报告，不能再混写。

每次正式实验启动前，必须先说明：

1. `50` 个候选节点的选取逻辑是什么
2. 当前候选节点文件是否发生变化
3. 当前正式缺陷矩阵实际激活了多少唯一缺陷节点
4. 这两个集合之间是什么关系：
   - 候选节点集
   - 真实激活缺陷节点集

也就是说：

> 候选节点数固定为 50，不等于正式缺陷矩阵已经合理覆盖了这 50 个节点。  
> 后续所有主表都必须同时报告这两层信息。

## Step 3. 生成正式缺陷矩阵

### 脚本
- [defect_matrix.py](/e:/11.16/script2_new/prep/defect_matrix.py)
  - 原始通用生成器
- [build_defect_matrix_v2.py](/e:/11.16/script2_new/prep/build_defect_matrix_v2.py)
  - 可控节点数/场景数的矩阵生成器

### 当前要求
正式矩阵生成后，必须先报告：

- 候选节点数
- 监测节点数
- `I` 类场景数与唯一缺陷节点数
- `E` 类场景数与唯一缺陷节点数
- 总唯一缺陷节点数
- 缺陷节点与监测节点交集
- 缺陷节点与候选节点关系

### 输出
- `input_1/defect_matrix_*.csv`

## Step 4. 重新提数并生成残差特征

### 脚本
- [extract_timeseries.py](/e:/11.16/script2_new/prep/extract_timeseries.py)
  - 基于正式缺陷矩阵重建全图时序数据
- [residual_features.py](/e:/11.16/script2_new/prep/residual_features.py)
  - 生成残差特征

### 输出
- `training_data_new/<subdir>/node_timeseries.parquet`
- `training_data_new/<subdir>/scenario_summary.csv`
- `training_data_new/<subdir>/dataset_manifest.json`
- `training_data_new/<subdir>/node_timeseries_with_residuals.parquet`

## Step 5. 训练第一章空间主线

### 脚本
- [train_privileged_teacher_student.py](/e:/11.16/script2_new/scripts/train_privileged_teacher_student.py)
  - 当前第一章 full-graph sparse-observation 正式训练入口

### 说明
虽然脚本名里有 `teacher_student`，但在蒸馏权重为 `0` 时，它就是当前 full-graph sparse-observation 主训练入口。

### 输出
- `outputs/model_checkpoints/best_model_<tag>.pth`
- `outputs/model_checkpoints/training_history_<tag>.csv`
- `outputs/model_checkpoints/by_scenario_val_<tag>.csv`
- `outputs/model_checkpoints/by_scenario_test_<tag>.csv`
- `outputs/reports/last_run_metrics_<tag>.json`

## Step 6. 汇总正式表与图

### 脚本
- [summarize_chapter1_model_comparison.py](/e:/11.16/script2_new/scripts/summarize_chapter1_model_comparison.py)
- [summarize_chapter1_node_holdout.py](/e:/11.16/script2_new/scripts/summarize_chapter1_node_holdout.py)
- [summarize_chapter1_topology_robustness.py](/e:/11.16/script2_new/scripts/summarize_chapter1_topology_robustness.py)
- [plot_chapter1_formal_figures.py](/e:/11.16/script2_new/group_meeting_figures/plot_chapter1_formal_figures.py)

### 输出
- `outputs/reports/chapter1_*.csv`
- `outputs/reports/chapter1_*.json`
- `outputs/reports/figures/chapter1_*.png`

## Step 7. 时间维度与过程级实验正式收口

### 当前状态
时间维度方案已经不只是想法，核心机制都已落地到代码里，包括：

- `time_gated` 基于 `overlap_ratio` 的 `phase_label / active_label`
- `always_on` 基于信号阈值的活跃状态判断
- 时间位置编码 `sin/cos(hour)`
- 趋势特征开关
- 过程级指标：
  - `active_period_recall`
  - `detection_latency_mean`
  - `event_level_top1/3/5`

### 当前还缺的正式实验
但时间维度还没有像空间主线一样完成最后一轮系统收口，后续正式实验必须补齐：

1. `time_gated` 与 `always_on` 的正式对比
2. 时间位置编码是否保留的正式多种子结果
3. 趋势特征是否进入正式主线的最终结论
4. 过程级指标在正式主线下的系统汇总表

也就是说：

> 第一章最终论文实验不能只剩空间定位主表，必须把时间维度和过程级指标正式接回主线。

---

## 4. 防错规则

后续任何正式实验都必须通过以下检查：

1. 必须显式报告唯一缺陷节点数，不能只报告总场景数。
2. 不允许把“探索用矩阵”直接改写成“正式矩阵”而不说明。
3. 不允许把当前真实缺陷节点动态加入观测输入集合。
4. `node_holdout` 只能从真实激活过的缺陷节点集合中抽样。
5. 任何会改变任务定义的修改，都必须先确认再执行。
6. 时间维度实验不能只停留在代码已实现，必须补齐正式主表和过程级结果表。

---

## 5. 当前正式重启重点

当前第一章正式重启的优先级是：

1. 重新设计正式 `I/E` 缺陷矩阵
2. 重建全图 sparse-observation 数据
3. 重跑空间主线
4. 重做 `node_holdout`
5. 如果 `node_holdout` 仍然差，再做专项优化

---

## 6. 当前判断：节点泛化问题怎么解决

当前对 `node_holdout` 的判断是：

- 问题不只是模型
- 更大的问题在于当前正式矩阵覆盖的真实缺陷节点太少
- 因此先修数据与协议，再修模型，才是稳路线

当前节点泛化专项计划是：

1. 扩大正式矩阵中的真实激活缺陷节点覆盖
2. 把 `node_holdout` 抽样母集限定为真实激活缺陷节点集合
3. 如果仍然差，再做：
   - `node_holdout-aware validation`
   - `candidate-conditioned matching`
   - `hard negative / confusion-aware ranking`

---

## 7. 一句话说明

从现在开始，`script2_new` 的第一章正式实验必须先回答：

> 在固定 `degree N25`、保留全网拓扑、仅输入稀疏观测的条件下，基于覆盖合理的正式 `I/E` 缺陷矩阵，模型能否实现稳定的场景泛化与可信的节点泛化？
