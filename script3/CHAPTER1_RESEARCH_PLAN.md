# 第一章研究计划

## 定位

`script3` 是第一章重做后的独立实验线，重点放在模型创新与实验验证上。

- 主任务：缺陷定位
- 配套任务：缺陷类型判别
- 当前默认可运行目标：节点级定位
- 后续计划扩展：在数据管线中明确“缺陷到上游管段”的映射后，推进管段级定位

## 当前基本设定

- 候选集 `C`：固定的工程先验候选集
- 监测集 `S`：主实验默认使用 `downstream_N25`
- 类型头：当前在节点主线中可作为辅助头使用；后续在管段主线中可扩展为联合主任务
- 水质输入：默认只保留一个水质变量 `pollut_BODf`
- `script3` 当前默认只保留 `I/E` 两类缺陷

## 待比较的模型家族

- `time_mean_linear`
- `gru_only`
- `gru_gcn`
- `hydraulic_inverse`
- `lstm_graphsage_edge`
- `tcn_graphsage_edge`

## 第一章核心实验模块

1. 基线模型对比  
在相同的 `S`、`C`、缺陷矩阵和特征预设下，对以上模型家族进行统一比较。

2. 泛化能力评估  
比较 `scenario` 与 `node_holdout` 两种数据划分方式。

3. 输入表示对比  
比较 `raw_residual`、`raw` 与 `residual` 三种输入方式。

4. 类型头贡献分析  
比较 `lambda_type > 0` 与 `lambda_type = 0` 的差异。

5. 候选空间敏感性  
在完整候选文件准备好之后，对主候选集与更大候选空间进行对比。

## Phase 1/2 当前进展

- `input_1/` 中已经生成 `segment_list.json`、`edge_static_features.csv` 和 `pipe_segment_map.json`
- 面向 `I/E` 主线的 `defect_matrix_ie.csv` 已经生成
- `extract_timeseries.py` 已支持同时输出 `node_timeseries.parquet` 和 `link_timeseries.parquet`
- `edge_feature_engineering.py` 已可将 link 动态、上下游节点差分和管段静态特征合成为 `edge_timeseries.parquet`
- 完整的管段模式 dataset/train 还未打通，目前可稳定运行的主线仍然是节点模式

## 执行说明

- 第一章批量实验入口见 [run_chapter1_main.py](/e:/11.16/script3/run_chapter1_main.py)
- 训练入口见 [train.py](/e:/11.16/script3/train/train.py)
- 在迁移阶段，`script3` 可以通过 `--training-data-dir` 暂时复用 `script2_new/training_data_new`
