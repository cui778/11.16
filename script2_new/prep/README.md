# prep/ 脚本目录说明

本目录存放数据预处理和数据集构建脚本。按角色分类如下。

## 核心脚本（正式管线入口）

| 脚本 | 用途 |
|---|---|
| `build_formal_ie_conservative_matrix.py` | IE420 缺陷矩阵构建 |
| `build_ie420_normal20_control_dataset.py` | 正式组合数据集构建（ie420_plus_normal20_v1） |

## 共享基础设施

| 脚本 | 用途 | 被依赖 |
|---|---|---|
| `build_48h_restructured_datasets.py` | normal/persistent 数据集构建函数 | 被正式脚本 import |
| `dataset_build_utils.py` | 共享工具函数（`_infer_ie_defect_csv` 等） | 被正式脚本 import |

## 工具模块

| 脚本 | 用途 |
|---|---|
| `inp_parser.py` | SWMM INP 文件解析 |
| `defect_matrix.py` | 缺陷矩阵工具 |
| `validate_node_sets.py` | 节点集合验证 |
| `extract_timeseries.py` | 时间序列提取 |
| `residual_features.py` | 残差特征计算 |
| `build_graph.py` | 图结构构建 |
| `build_graph_features.py` | 图特征提取 |
| `build_candidates.py` | 候选节点集合构建 |
| `build_monitors.py` | 监测节点构建 |

## 数据集构建辅助

| 脚本 | 用途 |
|---|---|
| `build_defect_matrix_v2.py` | 缺陷矩阵 v2 |
| `build_e_boost_matrix.py` | E-boost 矩阵 |
| `build_e_curriculum_matrix.py` | E-curriculum 矩阵 |
| `build_ie_curriculum_v3_matrix.py` | IE curriculum v3 |
| `build_fulltime_ie_matrix_from_formal.py` | fulltime IE 矩阵 |
| `build_topology_robustness_inputs.py` | 拓扑鲁棒性输入 |
| `build_balanced_observability_monitors.py` | 均衡可观测性监测 |
| `build_observability_aware_monitors.py` | 可观测性感知监测 |
| `repair_monitor_only_dataset.py` | 监测数据集修复 |
| `analyze_temporal_spatial_features.py` | 时空特征分析 |

## 已归档脚本

以下探索脚本已移入 `chapter3_data_generation/legacy_exploration/scripts/`：

| 脚本 | 原因 |
|---|---|
| `build_persistent_ie_matrix.py` | persistent 探索 |
| `build_mixed_persistent_training_dataset.py` | mixed 探索 |

另有 4 个探索脚本从 `scripts/` 目录归档：
- `plot_validate_persistent_normal.py`
- `summarize_mixed_persistent_results.py`
- `summarize_persistent_sampling_sweep.py`
- `diagnose_worst_localization_scenarios.py`
