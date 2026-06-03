# 第5章脚本目录说明

本目录存放第5章"监测点布局优化"的全部实验脚本。最终实验已完成（2026-06-01），以下按角色分类。

## 一、核心脚本（用于最终实验）

| 脚本 | 用途 | 输出 |
|---|---|---|
| `build_layout_suite.py` | 基线布局批量生成（Degree/Betweenness/Cand-Obs等） | `outputs/layouts/<strategy>/` |
| `build_two_stage_balanced_layout.py` | Two-stage v1 布局生成 | `outputs/layouts/two_stage_balanced_layout_v1/` |
| `build_embedding_guided_clean_layout.py` | Embedding-Guided 布局生成 | `outputs/layouts/embedding_guided_clean_fixed/` |
| `train_learnable_layout_network.py` | Node-Feedback 节点评分训练 + 布局选择 | `outputs/structural_innovation/learnable_layout_network_v0/` |
| `build_quality_dataset_fixed_normal20.py` | 构建 fixed protocol 下的 quality dataset | `outputs/structural_innovation/layout_quality_dataset_wide_fixed_normal20_*.csv` |
| `_mechanism_analysis.py` | 机制分析（I/E分组、Jaccard、结构指标） | `thesis_writing_repo/figures/ch5/source_data/` |
| `_run_all_budgets.py` | 预算曲线批量训练（4方法×4预算） | `script2_new/outputs/reports/` |

## 二、中间脚本（已由更新版本替代，保留供参考）

| 脚本 | 用途 | 说明 |
|---|---|---|
| `build_two_stage_balanced_layout_v2.py` | Two-stage v2 | 已被 generalization 版替代 |
| `build_two_stage_generalization_balanced_layout.py` | Two-stage v3/v3_1/v3_1a/v3_2 | 未用于最终实验 |
| `build_overlap_controlled_probe_layouts.py` | overlap-controlled 探针实验 | 机制验证用 |
| `build_layout_quality_dataset.py` | 旧版 quality dataset 构建 | 已被 fixed_normal20 版替代 |
| `build_learning_layout_multiseed.py` | 多seed布局生成 | 用于稳定性分析 |
| `train_layout_surrogate.py` | Surrogate 模型训练 | 已从主表移除 |

## 三、基础设施脚本

| 脚本 | 用途 |
|---|---|
| `build_evaluator_manifest.py` | 生成统一评估协议下的正式运行清单 |
| `build_dataset_views.py` | 为每个布局生成轻量 dataset-view 清单 |
| `build_ch5_core_budget_manifest.py` | 构建预算实验清单 |

## 四、最终实验协议

```
dataset = IE420 + normal20
feature_set = raw_plus_residual
lambda_loc = 0.5
model = hydraulic_inverse_deepattn
split = scenario
diagnosis_seed = 7 / 42 / 123
budget = N25（主表）, N5/N10/N15/N20（预算曲线）
```

## 五、最终实验矩阵

| 方法 | 布局来源 | seed 7 | seed 42 | seed 123 |
|---|---|---|---|---|
| Degree | `build_layout_suite.py` | ✅ | ✅ | ✅ |
| Betweenness | `build_layout_suite.py` | ✅ | ✅ | ✅ |
| Cand-Obs | `build_layout_suite.py` | ✅ | ✅ | ✅ |
| Two-stage v1 | `build_two_stage_balanced_layout.py` | ✅ | ✅ | ✅ |
| Node-Feedback | `train_learnable_layout_network.py` | ✅ | ✅ | ✅ |
| Embedding-Guided | `build_embedding_guided_clean_layout.py` | ✅ | ✅ | ✅ |
