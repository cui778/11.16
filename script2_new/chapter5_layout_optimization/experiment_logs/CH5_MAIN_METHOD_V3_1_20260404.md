# 第5章主方法强化实验：v3_1

## 基本信息

- 日期：2026-04-04
- 日志名称：主方法强化实验 v3_1
- 对应内容：两阶段主方法强化

## 目的

- 在 `v3` 过度全网正则化的基础上做折中回调
- 保留 `backbone / overlap regularization`
- 恢复更强的 candidate-side observability 与 separability
- 检查这种折中是否能改善 `node_holdout`

## 输入

- 脚本：
  - [build_two_stage_generalization_balanced_layout.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_two_stage_generalization_balanced_layout.py)
- 文件：
  - [two_stage_generalization_balanced_layout_v3_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_summary.csv)
  - [ch5_strengthened_main_method_v3_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_strengthened_main_method_v3_compare_N25_seed42.csv)

## 输出

- 文件：
  - [two_stage_generalization_balanced_layout_v3_1_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_1_summary.csv)
  - [monitor_nodes_two_stage_generalization_balanced_layout_v3_1_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/two_stage_generalization_balanced_layout_v3_1/monitor_nodes_two_stage_generalization_balanced_layout_v3_1_N25.json)
  - [last_run_metrics_ch5_tsgbalv31_N25_s42.json](/E:/11.16/script2_new/outputs/reports/last_run_metrics_ch5_tsgbalv31_N25_s42.json)
  - [last_run_metrics_ch5_nh_tsgbalv31_N25_s42.json](/E:/11.16/script2_new/outputs/reports/last_run_metrics_ch5_nh_tsgbalv31_N25_s42.json)
  - [ch5_strengthened_main_method_v3_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_strengthened_main_method_v3_compare_N25_seed42.csv)

## 过程

- 将 `build_two_stage_generalization_balanced_layout.py` 改成可配置权重版本。
- 新增 `weight_preset` 机制，保留原始 `v3`，并增加 `v3_1`。
- `v3_1` 使用更高的 candidate-side 权重、更低的 overlap penalty，并把 `overlap_cap` 放宽到 `anchor + 2`。
- 先生成 `N=25` 布局，再分别回插 `scenario split` 与 `node_holdout`。

## 关键发现

- 静态上，`v3_1` 相比 `v3`：
  - `direct: 13 -> 14`
  - `near: 29 -> 30`
  - `far: 8 -> 6`
  - `selected_overlap_count: 13 -> 14`
- `scenario split` 下，`v3_1` 没有超过 `v3`：
  - `MRR: 0.8451 -> 0.8285`
  - `Top-1: 0.7211 -> 0.7106`
- `node_holdout` 下，`v3_1` 明显好于 `v3`：
  - `MRR: 0.1972 -> 0.2446`
  - `Top-1: 0.0000 -> 0.0452`
  - `Top-3: 0.3022 -> 0.3188`
  - `Top-5: 0.4288 -> 0.5147`

## 对第5章写作的影响

- 这一轮证明 `v3` 的问题确实是“全网约束过重”，而不是方向完全错误。
- `v3_1` 说明折中回调是有效的，但当前仍未形成超过 `v2` 的新最优方案。
- 第5章主方法叙事可以更清楚地写成：
  - `v1` 建立框架
  - `v2` 增加敏感度/重构/鲁棒性
  - `v3` 验证过强全网正则化的风险
  - `v3_1` 验证折中回调的有效性

## 下一步

- 继续在 `v3_1` 附近小范围调权重，而不是回到 `v3` 的重约束配置。
- 下一轮优先尝试：
  - 再微升 candidate separability
  - 保持 moderate backbone
  - 不继续提高 overlap
