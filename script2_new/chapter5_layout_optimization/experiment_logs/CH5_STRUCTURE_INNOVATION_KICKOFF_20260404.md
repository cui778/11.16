# 第5章结构发明型创新启动记录

## 基本信息

- 日期：2026-04-04
- 日志名称：结构发明型创新启动
- 对应内容：从“结构发明型创新”角度启动第5章新路线

## 启动判断

当前不直接进入高风险的：

- 强化学习逐点选址
- evaluator-in-the-loop 高成本双层优化
- 端到端可学习 Top-K 选点

而是先搭两层基础：

1. 布局质量数据集
2. surrogate

原因是：

- 第5章已经积累了真实布局和真实 evaluator 回插结果
- 现在最缺的不是再多想一个模型名，而是先把可学习模型的训练底座搭起来
- 如果这个底座不先整理，后面的结构创新很容易又空转

## 本次动作

- 更新双轨创新主文档：
  - [CH5_DUAL_TRACK_INNOVATION_PLAN_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_DUAL_TRACK_INNOVATION_PLAN_20260404.md)
- 新增结构创新起步脚本：
  - [build_layout_quality_dataset.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_layout_quality_dataset.py)
  - [train_layout_surrogate.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/train_layout_surrogate.py)
- 更新脚本目录说明：
  - [README.md](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/README.md)

## 预期输出

结构创新起步层预期先产出：

- `layout_quality_dataset_long.csv`
- `layout_quality_dataset_wide.csv`
- `layout_quality_dataset_manifest.json`
- `surrogate_predictions.csv`
- `surrogate_coefficients.csv`
- `surrogate_training_report.json`

## 实际产出

本轮已经实际生成：

- [layout_quality_dataset_long.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/layout_quality_dataset_long.csv)
- [layout_quality_dataset_wide.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/layout_quality_dataset_wide.csv)
- [layout_quality_dataset_manifest.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/layout_quality_dataset_manifest.json)
- [surrogate_predictions.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/surrogate/surrogate_predictions.csv)
- [surrogate_coefficients.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/surrogate/surrogate_coefficients.csv)
- [surrogate_training_report.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/surrogate/surrogate_training_report.json)

当前数据集规模：

- 静态布局记录：`44`
- 带正式 evaluator 结果的布局记录：`20`
- 可直接用于 surrogate 的宽表样本：`10`
- 当前覆盖方法：`10`
- 当前预算：`N=25`

## 第一轮结果

第一轮 surrogate 只使用纯静态布局特征，不使用任何 evaluator 目标列作为输入。

当前结果显示：

- `scenario_top1` 的可预测性已经有一点信号：
  - `R2 = 0.3011`
- `scenario_mrr` 目前接近弱可预测：
  - `R2 = -0.0263`
- `node_holdout_mrr / node_holdout_top1` 目前仍然较难从这批静态特征稳定预测：
  - `R2 = -0.8799 / -0.6492`

第一轮系数里绝对值较大的静态项主要包括：

- `mean_hop`
- `near`
- `far`
- `signature_min_nn`
- `objective_overlap_penalty`
- `overlap_cap`

这说明结构创新下一步至少是有方向的：

- `scenario split` 这边，局部可观测性与 overlap 相关项已经开始显著
- `node_holdout` 这边，单靠现有静态特征还不够，后面更可能要补：
  - 更强的全网拓扑表征
  - 更稳的候选可分性表征
  - 或布局组合级特征

## 当前路线定位

这条线现在的定位不是“马上替代主方法”，而是：

- 先成为第5章的第二创新轨道
- 先证明“布局优化可以被建模”
- 再决定后续是走 surrogate 引导，还是走 learnable layout network

## 下一步

1. 在当前 `N=25` 数据集基础上，补更多正式回插样本
2. 优先把 `N=10 / N=15` 的主方法和基线结果也并进来
3. 再训练第二轮 surrogate
4. 根据 surrogate 报告判断：
   - 哪些静态布局特征值得保留
   - learnable layout network 应该学什么
