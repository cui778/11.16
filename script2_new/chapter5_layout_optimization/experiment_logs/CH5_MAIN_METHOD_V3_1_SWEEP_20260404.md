# 第5章主方法强化实验：`v3_1` 近邻 sweep

## 基本信息

- 日期：2026-04-04
- 日志名称：主方法强化实验 `v3_1` 近邻 sweep
- 对应内容：两阶段主方法强化后的局部权重/overlap 小范围试探

## 目的

- 在 `v3_1` 已经证明“折中回调有效”之后，再试 1 到 2 个近邻配置
- 重点判断：
  - `overlap` 再收一点，会不会让主方法更合理
  - `candidate separability` 再加一点，会不会带来新布局
  - 哪些变化值得正式回插，哪些可以在静态层停止

## 输入

- 脚本：
  - [build_two_stage_generalization_balanced_layout.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_two_stage_generalization_balanced_layout.py)
- 参考结果：
  - [two_stage_generalization_balanced_layout_v3_1_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_1_summary.csv)
  - [last_run_metrics_ch5_tsgbalv31_N25_s42.json](/E:/11.16/script2_new/outputs/reports/last_run_metrics_ch5_tsgbalv31_N25_s42.json)
  - [last_run_metrics_ch5_nh_tsgbalv31_N25_s42.json](/E:/11.16/script2_new/outputs/reports/last_run_metrics_ch5_nh_tsgbalv31_N25_s42.json)

## 尝试配置

### 1. `v3_1a`

- 权重：沿用 `v3_1`
- 变化：`overlap_cap = anchor + 1`
- 目标：只收紧 overlap，不额外改动主目标权重

输出：

- [two_stage_generalization_balanced_layout_v3_1a_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_1a_summary.csv)
- [monitor_nodes_two_stage_generalization_balanced_layout_v3_1a_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/two_stage_generalization_balanced_layout_v3_1a/monitor_nodes_two_stage_generalization_balanced_layout_v3_1a_N25.json)
- [last_run_metrics_ch5_tsgbalv31a_N25_s42.json](/E:/11.16/script2_new/outputs/reports/last_run_metrics_ch5_tsgbalv31a_N25_s42.json)
- [last_run_metrics_ch5_nh_tsgbalv31a_N25_s42.json](/E:/11.16/script2_new/outputs/reports/last_run_metrics_ch5_nh_tsgbalv31a_N25_s42.json)

关键结果：

- 静态：
  - `selected_overlap_count = 13`
  - `direct = 13`
  - `near = 31`
  - `far = 6`
- `scenario split`：
  - `MRR = 0.8584`
  - `Top-1 = 0.7561`
- `node_holdout`：
  - `MRR = 0.2045`
  - `Top-1 = 0.0317`
  - `Top-3 = 0.2720`
  - `Top-5 = 0.3708`

结论：

- 这版比 `v3_1` 更像“分布内修复版”
- 但它牺牲了 `node_holdout`
- 说明单纯再压 overlap，并不能同时保住分布内和未见节点泛化

### 2. `v3_2`

- 权重：在 `v3_1` 基础上继续微升 `candidate separability`，并做小幅权重平衡
- 变化：`overlap_cap = anchor + 2`
- 目标：看看更强判别性会不会带来新布局

输出：

- [two_stage_generalization_balanced_layout_v3_2_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_2_summary.csv)
- [monitor_nodes_two_stage_generalization_balanced_layout_v3_2_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/two_stage_generalization_balanced_layout_v3_2/monitor_nodes_two_stage_generalization_balanced_layout_v3_2_N25.json)

关键结果：

- 静态指标虽然目标值有变化
- 但最终选出的 25 个监测点集合与 `v3_1` 完全相同

结论：

- 这说明当前这一步权重微调还不足以让搜索过程跳到新的局部最优
- 因为布局集合没变，所以不再重复做正式回插

## 对当前主线的影响

1. `v3_1` 附近确实还有可探索空间，但已经进入“细调区”，不是大改区。
2. `v3_1a` 说明更紧的 overlap 约束会把结果往“更像分布内修复版”方向推。
3. `v3_2` 说明仅仅继续微升 `candidate separability` 权重，不一定能改变最终布局。
4. 当前最值得保留的判断仍然是：
   - `v3_1` 是这一小支线里最平衡的版本
   - `v3_1a` 证明了 trade-off 的存在
   - `v3_2` 证明这一步纯权重微调已经开始遇到平台

## 下一步

- 如果还继续试，优先：
  - 在 `v3_1` 和 `v3_1a` 之间做更小步长的 overlap / balance 联调
  - 不再单独为了 `v3_2` 这类“静态同构”配置重复正式训练
- 如果想收口这条支线，现在也已经足够写出：
  - `v3` 过重
  - `v3_1` 回调有效
  - `v3_1a` 揭示 trade-off
