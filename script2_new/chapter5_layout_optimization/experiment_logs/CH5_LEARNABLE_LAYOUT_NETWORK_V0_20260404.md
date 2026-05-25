# 第5章结构发明型创新实验：`learnable_layout_network_v0`

## 基本信息

- 日期：2026-04-04
- 日志名称：`learnable_layout_network_v0` 首轮实验
- 对应内容：结构发明型创新 `B1`

## 目标

- 不再直接手工写布局打分器
- 改为让模型从已有布局质量结果中学习“哪些节点更应该被选”
- 形成第5章第一版真正可运行的可学习布点模型

## 本轮方法

本轮采用的是轻量版本：

1. 读取第5章已有布局质量数据集
2. 把已有布局的正式 evaluator 结果转成节点级伪标签
3. 用节点图特征训练一个轻量节点打分网络
4. 再通过带约束的解码器生成 `N=25` 布局

输入特征包括：

- 候选标记
- 入度 / 出度 / 总度
- betweenness / closeness
- downstream reach
- 到候选集的最小 / 平均 hop
- `within2_count`
- 响应签名统计
- evaluator sensitivity

本轮尝试了三个预设：

- `balanced`
- `scenario`
- `generalization`

## 关键脚本与输出

- 脚本：
  - [train_learnable_layout_network.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/train_learnable_layout_network.py)
- 总表：
  - [learnable_layout_network_v0_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/structural_innovation/learnable_layout_network_v0_summary.csv)
  - [ch5_learnable_layout_network_v0_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_learnable_layout_network_v0_compare_N25_seed42.csv)

各预设布局：

- [monitor_nodes_learnable_layout_network_v0_balanced_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/learnable_layout_network_v0_balanced/monitor_nodes_learnable_layout_network_v0_balanced_N25.json)
- [monitor_nodes_learnable_layout_network_v0_scenario_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/learnable_layout_network_v0_scenario/monitor_nodes_learnable_layout_network_v0_scenario_N25.json)
- [monitor_nodes_learnable_layout_network_v0_generalization_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/learnable_layout_network_v0_generalization/monitor_nodes_learnable_layout_network_v0_generalization_N25.json)

## 先看静态结果

- `balanced`
  - `direct=14`, `near=31`, `far=5`, `mean_hop=1.40`, `overlap=14`
- `scenario`
  - 与 `balanced` 解码后得到同一套布局
- `generalization`
  - `direct=13`, `near=32`, `far=5`, `mean_hop=1.24`, `overlap=13`

这说明：

- `generalization` 预设已经在静态层面形成了不同布局
- 而 `balanced / scenario` 还停留在同一个局部最优

## 正式结果

### `scenario split`

- `learnable_layout_network_v0_balanced`
  - `MRR=0.8438`
  - `Top-1=0.7246`
- `learnable_layout_network_v0_generalization`
  - `MRR=0.9098`
  - `Top-1=0.8401`

### `node_holdout`

- `learnable_layout_network_v0_balanced`
  - `MRR=0.2079`
  - `Top-1=0.0226`
  - `Top-3=0.2344`
  - `Top-5=0.3821`
- `learnable_layout_network_v0_generalization`
  - `MRR=0.2864`
  - `Top-1=0.1047`
  - `Top-3=0.3323`
  - `Top-5=0.4748`

## 本轮结论

1. `learnable_layout_network_v0` 已经真正跑通，不再只是结构创新设想。
2. `scenario` 预设没有产生新布局，因此本轮没有必要单独再做正式回插。
3. `generalization` 预设是当前最有价值的一版：
   - 分布内结果几乎贴近 `candidate_observability`
   - `node_holdout` 也没有掉到比 `candidate_observability` 更差
4. `balanced` 说明“仅做平均折中”还不够，容易在 `node_holdout` 下塌掉。

## 对第5章写作的意义

这轮实验已经足以支撑一句很重要的话：

- 第5章的结构发明型创新不是空想
- 监测点布局确实可以被建模
- 而且第一版可学习布点网络已经能达到接近强任务驱动基线的水平

## 下一步

下一轮最值得继续的不是重新回到纯规则法，而是继续围绕 `generalization` 预设做：

1. 伪标签再设计
2. 节点图特征增强
3. 解码器约束再调
4. 如果数据样本变多，再尝试更像真正图模型的 `v0_1 / v1`
