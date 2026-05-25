# 第5章基线与任务驱动布局结果汇总

## 1. 文档作用

这份文档现在作为第5章“基线布局 + 任务驱动布局”结果总表使用。

它合并整理了原先分散的三部分内容：

- 静态布局初筛
- `scenario split` 下的首轮正式对比
- `node_holdout` 下的首轮泛化对比

后续凡是属于这一组方法的结果，优先继续更新这份文档，不再平行新建同类说明。

## 2. 对比对象

当前这一组方法包括：

- `random`
- `degree`
- `betweenness`
- `downstream`
- `candidate_observability`
- `identifiability_driven`

其中：

- `random / degree / betweenness / downstream`
  - 作为传统或拓扑启发式基线
- `candidate_observability / identifiability_driven`
  - 作为任务驱动基线

## 3. 静态布局初筛结论

静态汇总表：

- [layout_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layout_summary.csv)

`N=25` 时最关键的静态现象如下：

| 布局 | direct | near | far | mean_hop | overlap_count |
|---|---:|---:|---:|---:|---:|
| `degree` | 12 | 21 | 17 | 3.48 | 12 |
| `betweenness` | 11 | 16 | 23 | 3.56 | 11 |
| `downstream` | 9 | 10 | 31 | 4.90 | 9 |
| `random` | 9 | 29 | 12 | 1.96 | 9 |
| `candidate_observability` | 15 | 34 | 1 | 0.94 | 15 |
| `identifiability_driven` | 15 | 34 | 1 | 0.98 | 15 |

静态层可以先确认三点：

1. `degree` 不是候选可观测性最优布局。
2. `candidate_observability` 与 `identifiability_driven` 都能显著压低 `far`。
3. 静态指标只能做初筛，最终仍要由统一评估器裁决。

## 4. `scenario split` 首轮正式结果

结果表：

- [ch5_first_comparison_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_first_comparison_N25_seed42.csv)

对比图：

- [ch5_layout_compare_degree_vs_candidate_observability_vs_identifiability_driven_N25.png](/E:/11.16/script2_new/chapter5_layout_optimization/figures/ch5_layout_compare_degree_vs_candidate_observability_vs_identifiability_driven_N25.png)

核心结果：

| 布局 | MRR | Top-1 | Top-3 | Top-5 | event_level_top1 |
|---|---:|---:|---:|---:|---:|
| `degree` | 0.7933 | 0.6721 | 0.8926 | 0.9545 | 0.7302 |
| `candidate_observability` | 0.9097 | 0.8413 | 0.9802 | 0.9965 | 0.9206 |
| `identifiability_driven` | 0.8656 | 0.7713 | 0.9580 | 0.9860 | 0.8254 |

这一轮说明：

1. 只改变监测点布局，定位性能就会明显变化。
2. `candidate_observability` 在当前任务定义下表现最强。
3. `identifiability_driven` 也优于 `degree`，但第一轮还没超过 `candidate_observability`。

## 5. `node_holdout` 首轮正式结果

结果表：

- [ch5_node_holdout_comparison_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_node_holdout_comparison_N25_seed42.csv)
- [ch5_split_generalization_comparison_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_split_generalization_comparison_N25_seed42.csv)

对比图：

- [ch5_split_generalization_comparison_N25_seed42.png](/E:/11.16/script2_new/chapter5_layout_optimization/figures/ch5_split_generalization_comparison_N25_seed42.png)

核心结果：

| 布局 | MRR | Top-1 | Top-3 | Top-5 | event_level_top1 |
|---|---:|---:|---:|---:|---:|
| `degree` | 0.3655 | 0.1545 | 0.5049 | 0.7249 | 0.1778 |
| `candidate_observability` | 0.2936 | 0.0987 | 0.3873 | 0.5456 | 0.1111 |
| `identifiability_driven` | 0.1756 | 0.0407 | 0.2118 | 0.3075 | 0.0556 |

这一轮说明：

1. `candidate_observability` 在 `scenario split` 下最强，但没有在 `node_holdout` 下保持优势。
2. `degree` 在未见节点泛化下反而更稳。
3. 这说明“候选可观测性很重要”与“泛化最优”不是一回事。

## 6. 当前阶段结论

这组结果目前可以支持下面几条更稳的结论：

1. 第5章的核心问题成立，布局变量会显著影响定位效果。
2. `candidate_observability` 是必须正面承认的强任务驱动基线。
3. `candidate_observability` 的强势主要体现在当前候选假设空间内的任务性能。
4. `node_holdout` 结果表明，第5章后续主方法不能只停在纯候选导向上。

## 7. 与后续主方法的关系

这一组文档的作用不是给出最终主方法，而是把后续方法创新的起点讲清楚：

- 传统拓扑布局不等于任务最优
- 候选可观测性确实是有效机制
- 但泛化与更广义的全网监测意义仍需要更强方法来补足
