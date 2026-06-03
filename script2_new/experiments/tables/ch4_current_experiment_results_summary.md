# 第四章当前实验结果汇总

> 本文件汇总截至当前已完成的实验结果。由于 `two_stage_balanced_N5` 在训练过程中因内存问题中断，预算梯度表中暂不包含该项。部分低预算结果后续可在工具恢复后从对应 metrics JSON 中继续补齐。
>
> 正文主线说明：第4章正文采用 `IE420 + normal20` 的 time-gated 主线，不主动写 persistent / mixed persistent / full-window active。本文档中的 persistent、mixed persistent 和窗口采样相关结果仅作为内部探索实验记录，用于方法边界判断和后续方案备查，不进入当前第4章正文主结论。

## 1. 特征组对照结果

固定条件：

```text
layout = degree_N25
lambda_loc = 1.0
```

| 特征组 | MRR | Top-1 | Top-3 | Event Top-1 | Active F1 | Normal FPR | Scene Recall | Scene FPR | Scene F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_only | 0.2419 | 0.1438 | 0.2313 | 0.1452 | 0.4379 | 0.0564 | 0.3548 | 0.6667 | 0.5116 |
| residual_only | 0.8082 | 0.7158 | 0.8665 | 0.7258 | 0.9645 | 0.0027 | 0.9355 | 0.0000 | 0.9667 |
| raw_plus_residual | 0.8290 | 0.7319 | 0.9206 | 0.7903 | 0.9766 | 0.0145 | 1.0000 | 0.0000 | 1.0000 |

结论：`raw_plus_residual` 在主测试中综合表现最好，但性能提升主要来自 residual 特征。原始时序更适合作为背景状态补充，而不是单独承担缺陷定位任务。

## 2. lambda_loc 调优结果

固定条件：

```text
feature_set = raw_plus_residual
layout = degree_N25
```

| lambda_loc | MRR | Top-1 | Top-3 | Event Top-1 | Active F1 | Normal FPR | Scene Recall | Scene FPR | Scene F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.8115 | 0.6928 | 0.9229 | 0.7742 | 0.9914 | 0.0054 | 1.0000 | 0.0000 | 1.0000 |
| 0.5 | 0.8415 | 0.7399 | 0.9402 | 0.7742 | 0.9774 | 0.0064 | 0.9839 | 0.0000 | 0.9919 |
| 1.0 | 0.8290 | 0.7319 | 0.9206 | 0.7903 | 0.9766 | 0.0145 | 1.0000 | 0.0000 | 1.0000 |
| 2.0 | 0.8396 | 0.7480 | 0.9241 | 0.8226 | 0.9573 | 0.0317 | 1.0000 | 0.0000 | 1.0000 |

推荐主配置：

```text
feature_set = raw_plus_residual
lambda_loc = 0.5
```

选择理由：`lambda_loc=0.5` 的 MRR 最高、Top-3 最高，且 Normal FPR 较低，检测与定位之间较为均衡。虽然 `lambda_loc=2.0` 的 Top-1 和 Event Top-1 略高，但误报率更高，Active F1 有所下降。

## 3. Persistent IE Zero-shot 结果（内部探索，不进入正文主线）

| 实验设置 | Persistent Recall | Persistent Scene Recall | Persistent Top-1 | Persistent MRR |
|---|---:|---:|---:|---:|
| raw_only, lambda=1.0, degree_N25 | 0.2676 | 0.8690 | 0.1502 | 0.2471 |
| residual_only, lambda=1.0, degree_N25 | 0.9632 | 1.0000 | 0.0660 | 0.1517 |
| raw_plus_residual, lambda=1.0, degree_N25 | 0.9994 | 1.0000 | 0.0048 | 0.0639 |
| raw_plus_residual, lambda=0.5, degree_N25 | 0.9708 | 1.0000 | 0.0584 | 0.1271 |
| raw_plus_residual, lambda=2.0, degree_N25 | 0.9855 | 1.0000 | 0.0865 | 0.1631 |

结论：模型对 persistent IE 的场景级检出较稳定，但 persistent 条件下的定位迁移能力明显弱于 time-gated 主测试。这说明模型能够识别持续性缺陷存在，但在无 onset / offset 的 full-window 场景下，缺陷节点定位仍存在泛化边界。

## 4. 第四章桥接布局对比结果

固定条件：

```text
feature_set = raw_plus_residual
lambda_loc = 0.5
N = 25
```

| 布局 | N | MRR | Top-1 | Top-3 | Event Top-1 | Active F1 | Normal FPR | Scene Recall | Scene FPR | Scene F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| degree | 25 | 0.8415 | 0.7399 | 0.9402 | 0.7742 | 0.9774 | 0.0064 | 0.9839 | 0.0000 | 0.9919 |
| two_stage_balanced | 25 | 0.9032 | 0.8331 | 0.9689 | 0.8871 | 0.9908 | 0.0059 | 1.0000 | 0.0000 | 1.0000 |

结论：在监测节点数量相同的条件下，`two_stage_balanced` 布局明显优于 `degree` 布局，说明监测节点的空间分布会显著影响 I/E 缺陷诊断性能。该结果可作为第四章向第五章布局优化研究过渡的桥接实验。

## 5. 预算梯度结果：已确认部分

固定条件：

```text
feature_set = raw_plus_residual
lambda_loc = 0.5
```

| 布局 | N | MRR | Top-1 | Top-3 | Active F1 | Normal FPR | Scene F1 | Persistent Recall | Persistent Scene Recall | Persistent Top-1 | Persistent MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| degree | 15 | 0.8143 | 0.6997 | 0.9333 | 0.9822 | 0.0086 | 0.9919 | 0.9895 | 1.0000 | 0.0544 | 0.1436 |
| degree | 20 | 0.8114 | 0.6985 | 0.9079 | 0.9862 | 0.0075 | 1.0000 | 0.9986 | 1.0000 | 0.0230 | 0.1058 |
| degree | 25 | 0.8415 | 0.7399 | 0.9402 | 0.9774 | 0.0064 | 0.9919 | 0.9708 | 1.0000 | 0.0584 | 0.1271 |
| two_stage_balanced | 10 | 0.7685 | 0.6306 | 0.9137 | 0.9617 | 0.0059 | 0.9752 | 0.9991 | 1.0000 | 0.0587 | 0.1521 |
| two_stage_balanced | 15 | 0.8640 | 0.7745 | 0.9517 | 0.9908 | 0.0021 | 1.0000 | 0.8963 | 0.9762 | 0.0014 | 0.0627 |
| two_stage_balanced | 20 | 0.8713 | 0.7768 | 0.9712 | 0.9936 | 0.0011 | 0.9919 | 0.9986 | 1.0000 | 0.0323 | 0.1075 |
| two_stage_balanced | 25 | 0.9032 | 0.8331 | 0.9689 | 0.9908 | 0.0059 | 1.0000 | 0.9991 | 1.0000 | 0.0159 | 0.1047 |

当前缺失项：

```text
two_stage_balanced_N5
```

说明：`two_stage_balanced_N5` 因内存问题中断，暂不纳入当前表格。预算梯度在第四章中可作为辅助结果使用，不建议展开过多方法解释；系统性的预算和布局策略比较建议放入第五章。

## 6. 当前可写结论

1. `raw_only` 明显不足，说明原始水力水质时序本身难以直接支撑缺陷定位。
2. `residual_only` 已具备较强诊断能力，说明 baseline 对齐后的残差响应是主要有效信息来源。
3. `raw_plus_residual` 在主测试中表现最好，说明原始时序可以作为背景状态补充。
4. `lambda_loc=0.5` 在检测、定位和误报率之间取得较好平衡。
5. Persistent IE zero-shot 的场景级检出较好，但定位迁移较弱，说明模型对持续性缺陷定位仍存在泛化边界。
6. 在 `N=25` 条件下，`two_stage_balanced` 明显优于 `degree`，说明监测布局存在可优化空间。
7. 第四章应将布局实验作为桥接分析，不宜展开过多布局方法；第五章再系统讨论监测预算与布局优化策略。

## 7. Mixed Persistent Training v1 结果（内部探索，不进入正文主线）

本轮实验将 persistent IE 从 zero-shot 测试集改为训练分布的一部分，构建混合数据集：

```text
time-gated IE420 + persistent IE84 + normal20 + baseline
dataset = ie420_plus_persistent84_plus_normal20_v1
feature_set = raw_plus_residual
lambda_loc = 0.5
layout = degree_N25
epochs = 25
```

数据协议统计：

| 数据集 | 场景总数 | baseline | time-gated | persistent | normal | 节点数 |
|---|---:|---:|---:|---:|---:|---:|
| ie420_plus_persistent84_plus_normal20_v1 | 525 | 1 | 421 | 84 | 20 | 128 |

与原主模型对比：

| 实验 | MRR | Top-1 | Top-3 | Active F1 | Normal FPR | Scene Recall | Scene FPR | Scene F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IE420 + normal10 | 0.8415 | 0.7399 | 0.9402 | 0.9774 | 0.0064 | 0.9839 | 0.0000 | 0.9919 |
| IE420 + persistent84 + normal20 | 0.6483 | 0.5072 | 0.7220 | 0.9714 | 0.0043 | 0.9740 | 0.0000 | 0.9868 |

分组定位结果：

| 场景组 | 参与定位场景数 | 定位窗口数 | MRR | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| time-gated IE | 62 | 888 | 0.7885 | 0.6779 | 0.8637 | 0.9673 |
| persistent IE | 15 | 630 | 0.4507 | 0.2667 | 0.5222 | 0.6841 |

场景报警结果：

| 场景组 | 测试场景数 | Scene Recall | Scene FPR | Scene F1 | 平均场景分数 |
|---|---:|---:|---:|---:|---:|
| normal | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0012 |
| time-gated IE | 62 | 0.9839 | 0.0000 | 0.9919 | 0.9580 |
| persistent IE | 15 | 0.9333 | 0.0000 | 0.9655 | 0.9334 |

阶段性判断：

1. mixed v1 能够让 persistent 缺陷进入训练分布，persistent 定位明显强于此前 `raw_plus_residual, lambda=0.5` 的 zero-shot 结果（MRR 约 0.1271），但仍显著低于 time-gated 定位。
2. mixed v1 对场景级报警没有造成明显误报扩张，Normal FPR 和 Scene FPR 仍较低。
3. mixed v1 的总体定位性能较原主模型明显下降，说明直接混入 persistent84 会稀释 time-gated 主任务的定位学习。
4. 暂不建议立即扩展到 persistent168 或 persistent252。下一步应优先检查采样权重、分组损失或训练/测试拆分方式，再决定是否扩大 persistent 场景。

## 7.1 IE420 + normal20 对照实验

为确认 mixed v1 性能下降是否由 normal 场景扩展导致，补充构建 `IE420 + normal20` 对照数据集：

```text
dataset = ie420_plus_normal20_v1
feature_set = raw_plus_residual
lambda_loc = 0.5
layout = degree_N25
epochs = 25
```

数据协议：

| 数据集 | 场景总数 | baseline | time-gated | persistent | normal | 节点数 |
|---|---:|---:|---:|---:|---:|---:|
| ie420_plus_normal20_v1 | 441 | 1 | 421 | 0 | 20 | 128 |

实际划分：

| split | 场景数 | baseline/normal | I | E | 定位窗口数 |
|---|---:|---:|---:|---:|---:|
| train | 309 | 14 | 179 | 116 | 4224 |
| val | 66 | 2 | 37 | 27 | 910 |
| test | 66 | 5 | 34 | 27 | 876 |

对照结果：

| 实验 | MRR | Top-1 | Top-3 | Active F1 | Normal FPR | Scene Recall | Scene FPR | Scene F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IE420 + normal10 | 0.8415 | 0.7399 | 0.9402 | 0.9774 | 0.0064 | 0.9839 | 0.0000 | 0.9919 |
| IE420 + normal20 | 0.8457 | 0.7728 | 0.8973 | 0.9790 | 0.0005 | 0.9836 | 0.0000 | 0.9917 |
| IE420 + persistent84 + normal20 | 0.6483 | 0.5072 | 0.7220 | 0.9714 | 0.0043 | 0.9740 | 0.0000 | 0.9868 |

结论：

1. 仅将 normal10 扩展到 normal20 后，主定位性能没有下降，MRR 维持在约 0.84，Normal FPR 反而降低。
2. mixed v1 的性能下降主要不是由 normal20 引起，而是 persistent84 进入训练分布后改变了 active 定位窗口的组成。
3. 后续若继续纳入 persistent，不应简单扩大 persistent 场景数量，而应优先考虑 persistent/time-gated 的采样比例控制、分组损失或分形态定位策略。

## 7.2 Persistent 训练窗口下采样实验

为验证 mixed v1 的定位下降是否来自 persistent active 窗口占比过高，保持数据集不变，仅在训练集内对 persistent 窗口进行下采样：

```text
dataset = ie420_plus_persistent84_plus_normal20_v1
feature_set = raw_plus_residual
lambda_loc = 0.5
layout = degree_N25
persistent_train_sample_ratio = 0.25
```

训练集中 persistent 窗口变化：

```text
persistent train windows: 2352 -> 588
```

总体结果：

| 实验 | MRR | Top-1 | Top-3 | Active F1 | Normal FPR | Scene Recall | Scene FPR | Scene F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IE420 + normal20 | 0.8457 | 0.7728 | 0.8973 | 0.9790 | 0.0005 | 0.9836 | 0.0000 | 0.9917 |
| mixed full persistent | 0.6483 | 0.5072 | 0.7220 | 0.9714 | 0.0043 | 0.9740 | 0.0000 | 0.9868 |
| mixed persistent 25% train sample | 0.6820 | 0.5343 | 0.7905 | 0.9748 | 0.0027 | 0.9870 | 0.0000 | 0.9935 |

分组定位结果：

| 实验 | time-gated MRR | time-gated Top-1 | time-gated Top-3 | persistent MRR | persistent Top-1 | persistent Top-3 |
|---|---:|---:|---:|---:|---:|---:|
| mixed full persistent | 0.7885 | 0.6779 | 0.8637 | 0.4507 | 0.2667 | 0.5222 |
| mixed persistent 25% train sample | 0.8387 | 0.7365 | 0.9426 | 0.4612 | 0.2492 | 0.5762 |

结论：

1. 对 persistent 训练窗口做 25% 下采样后，time-gated 定位几乎恢复到原主任务水平（MRR 约 0.84）。
2. persistent 定位没有明显损失，MRR 从 0.4507 小幅提高到 0.4612，Top-3 也有所提高。
3. 因此 mixed v1 的主要问题不是 persistent 场景不可训练，而是 persistent 窗口占比过高导致 active 定位分布失衡。
4. 当前更合理的路线是保留统一模型，但在训练时控制 persistent/time-gated 窗口比例；暂不需要把任务拆成两个完全独立模型。

## 7.3 Persistent 窗口采样比例扫描

在 `IE420 + persistent84 + normal20` 数据集上进一步比较 persistent 训练窗口采样比例：

```text
persistent_train_sample_ratio ∈ {0.10, 0.25, 0.50, 1.00}
feature_set = raw_plus_residual
lambda_loc = 0.5
layout = degree_N25
```

结果如下：

| persistent 训练窗口比例 | Overall MRR | Overall Top-1 | Overall Top-3 | time-gated MRR | time-gated Top-1 | time-gated Top-3 | persistent MRR | persistent Top-1 | persistent Top-3 | Active F1 | Normal FPR | Scene F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.7591 | 0.6561 | 0.8294 | 0.8620 | 0.7883 | 0.9167 | 0.6141 | 0.4698 | 0.7063 | 0.9734 | 0.0027 | 0.9935 |
| 0.25 | 0.6820 | 0.5343 | 0.7905 | 0.8387 | 0.7365 | 0.9426 | 0.4612 | 0.2492 | 0.5762 | 0.9748 | 0.0027 | 0.9935 |
| 0.50 | 0.6814 | 0.5415 | 0.7866 | 0.8557 | 0.7725 | 0.9336 | 0.4357 | 0.2159 | 0.5794 | 0.9713 | 0.0027 | 0.9868 |
| 1.00 | 0.6483 | 0.5072 | 0.7220 | 0.7885 | 0.6779 | 0.8637 | 0.4507 | 0.2667 | 0.5222 | 0.9714 | 0.0043 | 0.9868 |

阶段性结论：

1. `persistent_train_sample_ratio=0.10` 在当前比例扫描中综合最优，既保持 time-gated 定位，又显著提高 persistent 定位。
2. full persistent 训练（1.00）不是最佳选择，说明全窗口 active 样本的重复窗口会造成定位学习分布失衡。
3. `0.25` 和 `0.50` 并未优于 `0.10`，说明 persistent 训练窗口不是越多越好。
4. 当前推荐第四章 mixed 版本采用 `persistent_train_sample_ratio=0.10` 作为统一训练框架下的主结果；`IE420 + normal20` 可作为无 persistent 持续形态补充前的对照结果。

## 8. 任务层级口径

当前模型仍然只有两个窗口级输出头：

```text
detect_logit(t) -> p_active(t)
candidate_logits(t, 50) -> node_scores(t)
```

论文中的诊断层级应表述为“由窗口级输出聚合得到的诊断结果”，而不是多个独立模型头：

| 层级 | 聚合对象 | 诊断问题 | 输出 |
|---|---|---|---|
| 窗口级证据 | 模型直接输出 | 当前窗口是否有缺陷响应，候选节点得分如何 | `p_active(t)`, `node_scores(t)` |
| 场景时间级 | 聚合 `p_active(t)` | 缺陷大致发生在哪段时间 | active 时间段 |
| 场景空间级 | 聚合 active 窗口的 `node_scores(t)` | 缺陷更可能位于哪些候选节点 | 候选节点 Top-K |
| 场景报警级 | 聚合 `p_active(t)` | 48 h 场景是否存在缺陷 | `has_defect` |
| 综合诊断级 | 时间 + 空间结果融合 | 给出最终诊断结果 | active 时间段 + 节点 Top-K |

推荐写作口径：

```text
上述层级并非多个独立输出头，而是由窗口级检测概率和候选节点得分经过时间聚合、空间聚合和场景聚合得到。
```
