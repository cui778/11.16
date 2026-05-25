# 第5章主方法迭代结果汇总

## 1. 文档作用

这份文档现在作为第5章“主方法迭代结果总表”使用。

它合并整理了原先分散的三部分内容：

- `two_stage_balanced_layout_v1 / v2`
- 两阶段方法在 `node_holdout` 下的对比
- 强化版主方法 `v3`

后续凡是属于主方法迭代的结果，优先继续更新这份文档。

## 2. 方法迭代序列

当前已经形成的主方法序列包括：

- `two_stage_balanced_layout_v1`
- `two_stage_balanced_layout_v2`
- `two_stage_generalization_balanced_layout_v3`
- `two_stage_generalization_balanced_layout_v3_1`
- `two_stage_generalization_balanced_layout_v3_1a`

相关脚本：

- [build_two_stage_balanced_layout.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_two_stage_balanced_layout.py)
- [build_two_stage_balanced_layout_v2.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_two_stage_balanced_layout_v2.py)
- [build_two_stage_generalization_balanced_layout.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_two_stage_generalization_balanced_layout.py)

## 3. v1：两阶段正式版

`v1` 的结构是：

1. 代表池压缩
2. 预算内优化

核心思想是把开题时的两阶段方法压成可执行正式版。

静态摘要：

- [two_stage_balanced_layout_v1_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_balanced_layout_v1_summary.csv)

`N=25` 时静态结果：

- `direct=18`
- `near=31`
- `far=1`
- `mean_hop=0.92`

## 4. v2：约束增强版

`v2` 在 `v1` 上增加：

- evaluator 节点敏感度
- reconstruction 约束
- 随机失效鲁棒性
- simulated annealing 微调

静态摘要：

- [two_stage_balanced_layout_v2_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_balanced_layout_v2_summary.csv)

关键缓存：

- [evaluator_node_sensitivity_ch5_tsbal25_s42.npz](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/cache/evaluator_node_sensitivity_ch5_tsbal25_s42.npz)

`N=25` 时静态结果：

- `direct=15`
- `near=29`
- `far=6`
- `mean_hop=1.56`
- `objective_reconstruction=0.8419`
- `objective_failure_robustness=0.5880`

## 5. v3：全网约束增强尝试

`v3` 是在 overlap probe 之后，为了强化全网意义而尝试的版本。

相比 `v2`，增加：

- non-candidate backbone coverage
- candidate / non-candidate coverage balance
- monitor dispersion
- overlap regularization

静态摘要：

- [two_stage_generalization_balanced_layout_v3_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_summary.csv)

`N=25` 时静态行为：

- `selected_overlap_count = 13`
- `direct = 13`
- `near = 29`
- `far = 8`
- `objective_noncandidate_backbone = 0.8454`
- `objective_coverage_balance = 0.7596`

## 6. v3.1：折中强化版

`v3_1` 不是新路线，而是在 `v3` 上做的折中强化。

调整思路是：

- 保留 backbone / overlap regularization
- 拉回 candidate-side observability 与 separability
- 降低全网约束过重造成的定位锐度损失

这版仍然使用同一个脚本：

- [build_two_stage_generalization_balanced_layout.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_two_stage_generalization_balanced_layout.py)

只是改为：

- `weight_preset = v3_1`
- `overlap_cap = overlap_anchor + 2`

静态摘要：

- [two_stage_generalization_balanced_layout_v3_1_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_1_summary.csv)

`N=25` 时静态行为：

- `selected_overlap_count = 14`
- `direct = 14`
- `near = 30`
- `far = 6`
- `objective_candidate_observability = 0.6300`
- `objective_candidate_separability = 0.8015`
- `objective_noncandidate_backbone = 0.8437`

与 `v3` 相比，它的特点是：

- 候选侧判别性更强
- overlap 略升，但仍未回到 `v1` 的 18
- 保留了一定全网支撑，而不是重新退化成纯 candidate-heavy

## 6.1 `v3_1` 附近小范围 sweep

在 `v3_1` 跑通之后，又补做了两个近邻配置：

- `v3_1a`：保持 `v3_1` 权重不变，但把 `overlap_cap` 从 `anchor + 2` 收紧到 `anchor + 1`
- `v3_2`：保留 `overlap_cap = anchor + 2`，同时继续微升 `candidate separability` 权重并微调其余权重

对应静态摘要：

- [two_stage_generalization_balanced_layout_v3_1a_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_1a_summary.csv)
- [two_stage_generalization_balanced_layout_v3_2_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/two_stage_generalization_balanced_layout_v3_2_summary.csv)

`v3_1a` 的关键静态行为：

- `selected_overlap_count = 13`
- `direct = 13`
- `near = 31`
- `far = 6`
- `objective_candidate_observability = 0.6225`
- `objective_candidate_separability = 0.8030`
- `objective_reconstruction = 0.7029`

这版相当于在 `v3_1` 基础上进一步收紧 candidate overlap，用来检验“略降 overlap 能否改善全网意义，同时不明显伤害定位”。

`v3_2` 的静态结果虽然目标值不同，但最终选出的 25 个监测点集合与 `v3_1` 完全相同，因此没有再重复做正式训练回插。

## 7. `scenario split` 下的主方法结果

结果表：

- [ch5_two_stage_framework_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_two_stage_framework_compare_N25_seed42.csv)
- [ch5_strengthened_main_method_v3_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_strengthened_main_method_v3_compare_N25_seed42.csv)

核心结果：

| 方法 | MRR | Top-1 |
|---|---:|---:|
| `candidate_observability` | 0.9097 | 0.8413 |
| `two_stage_balanced_layout_v1` | 0.8849 | 0.8133 |
| `two_stage_balanced_layout_v2` | 0.8751 | 0.7853 |
| `two_stage_generalization_balanced_layout_v3` | 0.8451 | 0.7211 |
| `two_stage_generalization_balanced_layout_v3_1` | 0.8285 | 0.7106 |
| `two_stage_generalization_balanced_layout_v3_1a` | 0.8584 | 0.7561 |
| `degree` | 0.7933 | 0.6721 |

解释：

1. `v1` 已经不是简单启发式，而是能稳定超过 `degree` 的正式方法。
2. `v2` 成功引入更多约束，但第一轮没有超过 `v1`。
3. `v3` 方向更偏全网与泛化，但第一版在当前任务上牺牲过多。
4. `v3_1` 没有把 `scenario split` 做得更高，说明候选侧拉回后，分布内性能也没有自动恢复。
5. `v3_1a` 把 overlap 从 14 收回到 13 后，`scenario split` 反而回升到 `MRR=0.8584 / Top-1=0.7561`，说明适度收紧 overlap 并不会自动伤害分布内表现。

## 8. `node_holdout` 下的主方法结果

结果表：

- [ch5_two_stage_framework_node_holdout_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_two_stage_framework_node_holdout_compare_N25_seed42.csv)
- [ch5_two_stage_framework_split_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_two_stage_framework_split_compare_N25_seed42.csv)
- [ch5_strengthened_main_method_v3_compare_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_strengthened_main_method_v3_compare_N25_seed42.csv)

核心结果：

| 方法 | MRR | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| `degree` | 0.3655 | 0.1545 | 0.5049 | 0.7249 |
| `candidate_observability` | 0.2936 | 0.0987 | 0.3873 | 0.5456 |
| `two_stage_balanced_layout_v1` | 0.2489 | 0.0543 | 0.3165 | 0.4778 |
| `two_stage_balanced_layout_v2` | 0.2843 | 0.0301 | 0.4748 | 0.6669 |
| `two_stage_generalization_balanced_layout_v3` | 0.1972 | 0.0000 | 0.3022 | 0.4288 |
| `two_stage_generalization_balanced_layout_v3_1` | 0.2446 | 0.0452 | 0.3188 | 0.5147 |
| `two_stage_generalization_balanced_layout_v3_1a` | 0.2045 | 0.0317 | 0.2720 | 0.3708 |

解释：

1. `v1` 在分布内表现更强，但未见节点泛化不够好。
2. `v2` 没把 `Top-1` 提起来，但提升了 `MRR / Top-3 / Top-5`。
3. `v3` 说明一味加全网约束会损伤定位锐度。
4. `v3_1` 把 `v3` 从过度正则化状态拉回来，`node_holdout` 下的 `MRR / Top-1 / Top-3 / Top-5` 都明显好于 `v3`，但仍未超过 `v2` 或 `degree`。
5. `v3_1a` 的 `node_holdout` 又掉回去了，说明“进一步压低 overlap”虽然能修复一部分 `scenario split`，但未必能保住未见节点泛化。

## 9. 当前阶段结论

这条主方法线目前能得出的更稳结论是：

1. 两阶段框架这条路线是成立的。
2. `v1` 是当前最稳的正式框架版。
3. `v2` 更像约束增强版，当前优势主要体现在排序深度而不是 Top-1。
4. `v3` 证明“更多全网约束”本身不是答案，关键是候选判别性和全网支撑之间的平衡。
5. `v3_1` 证明折中回调是有效的，但还没有形成新的最优版本。
6. `v3_1a` 说明 `v3_1` 附近确实存在“分布内性能”和“未见节点泛化”之间的细微拉扯，当前还没有找到同时压住两边的更优点。

## 10. 当前最值得继续的方向

主方法下一轮更应该做的是：

1. 保留两阶段框架主线
2. 保留 moderate backbone / overlap regularization
3. 恢复更强的 candidate-side discriminative term
4. 继续参考 `v3_1` 的折中思路，而不是回到 `v3` 的重约束配置
5. 如果再试下一轮，优先在 `v3_1` 和 `v3_1a` 之间做更小步长调节，而不是继续放大改动幅度
