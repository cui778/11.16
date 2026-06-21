# script2_new

## 1. 项目定位

`script2_new` 是学位论文第3章至第5章的实验代码与结果工作区，当前主线包括：

1. 第3章：基于 SWMM 的 I/E 缺陷多场景仿真数据生成；
2. 第4章：固定监测布局下的 I/E 缺陷时空诊断；
3. 第5章：固定诊断协议下的监测节点布局优化。

当前工作不再沿用早期“第一章实验重启”的章节命名。历史方案、旧运行入口和探索性口径统一保存在：

```text
docs/archive_root_legacy/
```

历史文件只用于追溯设计过程，不得直接作为当前正式实验入口。

## 2. 当前正式研究主线

### 2.1 第3章：数据生成

第3章负责构建缺陷场景、正常扰动场景和参考基线，并验证生成数据的时空响应合理性。

正式数据口径：

| 数据对象 | 当前正式来源 | 作用 |
|---|---|---|
| IE420 缺陷场景 | `time_gated_full_ie_v4_formal_conservative420_seed42` | I/E 缺陷仿真母数据 |
| normal20 | `normal_multibaseline_v2_seedset20` | 正常扰动与误报控制 |
| 正式组合数据 | `ie420_plus_normal20_v1` | 第4、5章训练与评估 |
| reference | 正式基准场景 | residual 对齐基线 |

正式缺陷矩阵必须来自 `formal_conservative420_seed42` 对应口径。不得自动切换到 `persistent`、`fulltime`、`legacy` 或 `seedset10` 数据。

详细说明：

- [chapter3_data_generation/README.md](chapter3_data_generation/README.md)
- [chapter3_data_generation/plans/CH3_PROTOCOL_FREEZE.md](chapter3_data_generation/plans/CH3_PROTOCOL_FREEZE.md)

### 2.2 第4章：固定布局下的诊断模型

第4章在固定 `degree_N25` 监测布局下研究场景报警、时间区间恢复和缺陷节点定位。

当前正式协议：

```text
dataset      = ie420_plus_normal20_v1
feature_set  = raw_plus_residual
model        = hydraulic_inverse_deepattn
layout       = degree_N25
lambda_loc   = 0.5
split        = scenario
seeds        = 7 / 42 / 123
```

核心评价包括：

- 报警识别：Scene F1；
- 误报控制：Normal FPR；
- 时间定位：Onset error、Active IoU；
- 空间定位：MRR、Top-1、Top-3；
- 事件级定位：Event Top-1、Event Top-3。

正式训练入口仍位于：

```text
scripts/train_privileged_teacher_student.py
```

详细说明：

- [chapter4_diagnosis_model/README.md](chapter4_diagnosis_model/README.md)

### 2.3 第5章：监测布局优化

第5章固定第4章的诊断模型、数据集、特征、划分和评价口径，仅改变监测节点集合及其观测掩膜，研究不同布局策略和预算条件下的定位性能。

N=25 主协议包含六类方法：

| 方法 | 主要依据 |
|---|---|
| Degree | 节点度中心性 |
| Betweenness | 介数中心性 |
| Cand-Obs | 缺陷节点邻近可观测性 |
| Two-stage v1 | 缺陷覆盖与结构约束 |
| Node-Feedback | 诊断反馈驱动的节点评分 |
| Embedding-Guided | 诊断表征驱动的布局选择 |

补充实验包括：

- N=5、10、15、20、25 预算变化；
- 覆盖率受控实验；
- I/E 类型分组；
- direct/near/far 结构分析；
- hard defect node 分析；
- 布局 Jaccard 相似度；
- 管网拓扑上的多预算布点分布。

详细说明：

- [chapter5_layout_optimization/README.md](chapter5_layout_optimization/README.md)
- [chapter5_layout_optimization/plans/CH5_PROTOCOL_FREEZE.md](chapter5_layout_optimization/plans/CH5_PROTOCOL_FREEZE.md)
- [chapter5_layout_optimization/scripts/README.md](chapter5_layout_optimization/scripts/README.md)

## 3. 目录结构

```text
script2_new/
├── chapter3_data_generation/        # 第3章协议、清单与历史探索归档
├── chapter4_diagnosis_model/        # 第4章正式实验与诊断分支
├── chapter5_layout_optimization/    # 第5章布局方法、实验与结果
├── prep/                            # SWMM 数据、图结构和特征构建
├── train/                           # 训练基础模块
├── models/                          # 模型定义
├── scripts/                         # 第4、5章共享训练与汇总入口
├── training_data_new/               # 正式及历史训练数据
├── input_1/                         # 图结构、节点集合和轻量输入
├── outputs/                         # checkpoint、指标与报告
├── experiments/                     # 实验清单和过程记录
├── docs/archive_root_legacy/        # 根目录历史方案与旧入口
├── config.py                        # 共享配置定义
└── README.md                        # 当前项目入口
```

论文正文、PPT 文稿和正式绘图位于相邻仓库：

```text
E:\11.16\thesis_writing_repo
```

其中图件按章节组织在：

```text
thesis_writing_repo/figures/ch3/
thesis_writing_repo/figures/ch4/
thesis_writing_repo/figures/ch5/
```

## 4. 正式实验约束

运行或汇总正式实验前必须检查：

1. 数据集是否为 `ie420_plus_normal20_v1`；
2. 缺陷母数据是否来自 `formal_conservative420_seed42`；
3. 是否误用了 `persistent`、`fulltime`、`legacy` 或 `seedset10` 数据；
4. 第4章是否固定 `degree_N25`；
5. 第5章是否只改变监测布局及其观测掩膜；
6. 特征是否为 `raw_plus_residual`；
7. `lambda_loc` 是否为 `0.5`；
8. 是否采用 scenario split；
9. 多 seed 结论是否来自 7、42、123；
10. 单 seed 预算实验是否明确标注其证据边界。

候选缺陷空间、真实缺陷节点和监测节点集合必须分别报告，不得把三者混为同一集合。

## 5. 配置与旧命名说明

`config.py` 是仍在使用的共享核心文件，不属于历史归档。由于代码由早期实验逐步演化，部分字段、checkpoint 前缀或脚本名称仍可能包含 `chapter1`、`ch1` 等历史命名。

判断实验是否属于当前正式主线时，应以以下内容为准：

1. 实际加载的数据目录；
2. 缺陷矩阵和 normal20 来源；
3. 实际模型、布局、特征和划分参数；
4. 对应章节的协议冻结文档；
5. 输出报告中的真实运行配置。

不得仅根据文件名中的旧章节编号判断实验口径。

## 6. 历史材料

2026年4月形成的 Chapter 1 重启方案、论文目录草案和旧运行入口已归档至：

```text
docs/archive_root_legacy/202604_chapter1_restart/
```

归档原因和可复用内容见：

- [docs/archive_root_legacy/202604_chapter1_restart/ARCHIVE_INDEX.md](docs/archive_root_legacy/202604_chapter1_restart/ARCHIVE_INDEX.md)

归档脚本默认不得直接运行。若需复现实验，应先核对数据路径、配置默认值和输出目录，避免覆盖当前正式结果。
