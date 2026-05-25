# script2_new 实验结果记录（B 路线）

每步改动后跑完 `train/train.py`，按下面格式追加一条记录，便于和上一版对比。

**B 路线设定**：候选集 C = 缺陷节点集（或覆盖全部缺陷节点），按 scenario 划分；不评估「未见过缺陷节点」泛化。

---

## 记录格式

```
实验名称：XXX | 日期：YYYY-MM-DD | 改动：XXX | 随机种子：XX

| 指标 | 本次 | 上一版 | 变化 |
|------|------|--------|------|
| 整体MRR | | | |
| 整体Top-1 | | | |
| 整体Top-3 | | | |
| I/E/P类Top-1 | | | |

预期是否符合：是/否
分析：XXX
```

---

## 实验 1：时间门控 + 相对残差（基线）

实验名称：时间门控数据 + residual_rel 特征 | 日期：2026-03-08 | 改动：时间门控（start_hour/duration_h）；候选集限制；27 特征 | 随机种子：42

| 指标 | 本次 | 上一版 | 变化 |
|------|------|--------|------|
| 整体MRR | 0.8998 | — | — |
| 整体Top-1 | 0.8237 | — | — |
| 整体Top-3 | 0.9733 | — | — |
| I/E/P类Top-1 | I=0.8726, E=0.7537, P=0.8165 | — | — |

预期是否符合：是
分析：B 路线下测试集缺陷节点均在训练集出现，指标反映「已知候选内定位」效果。后续可对比：全程注入、候选集/监测集/缺陷矩阵或 config 调参。

---

## 实验矩阵总览

所有实验**控制项**尽量固定：模型结构、lr/weight_decay/dropout/patience、sequence_length/window_stride/overlap_threshold、同一 candidate/monitor 文件（除非该实验就是改它们）。一次只改一个变量，改前记 baseline，改后对比。

| 层面 | 变量 | 水平/说明 | 备注 |
|------|------|-----------|------|
| **数据** | 场景数 \(N_{scen}\) | 100 / 300 / … | 缺陷矩阵行数或 max_scenarios |
| **数据** | 缺陷节点覆盖 | Partial（部分候选当缺陷）/ Full（全部候选当缺陷） | 缺陷矩阵中 defect_node 的覆盖范围 |
| **数据** | 时间门控 | on（start_hour/duration_h）/ off（全程注入） | extract_timeseries --time-gated、数据子目录 |
| **数据** | 监测节点范围 | monitor_only / full_nodes | 提取时序时的节点集合 |
| **划分** | 划分方式 | 按 scenario_id / 按 defect_node_id（node-holdout） | 默认 scenario；node-holdout 用于诊断泛化 |
| **模型** | 结构/方向 | GRU+GCN 隐层、层数、自环、A vs A^T | 一次只改一项 |
| **模型** | 空间编码器 | gru_gcn / hydraulic_inverse / **gru_only** / **time_mean_linear** | gru_only=无图；time_mean_linear=无GRU无图，最简基线 |
| **模型** | 输出空间 | 候选集限制 / 全图 | 当前为候选集限制 |
| **训练参数** | lr / dropout / weight_decay | 0.0003、0.2、5e-4 等 | 与当前 config 对齐为基线 |
| **训练参数** | patience / sigma / E 类过采样 / loc_ratio_clip | 早停、软标签、样本平衡 | 见 config |
| **策略** | 监测点策略 | degree / betweenness / downstream / observability | monitor_nodes_* 文件 |
| **策略** | 监测点数 N | 25 / 50 等 | 若做 chapter 对比 |

---

## 数据层面：场景数 × 缺陷覆盖（消融 D1–D4）

用于回答：**效果变好主要来自「增加场景」还是「缺陷矩阵覆盖全部候选节点」**。

| 实验ID | 场景数 | 缺陷覆盖 | 预期对比 |
|--------|--------|----------|----------|
| D1 | 100 | Partial | 旧版近似基线 |
| D2 | 300 | Partial | vs D1 → 纯场景数效应 |
| D3 | 100 | Full | vs D1 → 纯覆盖效应 |
| D4 | 300 | Full | 当前配置；vs D2/D3 看是否叠加 |

**操作**：固定时间门控、同一 candidate/monitor、同一模型与训练参数；只改缺陷矩阵（场景数 + 覆盖哪些节点）并重跑 extract → residual → train。记录格式同上，实验名称建议带 `D1`/`D2`/`D3`/`D4`。

---

## 模型对比：gru_gcn vs hydraulic_inverse（水力逆向注意力）

**方向**：用图路径物理特征（最短距离、管道长度、高程差、流向）做「候选→传感器」逆向注意力，替代 2 层 GCN，时间编码器 GRU 不变。

**对比实验**：同一数据（如 `--subdir time_gated`）、同一评估、只改 `--model-type`。
- Baseline：`gru_gcn`（当前默认）
- 新模型：`hydraulic_inverse`（需先跑 `python prep/build_graph_features.py` 生成 `input_1/graph_path_features.npz`）

**可执行命令**（在 PowerShell 中，先 `cd` 到 script2_new；若本机 `python` 未装 torch，改用 `D:\conda3\envs\swmm_gpu\python.exe`）：

```powershell
cd E:\11.16\script2_new

# 1）预计算图路径特征（只需跑一次）
python prep/build_graph_features.py

# 2）Baseline：GRU+GCN（跑完后会写 last_run_metrics_gru_gcn_time_gated.json）
python train/train.py --model-type gru_gcn --subdir time_gated --run-name gru_gcn_time_gated

# 3）水力逆向注意力（跑完后会写 last_run_metrics_hydraulic_inverse_time_gated.json）
python train/train.py --model-type hydraulic_inverse --subdir time_gated --run-name hydraulic_inverse_time_gated

# 4）自动对比两次结果（读上面两个 json，打表；加 --append-log 会追加到本文件下表）
python scripts/compare_model_runs.py gru_gcn_time_gated hydraulic_inverse_time_gated --append-log
```

**运行后怎么看对比？**
- **终端**：训练结束会在控制台打印「测试集最终评估」（MRR、Top-1、见过/未见过、I/E/P）。
- **JSON**：每次带 `--run-name` 跑完会生成 `outputs/reports/last_run_metrics_{run_name}.json`，内含 mrr、top1、top3、seen_nodes、unseen_nodes、by_type_top1 等。
- **自动对比**：执行上面第 4 条命令会**自动读取**这两份 JSON，在终端打印对比表；加 `--append-log` 会把该表**追加到本 STEP_RESULTS_LOG.md**，无需手抄日志。

| 实验名称 | 日期 | 模型类型 | 种子 | 整体MRR | 整体Top-1 | 见过Top-1 | 未见过Top-1 | I/E/P Top-1 |
|----------|------|----------|------|---------|-----------|-----------|-------------|-------------|
| gru_gcn 基线 | 待填 | gru_gcn | 42 | | | | | |
| hydraulic_inverse | 待填 | hydraulic_inverse | 42 | | | | | |

预期：水力逆向注意力利用物理结构，在未见过节点或弱信号场景下可能更稳；若整体 MRR 相当或略优则保留为可选模型。

---

## 简单基线：任务是否真的需要复杂模型？

**目的**：看“只时序、无图”甚至“只时间均值+线性”能否接近当前效果，判断任务难度与模型复杂度的必要性。

| 模型 | 含义 | 命令 |
|------|------|------|
| **time_mean_linear** | 对时间维取 mean，再一层线性 → 节点得分。无 GRU、无图。 | `--model-type time_mean_linear --run-name time_mean_linear_time_gated` |
| **gru_only** | GRU 做时间编码，单层线性投影到 hidden，无 GCN/无图。 | `--model-type gru_only --run-name gru_only_time_gated` |

**解读**：
- 若 **time_mean_linear** 已经 MRR 很高（如 >0.85）：说明任务可能较“容易”，当前特征+候选集下简单模型就够，复杂模型收益有限。
- 若 **gru_only** 接近 **gru_gcn**：说明**图（GCN）** 带来的增益有限，主要增益来自时序或水力注意力。
- 若两者都明显低于 gru_gcn/hydraulic_inverse：说明**时序+空间（图或路径注意力）** 都有必要，当前复杂度是合理的。

跑完后可用同一对比脚本看差异，例如：
`python scripts/compare_model_runs.py time_mean_linear_time_gated gru_gcn_time_gated`

---

## Node-holdout 评估（未见过缺陷节点泛化）

**目的**：测试集仅包含「训练/验证中从未出现过的缺陷节点」，用于拉开模型差距（gru_only 预期明显弱于 hydraulic_inverse）。

**划分逻辑**：从 50 个候选节点中随机留出 `n_holdout_nodes` 个（默认 10）作为 holdout；所有缺陷发生在这些节点上的**场景**全部进入测试集；其余场景按 train_ratio/val_ratio 划分训练/验证。训练和验证中不会出现 holdout 节点作为缺陷节点。

**命令示例**（与 scenario 划分唯一区别是 `--split-mode node_holdout`）：

```powershell
cd E:\11.16\script2_new

# 同一数据、同一种子，只改划分方式
python train/train.py --split-mode node_holdout --n-holdout-nodes 10 --subdir time_gated --run-name gru_only_node_holdout
python train/train.py --split-mode node_holdout --n-holdout-nodes 10 --subdir time_gated --model-type hydraulic_inverse --run-name hydraulic_inverse_node_holdout
```

**预期**：在 node-holdout 下，测试集 = 100% 未见过节点。gru_only 的测试 Top-1 可能明显下降；hydraulic_inverse 利用水力路径结构，有望保持更高 Top-1。对比两者即可验证「复杂模型在泛化上的价值」。

**记录表**（跑完后填）：

| 实验名称 | 划分 | 模型 | 种子 | 整体MRR | 整体Top-1 | 测试集=holdout节点 |
|----------|------|------|------|---------|-----------|---------------------|
| gru_only node_holdout | node_holdout | gru_only | 42 | | | 全部未见过 |
| hydraulic_inverse node_holdout | node_holdout | hydraulic_inverse | 42 | | | 全部未见过 |

---

### D1（100 场景 + Partial 覆盖）
（待跑，占位）

### D2（300 场景 + Partial 覆盖）
（待跑，占位）

### D3（100 场景 + Full 覆盖）
（待跑，占位）

### D4（300 场景 + Full 覆盖）
与「实验 1」一致，已记录在上方。

---

## 数据层面：时间门控 on/off

| 实验ID | 时间门控 | 数据子目录 / 说明 |
|--------|----------|-------------------|
| T0 | off（全程注入） | `full_injection`，需单独 extract+residual |
| T1 | on | `time_gated` 或根目录（当前实验 1） |

（待跑 T0 与 T1 同模型同参数对比时，在此追加记录。）

---

## 划分层面

- **默认**：按 `scenario_id` 划分（当前）。
- **可选诊断**：按 `defect_node_id` 做 node-holdout，检验跨节点泛化（若实现，在此追加一节记录）。

---

## 模型层面

每次只改一项，其余与实验 1 一致，记录同上格式。

- 隐层维度（time_hidden_dim / spatial_hidden_dim）
- GCN 层数、自环、A vs A^T
- 输入节点范围（monitor_only vs full）

（具体实验跑完后在下方追加，如「实验 M1：spatial_hidden_dim 256→512」等。）

---

## 训练参数层面

- lr、weight_decay、dropout、patience
- soft_label_sigma、dataset_loc_ratio_clip、dataset_e_class_oversample_ratio

（具体实验跑完后在下方追加，如「实验 P1：lr 0.0003→0.0001」等。）

---

## 策略层面（监测点 / 候选集）

- 监测点策略：degree / betweenness / downstream / observability
- 监测点数 N、候选集生成策略

（具体实验跑完后在下方追加。）

---

## 后续实验

每新增一步（数据/划分/模型/参数/策略任一维度），在对应层面下方追加一节，用统一记录格式（实验名称、日期、改动、种子、指标表、预期是否符合、分析）。


## 第一章批量实验（run_chapter1.py）

| 策略 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |
|------|--------|--------|----------|----------|
| degree | 0.8796 | 0.0341 | 0.8003 | 0.0670 |
| betweenness | 0.8921 | 0.0714 | 0.8105 | 0.1217 |
| downstream | 0.9097 | 0.0417 | 0.8344 | 0.0833 |
| random | 0.8406 | 0.0604 | 0.7690 | 0.0883 |
| full | 0.6351 | 0.0509 | 0.5451 | 0.0490 |
| observability | 0.8768 | 0.0442 | 0.7775 | 0.0754 |


## 第一章批量实验（run_chapter1.py）

| 策略 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |
|------|--------|--------|----------|----------|
| degree | 0.8796 | 0.0341 | 0.8003 | 0.0670 |
| betweenness | 0.8921 | 0.0714 | 0.8105 | 0.1217 |
| downstream | 0.9097 | 0.0417 | 0.8344 | 0.0833 |
| random | 0.8406 | 0.0604 | 0.7690 | 0.0883 |
| full | 0.6351 | 0.0509 | 0.5451 | 0.0490 |
| observability | 0.8768 | 0.0442 | 0.7775 | 0.0754 |


## 第一章批量实验（run_chapter1.py）

| 策略 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |
|------|--------|--------|----------|----------|
| degree | 0.8796 | 0.0341 | 0.8003 | 0.0670 |
| betweenness | 0.8921 | 0.0714 | 0.8105 | 0.1217 |
| downstream | 0.9097 | 0.0417 | 0.8344 | 0.0833 |
| random | 0.8406 | 0.0604 | 0.7690 | 0.0883 |
| full | 0.6351 | 0.0509 | 0.5451 | 0.0490 |
| observability | 0.8768 | 0.0442 | 0.7775 | 0.0754 |


## 第一章批量实验（run_chapter1.py）

| 策略 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |
|------|--------|--------|----------|----------|
| degree | 0.8796 | 0.0341 | 0.8003 | 0.0670 |
| betweenness | 0.8921 | 0.0714 | 0.8105 | 0.1217 |
| downstream | 0.9097 | 0.0417 | 0.8344 | 0.0833 |
| random | 0.8406 | 0.0604 | 0.7690 | 0.0883 |
| full | 0.6351 | 0.0509 | 0.5451 | 0.0490 |
| observability | 0.8768 | 0.0442 | 0.7775 | 0.0754 |


## 第一章批量实验（run_chapter1.py）

| 策略 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |
|------|--------|--------|----------|----------|
| degree | 0.8796 | 0.0341 | 0.8003 | 0.0670 |
| betweenness | 0.8921 | 0.0714 | 0.8105 | 0.1217 |
| downstream | 0.9097 | 0.0417 | 0.8344 | 0.0833 |
| random | 0.8406 | 0.0604 | 0.7690 | 0.0883 |
| full | 0.6351 | 0.0509 | 0.5451 | 0.0490 |
| observability | 0.8768 | 0.0442 | 0.7775 | 0.0754 |

### 对比结果（自动追加）
日期: 2026-03-10 19:53

| 指标 | 基线 | 当前 | 变化 |
|------|------|------|------|
| 整体 MRR | 0.9203 | 0.9898 | +0.0695 |
| 整体 Top-1 | 0.8609 | 0.9807 | +0.1198 |
| 整体 Top-3 | 0.9821 | 0.9986 | +0.0165 |
| 整体 Top-5 | 0.9931 | 1.0000 | +0.0069 |
| 见过节点 Top-1 | 0.8754 | 0.9804 | +0.1050 |
| 见过节点 MRR | 0.9274 | 0.9896 | +0.0622 |
| 未见过节点 Top-1 | 0.0000 | 1.0000 | +1.0000 |
| 未见过节点 MRR | 0.5000 | 1.0000 | +0.5000 |
|   I 类 Top-1 | 0.8905 | 0.9793 | +0.0888 |
|   E 类 Top-1 | 0.8853 | 0.9725 | +0.0872 |
|   P 类 Top-1 | 0.7706 | 0.9941 | +0.2235 |

### 对比结果（自动追加）
日期: 2026-03-11 09:20

| 指标 | 基线 | 当前 | 变化 |
|------|------|------|------|
| 整体 MRR | 0.8763 | 0.9299 | +0.0536 |
| 整体 Top-1 | 0.8255 | 0.8989 | +0.0733 |
| 整体 Top-3 | 0.9178 | 0.9444 | +0.0265 |
| 整体 Top-5 | 0.9520 | 0.9810 | +0.0291 |
| 见过节点 Top-1 | 0.0000 | 0.0000 | +0.0000 |
| 见过节点 MRR | 0.0000 | 0.0000 | +0.0000 |
| 未见过节点 Top-1 | 0.8255 | 0.8989 | +0.0733 |
| 未见过节点 MRR | 0.8763 | 0.9299 | +0.0536 |
|   I 类 Top-1 | 0.9170 | 0.9509 | +0.0340 |
|   E 类 Top-1 | 0.7670 | 0.8586 | +0.0916 |
|   P 类 Top-1 | 0.8125 | 0.9097 | +0.0972 |

### 对比结果（多 run，自动追加）
日期: 2026-03-11 09:42

| 指标 | gru_gcn_node_holdout | gru_only_node_holdout | hydraulic_inverse_node_holdout |
|------|------|------|------|
| 整体 MRR | 0.5916 | 0.8763 | 0.9299 |
| 整体 Top-1 | 0.3489 | 0.8255 | 0.8989 |
| 整体 Top-3 | 0.8116 | 0.9178 | 0.9444 |
| 整体 Top-5 | 0.8369 | 0.9520 | 0.9810 |
| 见过节点 Top-1 | 0.0000 | 0.0000 | 0.0000 |
| 见过节点 MRR | 0.0000 | 0.0000 | 0.0000 |
| 未见过节点 Top-1 | 0.3489 | 0.8255 | 0.8989 |
| 未见过节点 MRR | 0.5916 | 0.8763 | 0.9299 |
|   I 类 Top-1 | 0.2075 | 0.9170 | 0.9509 |
|   E 类 Top-1 | 0.4843 | 0.7670 | 0.8586 |
|   P 类 Top-1 | 0.2500 | 0.8125 | 0.9097 |

---

## 实验：删除 flooding 系列特征（27 -> 24）

实验名称：删除 flooding / flooding_residual / flooding_residual_rel | 日期：2026-03-18 | 改动：仅删除 flooding 系列 3 个特征，其余 24 个特征保持不变 | 随机种子：42

| 指标 | 本次 | 上一版 | 变化 |
|------|------|--------|------|
| 整体MRR | 0.9297 | 0.9174 | +0.0123 |
| 整体Top-1 | 0.8788 | 0.8609 | +0.0179 |
| 整体Top-3 | 0.9890 | 0.9780 | +0.0110 |
| I/E/P类Top-1 | I=0.9053, E=0.8807, P=0.8235 | I=0.8669, E=0.8899, P=0.8118 | I=+0.0384, E=-0.0092, P=+0.0117 |

预期是否符合：是
分析：本次只删除了全局零信息的 flooding 系列，整体指标小幅提升，说明这 3 个特征确实属于噪声或无效输入。E 类 Top-1 略有下降，但幅度很小；整体看删除 flooding 系列是安全的。下一步可继续按单变量原则测试删除 head 系列（与 depth 冗余）或 volume 系列（弱信号）。

### 对比结果（多 run，自动追加）
日期: 2026-03-11 09:42

| 指标 | gru_gcn_node_holdout | gru_only_node_holdout | hydraulic_inverse_node_holdout |
|------|------|------|------|
| 整体 MRR | 0.5916 | 0.8763 | 0.9299 |
| 整体 Top-1 | 0.3489 | 0.8255 | 0.8989 |
| 整体 Top-3 | 0.8116 | 0.9178 | 0.9444 |
| 整体 Top-5 | 0.8369 | 0.9520 | 0.9810 |
| 见过节点 Top-1 | 0.0000 | 0.0000 | 0.0000 |
| 见过节点 MRR | 0.0000 | 0.0000 | 0.0000 |
| 未见过节点 Top-1 | 0.3489 | 0.8255 | 0.8989 |
| 未见过节点 MRR | 0.5916 | 0.8763 | 0.9299 |
|   I 类 Top-1 | 0.2075 | 0.9170 | 0.9509 |
|   E 类 Top-1 | 0.4843 | 0.7670 | 0.8586 |
|   P 类 Top-1 | 0.2500 | 0.8125 | 0.9097 |
> Warning
>
> This log aggregates many historical experiments across old and new pipeline stages.
> Entries produced before the monitor-node leakage fix, before the clean data rebuild, or before the current provenance alignment must not be used directly as formal thesis evidence.
> Keep this file as history only; current formal results should be read from the clean summary files in `process_diagnosis_revision_20260321`.
