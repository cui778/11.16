# 第一章：已做实验梳理与系统安排

日期：2026-03-10

---

## 一、变量角色与第一章控制原则

### 1.1 各变量在章节中的定位

| 变量 | 第一章 | 第二章 | 说明 |
|------|--------|--------|------|
| **监测点集 S** | **只做简单对比**（degree / betweenness / downstream / observability / random / full） | **主内容**：在最优规则基础上优化布置 | 第一章不深挖，结论给第二章当起点 |
| **候选节点集 C** | **固定，不改**（对比矩阵内一致） | 同左 | 是否 50 个要单独定：可做小实验（如 30/50/70）选一个数，定下来后第一章全程用该数 |
| **缺陷矩阵** | **策略要明确**：生成方式多种（I:E:P、强度、时间模式、场景数），第一章建议**固定一种**做主实验；若对比「不同生成策略」可单列一组 | 同左或沿用 | 见下 1.2 |
| **时序 / 残差** | 是否用残差、**是否多 baseline** 都要先定 | — | 见下 1.3 |
| **序列长度、训练参数** | **会调**：T、stride、lr、patience 等可做敏感性或网格搜索，定一套主配置 | — | 建议先固定一套，再用 D 组微调 |
| **划分** | **不改**，按 scenario 70/15/15 | — | 默认即可 |

### 1.2 缺陷矩阵：多种策略怎么处理？

缺陷矩阵生成本身有多维策略，例如：

- **场景数**：100 / 300（B1 已做，定 300）
- **类型比例**：I:E:P = 40:40:20 或其它
- **强度**：intensity_choices、范围
- **时间模式**：fixed / diverse / realistic
- **覆盖**：50 节点全覆盖 vs 部分节点（B2 已做）

**建议**：

- **第一章主实验（A 组策略对比）**：**固定一种**缺陷矩阵（如当前 defect_matrix_diverse.csv：300 场景、I:E:P=40:40:20、diverse 时间、强度 30/40/50）。所有策略在同一矩阵下比，否则不可比。
- **若要做「缺陷矩阵生成策略」的对比**：单独设一组实验（如 B0 或 E 组），变量为「另一种矩阵」（如 100 场景 / 另一种 I:E:P / realistic 时间），其余（监测策略、模型、划分）固定。这样不会和 A 组混在一起。

结论：**对比矩阵里缺陷矩阵不能混用**；先定一个主矩阵，再决定是否加一组「不同生成策略」的消融。

### 1.3 时序：要不要残差？要不要多 baseline？

- **要不要残差**：B3 已做，残差 vs 原始差距不大，可**固定用残差**（更稳定）或固定用原始，第一章统一即可。
- **要不要多 baseline**：管网动态变化 → 理想情况是多个无缺陷工况（不同季节/天气）各算一个 baseline，残差 = 当前仿真 − 对应 baseline。当前只有**单 baseline**。  
  - 第一章：**单 baseline 即可**，在文中写明「本研究在单一 baseline 下进行；多 baseline 作为局限与后续工作」。  
  - 若有余力：可做 1～2 个额外 baseline（如另一种降雨曲线），算两套残差，看模型是否对 baseline 敏感（探索性），不必进主表。

### 1.4 当前「主配置」建议（固定后不再在对比中改）

| 变量 | 建议固定值 | 备注 |
|------|-------------|------|
| 候选节点数 | **50**（或先做 30/50/70 小实验再定） | 定下来后全章一致 |
| 缺陷矩阵 | defect_matrix_diverse.csv（300 场景，I:E:P=40:40:20，diverse） | 主实验唯一矩阵 |
| 时序特征 | 残差 27 维（或按 B3 结论二选一固定） | 不做多 baseline 则单 baseline 残差 |
| 序列长度 / 步长 | T=36, stride=6 | 可后续 D 组调 |
| 训练参数 | lr=0.0003, patience=10, batch=32 | 可后续 D 组调 |
| 划分 | scenario 70/15/15 | 不改 |
| 模型（主表） | GRU+GCN | 模型对比单独做，不混入 A 组 |

---

## 二、实验顺序与先控制谁（推荐）

原则：**先固定「数据与模型」再比「监测策略」**；再补「设计选择」的验证；最后可选调参/消融。

| 顺序 | 做什么 | 控制住什么 | 目的 |
|------|--------|------------|------|
| **0** | 定候选节点数（若未定） | 可选：30/50/70 各跑一次，看 MRR 与稳定性，选一个 | 全章统一 C 的大小 |
| **1** | 定缺陷矩阵、特征、baseline | 一种矩阵、残差或原始二选一、单 baseline | 确保 A 组所有策略用同一套数据 |
| **2** | 定模型与超参 | GRU+GCN，T=36，lr=0.0003 等 | 避免策略对比被模型/超参干扰 |
| **3** | **A 组：监测策略对比** | 以上全固定，只改 monitor_nodes_* + 对应 subdir | 第一章主表：哪种布置规则更好 |
| **4** | B1/B2/B3（设计验证） | 已做：场景数、覆盖度、残差 vs 原始 | 支撑「为什么 300、为什么这样设计」 |
| **5** | 可选：序列/训练参数（D 组） | 固定策略（如 downstream） | 找更稳的 T、lr、patience |
| **6** | 可选：C 组模型消融、缺陷矩阵策略对比 | 固定策略与数据 | 加分项，非必须 |

**一句话**：先控制**候选集大小、缺陷矩阵、特征与 baseline、模型与超参、划分**，再跑 **A 组**；B 组验证设计；D/C 可选。

---

## 三、已完成的实验（按组与设置逐项列出）

### 3.1 A 组：监测节点策略对比（主实验）✅

**目的**：比较不同传感器布置规则的定位性能。

| 实验编号 | 策略 | 监测节点文件 | 数据子目录 | 种子 | 设置要点 |
|----------|------|--------------|------------|------|----------|
| A0 | full | monitor_nodes_full_N128.json | time_gated_full | 42/7/123 | 输入 128 节点，作对照 |
| A1 | degree | monitor_nodes_degree_N25.json | time_gated | 42/7/123 | 度中心性 Top-25 |
| A2 | betweenness | monitor_nodes_betweenness_N25.json | time_gated_betweenness | 42/7/123 | 介数中心性 Top-25 |
| A3 | downstream | monitor_nodes_downstream_N25.json | time_gated_downstream | 42/7/123 | 下游覆盖 Top-25 |
| A4 | observability | monitor_nodes_observability_N25.json | time_gated_observability | 42/7/123 | 可观测性贪心 Top-25 |
| A5 | random | monitor_nodes_random_N25.json | time_gated_random | 42/7/123 | 随机 25，下限基线 |

**模型**：GRU+GCN，按 scenario 划分。

**结果摘要**：downstream 最优（MRR≈0.91），full 最差（0.64）；前四名策略差距约 3 个百分点，在标准差内。

---

### 3.2 B1：场景数 100 vs 300 ✅

| 实验 | 场景数 | 策略 | 数据子目录 | 种子 |
|------|--------|------|------------|------|
| B1_100scen | 100 | observability | （对应 100 场景的 parquet） | 42/7/123 |
| B1_300scen | 300 | observability | time_gated_observability | 42/7/123 |

**结论**：300 场景远优于 100（MRR 0.42→0.88），支撑「300 场景」设计。

---

### 3.3 B2：缺陷矩阵覆盖度（全覆盖 vs 部分覆盖）✅

| 实验 | 缺陷矩阵 | 覆盖 | 策略 | 种子 |
|------|----------|------|------|------|
| B2_full_coverage | b2_full_coverage.csv | 50 节点全覆盖 | observability | 42/7/123 |
| B2_partial_k25 | b2_partial_k25.csv | 仅 25 节点 | observability | 42/7/123 |

**结论**：固定 300 场景下，部分覆盖略优于全覆盖（样本更集中）；B2 与 A 组缺陷矩阵不同，结果低于 A 组 observability。

---

### 3.4 B3：特征 残差 vs 原始 ✅

| 实验 | 特征 | 命令/设置 | 策略 | 种子 |
|------|------|-----------|------|------|
| B3_residual | 27 维（含残差） | 默认 node_timeseries_with_residuals.parquet | observability | 42/7/123 |
| B3_raw | ~11 维（无残差列） | --raw-features | observability | 42/7/123 |

**结论**：原始特征略优但波动更大；两者差距不显著，可表述为「残差与原始相当，残差更稳定」。

---

### 3.5 模型对比（scenario 划分）✅

**数据**：time_gated，scenario 划分，seed=42（单种子）。

| 实验 | 模型 | run-name | 整体 MRR | 整体 Top-1 |
|------|------|----------|----------|------------|
| — | gru_gcn | gru_gcn_time_gated | 0.920 | 0.861 |
| — | gru_only | gru_only_time_gated | 0.979 | 0.963 |
| — | hydraulic_inverse | hydraulic_inverse_time_gated | 0.990 | 0.981 |

**结论**：scenario 划分下 gru_only 已很强，GCN 未带来收益；hydraulic_inverse 略优于 gru_only（约 +2% Top-1）。

---

### 3.6 模型对比（node-holdout 划分）✅

**数据**：time_gated，**split_mode=node_holdout, n_holdout_nodes=10**，seed=42。

| 实验 | 模型 | run-name | 整体 MRR | 整体 Top-1 |
|------|------|----------|----------|------------|
| — | gru_gcn | gru_gcn_node_holdout | 0.592 | 0.349 |
| — | gru_only | gru_only_node_holdout | 0.876 | 0.826 |
| — | hydraulic_inverse | hydraulic_inverse_node_holdout | 0.930 | 0.899 |

**结论**：在未见过节点上，GCN 严重拖后腿；hydraulic_inverse 明显优于 gru_only（约 +7% Top-1），支撑「水力结构先验有利于泛化」。

---

### 3.7 简单基线（可选）✅

| 模型 | 含义 | run-name |
|------|------|----------|
| time_mean_linear | 时间均值 + 线性，无 GRU 无图 | time_mean_linear_time_gated |
| gru_only | GRU，无 GCN | gru_only_time_gated |

已跑过，用于判断「任务是否需复杂模型」；结论见上。

---

## 四、未做或未系统完成的实验

| 组 | 内容 | 状态 |
|----|------|------|
| C | 模型消融（GCN 层数/方向/类型头） | 未做 |
| D | 训练参数搜索（lr/dropout/weight_decay） | 未做 |
| A 组 | 5 种子补充（加强策略间显著性） | 可选 |
| 时间门控 T0/T1 | 全程注入 vs 时间门控 | 未做 |
| D1–D4 | 场景数×覆盖度四格 | 部分被 B1/B2 覆盖，未单独成体系 |

---

## 五、第一章系统实验安排（建议）

在「已做 A/B1/B2/B3 + 模型/node-holdout」基础上，按论文叙事做**收尾与补全**，建议顺序如下。

### 5.1 必做（支撑主结论）

1. **锁定主模型与划分**
   - 第一章主表：**scenario 划分 + GRU+GCN**（与 research_plan 一致），A/B 组已满足。
   - 若论文单独有一节「泛化与未见过节点」：采用 **node-holdout + hydraulic_inverse vs gru_gcn** 的对比（已做，补 3 种子更稳）。

2. **A 组收尾**
   - 确认 A1–A5 均有三种子记录（已有）。
   - 如需更强结论：对 downstream / degree / observability 补 2 种子（如 99、1234），报告 mean±std，看 downstream 是否稳定领先。

3. **B 组收尾**
   - B1/B2/B3 已有结论，写入论文「实验设置与消融」即可。
   - 可选：B3 补 2 种子确认 raw vs residual 稳定性。

### 5.2 可选（加分项）

4. **C 组模型消融**（时间允许）
   - C2：GCN 方向 A^T vs A（固定 A4 或 downstream）。
   - C3：有无类型辅助头（lambda_type=0 vs 0.1）。
   - 注：当前现象是「GCN 在 node-holdout 下变差」，C 组仍可做 scenario 下的消融，说明「在见过节点上 GCN 的贡献有限」。

5. **D 组训练参数**
   - 用 downstream（或 A4）固定跑 D1/D2/D3 单变量，找较优 lr/dropout/weight_decay，再用三种子确认。

6. **时间门控 T0 vs T1**
   - 若有 full_injection 数据：同模型同参数对比 time_gated vs full_injection，说明时间门控的价值。

### 5.3 不必要再做

- 不再为「拉大模型差距」增加新设定；node-holdout 已给出清晰结论。
- 缺陷矩阵不必为第一章再增多种「定义」；一种主矩阵 + B2 覆盖度已够。

---

## 六、每步实验设置速查（复现用）

| 步骤 | 命令/设置要点 |
|------|----------------|
| 数据 | 对应策略的 monitor_nodes_*.json + time_gated_* 子目录的 parquet |
| 划分 | 默认 scenario；node-holdout 时加 --split-mode node_holdout --n-holdout-nodes 10 |
| 模型 | --model-type gru_gcn / gru_only / hydraulic_inverse |
| 策略 | --subdir time_gated / time_gated_downstream / …（由 run_chapter1 或手动指定） |
| 种子 | --seed 42（或 7、123）；三种子即跑 3 次取 mean±std |
| 对比 | python scripts/compare_model_runs.py run1 run2 [run3] [--append-log] |

---

## 七、与 research_plan_full 的对应关系

| research_plan 节 | 已做 | 待做/可选 |
|------------------|------|-----------|
| 4.2 控制变量 | 全部满足 | — |
| 4.3 实验组 A | A0–A5 三种子 ✅ | 可选：补种子 |
| 4.3 实验组 B | B1/B2/B3 ✅ | — |
| 4.3 实验组 C | — | C2/C3 可选 |
| 4.3 实验组 D | — | D1–D3 可选 |
| 4.4 执行顺序 | Step 0–5 实质已完成 | Step 6–7 可选 |

以上为已做实验的完整梳理与第一章系统安排；你可据此在 calendar 或任务列表里逐项打勾推进。
