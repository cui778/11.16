# 项目背景与调试任务

## 项目简介

这是一个基于时空图卷积网络（STGCN）的城市排水管网缺陷定位系统。
- 用PySWMM对真实管网进行机理仿真，生成缺陷场景数据
- 缺陷类型：I类（渗入）、E类（渗漏）、P类（点源污染）
- 任务：给定128个节点的时序特征，定位50个候选节点中哪个节点发生了缺陷
- 评估指标：MRR、Top-1/3/5召回率

## 当前训练结果（问题所在）

最新一次训练结果：
- 测试集 MRR=0.2336，Top-1=0.1549，Top-3=0.2137
- **严重过拟合**：训练Loss从3.49降到1.62，验证Loss从4.16涨到10.32
- **模型坍缩**：预测时反复输出固定几个节点（70、120、40），忽略真实缺陷节点
- **场景间方差极大**：场景15 Top-1=1.0，场景59 Top-1=0.0，差异悬殊

## 已确认的核心问题清单

### 问题1（最高优先级）：GCN自环缺失 —— 架构Bug
**文件**：`anomaly_detection_model.py`，`forward()`方法，空间编码部分

当前代码：
```python
D = adj_norm.sum(dim=2, keepdim=True) + 1e-8
adj_norm = adj_norm / D
x_agg = torch.bmm(adj_norm, x_spatial)
```

问题：GCN只聚合邻居特征，完全丢弃节点自身特征。对于缺陷定位，缺陷节点自身残差才是最重要的信号。

**修复方案**：加自环 A_hat = A + I
```python
I = torch.eye(adj_norm.shape[-1], device=adj_norm.device).unsqueeze(0).expand_as(adj_norm)
adj_norm = adj_norm + I
D = adj_norm.sum(dim=2, keepdim=True) + 1e-8
adj_norm = adj_norm / D
x_agg = torch.bmm(adj_norm, x_spatial)
```

### 问题2（最高优先级）：有效训练样本严重不足
- 仿真48小时，缺陷只激活4-24小时
- overlap>=0.5的有效窗口只占25.8%（541/2100个）
- 训练集70%中有效样本约379个，50个候选节点平均每节点不到8个样本
- 大量窗口残差≈0但标签仍为"有缺陷"，形成矛盾标签

**修复方案**：在`extract_with_fixed_time_intervals_v3.py`中，提取数据时只保留缺陷激活期±2小时的时间段，其余丢弃。这样：
- 数据量降到原来约1/4，内存大幅减少
- 每个窗口天然有缺陷信号，overlap接近1.0
- 可以用节省的内存增加场景数到300-500个

### 问题3（高优先级）：数据信号强度极不均匀
- signal_score中位数=0.0014，均值=0.0323
- 说明少数场景信号极强，大多数场景信号接近零
- 模型只学会了强信号场景，弱信号场景完全失效

**需要分析**：找出好场景（场景15、92）和差场景（场景59、49、80）的缺陷参数差异：
```python
import pandas as pd
dm = pd.read_csv(r"E:\11.16\input2\defect_matrix_realistic.csv")
print(dm[dm['defect_id'].isin([15, 92, 59, 49, 80])])
```

### 问题4（中优先级）：场景划分导致5个测试节点训练集从未见过
```
TEST中训练未见过的缺陷节点: 5/14
```
这5个节点模型完全没学过，拉低整体MRR。需要在评估时分开报告"见过的节点"和"未见过的节点"的性能。

### 问题5（中优先级）：验证集评估不稳定
MRR在epoch间剧烈波动（0.08→0.24→0.14→0.24），说明早停策略在噪声上触发，选出的"最佳模型"不可靠。

## 文件结构

```
E:\11.16\script2\
├── config.py                          # 所有超参数配置
├── anomaly_detection_model.py         # 模型架构（含Bug）
├── 22_dataset_processor_fixed_patched.py  # 数据集构建
├── 24_train_anomaly_detection_patched.py  # 训练主脚本
├── 25_baseline_feature_engineering.py    # 残差特征工程
├── 26_check_labels.py                    # 数据质量检查
├── extract_with_fixed_time_intervals_v3.py  # 仿真数据提取
├── defect_matrix_generator_unified.py   # 缺陷矩阵生成
└── evaluation_patched.py                # 评估指标

E:\11.16\
├── input_data\2_tuned_v3_merged1.inp   # SWMM管网模型文件
├── input2\
│   ├── defect_matrix_realistic.csv     # 缺陷矩阵（100个场景）
│   ├── candidate_nodes_50.json         # 50个候选节点
│   ├── adj_matrix.npy                  # 邻接矩阵(128x128)
│   └── node_list.json                  # 节点列表
└── training_data_2\
    └── node_timeseries_with_residuals.parquet  # 训练数据(3.7M条)
```

## 关键配置参数（config.py）

```python
sequence_length = 36      # 输入序列长度（36步=6小时）
window_stride = 12        # 滑动窗口步长
dataset_label_mode = "time_gated"   # 标签模式
dataset_overlap_threshold = 0.5     # 有效窗口阈值
num_epochs = 50
batch_size = 32
learning_rate = 0.001
```

## 当前数据统计

- 总样本数：4200（100场景×42窗口）
- 训练集：2940样本（70场景）
- 有效定位样本（overlap≥0.5）：仅1077个（25.6%）
- 节点数：128，候选节点：50
- 特征数：18（包含残差特征）

---

## 调试任务列表（按优先级）

### Task 1：修复GCN自环Bug
- 修改`anomaly_detection_model.py`中所有GCN聚合处（`forward`和`get_node_embeddings`方法）
- 加入自环后重新训练，记录MRR变化

### Task 2：分析好/差场景的缺陷参数差异
- 读取defect_matrix_realistic.csv
- 对比场景15、92（好）vs 场景59、49、80（差）的type/intensity/start_hour/duration_h
- 输出分析报告

### Task 3：修改数据提取脚本，只保留缺陷激活期数据
- 修改`extract_with_fixed_time_intervals_v3.py`
- 提取时过滤掉缺陷激活期之外的时间步（保留start_hour-2到start_hour+duration_h+2小时的数据）
- 验证修改后每场景的数据量和overlap分布

### Task 4：评估脚本改进
- 修改`evaluation_patched.py`，分开报告：
  - 训练集见过的节点 vs 未见过的节点
  - 按缺陷类型（I/E/P）分别报告MRR和Top-K

### Task 5：增加场景数量
- 在Task3完成后（内存降低），将场景数从100增加到300
- 重新运行完整流程：生成缺陷矩阵→仿真提取→特征工程→训练→评估

---

## 调试时需要注意的事项

1. **每次实验必须固定缺陷矩阵**，不要重新随机生成，否则结果不可比
2. **一次只改一个变量**，改完训练后记录MRR再改下一个
3. **PySWMM仿真**：管网只有100多个节点，单次仿真不到1分钟，可以批量跑
4. **内存限制**：当前100场景数据约1GB，增加场景前必须先压缩单场景数据量
5. **早停问题**：验证MRR波动很大，建议改为取最后10个epoch的平均MRR作为模型选择标准，而不是单epoch最高值

---

## 期望的最终效果参考

参考论文（类似任务）：
- 5000个场景，250个候选节点
- Top-1=60.1%，Top-2=85.0%，Top-5=90.6%

当前我们的目标（100场景，50候选节点）：
- 短期目标：修复Bug后Top-1>30%，MRR>0.35
- 中期目标：增加场景到300后Top-1>45%，MRR>0.50

---

## 【更新】当前状态与完整文档（供交接/第二意见）

**完整修改、诊断、运行结果与建议下一步** 已汇总至：

- **`E:\11.16\script2\reports\MASTER_DEBUG_AND_STATUS.md`**

该文档包含：项目与任务定义（含「识别不易发现缺陷、不通过调大缺陷提升指标」的约束）、**已完成的全部修改清单**（GCN 自环、Task4 评估拆分、E 类过采样、seen_mrr 早停）、**诊断与报告**（Task2 好/差场景、数据统计、Task4 指标表）、**关键运行结果与日志摘要**、**当前问题与结论**、**建议的下一步**（弱信号样本加权、Task3 只保留激活期、特征敏感性；不推荐调大缺陷）。

**当前结果摘要（最近一次评估）**：
- 见过节点：MRR≈0.53，Top-1≈0.39
- 未见过节点：MRR≈0.08，Top-1=0
- I 类 Top-1≈0.56，E 类 Top-1=0，P 类 Top-1≈0.31
- GCN 自环已修复；E 类 2x 过采样已做，E Top-1 仍为 0（瓶颈在信号弱）

**其他报告**：`script2/reports/` 下另有 `PHASE1_DEBUG_SUMMARY.md`、`TASK2_scenario_good_bad_analysis.md`、`NEXT_STEPS_ANALYSIS.md`；以 **MASTER_DEBUG_AND_STATUS.md** 为准做交接或征求第二意见。
