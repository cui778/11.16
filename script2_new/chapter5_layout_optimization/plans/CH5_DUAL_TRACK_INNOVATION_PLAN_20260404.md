# 第5章双轨创新方案：方法框架创新 + 结构发明型创新

## 1. 目标

第5章不再只停留在启发式布点比较，而是同时推进两条创新主线：

1. **方法框架创新主线**
   - 目标：形成一套稳妥、可解释、可复现实验的正式方法
   - 定位：第5章主方法，优先保证论文可交付
2. **结构发明型创新探索线**
   - 目标：尝试真正“模型化”的可学习布点方法
   - 定位：高风险高收益的增强方案，争取更强新意

这两条线共享同一任务设置、同一评估协议和同一实验口径，但创新层次不同。

## 2. 共同约束

为保证两条线的结果可比较，第5章主实验统一遵守以下条件：

- 候选节点集合保持一致
- 缺陷场景集合保持一致
- 输入数据口径保持一致
- 统一使用同一诊断评估器
- 统一窗长、随机种子与 split 设置
- 自变量主要是：
  - 监测点布局方案
  - 布局生成策略

换句话说：

- 双轨创新可以并行尝试
- 但都必须回到同一评估体系下比较

## 3. 当前已有基础

### 3.1 已有基线

当前已经具备以下基线布局：

- `random`
- `degree`
- `betweenness`
- `downstream`
- `candidate_observability`
- `identifiability_driven`

这些方法将继续作为第5章比较组。

### 3.3 当前方法层级判断

为了避免后续再次把“基线、主方法、探索线”混在一起，当前统一按下面分层：

- 基线层
  - `random`
  - `degree`
  - `betweenness`
  - `downstream`
- 任务驱动基线层
  - `candidate_observability`
  - `identifiability_driven`
- 主方法层
  - `two_stage_balanced_layout_v1`
  - `two_stage_balanced_layout_v2`
- 强化探索层
  - `two_stage_generalization_balanced_layout_v3`
  - `learnable_layout_network_v0`

这意味着：

- `candidate_observability` 必须承认其结果价值
- 但它当前更适合作为强任务驱动基线
- 第5章主方法仍然优先落在两阶段框架线上

### 3.2 已有脚本基础

可复用脚本主要包括：

- [build_layout_suite.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_layout_suite.py)
- [build_evaluator_manifest.py](/E:/11.16/script2_new/chapter5_layout_optimization/scripts/build_evaluator_manifest.py)
- [train_privileged_teacher_student.py](/E:/11.16/script2_new/scripts/train_privileged_teacher_student.py)

因此，第5章后续新增方法最好都输出为统一的：

- `monitor_nodes_*.json`

这样可以直接回插现有评估流程。

## 4. 轨道 A：方法框架创新

## 4.1 方法名称建议

建议命名为：

- `two_stage_balanced_layout_v1`

如果后续有增强版，可继续扩展为：

- `two_stage_balanced_layout_v2`

## 4.2 核心思想

该方法继承开题报告中的两阶段思想，但做成可执行简化正式版：

### 阶段 1：候选池压缩

目标：

- 在预算 `M` 下，先把候选池压缩成约 `1.2~1.5 x M` 的代表集
- 去除空间和功能上的冗余点

候选相似度建议由以下部分构成：

1. 拓扑距离相似度
   - 最短路距离
   - 分区/边界信息
2. 响应相似度
   - 候选节点在场景中的观测签名
   - 或节点对不同场景的影响模式
3. 可选增强项
   - evaluator 敏感度
   - attention / attribution 派生敏感度

候选压缩方式：

- `k-medoids`
- 或谱聚类后取簇中心/medoid

### 阶段 2：预算内组合优化

在代表集中求解大小为 `M` 的布局集合 `S`。

目标函数建议综合：

1. 任务效能项
   - 定位性能 surrogate
   - 或基于静态指标近似 evaluator 效果
2. 覆盖/重构项
   - 候选可观测性
   - 全网结构覆盖
   - 可选的未观测恢复约束
3. 冗余惩罚项
   - 相似度过高惩罚
   - 距离过近惩罚
4. 结构性可观测项
   - 主干边
   - 边界节点
   - 关键路径覆盖

求解策略建议：

- 先 `greedy add`
- 再 `local swap`
- 如有余力，再尝试：
  - `simulated annealing`
  - `genetic algorithm`

## 4.3 它的创新点

这条线的创新点不是神经网络结构，而是：

1. 将监测点优化构造成“压缩 + 联合优化”的完整框架
2. 将拓扑、任务、冗余、结构覆盖联合建模
3. 将统一诊断评估器回插纳入闭环

这属于：

- **方法框架创新**
- **任务导向优化框架创新**

## 4.4 这条线的优点

- 工程可实现性高
- 与开题报告衔接自然
- 容易形成第5章主结果
- 可解释性强

## 4.5 这条线的风险

- 如果目标设计过于复杂，容易调不动
- 如果 evaluator 回插过慢，优化成本高
- 如果压缩阶段做得太强，可能伤害最优解空间

## 4.6 建议实现顺序

### A1：正式版

- `拓扑距离 + 响应相似度` 做候选压缩
- `greedy + local swap` 做预算内优化
- 输出 `two_stage_balanced_layout_v1`

### A2：增强版

- 加入敏感度项
- 加入鲁棒性项
- 加入更强全网约束
- 输出 `two_stage_balanced_layout_v2`

## 5. 轨道 B：结构发明型创新

## 5.1 目标

这条线不再手工设计布局打分器，而是让模型学习“应该选哪些点”。

建议先做一个轻量版本，命名为：

- `learnable_layout_network_v0`

如果有效，再升级。

## 5.2 可选方向

### 方向 B1：可学习布点网络

最推荐首先尝试这一条。

基本思路：

- 输入：
  - 128 节点图结构
  - 候选节点标记
  - 节点图特征
  - 候选静态签名 / 场景响应统计
- 输出：
  - 每个节点一个布点得分
- 选点：
  - Top-K
  - 或 soft-topk / Gumbel-TopK
- 训练信号：
  - evaluator surrogate
  - 或布局质量回归

核心逻辑：

- 不是人工规定“什么点更好”
- 而是由模型从历史布局效果中学习“什么样的点组合更好”

### 方向 B2：双层优化 + surrogate

思路：

- 外层：搜索布局
- 内层：统一评估器评分
- 中间：训练一个 surrogate 近似布局优劣

优点：

- 保持 evaluator 导向
- 比直接把 evaluator 放进搜索循环更省算力

### 方向 B3：强化学习逐点选址

思路：

- 状态：当前已选点集 + 剩余预算 + 图嵌入
- 动作：选下一个监测点
- 奖励：定位指标收益

优点：

- 很像“策略模型”
- 创新感强

不足：

- 工程复杂
- 不适合作为最先落地版本

### 方向 B4：表示学习型可分性布局

思路：

- 先学习候选缺陷响应的嵌入表示
- 再选择最能把候选缺陷拉开的点集

优点：

- 与定位任务强相关
- 比单纯 coverage 更有模型感

## 5.3 这条线的创新点

这条线的创新点是：

- 监测点选择器本身成为一个模型
- 布局生成不再完全依赖规则或手工目标
- 可形成真正的“结构发明型创新”表述

## 5.4 这条线的优点

- 新意更强
- 更接近“模型创新”
- 如果有效，论文亮点会明显增强

## 5.5 这条线的风险

- 开发周期长
- 训练不稳定
- 可能需要 surrogate 才能跑得动
- 容易出现“模型很新但结果不稳定”

## 5.6 建议实现顺序

### B1：轻量起步版

- 先训练一个布局质量 surrogate
- 再训练一个节点打分网络去拟合高质量布局
- 输出 `learnable_layout_network_v0`

### B2：增强版

- 加入 soft-topk / differentiable subset
- 加入 evaluator-in-the-loop 微调
- 输出 `learnable_layout_network_v1`

## 6. 双轨共享的中间件

为了避免两条线分别重造轮子，第5章建议先做以下共享模块：

1. **布局特征提取模块**
   - 输入节点集合
   - 输出：
     - 候选 hop 指标
     - 全网覆盖指标
     - 冗余指标
     - 结构覆盖指标
2. **布局评分数据集构建模块**
   - 收集：
     - 布局 JSON
     - 静态特征
     - evaluator 得分
   - 用于：
     - surrogate
     - learnable layout network
3. **统一回插与汇总模块**
   - 自动跑 evaluator
   - 自动写结果表
4. **布局候选压缩模块**
   - 轨道 A 直接用
   - 轨道 B 也可用作动作空间压缩

## 7. 建议的正式推进顺序

建议按下面顺序推进，而不是同时无序开发：

### Step 1

先完成轨道 A 的 `two_stage_balanced_layout_v1`

原因：

- 风险低
- 能尽快形成第5章主方法
- 还能为轨道 B 产生高质量布局样本

### Step 2

基于轨道 A 与现有基线，构建“布局 -> evaluator 得分”的训练集

当前已经决定从这里正式启动，起步层拆成两步：

1. `build_layout_quality_dataset.py`
   - 把已有布局静态特征、`scenario split` 结果、`node_holdout` 结果合并成结构创新训练底座
2. `train_layout_surrogate.py`
   - 先训练一个轻量 surrogate
   - 用它判断哪些静态布局特征最能预测 evaluator 指标
   - 再决定 learnable layout network 的输入和监督信号

这一步已经完成，当前状态是：

- 布局质量数据集已生成
- 第一轮 surrogate 已训练完成
- `scenario_top1` 已出现初步可预测性信号
- `node_holdout` 仍然较难从现有静态特征稳定预测

### Step 3

再启动轨道 B 的 `learnable_layout_network_v0`

这样做的好处是：

- 轨道 B 有监督信号来源
- 不会一开始就陷入高风险模型实验

这一步现在也已经启动，且完成了第一轮真实尝试：

- `learnable_layout_network_v0_balanced`
- `learnable_layout_network_v0_scenario`
- `learnable_layout_network_v0_generalization`

其中：

- `balanced` 与 `scenario` 最终解码成同一套布局
- `generalization` 形成了不同布局，并在首轮正式实验中表现最好

## 8. 第5章最终可以形成的创新叙事

如果双轨都推进成功，第5章可以写成：

1. 先证明传统拓扑布点与简单任务驱动启发式的局限
2. 提出一套两阶段平衡型监测点优化框架，作为稳妥主方法
3. 在此基础上进一步提出可学习布点模型，探索监测点优化的模型化方向
4. 对比说明：
   - 规则法
   - 框架法
   - 模型法
   在定位性能、鲁棒性、泛化与工程可实施性上的差异

这样第5章的创新层次会明显更完整。

## 9. 当前建议结论

可以正式采用如下定位：

- **主线方法**：
  - `two_stage_balanced_layout_v1`
- **增强方法**：
  - `two_stage_balanced_layout_v2`
- **结构创新探索线**：
  - `learnable_layout_network_v0`

这意味着：

1. 方法框架创新和结构发明型创新都试
2. 但先后顺序明确
3. 第5章先保住可交付主线，再冲更强创新

## 10. 下一步脚本落点建议

建议新增以下脚本：

### 轨道 A

- `chapter5_layout_optimization/scripts/build_two_stage_balanced_layout.py`
- `chapter5_layout_optimization/scripts/extract_layout_features.py`

### 轨道 B

- `chapter5_layout_optimization/scripts/build_layout_quality_dataset.py`
- `chapter5_layout_optimization/scripts/train_layout_surrogate.py`
- `chapter5_layout_optimization/scripts/train_learnable_layout_network.py`

### 汇总

- `chapter5_layout_optimization/scripts/run_ch5_dual_track_eval.py`

## 11. 最终一句话

第5章后续不再只做“启发式布点比较”，而是正式按双轨推进：

- 一条做**可交付的两阶段平衡型监测点优化框架**
- 一条做**可学习的监测点布局模型**

两条线共享同一评估体系，最后统一比较。

## 12. 当前启动决定

结构发明型创新这条线，当前不直接上高风险策略模型，而是先走：

1. 布局质量数据集
2. surrogate
3. 再上 learnable layout network

这样做的原因是：

- 现有第5章已经积累了一批真实布局与真实回插结果
- 可以先把这些结果组织成模型训练数据
- 再决定哪些结构创新方向最值得继续深挖

当前最新判断：

- `learnable_layout_network_v0` 已经不是纸面方案，而是已落地并完成首轮回插
- 其中 `generalization` 预设最值得继续
- 后续更合理的方向不是重新回到纯规则法，而是继续围绕：
  - 伪标签构造
  - 节点图特征
  - 解码器约束
  做下一轮结构创新迭代
