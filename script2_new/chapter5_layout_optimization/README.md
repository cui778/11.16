# Chapter 5 Layout Optimization

这个工作区专门承接**学位论文第5章“监测点优化研究”**的全部材料。

请统一按下面这条口径理解：

- 你有时提到的“第二章研究内容”
- 在当前工作组织上
- 就对应这里这条监测点优化主线
- 在论文正式章节里对应**第5章**

因此，后续不再为这条主线另开并行目录。

## Directory

- `plans/`
  - 放第5章主协议、实验设计、方法方案、执行顺序
- `docs/`
  - 放阶段总结、结果分析、写作说明
- `experiment_logs/`
  - 每次设计调整、正式实验、复核都记录一个 markdown
- `outputs/`
  - 放布局文件、汇总表、对比结果
- `figures/`
  - 放第5章图件
- `scripts/`
  - 放第5章专属脚本
- `visualizations/`
  - 放第5章专属可视化脚本

## Main Rule

这个目录后续遵循两条规则：

1. **优先更新已有主文档，不要为同一主线反复平行新建说明文件。**
2. **每次有实际推进的设计或实验，都在 `experiment_logs/` 留下一份 markdown 记录。**

## Data / Output Convention

- 布局文件放在：
  - `outputs/layouts/<strategy>/monitor_nodes_<strategy>_N<n>.json`
- 第5章当前默认复用的正式 full-graph 数据目录是：
  - `E:\11.16\script2_new\training_data_new\time_gated_full_ie_v4_formal_conservative420_seed42`
- 第5章主实验研究的是：
  - 在统一评估协议下，只改变监测点布局及其对应观测掩膜，比较定位效果变化

## Reading Order

后续优先看这几份主文档：

1. [CH5_PROTOCOL_FREEZE.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_PROTOCOL_FREEZE.md)
2. [CH5_INNOVATION_TO_EXPERIMENT_MAP.md](/E:/11.16/script2_new/CH5_INNOVATION_TO_EXPERIMENT_MAP.md)
3. [plans/README.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/README.md)

这三份负责管住：

- 第5章口径
- 创新点与实验映射
- 现有计划文档怎么继续更新
