# 第5章脚本目录说明

本目录专门存放第5章“监测点优化研究”的脚本。

当前脚本按两条主线组织：

1. 布局生成与方法框架创新
2. 结构发明型创新的中间件与模型脚手架

## 一、已有布局与框架脚本

- `build_layout_suite.py`
  - 批量生成基线组与任务驱动组布局
  - 输出：
    - `outputs/layouts/<strategy>/monitor_nodes_<strategy>_N<n>.json`
    - `outputs/layout_summary.csv`

- `build_evaluator_manifest.py`
  - 生成统一评估协议下的正式运行清单

- `build_dataset_views.py`
  - 为每个布局生成轻量 dataset-view 清单
  - 不复制共享 full-graph parquet

- `build_two_stage_balanced_layout.py`
  - 第5章方法框架创新 `v1`
  - 两阶段：
    - 阶段1：拓扑-响应联合代表池压缩
    - 阶段2：`greedy + local swap` 预算内优化

- `build_two_stage_balanced_layout_v2.py`
  - 第5章方法框架创新 `v2`
  - 在 `v1` 上增加：
    - evaluator 节点敏感度
    - reconstruction 约束
    - 随机失效鲁棒性
    - simulated annealing 微调

- `build_overlap_controlled_probe_layouts.py`
  - overlap-controlled 机制验证脚本
  - 在固定预算、固定 `S ∩ C` 的前提下比较布局结构差异

- `build_two_stage_generalization_balanced_layout.py`
  - 第5章强化版主方法脚本
  - 当前包含：
    - `v3`
    - `v3_1`
    - `v3_1a`
    - `v3_2`
  - 核心增强：
    - non-candidate backbone coverage
    - candidate / non-candidate balance
    - monitor dispersion
    - overlap regularization

## 二、结构发明型创新起步脚本

- `build_layout_quality_dataset.py`
  - 把现有布局静态特征与 evaluator 回插结果整理成结构创新训练底座
  - 输出：
    - `outputs/structural_innovation/layout_quality_dataset_long.csv`
    - `outputs/structural_innovation/layout_quality_dataset_wide.csv`
    - `outputs/structural_innovation/layout_quality_dataset_manifest.json`

- `train_layout_surrogate.py`
  - 基于当前布局质量数据集训练轻量 surrogate
  - 先做低风险起步版，用于回答：
    - 哪些静态布局特征最能预测 evaluator 得分
    - 结构创新后续应优先学什么
  - 输出：
    - `outputs/structural_innovation/surrogate/surrogate_predictions.csv`
    - `outputs/structural_innovation/surrogate/surrogate_coefficients.csv`
    - `outputs/structural_innovation/surrogate/surrogate_training_report.json`

- `train_learnable_layout_network.py`
  - 第5章结构发明型创新 `learnable_layout_network_v0`
  - 用已有布局质量结果构造节点伪标签
  - 训练轻量节点打分网络
  - 再用带 overlap / 分散性约束的解码器生成布局
  - 当前支持：
    - `balanced`
    - `scenario`
    - `generalization`
  - 输出：
    - `outputs/structural_innovation/learnable_layout_network_v0_summary.csv`
    - `outputs/structural_innovation/learnable_layout_network_v0/<preset>/`
    - `outputs/layouts/learnable_layout_network_v0_<preset>/`

## 三、共享数据口径

第5章统一复用共享 full-graph 数据目录：

- `E:\11.16\script2_new\training_data_new\time_gated_full_ie_v4_formal_conservative420_seed42`

主实验保持：

- 缺陷场景集合一致
- 候选节点集合一致
- 主模型与输入口径一致
- 主要变化量是监测点布局与诱导的 observed mask

## 四、当前推荐顺序

1. 先用 `build_layout_suite.py` 生成基线布局
2. 再用两阶段脚本推进方法框架创新
3. 然后用 `build_layout_quality_dataset.py` 汇总现有布局质量数据
4. 再用 `train_layout_surrogate.py` 启动结构发明型创新的起步层

当前结构创新不建议一开始就直接上：

- 强化学习逐点选址
- evaluator-in-the-loop 高成本双层优化

更稳的起点是：

- 先有布局质量数据集
- 再有 surrogate
- 最后再上 learnable layout network
