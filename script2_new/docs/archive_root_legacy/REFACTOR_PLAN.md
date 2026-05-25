# script2_new 重构计划

## 为什么要新建 script2_new

- 旧目录 **script2** 承担了多轮预实验与修 bug，候选节点、缺陷矩阵、监测节点、训练输入输出等概念混在一起，不利于清晰做「监测点布置 vs 定位性能」的对照实验。
- 新实验设计需要显式区分 **V（全网）、C（候选缺陷）、S（监测）** 三个集合，并固定流程顺序，因此用独立目录 **script2_new** 承接新路线，与旧目录隔离、互不影响。

## 新路线与旧路线的区别

| 维度 | 旧路线 (script2) | 新路线 (script2_new) |
|------|------------------|----------------------|
| 候选集 C | 由脚本从 INP 等生成，但 **必须包含 defect_csv 中所有缺陷节点**，即依赖缺陷矩阵反向约束 C | C **先**由 build_candidates 独立生成，**再**基于 C 生成缺陷矩阵，不依赖 defect 反向定义 |
| 监测节点 | 多与“候选节点”或训练用节点混用，未单独成集 | 单独 **监测节点集 S**，由 build_monitors 生成，支持多种布置策略 |
| 输入/输出 | 输入输出节点集合边界不统一 | 明确：输入空间 S（或 pretest 下 V），输出空间 C |
| 流程 | 脚本分散，顺序依赖隐含 | 固定顺序：图 → C → S → 缺陷矩阵 → 时序 → 残差 → 标签检查 → 训练 |

## 为什么要显式区分 V / C / S

- **V**：全网节点，图与仿真的基础。
- **C**：模型输出空间，即“要预测哪几个节点可能缺陷”；缺陷矩阵中的缺陷必须落在 C 内，否则任务定义不一致。
- **S**：模型输入空间，即“只在哪些节点上有观测”；正式实验（chapter1/2）比较的是不同 S 的布置策略对定位性能的影响，因此 S 必须独立于 C 且可配置。

S 与 C 可部分重叠，但不要求 S=C 或 C⊆S。

## 为什么候选集 C 不能再依赖 defect_matrix 反向定义

- 旧实现中 `build_candidate_nodes_from_inp.py` 要求 “Must-include: all defect nodes from defect matrix CSV”，即先有缺陷矩阵再反推候选集，导致「缺陷场景」与「候选集」耦合，无法先定输出空间再生成缺陷。
- 新逻辑要求：**先定义 C**，再在 C 上生成缺陷矩阵（D ⊆ C），这样实验可控、可复现，且便于做“固定 C、变化 S”的对照。

## 为什么要单独生成监测节点集 S

- 正式实验要比较「不同监测点布置规则」下的定位性能，S 需作为独立配置项存在。
- 若 S 与 C 混在一起或由候选脚本一并产出，则无法单独调节 S 的生成策略（如覆盖率、关键节点优先等），故单独 **build_monitors** 只负责 S。

## 阶段划分 TODO

### 第一阶段（本轮）

- [x] 创建 script2_new 目录骨架
- [x] 各模块占位文件、README、REFACTOR_PLAN、config 基础结构
- [x] 梳理旧文件到新文件映射、职责边界
- **不做**：大规模改训练逻辑、重写模型、完整迁移旧代码、改数据处理细节、直接复制并改写所有旧脚本、自动跑训练

### 第二阶段（后续）

- 按新流程顺序迁移/重写 prep 各脚本（inp_parser → build_graph → build_candidates → build_monitors → defect_matrix → extract_timeseries → residual_features）
- 实现 validate_node_sets（C⊆V, S⊆V, D⊆C, S∩C 及 train/val/test 覆盖统计）
- dataset/processor、train、check_labels 对接新 C/S 与 config
- 模型与评估可沿用或小改，不激进重写

### 第三阶段（后续）

- pretest / chapter1 / chapter2 的完整实验配置与跑通
- 监测点布置策略扩展与对比实验

## 当前仅占位、后续才实现的逻辑

- **prep/**：除 build_candidates、build_monitors、validate_node_sets 的职责说明与骨架外，其余为占位+旧文件来源注释，未做完整迁移。
- **dataset/processor.py**、**train/train.py**、**train/check_labels.py**：占位，注明对应旧脚本，逻辑在第二阶段对接。
- **models/anomaly_detection_model.py**、**utils/evaluation.py**、**utils/metrics.py**：占位，注明来源，不在此阶段改模型结构。

## 旧项目注释与实现不一致（已发现）

1. **build_candidate_nodes_from_inp.py**  
   - 注释/文档：称 “Build a candidate node set using INP + optional adj/node_list”。  
   - 实现：同时要求 “Must-include: all defect nodes from defect matrix CSV”，即候选集依赖 `defect_csv` 反向定义。  
   - **不一致点**：文档未强调“候选集由缺陷矩阵约束”，而实现上 C 并非纯由 INP/图生成，与“先定 C 再生成缺陷矩阵”的新设计冲突。

2. **defect_matrix_generator_unified.py**  
   - 注释：统一缺陷矩阵生成器，生成 I/E/P 缺陷矩阵。  
   - 实现：未在文档中明确“缺陷节点必须来自事先给定的候选集 C”；当前用法多与旧候选脚本（依赖 defect_csv）配合。  
   - **建议**：新路线在 defect_matrix 步骤明确输入为 C，输出缺陷矩阵中 node_id ∈ C。

以上记录便于第二阶段迁移时对齐语义。
