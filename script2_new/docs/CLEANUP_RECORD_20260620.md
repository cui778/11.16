# 实验资产整理记录

整理日期：2026-06-20。

## 已完成

- 建立根项目当前 README；
- 归档早期 Chapter 1 方案和运行入口；
- 移除根级旧 `visualizations` 入口；
- 将 seedset10、persistent、fulltime 和 smoke 数据归入 `training_data_new/_legacy_exploration`；
- 为 `training_data_new`、`input_1` 和 `outputs` 增加目录说明；
- 为第3、4、5章分别建立 `RESULT_INDEX.md`；
- 为论文绘图增加 `DATA_SOURCE_POLICY.md`；
- 增加第5章当前协议补充说明。

## 暂不移动

`outputs/reports` 和 `outputs/model_checkpoints` 仍由共享训练入口按固定路径写入。为避免破坏训练、预算实验和复现实验，本轮不进行大规模物理搬迁，改用章节结果索引区分正式与历史输出。

## 使用原则

- 正文结果从章节 `RESULT_INDEX.md` 进入；
- 大型数据从 `training_data_new/README.md` 进入；
- 共享模型输出从 `outputs/README.md` 进入；
- 论文绘图遵循 `thesis_writing_repo/figures/DATA_SOURCE_POLICY.md`；
- 未出现在正式索引中的文件默认不得直接用于正文。
