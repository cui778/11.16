# script3

`script3` 是第一章重置后的独立实验线。

它用于承载重新组织后的“候选节点/候选管段缺陷定位”研究流程，并与 `script2`、`script2_new` 分开维护。

## 研究范围

- 主任务：缺陷定位
- 扩展任务：缺陷类型判别（`I/E/P`，当前可按需要筛成 `I/E`）
- 第一章主线：聚焦模型创新，而不是完整诊断系统

## 目录结构

- `prep/`：数据预处理与图结构构建
- `dataset/`：数据集构造与数据划分逻辑
- `models/`：模型定义
- `train/`：训练与评估入口
- `utils/`：指标计算与评估辅助函数
- `input_1/`：本实验线使用的中间输入文件
- `training_data/`：本实验线生成的训练数据
- `outputs/reports/`：实验报告与汇总结果
- `outputs/model_checkpoints/`：模型检查点
- `outputs/logs/`：运行日志

## 说明

- 不要把旧实验的 checkpoint 或 report 直接复制到这里。
- `script3` 的命名应与 `script2_new` 保持独立，避免新旧实验混淆。
- 可以有选择地复用 `script2_new` 中的代码，但不要直接继承旧第一章的实验命名体系。
- 当前 `script3` 的默认主线仍是“先做定位，再做类型判别扩展”。
- 默认水质特征预设仅保留 `pollut_BODf` 及其残差特征。
- 第一章批量对比入口见 [run_chapter1_main.py](/e:/11.16/script3/run_chapter1_main.py)
- 管段级预处理目前从 [build_segments.py](/e:/11.16/script3/prep/build_segments.py) 和 [edge_feature_engineering.py](/e:/11.16/script3/prep/edge_feature_engineering.py) 开始
- Phase 1/2 预处理串联入口见 [run_phase12_prep.py](/e:/11.16/script3/run_phase12_prep.py)
