# 第5章文档收口记录

## 基本信息

- 日期：2026-04-04
- 日志名称：第5章 plans/docs 文档收口
- 对应内容：第5章主文档整理

## 目的

- 压缩 `plans/` 与 `docs/` 下的平行说明文档
- 保留主线文档，删除被吸收的派生说明
- 以后优先修改旧文档，不再横向增加同类说明

## 输入

- 文件：
  - [CH5_PROTOCOL_FREEZE.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_PROTOCOL_FREEZE.md)
  - [CH5_DUAL_TRACK_INNOVATION_PLAN_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_DUAL_TRACK_INNOVATION_PLAN_20260404.md)
  - [CH5_C_AND_S_REVIEW_AND_NEXT_STEPS_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_C_AND_S_REVIEW_AND_NEXT_STEPS_20260404.md)
  - [CH5_INNOVATION_METHOD_OPTIONS_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_INNOVATION_METHOD_OPTIONS_20260404.md)
  - [CH5_STATIC_LAYOUT_SCREENING_20260403.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_STATIC_LAYOUT_SCREENING_20260403.md)
  - [CH5_FIRST_COMPARISON_N25_SEED42_20260403.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_FIRST_COMPARISON_N25_SEED42_20260403.md)
  - [CH5_NODE_HOLDOUT_COMPARISON_N25_SEED42_20260403.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_NODE_HOLDOUT_COMPARISON_N25_SEED42_20260403.md)
  - [CH5_TWO_STAGE_FRAMEWORK_PROGRESS_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_TWO_STAGE_FRAMEWORK_PROGRESS_20260404.md)
  - [CH5_TWO_STAGE_NODE_HOLDOUT_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_TWO_STAGE_NODE_HOLDOUT_20260404.md)
  - [CH5_STRENGTHENED_MAIN_METHOD_V3_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_STRENGTHENED_MAIN_METHOD_V3_20260404.md)

## 输出

- 更新：
  - [README.md](/E:/11.16/script2_new/chapter5_layout_optimization/README.md)
  - [plans/README.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/README.md)
  - [CH5_PROTOCOL_FREEZE.md](/E:/11.16/script2_new/chapter5_layout_optimization/plans/CH5_PROTOCOL_FREEZE.md)
  - [CH5_FIRST_COMPARISON_N25_SEED42_20260403.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_FIRST_COMPARISON_N25_SEED42_20260403.md)
  - [CH5_TWO_STAGE_FRAMEWORK_PROGRESS_20260404.md](/E:/11.16/script2_new/chapter5_layout_optimization/docs/CH5_TWO_STAGE_FRAMEWORK_PROGRESS_20260404.md)

## 过程

- 先确认哪些文档属于主线文档，哪些只是阶段派生说明。
- 采用“借旧壳收内容”的方式，把相关结果并入已有主文档。
- 计划在完成内容吸收后删除被并入的冗余说明文件。

## 关键发现

- 当前最适合保留的结构，是“少量主文档 + 每次实验一条日志”。
- `docs/` 里最容易压缩的是基线结果组和主方法迭代组。
- `plans/` 里最容易压缩的是 `C/S` 评审说明和创新选项说明。

## 对第5章写作的影响

- 后续第5章的阅读入口更清楚。
- 实验轨迹仍然保留在 `experiment_logs/`，但不再让 `plans/` 和 `docs/` 横向膨胀。

## 下一步

- 删除已经被吸收掉的冗余计划文档与结果文档。
- 后续继续只修改主文档，不再平行新开同类说明。
