# 第4章 诊断模型目录说明

本目录存放第4章"固定布局下的I/E缺陷诊断模型"的文档和归档结构。

## 正式协议

```text
数据集：ie420_plus_normal20_v1 (441 场景)
特征：raw_plus_residual (12 特征)
lambda_loc：0.5
模型：hydraulic_inverse_deepattn
布局：degree_N25 (25 监测节点)
划分：scenario split
种子：7 / 42 / 123
```

## 目录结构

```text
chapter4_diagnosis_model/
  README.md                    — 本文件
  docs/                        — 内部文档
  outputs/                     — 正式输出（候选空间分析等）
  time_process_diagnosis_branch/ — 时间过程诊断分支（独立 BiGRU checkpoint）
  legacy_exploration/
    scripts/                   — 已归档的探索脚本
    reports/                   — 已归档的探索报告
    indexes/                   — 过期索引文件
    checkpoints_index/         — checkpoint 分类索引
```

## 正式训练入口

```text
E:\11.16\script2_new\scripts\train_privileged_teacher_student.py
```

该脚本同时被第5章使用，不移动。

## 正式 checkpoint

文件名前缀：`ch1_fullgraph_degree_ie420_*`
位于：`E:\11.16\script2_new\outputs\model_checkpoints\`

## 正式报告

位于 `E:\11.16\script2_new\outputs\reports\`：
- `last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s{7,42,123}.json`
- `last_run_metrics_ch1_fullgraph_degree_ie420_s{7,42,123}_fix1.json`
- `clean_*` 系列正式汇总表
- `feasible_candidate_subsets_*` per-seed 候选子集

## 图源包

论文图源位于 `thesis_writing_repo/figures/ch4/source_data/`：
- CH4-F06 ~ CH4-F10b（详见 figure_catalog.md）
