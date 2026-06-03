# 第3章 数据生成目录说明

本目录存放第3章"SWMM仿真数据生成"的归档结构。第3章的数据生成脚本位于 `script2_new/prep/`，大型 Parquet 数据保持原位于 `script2_new/training_data_new/`。

## 目录结构

```text
chapter3_data_generation/
  README.md                    — 本文件
  plans/
    CH3_PROTOCOL_FREEZE.md     — 正式协议冻结文档
  outputs/
    CH3_DATASET_MANIFEST.csv   — 9个数据集目录的冻结清单
    CH3_SCRIPT_STATUS.csv      — 29个脚本的分类状态
  legacy_exploration/
    scripts/                   — 已归档的探索脚本（6个）
    inputs/                    — 探索性轻量输入
    reports/                   — 探索性报告
    indexes/                   — 过期索引文件
```

## 正式数据集

| 数据集 | 场景数 | 用途 |
|---|---|---|
| `time_gated_full_ie_v4_formal_conservative420_seed42` | 421 | IE420 缺陷母数据 |
| `normal_multibaseline_v2_seedset20` | 20 | 无缺陷正常工况层 |
| `ie420_plus_normal20_v1` | 441 | 第4、5章正式训练组合 |

## 正式脚本（位于 prep/）

| 脚本 | 角色 |
|---|---|
| `build_formal_ie_conservative_matrix.py` | IE420 缺陷矩阵构建 |
| `build_ie420_normal20_control_dataset.py` | 正式组合数据集构建 |
| `build_48h_restructured_datasets.py` | 共享基础设施（不归档） |

## 图源包

论文图源位于 `thesis_writing_repo/figures/ch3/source_data/`：
- CH3-F01 数据集协议汇总
- CH3-F02 IE420 缺陷矩阵统计
- CH3-F03 正常工况层参数
- CH3-F04 采样与记录统计
- CH3-F05 残差特征定义
- CH3-F06 数据完整性审计
