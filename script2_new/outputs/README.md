# outputs 共享训练输出说明

## 1. 目录用途

本目录是共享训练入口产生的运行时输出，不是单一章节的最终结果目录。

```text
outputs/
├── logs/               # 运行日志
├── model_checkpoints/  # 模型权重、训练历史和部分逐场景输出
└── reports/            # JSON 指标、CSV 汇总和审计结果
```

这里同时包含正式实验、补充实验和历史运行，因此不能直接按“最新文件”或文件数量选择论文结果。

## 2. 第4章正式结果

第4章固定协议的核心报告：

```text
reports/
last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s7.json
last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s42.json
last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s123.json
```

窗口级和事件级预测明细位于：

```text
E:\11.16\script2_new\chapter4_diagnosis_model\outputs
```

完整入口见：

- `E:\11.16\script2_new\chapter4_diagnosis_model\RESULT_INDEX.md`

## 3. 第5章正式结果

第5章训练指标仍由共享训练入口写入 `reports/`，常见前缀包括：

```text
last_run_metrics_ch5_fixed_*
last_run_metrics_ch5_embedding_guided_*
last_run_metrics_ch5_coverage_rate_controlled_*
last_run_metrics_48h_control_ie420_normal20_*  # Degree/Betweenness 预算结果
```

布局文件、实验清单和章节汇总位于：

```text
E:\11.16\script2_new\chapter5_layout_optimization\outputs
```

完整入口见：

- `E:\11.16\script2_new\chapter5_layout_optimization\RESULT_INDEX.md`

## 4. 历史与非正式结果

以下命名通常表示历史、调试或已被替代的实验：

```text
chapter1_restart_*
process_diagnosis_*
v2e_dense_*
*_persistent_*
*_smoke_*
*_debug_*
48h_feature_*              # seedset10 历史调参
48h_lambda_*               # seedset10 历史调参
48h_layout_*               # 早期布局实验
```

旧结果暂不批量删除，以免破坏复现实验；但不得作为论文正文数据源。已经完成分类的旧汇总表位于：

```text
E:\11.16\script2_new\chapter4_diagnosis_model\legacy_exploration
```

## 5. 文件格式约定

| 文件类型 | 作用 |
|---|---|
| `last_run_metrics_<tag>.json` | 单次运行完整指标与配置 |
| `training_history_<tag>.csv` | epoch 级训练历史 |
| `best_model_<tag>.pth` | 最优模型权重 |
| `*_BY_SEED.csv` | 多 seed 原始结果 |
| `*_SUMMARY.csv` | 汇总均值、标准差或章节表格 |
| `*_AUDIT.md` | 数据源、协议和运行完整性说明 |

论文绘图优先读取经审计的 `*_SUMMARY.csv` 或 `figures/ch*/source_data`，不得直接混合不同协议的 JSON。
