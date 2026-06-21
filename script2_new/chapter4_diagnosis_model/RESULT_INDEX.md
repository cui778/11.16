# 第4章结果索引

## 正式协议

```text
dataset     = ie420_plus_normal20_v1
model       = hydraulic_inverse_deepattn
layout      = degree_N25
features    = raw_plus_residual
lambda_loc  = 0.5
split       = scenario
seeds       = 7 / 42 / 123
```

## 正式结果入口

| 结果 | 路径 |
|---|---|
| 主模型单次报告 | `E:\11.16\script2_new\outputs\reports\last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s{seed}.json` |
| 模型对比 | `outputs\formal_model_comparison` |
| 窗口长度 | `outputs\formal_window_length` |
| I/E 分组 | `outputs\formal_ie_group` |
| 方法消融（深度/路径先验/LSTM） | `outputs\formal_method_ablation` |
| 可观测性关系 | `outputs\candidate_observability_tiers.csv` |
| 协议冻结 | `plans\CH4_PROTOCOL_FREEZE.md` |

## 论文绘图

| 内容 | 路径 |
|---|---|
| 绘图脚本 | `E:\11.16\thesis_writing_repo\figures\scripts\ch4` |
| 汇总源表 | `E:\11.16\thesis_writing_repo\figures\ch4\source_data` |
| 正式图片 | `E:\11.16\thesis_writing_repo\figures\ch4\generated_results` |
| 图件索引 | `E:\11.16\thesis_writing_repo\figures\ch4\CH4_OFFICIAL_FIGURE_INDEX.md` |

`draw_13` 至 `draw_15` 直接读取本目录的预测明细和拓扑输入；其余正式汇总图优先读取 `source_data`。

## 结果等级

| 等级 | 内容 |
|---|---|
| formal | 主模型、多 seed、正式模型对比 |
| supplementary | 方法消融、窗口长度、I/E 分组、案例诊断、可观测性边界 |
| legacy | chapter1 restart、persistent、process_diagnosis、旧 node-holdout |

历史结果统一保存在 `legacy_exploration/`，不得回填正文主表。
