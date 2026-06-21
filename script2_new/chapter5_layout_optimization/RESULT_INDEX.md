# 第5章结果索引

## 研究边界

第5章固定第4章诊断方法和正式数据协议，仅改变监测节点集合、监测预算和相应观测掩膜。

## 正式主协议

```text
dataset     = ie420_plus_normal20_v1
model       = hydraulic_inverse_deepattn
features    = raw_plus_residual
lambda_loc  = 0.5
split       = scenario
N25 seeds   = 7 / 42 / 123
budgets     = 5 / 10 / 15 / 20 / 25
```

## 正式结果入口

| 结果 | 路径 |
|---|---|
| N25 主协议布局 | `outputs\layouts` |
| N25 主结果汇总 | `E:\11.16\thesis_writing_repo\figures\ch5\source_data\CH5-EXPT_fixed_protocol_N25_main_table.csv` |
| 正式预算实验 | `outputs\budget_sweep_formal` |
| 覆盖率受控实验 | `outputs\coverage_rate_controlled` |
| 结构与学习方法 | `outputs\structural_innovation` |
| 布局稳定性 | `outputs\layout_stability` |
| 单次训练指标 | `E:\11.16\script2_new\outputs\reports` |
| 协议冻结 | `plans\CH5_PROTOCOL_FREEZE.md` |

## 论文绘图

| 内容 | 路径 |
|---|---|
| 绘图脚本 | `E:\11.16\thesis_writing_repo\figures\scripts\ch5` |
| 汇总源表 | `E:\11.16\thesis_writing_repo\figures\ch5\source_data` |
| 正式图片 | `E:\11.16\thesis_writing_repo\figures\ch5\generated_results` |
| 源表索引 | `E:\11.16\thesis_writing_repo\figures\ch5\source_data\TABLE_INDEX.md` |

## 正式方法

- Degree；
- Betweenness；
- Cand-Obs；
- Two-stage v1；
- Node-Feedback；
- Embedding-Guided。

Two-stage v2/v3、overlap-controlled probe、surrogate 和其他早期方案属于方法探索，不进入当前 N25 主结果表。
