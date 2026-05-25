# 第一章 Segment Smoke 小结

日期：2026-03-18
任务：segment mode 管段级定位 + I/E/P 图级类型辅助头
数据来源：E:\11.16\script2_new\training_data_new\time_gated_downstream
缺陷集合：E:\11.16\script3\input_1\defect_matrix_ie.csv
说明：保持现有仿真定义不变，先用 node timeseries 与 segment 静态/上下游差分特征补齐边级训练闭环。

## Smoke 结果

- 运行名：segment_ie_smoke1
- 训练轮数：1
- hidden_dim：32
- MRR：0.6206
- Top-1：0.4762
- Top-3：0.7423
- Top-5：0.7605
- type_macro_f1：0.3200
- I 类 Top-1：0.7870
- E 类 Top-1：0.1265

## 正式版结果

- 运行名：segment_ie_full_v1
- 训练设置：num_epochs=10, patience=3, hidden_dim=64
- MRR：0.7094
- Top-1：0.6246
- Top-3：0.7885
- Top-5：0.8011
- type_macro_f1：0.3194
- I 类 Top-1：0.7910
- E 类 Top-1：0.4375

## 结论

- segment 训练、验证、测试与报告输出已经闭环。
- 正式版相较 smoke 有明显提升，尤其 E 类 Top-1 从 0.1265 提升到 0.4375。
- 当前边级结果已经具备进入总表和总报告的条件。
- 下一步应转入更系统的模型对比与特征/结构调参，而不是继续停留在“能不能跑通”。
