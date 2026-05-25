# 第一章总报告

节点级主线已经完成，当前建议将 node_holdout 作为第一章主比较划分，并将 lstm_graphsage_edge 作为节点级主模型。代表结果为 MRR=0.9147、Top-1=0.8764、Top-3=0.9521、Top-5=0.9969。

segment mode 也已经完成正式版闭环。segment_ie_full_v1 当前结果为 MRR=0.7094、Top-1=0.6246、Top-3=0.7885、Top-5=0.8011、type_macro_f1=0.3194，其中 I 类 Top-1=0.7910、E 类 Top-1=0.4375。

截至 2026-03-18，可以把第一章状态概括为：节点级主模型已确认，节点级总表已完成，segment 训练与评估已打通，边级定位和联合类型头已有正式结果，当前重点应转向统一总表整理与 E 类增强。

对应文件：CHAPTER1_TOTAL_EXPERIMENT_TABLE.json、CHAPTER1_NODE_EXPERIMENT_TABLE.json、CHAPTER1_NODE_FINAL_SUMMARY_CN.md、CHAPTER1_SEGMENT_SMOKE_SUMMARY.md、last_run_metrics_segment_segment_ie_full_v1.json。
