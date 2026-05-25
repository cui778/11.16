# 第一章节点级实验总结
 
节点级主结果建议统一采用 node_holdout 划分，主指标为 MRR、Top-1、Top-3、Top-5。
当前节点级主模型建议定为 lstm_graphsage_edge，hydraulic_inverse 作为重要对照，tcn_graphsage_edge 作为补充模型。
输入表示存在显著模型依赖：hydraulic_inverse 更依赖 residual，GraphSAGE 两个模型在 raw 或 raw_residual 下更强。
raw + lambda=0.2 下，lstm_graphsage_edge 在 node_holdout 上达到 MRR=0.9982、Top-1=0.9969；hydraulic_inverse 仅 MRR=0.0578。
raw_residual + lambda=0.2 下，lstm_graphsage_edge 为当前综合最优；residual + lambda=0.2 下，hydraulic_inverse 反超，说明其更依赖残差特征。
已完成的关键消融显示，当前节点级类型头未带来稳定定位增益；建议保留 I/E 分类型分析，但不把类型头作为节点级主结论。
segment mode 已完成正式版闭环：segment_ie_full_v1 在 I/E 集合上达到 MRR=0.7094、Top-1=0.6246、Top-3=0.7885、Top-5=0.8011，type_macro_f1=0.3194；其中 I 类 Top-1=0.7910，E 类 Top-1=0.4375，说明边级线已经从“能跑通”进入“可纳入主报告比较”的阶段。
下一步重点不再是闭环本身，而是把 segment 结果系统化整理进总表，并继续做正式模型对比与 E 类增强。
原始总表见 CHAPTER1_NODE_EXPERIMENT_TABLE.json，segment 结果见 last_run_metrics_segment_segment_ie_full_v1.json 与 CHAPTER1_SEGMENT_SMOKE_SUMMARY.md。
