# 第5章当前协议补充说明

本文件修正 `CH5_PROTOCOL_FREEZE.md` 中仍保留的早期表述。发生冲突时，以本文件和 `chapter5_layout_optimization/RESULT_INDEX.md` 为准。

## 正式数据

```text
training/evaluation dataset:
  E:\11.16\script2_new\training_data_new\ie420_plus_normal20_v1

IE420 defect mother:
  E:\11.16\script2_new\training_data_new\
  time_gated_full_ie_v4_formal_conservative420_seed42

normal layer:
  E:\11.16\script2_new\training_data_new\
  normal_multibaseline_v2_seedset20
```

第5章正式评价必须包含 normal20，不再将 IE420 母数据单独表述为完整训练与评价集。

## 正式方法

N=25 主结果包含：

1. Degree；
2. Betweenness；
3. Cand-Obs；
4. Two-stage v1；
5. Node-Feedback；
6. Embedding-Guided。

random、downstream、identifiability-driven、Two-stage v2/v3、surrogate 和 overlap-controlled probe 属于历史基线、方法探索或机制实验，不进入当前六方法主表。

## 正式实验顺序

1. 固定第4章诊断模型和正式数据；
2. 比较 N=25 多 seed 主结果；
3. 分析布局结构、缺陷节点空间关系和方法差异；
4. 比较 N=5 至 N=25 的预算变化；
5. 通过覆盖率受控实验分析“缺陷节点邻近覆盖”与诊断性能的关系。
