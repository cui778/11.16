# 第4章候选节点机制分析

日期：2026-05-05

## 正式口径

- V：128 个全网节点，是图结构输入空间。
- S：25 个 degree 监测节点，是动态观测空间。
- C：50 个候选缺陷节点，是正式定位输出空间。
- D：正式 IE420 矩阵中的实际激活缺陷节点。

第4章主线是全图拓扑约束下的候选缺陷节点排序定位，不是 128 节点全网自由定位。

## 核验结果

- |V| = 128
- |S| = 25
- |C| = 50
- |D| = 50
- |S ∩ C| = 12
- |C ∩ D| = 50
- I 覆盖节点数 = 50
- E 覆盖节点数 = 34
- E 类未覆盖候选节点数 = 16
- 可观测性分层 = {'near': 21, 'far': 17, 'direct': 12}

## 边界探索

`C128` 与 `C50+Neg` 仅作为后续边界探索定义，不进入第4章正式主表。第5章继续固定 V 与 C，只优化 S。

## 输出文件

- `E:\11.16\script2_new\chapter4_diagnosis_model\outputs\candidate_space_relationships.csv`
- `E:\11.16\script2_new\chapter4_diagnosis_model\outputs\candidate_observability_tiers.csv`
- `E:\11.16\script2_new\chapter4_diagnosis_model\outputs\candidate_space_boundary_probe_definitions.csv`
