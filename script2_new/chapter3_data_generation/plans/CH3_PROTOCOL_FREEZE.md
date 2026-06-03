# 第3章协议冻结文档

## 正式数据生成管线

```text
SWMM 基线模型 (2_tuned_v3_merged1.inp)
  → IE420 time-gated 缺陷矩阵 (defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv)
  → PySWMM 批量仿真
  → time_gated_full_ie_v4_formal_conservative420_seed42 (421 场景)

SWMM 基线模型
  → normal_multibaseline_v2_seedset20 (20 个无缺陷正常扰动场景)

ie420_plus_normal20_v1 = IE420(421) + normal20(20) = 441 场景
```

## 正式文件路径

| 项目 | 路径 |
|---|---|
| SWMM 模型文件 | `E:\11.16\input_data\2_tuned_v3_merged1.inp` (1.8 MB) |
| IE420 缺陷矩阵 | `E:\11.16\script2_new\input_1\defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv` |
| IE420 母数据 | `E:\11.16\script2_new\training_data_new\time_gated_full_ie_v4_formal_conservative420_seed42` |
| normal20 数据 | `E:\11.16\script2_new\training_data_new\normal_multibaseline_v2_seedset20` |
| 正式训练组合 | `E:\11.16\script2_new\training_data_new\ie420_plus_normal20_v1` |

## 统计口径

```text
IE420 母数据：
  1 个正常参照场景 (BASELINE, scenario_id=0)
  + 250 个 I 类缺陷场景
  + 170 个 E 类缺陷场景
  = 421 场景

normal20：
  20 个无缺陷正常扰动场景 (seeds: 42,101,202,303,404,505,606,707,808,909,1001,1102,1203,1304,1405,1506,1607,1708,1809,1910)
  扰动参数：flow_noise=6%, quality_noise=4%, daily_variation=3%

正式组合：
  421 + 20 = 441 场景
  其中 persistent_count = 0

每场景：
  128 节点 × 287 有效时刻 = 36736 条节点记录
  采样间隔：10 min
  时间窗口：48 h（2025-01-01 00:10:00 至 2025-01-02 23:50:00）
```

## 关键设计说明

1. **正常场景与缺陷场景处于同一层级。** 正常场景不是"背景噪声"，而是独立的诊断类别。
2. **其中一个正常响应作为 residual 对齐参照。** IE420 母数据中 scenario_id=0 的 BASELINE 场景用于计算残差特征。
3. **48 h 是连续观测窗口，不是缺陷寿命。** 缺陷在窗口内的 start_hour 和 duration_h 由缺陷矩阵定义。
4. **time-gated 协议。** 缺陷仅在 [start_hour, start_hour+duration_h) 时间段内注入，其余时间与正常工况一致。

## 缺陷矩阵统计

| 字段 | 值 |
|---|---|
| 总行数 | 420 |
| I 类数量 | 250 |
| E 类数量 | 170 |
| 唯一缺陷节点数 | 50 |
| 强度范围 | 40%–60% |
| 持续时间范围 | 12–18 h |
| 开始时间范围 | 2–26 h |
| 列名 | defect_id, defect_type, node_id, link_id, intensity_pct, start_hour, duration_h, flow, BODf, NH4, DO, baseline_flow |

## 不属于正式管线的数据集

以下数据集属于探索或调试，不进入正式论文主线：

| 数据集 | 状态 | 原因 |
|---|---|---|
| `ie420_plus_normal_multibaseline_v1_seedset10` | frozen_legacy | 旧 seedset10，已被 normal20 替代 |
| `ie420_plus_persistent84_plus_normal20_v1` | frozen_legacy | 包含 persistent 缺陷，非正式协议 |
| `normal_multibaseline_v1_seedset10` | frozen_legacy | 旧 normal 层，仅 10 场景 |
| `persistent_ie_fullwindow_v1_seed42` | frozen_legacy | persistent 缺陷探索 |
| `fulltime_full_ie_v4_formal_conservative420_seed42` | frozen_legacy | 目录为空，未生成 |
| `fulltime_full_ie_v4_formal_conservative420_seed42_smoke5` | debug | fulltime smoke 测试 |
