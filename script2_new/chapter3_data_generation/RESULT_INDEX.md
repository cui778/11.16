# 第3章结果索引

## 正式研究对象

第3章验证 SWMM 生成的 I/E 缺陷数据是否具备合理的时间响应、空间传播和正常扰动边界。

## 正式原始资产

| 资产 | 路径 |
|---|---|
| SWMM INP | `E:\11.16\input_data\2_tuned_v3_merged1.inp` |
| 正式缺陷矩阵 | `E:\11.16\script2_new\input_1\defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv` |
| IE420 母数据 | `E:\11.16\script2_new\training_data_new\time_gated_full_ie_v4_formal_conservative420_seed42` |
| normal20 | `E:\11.16\script2_new\training_data_new\normal_multibaseline_v2_seedset20` |
| 正式组合 | `E:\11.16\script2_new\training_data_new\ie420_plus_normal20_v1` |
| 管网拓扑 | `E:\11.16\script2_new\input_1\parsed_inp_data.json` |

## 论文绘图

| 内容 | 路径 |
|---|---|
| 绘图脚本 | `E:\11.16\thesis_writing_repo\figures\scripts\ch3` |
| 轻量源表 | `E:\11.16\thesis_writing_repo\figures\ch3\source_data` |
| 正式图片 | `E:\11.16\thesis_writing_repo\figures\ch3\generated_results` |
| 图件索引 | `E:\11.16\thesis_writing_repo\figures\ch3\CH3_OFFICIAL_FIGURE_INDEX.md` |

第3章的残差曲线、空间图和动画会直接读取正式 Parquet，不可能只依靠 `source_data` 完整复现。

## 禁止口径

- persistent；
- fulltime；
- seedset10；
- legacy defect matrix；
- 自动搜索任意 `defect_matrix_*.csv`。
