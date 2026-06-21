# input_1 输入资产说明

## 1. 目录用途

本目录只保留跨章节共享的轻量输入：管网结构、正式缺陷矩阵、候选缺陷空间和第4章固定监测布局。

## 2. 正式共享输入

| 文件 | 状态 | 用途 |
|---|---|---|
| `parsed_inp_data.json` | formal | SWMM 管网节点、管段与坐标 |
| `node_list.json` | formal | 全网节点顺序 |
| `adj_matrix.npy` | formal | 邻接矩阵 |
| `graph_path_features.npz` | formal | 最短跳数、管长、方向和高程差等路径先验 |
| `baseline_flow_stats.json` | formal | 基线流量统计 |
| `candidate_nodes_new.json` | formal | 50 个候选缺陷节点 |
| `candidate_nodes_50_report.csv` | formal | 候选缺陷节点审计表 |
| `defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv` | formal | IE420 正式缺陷矩阵 |
| `monitor_nodes_degree_N25.json` | formal | 第4章固定 Degree-N25 布局 |

## 3. 第5章布局文件

第5章不同方法和预算的布局不再以本目录为正式来源，统一从以下目录读取：

```text
E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts
```

## 4. 历史输入

`_legacy_historical_20260326/` 保存：

- 早期缺陷矩阵；
- curriculum、fulltime 和旧覆盖实验输入；
- 旧监测布局；
- 已被第5章独立布局目录替代的布局文件。

历史输入不得被正式脚本自动搜索或作为 fallback 使用。

## 5. 使用约束

正式脚本应显式指定输入文件，不得使用以下模糊逻辑：

```text
搜索第一个 defect_matrix_*.csv
搜索第一个 monitor_nodes_*.json
找不到正式文件后自动回退到 legacy 文件
```

正式第3至第5章共同使用的缺陷矩阵只能是：

```text
defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv
```
