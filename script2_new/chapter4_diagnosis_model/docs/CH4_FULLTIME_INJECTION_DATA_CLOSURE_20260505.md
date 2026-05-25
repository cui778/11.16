# 第4章全程注入数据闭环说明

日期：2026-05-05

## 1. 目的

此前 `always_on` 只是对 time-gated 数据修改标签，并不是真正的全程注入实验。为避免标签协议与物理仿真不一致，本轮新增真正的 full-time injection 矩阵：

```text
start_hour = 0
duration_h = 48 h
```

该矩阵应配合 `extract_timeseries.py` 在不使用 `--time-gated` 的情况下重新提取时序数据。

## 2. 已完成内容

已生成独立全程注入矩阵：

```text
E:\11.16\script2_new\input_1\defect_matrix_diverse_ie_v4_formal_conservative420_fulltime_seed42.csv
E:\11.16\script2_new\input_1\defect_matrix_diverse_ie_v4_formal_conservative420_fulltime_seed42.summary.json
```

该矩阵保持正式 IE420 的节点与强度覆盖不变，只改变时间协议：

| 项目 | 数值 |
|---|---:|
| 缺陷场景总数 | 420 |
| I 场景数 | 250 |
| E 场景数 | 170 |
| I 覆盖节点数 | 50 |
| E 覆盖节点数 | 34 |
| 总激活节点数 | 50 |
| start_hour | 0 |
| duration_h | 48 |

## 3. smoke 数据闭环

全 420 场景 SWMM 提取在本轮运行 1 小时后未完成，输出目录为空，因此已停止后台进程，避免继续占用资源。

为验证流程，本轮完成了 `max_scenarios=5` 的 smoke 数据闭环：

```text
E:\11.16\script2_new\training_data_new\fulltime_full_ie_v4_formal_conservative420_seed42_smoke5
```

该目录包含：

| 文件 | 状态 |
|---|---|
| `dataset_manifest.json` | 已生成 |
| `scenario_summary.csv` | 已生成 |
| `node_timeseries.parquet` | 已生成 |
| `node_timeseries_with_residuals.parquet` | 已生成 |

manifest 核验结果：

| 项目 | 数值 |
|---|---:|
| time_gated | false |
| scenario_count | 6 |
| unique_node_count | 128 |
| sample_interval_minutes | 10 |
| max_scenarios | 5 |

其中 6 个场景包括 baseline + 5 个 fulltime 缺陷场景。每个场景均有 287 个时间点和 36736 条节点-时间记录。

## 4. 当前边界

本轮已经完成：

- 真正 fulltime 矩阵生成。
- 独立 smoke 数据目录。
- 残差特征文件生成。
- 可训练性所需基本文件闭环。

本轮尚未完成：

- 全 420 缺陷场景 fulltime 数据提取。
- fulltime 模型训练。
- fulltime 与 time-gated 的正式性能对比。

因此，当前 fulltime 只能写成“数据闭环已打通，完整数据生成需作为长任务继续运行”，不能写成已经完成正式全程注入模型实验。

## 5. 后续命令

完整 fulltime 数据提取建议作为单独长任务运行：

```powershell
conda run -n swmm_gpu python E:\11.16\script2_new\prep\extract_timeseries.py `
  --defect-csv E:\11.16\script2_new\input_1\defect_matrix_diverse_ie_v4_formal_conservative420_fulltime_seed42.csv `
  --output-dir E:\11.16\script2_new\training_data_new\fulltime_full_ie_v4_formal_conservative420_seed42 `
  --full-nodes
```

随后运行残差特征：

```powershell
conda run -n swmm_gpu python E:\11.16\script2_new\prep\residual_features.py `
  --input-dir E:\11.16\script2_new\training_data_new\fulltime_full_ie_v4_formal_conservative420_seed42 `
  --output-dir E:\11.16\script2_new\training_data_new\fulltime_full_ie_v4_formal_conservative420_seed42
```

## 6. 论文口径

可以写：

> 本文进一步区分了 time-gated 标签协议与真正全程注入数据。当前正式实验采用 time-gated 注入；全程注入对照需基于重新生成的 fulltime 仿真数据，而不能仅通过修改窗口标签获得。

不能写：

> `always_on` 结果就是全程注入结果。

