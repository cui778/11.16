# 特征有效性分析报告

生成时间: 2026-03-18 21:20
数据目录: `E:\11.16\script2_new\training_data_new\full_injection`
分析场景数: 100

## 判断标准
- **ZERO**：残差绝对均值 < 1e-06（恒为0，完全无用）
- **WEAK**：残差绝对均值 < 0.001（信号极弱，建议删除）
- **SMALL**：残差绝对均值 < 0.01（信号较小，可选删除）
- **GOOD**：残差有明显变化，保留
- **DUPLICATE**：与另一特征残差相关系数 ≥ 0.9999（线性冗余）

## 各特征统计

| 特征 | 残差均值 | 残差std | 等级 | 建议 |
|------|---------|---------|------|------|
| `depth` | 0.001269 | 0.001833 | DUPLICATE | 删除 - 与其他特征线性冗余 |
| `head` | 0.001269 | 0.001833 | SMALL | 可选删除 - 信号较小 |
| `volume` | 0.000588 | 0.001028 | WEAK | 建议删除 - 信号极弱 |
| `lateral_inflow` | 0.160506 | 0.465822 | GOOD | 保留 |
| `total_inflow` | 0.994329 | 1.614315 | GOOD | 保留 |
| `flooding` | 0.000000 | 0.000000 | ZERO | 删除 - 残差恒为0 |
| `pollut_BODf` | 0.435345 | 1.000240 | GOOD | 保留 |
| `pollut_NH4` | 0.070364 | 0.153918 | GOOD | 保留 |
| `pollut_DO` | 0.005570 | 0.011056 | SMALL | 可选删除 - 信号较小 |

## 结论

**建议删除 5 个特征：**
- `depth` (DUPLICATE): 删除 - 与其他特征线性冗余
- `head` (SMALL): 可选删除 - 信号较小
- `volume` (WEAK): 建议删除 - 信号极弱
- `flooding` (ZERO): 删除 - 残差恒为0
- `pollut_DO` (SMALL): 可选删除 - 信号较小

**建议保留 4 个特征：**
- `lateral_inflow`
- `total_inflow`
- `pollut_BODf`
- `pollut_NH4`

## 推荐特征列表（可直接用于训练命令）

```
# 原始特征（保留）：
['lateral_inflow', 'total_inflow', 'pollut_BODf', 'pollut_NH4']

# 对应残差特征（如已计算）：
['lateral_inflow_residual', 'total_inflow_residual', 'pollut_BODf_residual', 'pollut_NH4_residual']
```