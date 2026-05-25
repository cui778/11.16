# 第4章场景级时间线诊断补充结果

日期：2026-05-05

## 1. 目的

本补充不修改模型结构、不重训模型，而是在现有滑动窗口模型输出之上恢复场景级时间线：

```text
完整场景 -> 滑动窗口 -> 每窗 p_active 与 node scores
-> 按 scenario_id + window_start_time 排序
-> 得到 p_active(t)
-> 预测缺陷开始时间、活跃区间和场景级缺陷节点
```

这一步用于检验当前窗口模型是否已经能够支持“缺陷时间线定位”。

## 2. 默认规则

- 模型：`best_model_ch1_fullgraph_degree_ie420_s42_fix1.pth`
- 数据：`time_gated_full_ie_v4_formal_conservative420_seed42`
- 阈值：`p_active > 0.5`
- 连续窗口要求：连续 2 个窗口超过阈值
- 预测开始时间：第一段连续 active 窗口的首窗口起点
- 预测结束时间：该连续段最后一个窗口终点
- 空间定位：只使用预测 active 窗口聚合节点分数
- 重要边界：不使用真实 active 标签筛选窗口

## 3. 主要结果

| 指标 | 数值 |
|---|---:|
| 测试事件数 | 63 |
| 缺陷事件数 | 63 |
| scene defect recall | 1.0000 |
| missed detection rate | 0.0000 |
| onset error mean | 9.7963 h |
| onset accuracy within 1 window | 0.1746 |
| onset accuracy within 2 windows | 0.3333 |
| active interval IoU mean | 0.3500 |
| duration error mean | 19.1772 h |
| false alarm before start rate | 0.8254 |
| scene node MRR | 0.0628 |
| scene node Top-1 | 0.0000 |
| scene node Top-3 | 0.0952 |
| scene node Top-5 | 0.1111 |

## 4. 解释

结果表明，现有窗口模型可以较容易地在测试缺陷场景中给出 active 响应，因此场景级“是否有缺陷”的召回为 1.0。但它并没有可靠地定位缺陷开始时间：

- 平均开始时间误差接近 9.8 小时。
- 仅 17.46% 的事件能在 ±1 个窗口内命中开始时间。
- 82.54% 的检测发生在真实开始时间之前。
- 使用预测 active 窗口聚合节点分数后，空间定位指标明显低于原窗口级/真实活跃期评价。

这说明当前 `active_label` 训练目标和现有 `time_pos` 输入尚不足以支撑严格的场景级缺陷开始时间定位。此前的窗口级 active recall 不能等同于真正的 onset localization。

## 5. 对论文口径的影响

当前第4章可以写：

> 模型已具备窗口级活跃状态识别和候选节点定位能力；进一步将窗口预测恢复为场景级时间线后发现，严格缺陷开始时间定位仍不稳定，是后续需要补强的时间诊断环节。

当前第4章不应写：

> 现有模型已经准确完成缺陷开始时间定位。

## 6. 后续建议

下一步若要真正提高时间线定位，应优先考虑：

1. 增加场景相对时间特征，而不仅是 `sin(hour), cos(hour)` 日内编码。
2. 调整 active 标签或损失，使模型区分真实注入前、过渡期、稳定活跃期。
3. 在场景级评价中调参阈值与连续窗口规则，而不是只看窗口级 active recall。
4. 重新检查 `always_on` 标签协议，不再将其视为正式全程注入对照。

## 7. 输出文件

- `E:\11.16\script2_new\outputs\reports\scene_timeline_ch1_s42_window_predictions.csv`
- `E:\11.16\script2_new\outputs\reports\scene_timeline_ch1_s42_event_predictions.csv`
- `E:\11.16\script2_new\outputs\reports\scene_timeline_ch1_s42_metrics.json`

