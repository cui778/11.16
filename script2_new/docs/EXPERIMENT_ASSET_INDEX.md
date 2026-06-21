# 第3至第5章实验资产总索引

## 快速入口

| 章节 | 研究内容 | 实验结果入口 | 论文图件入口 |
|---|---|---|---|
| 第3章 | SWMM 缺陷数据生成与可观测性验证 | `chapter3_data_generation/RESULT_INDEX.md` | `thesis_writing_repo/figures/ch3` |
| 第4章 | 固定 Degree-N25 下的诊断模型 | `chapter4_diagnosis_model/RESULT_INDEX.md` | `thesis_writing_repo/figures/ch4` |
| 第5章 | 固定诊断协议下的布局优化 | `chapter5_layout_optimization/RESULT_INDEX.md` | `thesis_writing_repo/figures/ch5` |

## 唯一正式数据链

```text
formal_conservative420_seed42 defect matrix
        ↓
time_gated_full_ie_v4_formal_conservative420_seed42
        +
normal_multibaseline_v2_seedset20
        ↓
ie420_plus_normal20_v1
        ↓
Chapter 4 fixed-layout diagnosis
        ↓
Chapter 5 layout optimization
```

## 状态标签

| 标签 | 含义 |
|---|---|
| `formal` | 正文主结果 |
| `supplementary` | 补充分析或答辩材料 |
| `historical` | 历史调参依据 |
| `legacy` | 已被正式管线替代 |
| `debug` | smoke/debug |
| `invalid` | 协议错误或结果失效 |

## 整理规则

1. 正式文件保留在章节主目录；
2. 历史文件进入 `legacy_exploration` 或 `_legacy_*`；
3. 大型原始数据不复制到论文仓库；
4. `source_data` 必须能够追溯到原始来源；
5. 绘图脚本不得自动回退到历史数据；
6. 删除资产前先检查当前脚本和索引依赖。
