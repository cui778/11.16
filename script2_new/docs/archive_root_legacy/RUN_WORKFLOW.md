# script2_new 完整运行流程（B 路线）

建议在项目根或 `script2_new` 下执行，保证能正确加载 `config`。以下命令默认在 **`E:\11.16\script2_new`** 目录下执行。

---

## 第一章一键跑完（推荐一晚上挂机）

在**已安装 pyswmm、torch、pandas 等依赖**的 Python 环境下，在 `script2_new` 目录执行：

```bash
cd E:\11.16\script2_new
python run_chapter1.py
```

该脚本会依次：生成 full/random 监测集 → 各策略（degree/betweenness/downstream/random/full）提取时序+残差 → A4 observability（full-nodes→observability→25 节点）→ 各策略三种子 (42/7/123) 训练 → 汇总写入 `outputs/reports/CHAPTER1_RESULTS.json` 并追加 `STEP_RESULTS_LOG.md`。  
可选：`--skip-extract` 跳过提取与残差（用已有 parquet）；`--only-train` 仅训练；`--max-strategies N` 只跑前 N 个策略（调试用）。

---

## 前置条件

- 项目根目录存在 **`input_data/`**（至少包含 INP，如 `2_tuned_v3_merged1.inp`）。
- Python 环境已安装：`pandas`、`numpy`、`networkx`、`pyswmm`、`torch`、`pyarrow` 等（与 script2 一致）。
- **不依赖** 项目根下的 `input2/`、`training_data_time_gated/`；B 路线中间数据全部落在 **`script2_new/input_1/`** 和 **`script2_new/training_data_new/`**。

---

## 一、数据与图准备（顺序不可颠倒）

**依赖关系简要**：
- **baseline_flow_stats.json 必须先算**：跑一遍无缺陷 SWMM 得到节点/管道流量统计。  
  - **build_candidates 依赖它**：候选节点与原始流量相关，会按流量过滤（剔除近零流、末端低流）；出口（出流口）已在脚本中排除，不会进入候选集。  
  - **defect_matrix 依赖它**：用节点/管道 baseline 流量计算缺陷流量并选 E 类管道。  
- **监测集 build_monitors**（degree/betweenness/downstream）不依赖 baseline_flow_stats；observability 依赖后面的残差 parquet。

| 步骤 | 脚本 | 作用 | 产出 |
|------|------|------|------|
| 0 | `prep/defect_matrix.py --baseline-only` | **先跑**无缺陷 SWMM，写节点/管道流量统计 | `script2_new/input_1/baseline_flow_stats.json` |
| 1 | `prep/inp_parser.py` | 解析 INP | `script2_new/input_1/parsed_inp_data.json` |
| 2 | `prep/build_graph.py` | 建图 V | `script2_new/input_1/adj_matrix.npy`、`node_list.json` |
| 3 | `prep/build_candidates.py` | 生成候选集 C（**依赖 baseline**：按流量过滤，剔除近零流/末端低流；出口已排除） | `script2_new/input_1/candidate_nodes_new.json`（及可选 report） |
| 4 | `prep/build_monitors.py` | 生成监测集 S | `script2_new/input_1/monitor_nodes_{strategy}_N{n}.json` |
| 5 | `prep/defect_matrix.py` | 基于 C + baseline 统计生成缺陷矩阵 | `script2_new/input_1/defect_matrix_diverse.csv`（默认，可用 `--output-name` 改名） |
| 6 | `prep/extract_timeseries.py` | 按时序提取 | `script2_new/training_data_new/node_timeseries.parquet` 等 |
| 7 | `prep/residual_features.py` | 残差特征 | `script2_new/training_data_new/node_timeseries_with_residuals.parquet` |

### 0. 生成 baseline 流量统计（必须先做）

```bash
python prep/defect_matrix.py --baseline-only
```

跑一遍无缺陷 SWMM，在 `script2_new/input_1/` 下生成 `baseline_flow_stats.json`。  
- **build_candidates** 依赖该文件：候选节点与原始流量相关，会剔除近零流、末端低流节点，出口已在脚本中排除。  
- **defect_matrix** 生成缺陷矩阵时也依赖该文件。  
未先执行本步时，运行 build_candidates 会直接报错并提示先执行 `--baseline-only`。

### 1. 解析 INP

```bash
cd E:\11.16\script2_new
python prep/inp_parser.py
```

### 2. 构建图（得到 V）

```bash
python prep/build_graph.py
```

### 2.5. 构建图路径特征（hydraulic_inverse 模型专用，可选）

仅在使用 `hydraulic_inverse` 模型时需要，其他模型可跳过：

```bash
python prep/build_graph_features.py
```

产出：`input_1/graph_path_features.npz`（含最短距离、管道长度、高程差、流向等路径特征）。

### 3. 生成候选缺陷节点集 C

```bash
python prep/build_candidates.py
```

无参数时使用 config 默认（如 `--k 50`、输出 `candidate_nodes_new.json`）。需要时可加 `--adj`、`--node_list` 指向 `input_1` 下文件。

### 4. 生成监测节点集 S（chapter1 对比用）

**默认推荐**：度中心性（degree），分布均匀、覆盖主干交汇处。config 默认指向 `monitor_nodes_degree_N25.json`。

**不依赖时序**的策略（有图 + C 即可）：

```bash
python prep/build_monitors.py --strategy degree --n 25    # 推荐，默认使用
python prep/build_monitors.py --strategy betweenness --n 25
python prep/build_monitors.py --strategy downstream --n 25
```

**依赖时序+残差**的策略（需先完成步骤 5–7）：

```bash
python prep/build_monitors.py --strategy observability --n 25
```

**observability 所需数据**：
| 文件 | 来源 | 用途 |
|------|------|------|
| `node_timeseries_with_residuals.parquet` | 步骤 6 + 7 | 含 `depth_residual_rel` 等残差列，用于判断缺陷影响范围 |
| `defect_matrix_*.csv` | 步骤 5 | 若 parquet 无 `defect_node` 列，按 scenario_id 合并得到缺陷节点 |

**注意**：observability 需计算「每个候选缺陷影响哪些节点」，因此步骤 6 提取时序时**必须用全网节点**，加 `--full-nodes`。建议用独立输出目录避免覆盖：

```bash
python prep/extract_timeseries.py --full-nodes --time-gated --output-dir training_data_new/observability_full_nodes
python prep/residual_features.py --input-dir training_data_new/observability_full_nodes --output-dir training_data_new/observability_full_nodes
python prep/build_monitors.py --strategy observability --n 25
```

（observability 需指定 `--timeseries-path` 指向上述残差 parquet，或修改 config 的 `node_timeseries_file`。）

跑完 observability 后，若需为训练提取监测节点数据，可再跑一次 `extract_timeseries`（不加 `--full-nodes`，用默认 degree 监测节点，并指定 `--output-dir training_data_new/time_gated` 等）。

### 5. 生成缺陷矩阵（基于 C + baseline 统计）

需先完成 **步骤 0** 得到 `baseline_flow_stats.json`。然后：

```bash
python prep/defect_matrix.py
```

可加 `--output-name`、`--num-scenarios`、`--time-mode` 等。

### 6. 提取时序

**⚠️ 时间门控 vs 全程注入会产出同名文件，必须用不同 `--output-dir` 避免覆盖：**

| 模式 | 命令 | 输出目录 |
|------|------|----------|
| **全程注入**（缺陷整个仿真期间存在） | `python prep/extract_timeseries.py --output-dir training_data_new/full_injection` | `training_data_new/full_injection/node_timeseries.parquet` |
| **时间门控**（缺陷按 start_hour/duration_h 注入） | `python prep/extract_timeseries.py --time-gated --output-dir training_data_new/time_gated` | `training_data_new/time_gated/node_timeseries.parquet` |

```bash
# 全程注入（默认）
python prep/extract_timeseries.py --output-dir training_data_new/full_injection

# 时间门控
python prep/extract_timeseries.py --time-gated --output-dir training_data_new/time_gated
```

可加 `--max-scenarios`、`--full-nodes` 等。若已生成监测节点集 S，config 会自动读取，只提取监测节点数据。

### 7. 残差特征

残差脚本需指定与 extract 相同的目录，用 `--input-dir` / `--output-dir`：

```bash
# 全程注入数据
python prep/residual_features.py --input-dir training_data_new/full_injection --output-dir training_data_new/full_injection

# 时间门控数据
python prep/residual_features.py --input-dir training_data_new/time_gated --output-dir training_data_new/time_gated
```

不传参数时默认用 config 的 `training_data_dir`（兼容旧用法）。

---

## 二、检查与训练

| 步骤 | 脚本 | 作用 |
|------|------|------|
| 8 | `train/check_labels.py` | 检查标签与 DataLoader 是否正常 |
| 9 | `train/train.py` | 训练模型 |

### 8. 标签/数据检查（建议先跑）

```bash
python train/check_labels.py
```

确认无报错、batch 中 `target_node_idx`/`candidate_mask` 等符合预期后再训练。

### 9. 训练

训练前在 `config.py` 中设置 `training_data_subdir`，指向要用的数据子目录：

| 数据来源 | config 设置 |
|----------|-------------|
| 全程注入 | `training_data_subdir = "full_injection"` |
| 时间门控 | `training_data_subdir = "time_gated"` |
| 根目录（旧用法） | `training_data_subdir = ""` |

**常用命令行参数**（可覆盖 config，无需每次改文件）：

```bash
# 基础训练（使用 config 默认值）
python train/train.py

# 指定模型类型
python train/train.py --model-type gru_gcn
python train/train.py --model-type gru_only
python train/train.py --model-type hydraulic_inverse   # 需先跑步骤 2.5
python train/train.py --model-type time_mean_linear

# 指定数据子目录（覆盖 config.training_data_subdir）
python train/train.py --subdir time_gated
python train/train.py --subdir time_gated_downstream

# 指定随机种子
python train/train.py --seed 42
python train/train.py --seed 7
python train/train.py --seed 123

# 指定划分方式（scenario 默认；node_holdout 用于泛化评估）
python train/train.py --split-mode scenario
python train/train.py --split-mode node_holdout --n-holdout-nodes 10

# 命名本次运行（结果写入 outputs/reports/last_run_metrics_{run_name}.json）
python train/train.py --run-name gru_gcn_time_gated

# 组合示例：hydraulic_inverse + node_holdout + 命名
python train/train.py --model-type hydraulic_inverse --split-mode node_holdout --n-holdout-nodes 10 --seed 42 --run-name hydraulic_inverse_node_holdout
```

模型与日志落在 config 的 `model_save_dir`、`log_dir`（默认在 `script2_new/outputs/` 下）。

---

## 三、chapter1 多策略 S 的典型流程

若要做「不同 S 策略」的对比实验：

1. 完成步骤 1–3（图 + C）。
2. 对 **degree / betweenness / downstream** 各跑一次 `build_monitors.py`，得到多份 `monitor_nodes_*_N25.json`。
3. 完成步骤 5–7（缺陷矩阵 + 时序 + 残差）。
4. 再跑 **observability** 的 `build_monitors.py`。
5. 在 **dataset/processor** 与 **train** 中按当前实验选定一份 `monitor_nodes_*.json`（或通过 config 的 `monitor_nodes_file` 指向该文件）。
6. 运行 `check_labels.py` → `train.py`，记录该策略的指标。
7. 换另一份 `monitor_nodes_*.json`，重复 5–6。

（当前 dataset 与 config 若尚未接「指定某份 S 文件」的开关，可在 config 里增加 `monitor_nodes_file` 并在 processor 中读该文件，用于限定输入节点或采样。）

---

## 四、可视化

### 候选节点（C）与监测节点（S）分布

```bash
python visualizations/visualize_candidates_and_monitors.py
```

在管网拓扑上同时展示：候选缺陷节点集 C（橙色）、监测节点集 S（蓝色）、重叠节点 C∩S（红星）。加 `--save` 可保存为 PNG。

### 所有监测节点布置方案对比

```bash
# 仅可视化已有 monitor_nodes_*.json
python visualizations/visualize_all_monitor_strategies.py

# 先生成 degree / betweenness / downstream，再可视化
python visualizations/visualize_all_monitor_strategies.py --generate

# 包含 observability（需先完成 extract_timeseries + residual_features）
python visualizations/visualize_all_monitor_strategies.py --generate --include-observability

# 保存为 PNG
python visualizations/visualize_all_monitor_strategies.py --generate --save
```

并排展示各策略的监测点分布，便于对比选点逻辑。

### 缺陷影响范围热力图

```bash
# 需先完成 extract_timeseries + residual_features
python visualizations/defect_impact_heatmap.py

# 只画前 20 个场景（快速预览）
python visualizations/defect_impact_heatmap.py --max-scenarios 20
```

输出到 `visualizations/defect_impact_figures/`，红色越深表示该节点受缺陷影响越大。可用于评估：
- 缺陷影响是否主要向下游传播
- 边界入流点作为缺陷时影响范围是否异常
- 候选节点布置是否合理（避免过多边界入流点）

---

## 五、结果对比

### 多次运行结果对比

```bash
# 对比两次运行（读取 last_run_metrics_{name}.json）
python scripts/compare_model_runs.py gru_gcn_time_gated hydraulic_inverse_time_gated

# 对比三次运行
python scripts/compare_model_runs.py gru_gcn_node_holdout gru_only_node_holdout hydraulic_inverse_node_holdout

# 对比并追加到 STEP_RESULTS_LOG.md
python scripts/compare_model_runs.py gru_gcn_time_gated hydraulic_inverse_time_gated --append-log
```

### 组会汇报图生成

```bash
# 一键生成所有组会图（输出到 group_meeting_figures/outputs/）
python group_meeting_figures/generate_all_figures.py

# 单独生成某张图
python group_meeting_figures/plot_vcs_overview.py
python group_meeting_figures/plot_strategy_performance.py
python group_meeting_figures/plot_model_comparison.py
python group_meeting_figures/plot_b_experiments.py
```

---

## 六、路径与 config 速查

- **中间数据目录**：`config.input_dir_new` → 默认 **`script2_new/input_1`**。
- **时序/训练数据目录**：`config.training_data_dir` → 默认 **`script2_new/training_data_new`**。
- **时序数据子目录**：`config.training_data_subdir` → `"full_injection"` | `"time_gated"` | `""`，训练时 `node_timeseries_file` 指向 `training_data_dir/<subdir>/node_timeseries_with_residuals.parquet`。
- **只读输入**：INP 等来自 `data_root/input_data/`；不与 script2 共用 `input2`。

修改根目录或目录名时，改 `script2_new/config.py` 里的 `data_root`、`script_root` 即可。
> Warning
>
> This workflow file still documents historical commands and directory conventions from pre-clean stages.
> Before using any command here as a formal experiment command, check whether it points to old directories such as `time_gated`, `time_gated_downstream`, or old `node_holdout` assumptions.
> The current clean main line should be interpreted together with:
> - `process_diagnosis_revision_20260321/24_clean_scenario_rerun_summary_20260325.md`
> - `process_diagnosis_revision_20260321/28_project_alignment_checkpoint_20260325.md`
