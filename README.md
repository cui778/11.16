# script2_new — 新路线（按职责整理的目录）

本目录是**新方法/新路线**的完整、自包含脚本集合；目录按**职责**重新规划，不再沿用 script2 的 `visualizations/缺陷矩阵` 等嵌套。  
数据仍使用项目根目录的 `input_data`、`input2`。

---

## 目录与文件列表

```
script2_new/
  config.py                 # 全局配置（路径、超参、特征等）
  README.md

  prep/                     # 数据与图准备（INP → 图、候选、缺陷矩阵、时序、残差）
    __init__.py
    inp_parser.py           # 统一 INP 解析（原 1_inp_parser.py）
    build_graph.py          # 构建 adj_matrix + node_list（原 script/ 与 script2 的 6_build_graph_parquet.py）
    build_candidates.py     # 候选节点集（原 build_candidate_nodes_from_inp.py）
    defect_matrix.py       # 缺陷矩阵生成（原 defect_matrix_generator_unified.py）
    extract_timeseries.py  # 时序提取（原 extract_with_fixed_time_intervals_v3.py）
    residual_features.py   # 残差特征（原 25_baseline_feature_engineering.py）

  dataset/                  # 数据集
    __init__.py
    processor.py            # 数据集构建与 DataLoader（原 22_dataset_processor_fixed_patched.py）

  train/                    # 训练与检查
    __init__.py
    train.py                # 训练入口（原 24_train_anomaly_detection_patched.py）
    check_labels.py        # 标签覆盖率检查（原 26_check_labels.py）

  models/                   # 模型定义
    __init__.py
    anomaly_detection_model.py

  utils/                    # 评估与指标
    __init__.py
    evaluation_patched.py
    metrics_patched.py

  model_checkpoints/        # 训练输出（自动创建）
```

---

## 路径说明

| 项 | 路径 |
|----|------|
| 数据根目录 | `E:\11.16`（config 中 `data_root`） |
| 本脚本根目录 | `E:\11.16\script2_new`（config 中 `script_root`） |
| 数据集处理器 | `script2_new/dataset/processor.py`（config 中 `dataset_processor_path`） |
| 模型保存 | `script2_new/model_checkpoints/` |

输入数据（INP、adj、node_list、defect_matrix、candidate_nodes 等）仍在 `input_data/`、`input2/`，与 script2 共用。

---

## 运行方式

**推荐先进入本目录再运行**，以便正确加载 `config.py` 和子模块：

```bash
cd E:\11.16\script2_new
```

### 数据准备（prep）

```bash
# 1. 构建图结构 → input2/adj_matrix.npy, node_list.json
python prep/build_graph.py

# 2. 候选节点 → input2/candidate_nodes_*.json
python prep/build_candidates.py --inp "E:\11.16\input_data\2_tuned_v3_merged1.inp" --defect_csv "E:\11.16\input2\defect_matrix_diverse.csv" --adj "E:\11.16\input2\adj_matrix.npy" --node_list "E:\11.16\input2\node_list.json" --k 50 --out_json "E:\11.16\input2\candidate_nodes_50.json" ...

# 3. 缺陷矩阵（若尚未生成）
python prep/defect_matrix.py --base-inp "E:\11.16\input_data\2_tuned_v3_merged1.inp" ...

# 4. 时序提取
python prep/extract_timeseries.py --base-inp "..." --defect-csv "..." --output-dir "E:\11.16\training_data_full" ...

# 5. 残差特征
python prep/residual_features.py
# 注意：__main__ 中输入/输出路径需按你当前数据目录修改
```

### 数据集与训练（dataset + train）

```bash
# 标签检查
python train/check_labels.py

# 训练
# 第4章/固定布局诊断模型训练入口
python script2_new/scripts/train_privileged_teacher_student.py \
  --student-model-type hydraulic_inverse_deepattn \
  --student-monitors script2_new/input_1/monitor_nodes_degree_N25.json \
  --split-mode scenario \
  --seed 42
```

从项目根目录运行也可（需能解析到 `script2_new` 的 config）：

```bash
cd E:\11.16
python script2_new/train/check_labels.py
python E:\11.16\script2_new\chapter5_layout_optimization\scripts\train_learnable_layout_network.py
```

---

## 主流程顺序

1. **prep/build_graph.py** → 生成 `input2/adj_matrix.npy`、`input2/node_list.json`
2. **prep/build_candidates.py** → 生成 `input2/candidate_nodes_*.json`
3. **prep/defect_matrix.py** → 生成 `input2/defect_matrix_*.csv`
4. **prep/extract_timeseries.py** → 生成 `training_data_*/node_timeseries.parquet`
5. **prep/residual_features.py** → 生成 `node_timeseries_with_residuals.parquet`
6. **train/check_labels.py** → 检查标签与覆盖率
7. # 第4章/固定布局诊断模型训练入口
python script2_new/scripts/train_privileged_teacher_student.py \
  --student-model-type hydraulic_inverse_deepattn \
  --student-monitors script2_new/input_1/monitor_nodes_degree_N25.json \
  --split-mode scenario \
  --seed 42

---

## 与旧版的对应关系

| script2_new | 原 script2 / script |
|-------------|----------------------|
| prep/inp_parser.py | visualizations/缺陷矩阵/1_inp_parser.py |
| prep/build_graph.py | script/6_build_graph_parquet.py 或 script2/.../6_build_graph_parquet.py |
| prep/build_candidates.py | visualizations/缺陷矩阵/build_candidate_nodes_from_inp.py |
| prep/defect_matrix.py | visualizations/缺陷矩阵/defect_matrix_generator_unified.py |
| prep/extract_timeseries.py | visualizations/缺陷矩阵/extract_with_fixed_time_intervals_v3.py |
| prep/residual_features.py | 25_baseline_feature_engineering.py |
| dataset/processor.py | 22_dataset_processor_fixed_patched.py |
| train/train.py | 24_train_anomaly_detection_patched.py |
| train/check_labels.py | 26_check_labels.py |

---

修改本目录内任意脚本不会影响 `script2`；配置只改本目录下的 `config.py` 即可。
