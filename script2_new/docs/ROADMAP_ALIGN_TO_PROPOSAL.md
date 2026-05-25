# 将当前实现对齐开题报告的分步修改路线图

目标：把现有「候选节点定位 + 图级 I/E/P 辅助头」改成开题中的「**疑似管段的空间定位 + I/E 类型判别**」，并采用 **TCN/LSTM + GraphSAGE + 边表示**，输入包含**边向等效流量/流速、上下游差分、通量、直径坡度材质**等。

---

## 总体依赖关系（建议顺序）

```
Phase 1 任务与数据定义 → Phase 2 边/管段级特征与样本 → Phase 3 模型（时序+空间+边表示）→ Phase 4 训练与评估
```

每一步都依赖前一步产出；建议按 Phase 1 → 2 → 3 → 4 顺序执行，每阶段做完再进入下一阶段。

---

## Phase 1：任务与数据定义（管段 = 边/管道）

### 1.1 明确「管段」与候选空间

- **管段定义**：与开题一致时，**管段 = 有向管道（link）**。  
  - I 类（渗入）：可挂在「该管道上游节点」或「该管道」上，需约定一种（见下）。  
  - E 类（渗漏）：缺陷在管道上，天然是「某条 link」。  
- **候选管段集**：  
  - 方案 A：候选空间 = **候选管道集合**（如从 `build_candidates` 或 INP 得到 link 列表，再按长度/流量等筛一批）。  
  - 方案 B：保留「候选节点」概念，但**每个样本的定位目标改为「缺陷所在管段」**：I/P 对应「该节点某条入边或出边」（需约定取哪条边，例如主边或按流量最大），E 对应 link_id。  
- **建议**：先采用**方案 B**，即缺陷矩阵仍用 node_id / link_id，但在数据集层增加「**segment_id → 边 (from_node, to_node) 或 link_id**」的映射；这样无需大改缺陷矩阵生成逻辑，只改「谁作为定位目标」。

**要改的文件/配置：**

- 新建或扩展：`prep/build_segments.py`（或等价脚本）  
  - 输入：`node_list.json`、`adj_matrix.npy`、`parsed_inp_data.json`（或 INP 解析结果）、可选 `candidate_nodes`。  
  - 输出：`segment_list.json`（或 csv）：每行一条「管段」，字段至少包含 `segment_id, link_id, from_node, to_node`；若 I 挂在节点上，可增加 `primary_link_for_node` 等。  
- 缺陷矩阵：已有 `node_id` / `link_id`，保持不变；在 **dataset 构建时** 用 1.2 的规则把「缺陷位置」转成 `target_segment_idx`。

### 1.2 缺陷类型只做 I/E（可选：P 合并或单独）

- 开题写的是「I/E 类型判别」。若要严格对齐：  
  - **训练/评估时** 只使用缺陷类型为 I 和 E 的场景（或对 P 做单独分支，但主任务写 I/E）。  
- 在 `defect_matrix` 或 `DefectDetectionDataset` 的过滤逻辑里：  
  - 增加「仅 I/E」模式（例如 `defect_types_for_task = ['I','E']`），过滤掉 P 或把 P 映射到 I。  
- **要改**：`config.py` 增加开关；`dataset/processor.py` 在读取 defect_info 后按该开关过滤/映射场景。

### 1.3 数据管线：持久化管道时序

- 当前 `extract_timeseries.py` 在 `run_single_scenario_fixed_time` 里已经提取了 **link 级** `link_record`（flow, depth, velocity），但主流程只合并并保存了 **node** 数据，**link 数据被丢弃**。  
- **步骤**：  
  1. 在 `simulate_and_extract_timeseries_fixed_time` 中：  
     - 维护 `all_link_data = []`，每场景的 `link_data` append 进去。  
     - 最后 `df_links_all = pd.concat(all_link_data)`，保存为 `link_timeseries.parquet`（与 `node_timeseries.parquet` 同目录或同 subdir）。  
  2. 保证与 node 一致：同一 scenario_id、同一 datetime 对齐，便于后面做「上下游差分」等。  

**要改的文件：**

- `prep/extract_timeseries.py`：在循环里收集 `link_data`；合并后写 `link_timeseries.parquet`；返回或内部使用 `(df_nodes_all, df_links_all, df_summary)`。

---

## Phase 2：边/管段级特征与样本

### 2.1 边静态属性（直径、坡度、长度、材质）

- 从 INP 或 `prep/build_candidates.py` 的 `links_df` / `parsed_inp_data` 中已有：`length_m`、`diameter_m` 等。  
- **步骤**：  
  - 在 `build_segments` 或新建 `prep/build_edge_static_features.py` 中，为每条 segment（link）生成静态向量：`[length, diameter, slope, material_id]`（若无材质可省或填 0）。  
  - 输出：`edge_static_features.npy` 或按 segment_list 顺序的 csv，供 dataset 按 `segment_id` 读取。  
- **要改/新增**：  
  - `prep/build_candidates.py` 或 `prep/build_graph.py` 中已有 conduit 解析，可复用；  
  - 新建 `prep/build_edge_static_features.py`，输出与 `segment_list` 一一对应的静态特征矩阵。

### 2.2 边动态特征（等效流量/流速、上下游差分、通量）

- **等效流量/流速**：管道时序里已有 `flow`、`velocity`，可直接用；若需「等效」可再按截面换算（当前已有 velocity 即可先不做）。  
- **上下游差分**：对每条边 (u, v)，每个时间步：  
  - 从 node 时序取 `depth_u`, `depth_v`（及可选 inflow 等），计算 `diff_depth = depth_u - depth_v`，同理可做 `diff_*`。  
- **通量**：可用 `flow` 或 `velocity * 截面` 等，与开题表述一致即可。  
- **步骤**：  
  - 在 **残差阶段** 或新建「边特征工程」脚本：输入 `node_timeseries.parquet` + `link_timeseries.parquet` + baseline，按 (scenario_id, datetime) 对齐；  
  - 对每条 link，每时间步生成：`[flow, velocity, depth_up - depth_down, flow_residual, velocity_residual, ...]`；  
  - 输出：`edge_timeseries.parquet`（列：datetime, scenario_id, link_id/segment_id, flow, velocity, diff_depth, flow_residual, ...）或按 (T, num_edges, F_edge) 的 npy（需与 segment 顺序一致）。  
- **要改/新增**：  
  - `prep/residual_features.py` 扩展：在 node residual 之外，增加 link 的 baseline 与 residual、以及上下游差分计算（需 link 与 node 的对应关系 from_node/to_node）；  
  - 或新建 `prep/edge_feature_engineering.py`，读入 node + link parquet，写出 edge 时序（含残差与差分）。

### 2.3 样本从「节点窗口」改为「管段窗口」

- 当前样本：每个样本 = (scenario_id, 时间窗口)，标签 = `target_node_idx`（在候选节点中的哪个节点）。  
- 目标样本：每个样本 = (scenario_id, 时间窗口)，标签 = `target_segment_idx`（在候选管段中的哪条管段），且类型标签为 I 或 E。  
- **步骤**：  
  1. **Segment 列表与候选**：加载 `segment_list.json`，候选管段集 = 全部或按规则子集（如只保留「与候选节点相邻的边」）。  
  2. **标签**：对每个 (scenario_id, 时间窗口)，根据 defect_info：  
     - 若 defect_type == 'E'，defect 在 link_id → 映射为 `target_segment_idx`；  
     - 若 defect_type == 'I'（或 P），defect 在 node_id → 按 1.1 约定映射到一条 segment（如该节点的主入边或主出边），得到 `target_segment_idx`。  
  3. **特征**：每个样本包含：  
     - **节点时序** `[T, N_node, F_node]`（保留，供 GraphSAGE 节点聚合）；  
     - **边时序** `[T, N_edge, F_edge]`（新建，F_edge = 动态 + 可拼接静态）；  
     - 图结构：邻接矩阵或 edge_index（节点级），以及「边-节点」关系（每条边 (u,v) 对应哪两个节点）。  
- **要改的文件**：  
  - `dataset/processor.py`：  
    - 增加「管段模式」：读入 `segment_list`、`edge_timeseries`（或 edge 特征 npy）、`edge_static_features`；  
    - 样本字段增加 `edge_features`、`target_segment_idx`、`candidate_segment_mask`；  
    - 若仍用节点级图，保留 `adj_matrix`；同时提供 `edge_index`（2 x E）和 `segment_id_to_edge_idx` 等，供模型使用。

### 2.4 仅 I/E 时的数据过滤

- 在 `create_dataloaders` 或 `_build_dataset` 中：若 `defect_types_for_task == ['I','E']`，则只保留 defect_type 为 I 或 E 的场景的窗口，且类型标签改为二分类（I=0, E=1）。

---

## Phase 3：模型结构（TCN/LSTM + GraphSAGE + 边表示）

### 3.1 时间模块：GRU → TCN 或 LSTM

- **TCN**：可引入 `torch.nn` 或第三方 TCN 实现，输入 `[B*N, T, F]` 或 `[B*E, T, F]`，输出同长度时序的隐藏表示，再取最后时间步或做 pooling 得到每个节点/边的向量。  
- **LSTM**：将当前 `time_encoder` 从 GRU 换成 LSTM（`nn.LSTM`），接口保持（input/output shape 一致），便于 ablation。  
- **步骤**：  
  - 在 `models/anomaly_detection_model.py` 中新增 `TimeEncoderTCN` 和 `TimeEncoderLSTM`（或通过 `model_type` 切换）；  
  - 配置项：`time_encoder_type: 'gru' | 'lstm' | 'tcn'`。  
- **要改**：`config.py` 增加 `time_encoder_type`；`models/anomaly_detection_model.py` 实现 TCN/LSTM 分支并在 `create_model` 里根据 config 选择。

### 3.2 空间模块：GCN → GraphSAGE（节点聚合），再得到边表示

- **GraphSAGE**：对**节点**做采样邻域聚合（如 2 层），得到每个节点的表征 h_u；  
  - 边 (u,v) 的表示 = 拼接上下游节点表征 + 边自身特征：  
  - `e_uv = concat(h_u, h_v, x_edge_uv)`，其中 `x_edge_uv` 来自 2.2 的边动态特征（经时序编码后的向量）和 2.1 的静态特征。  
- **步骤**：  
  - 实现 GraphSAGE 层（采样邻居、聚合、归一化），替换当前 GCN 的「全邻接矩阵乘」；  
  - 前向：先对节点做时间编码 → 得到 node features；再对节点做 GraphSAGE 得到 node embeddings；再按 edge_index 拼成 edge 表示 `e_uv`。  
- **要改**：`models/anomaly_detection_model.py`：  
  - 新增 `GraphSAGELayer` 或使用 PyG 的 `SAGEConv`；  
  - 新增模型类（如 `SegmentLocalizationModel`）：时间编码(节点+边) → 节点 GraphSAGE → 边表示拼接 → 边级分类/排序。

### 3.3 边表示作为主干的输出

- **定位头**：对「候选管段」做分类或排序：输入为边表示 `[B, E, D]`，输出每个边的「是否异常」概率或得分，再在候选段上做 softmax/排序得到 MRR、Top-k。  
- **类型头（I/E）**：对**当前样本**预测缺陷类型 I 或 E（二分类）；可对「预测为最可能缺陷的那条边」再预测类型，或图级池化后预测（与开题「联合输出」一致即可）。  
- **步骤**：  
  - 模型输出：`logits_segment`（或 scores_segment）形状 `[B, num_edges]`，以及 `logits_type` 形状 `[B, 2]`（I/E）；  
  - 训练时：定位损失（如 CE 或 listwise ranking）+ 类型分类损失，**两者都作为主任务**（权重可 1:1 或可配置）。  
- **要改**：`models/anomaly_detection_model.py` 中新模型的 forward 返回 `(logits_segment, logits_type)`；`train/train.py` 中根据新输出计算两个 loss 并相加。

### 3.4 输入张量形状与 DataLoader

- 模型输入需包含：  
  - 节点时序 `x_node`: `[B, T, N_node, F_node]`；  
  - 边时序/边特征 `x_edge`: `[B, T, N_edge, F_edge]` 或先做时序编码再给模型 `[B, N_edge, D_edge]`；  
  - 邻接/边表：`edge_index` `[2, num_edges]`（节点索引），以及可选 `batch_edge_index`。  
- DataLoader 的 batch 里需提供上述键；`processor.py` 的 `__getitem__` 和 `collate_fn` 需对应修改。

---

## Phase 4：训练与评估

### 4.1 损失与权重

- **定位损失**：在候选管段上的 CE 或 KL（与当前节点级类似），或 listwise ranking loss。  
- **类型损失**：I/E 二分类 CE，仅对有缺陷样本（has_defect=1）计算。  
- 总损失：`L = L_loc + λ_type * L_type`，λ_type 建议从 1.0 起调。  
- **要改**：`train/train.py` 中前向得到 `logits_segment`、`logits_type`，计算两个 loss，相加后反传。

### 4.2 评估指标（与开题一致）

- **定位**：PR-AUC、F1、Top-k 命中、最小拓扑距离（在**管段图**上定义距离，如边与边之间跳数）。  
- **类型**：I/E 二分类的 AUC、宏 F1。  
- **要改**：`utils/evaluation.py` 增加「管段模式」：收集 `target_segment_idx`、`pred_segment_rank`、`true_type`、`pred_type`，汇总上述指标；`utils/metrics.py` 可增加边-边拓扑距离（若用线图或邻接边定义）。

### 4.3 可解释展示（可选）

- 开题提到的「典型案例的证据链：上下游差分、稀释/通量线索、邻域关注」：  
  - 可在预测阶段保存「预测缺陷边」的上下游节点、该边的 flow/velocity/diff 等，写成短文或图表；  
  - 可用 GraphSAGE 或边表示的注意力权重做简单可视化（若后续加 attention）。

---

## 简要检查清单（按顺序做）

| 顺序 | 内容 | 主要改动位置 |
|------|------|--------------|
| 1 | 管段/候选定义与 segment_list | 新建 `prep/build_segments.py`，输出 segment_list |
| 2 | 缺陷类型仅 I/E（可选） | config + dataset 过滤/映射 |
| 3 | 持久化管道时序 | `prep/extract_timeseries.py` 保存 link_timeseries.parquet |
| 4 | 边静态特征 | `prep/build_edge_static_features.py` 或扩展现有 build |
| 5 | 边动态特征（流量/流速/差分/残差） | 扩展 `prep/residual_features.py` 或新建 edge 特征脚本 |
| 6 | 样本改为管段窗口 + target_segment_idx | `dataset/processor.py` 管段模式 |
| 7 | 时间编码 TCN/LSTM | `models/anomaly_detection_model.py` + config |
| 8 | 空间 GraphSAGE + 边表示 | 同上，新模型类 |
| 9 | 定位头 + I/E 类型头（双主任务） | 模型 forward + train 双 loss |
| 10 | 评估：PR-AUC/F1/Top-k/拓扑距离 + I/E AUC/宏F1 | `utils/evaluation.py` + metrics |

---

## 注意事项

- **先不删现有节点级流程**：建议保留现有 `DefectDetectionDataset` 和 GRU+GCN/hydraulic_inverse，用**新配置或新 Dataset 类**（如 `SegmentDefectDataset`）做管段分支，便于对比和回退。  
- **E 类缺陷当前是 link_id**：缺陷矩阵里 E 已是 link，只需在 dataset 里把 link_id 映射到 segment_idx；I 类需要你约定「节点→主边」的规则（例如该节点最大流量的一条出边）。  
- **P 类**：若严格对齐开题只做 I/E，P 可不参与训练，或合并进 I 作为「入流类」再单独评估。

按上述顺序，每一步只做该步所列修改，做完再跑通数据/训练/评估，再进入下一步，即可逐步把实现对齐到开题报告描述。
