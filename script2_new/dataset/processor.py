# -*- coding: utf-8 -*-
"""
数据集处理器 - 优化版本。来源：原 script2/22_dataset_processor_fixed_patched.py
主要优化: 快速样本创建 (比原版快10-50倍)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
import time
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_soft_labels(adj_matrix: np.ndarray, target_idx: int, sigma: float = 1.0) -> np.ndarray:
    """根据图距离生成高斯平滑的软标签（返回长度为N的概率分布）。"""
    num_nodes = int(adj_matrix.shape[0])
    try:
        import networkx as nx
        G = nx.from_numpy_array((adj_matrix > 0).astype(np.int32))
        lengths = nx.single_source_shortest_path_length(G, int(target_idx))
    except Exception:
        labels = np.zeros(num_nodes, dtype=np.float32)
        if 0 <= int(target_idx) < num_nodes:
            labels[int(target_idx)] = 1.0
        return labels

    soft = np.zeros(num_nodes, dtype=np.float32)
    sigma = float(max(sigma, 1e-6))
    for node_idx, dist in lengths.items():
        soft[int(node_idx)] = float(np.exp(- (float(dist) ** 2) / (2.0 * sigma ** 2)))

    s = float(soft.sum())
    if s <= 0:
        soft = np.zeros(num_nodes, dtype=np.float32)
        soft[int(target_idx)] = 1.0
        return soft
    return soft / s


class DefectDetectionDataset(Dataset):
    """缺陷检测数据集 - 优化版本"""

    def __init__(self,
                 node_timeseries_file: str,
                 adjacency_matrix_file: str,
                 node_list_file: str,
                 defect_matrix_file: str,
                 sequence_length: int = 30,
                 window_stride: int = 1,
                 cls_overlap_threshold: float = 0.1,
                 loc_overlap_threshold: float = 0.5,
                 normalize: bool = True,
                 random_seed: int = 42,
                 feature_names: Optional[List[str]] = None,
                 candidate_nodes_file: Optional[str] = None,
                 soft_label_sigma: float = 1.0,
                 signal_threshold: Optional[float] = None,
                 label_mode: str = "auto",
                 overlap_threshold: Optional[float] = None,
                 always_on_force_target: bool = True,
                 use_time_pos_encoding: bool = True,
                 use_observed_mask_feature: bool = True,
                 observed_nodes_file: Optional[str] = None,
                 use_trend_feature: bool = False,
                 active_overlap_threshold: float = 0.5,
                 transition_overlap_threshold: float = 0.0,
                 active_signal_threshold: Optional[float] = None,
                 transition_signal_threshold: Optional[float] = None):

        self.sequence_length = sequence_length
        self.window_stride = max(int(window_stride), 1)
        self.cls_overlap_threshold = float(cls_overlap_threshold)
        self.loc_overlap_threshold = float(loc_overlap_threshold)
        self.normalize = normalize
        self.random_seed = random_seed
        self.feature_cols = None
        self.candidate_nodes_file = candidate_nodes_file
        self.candidate_nodes = None  # 将在_build_dataset中加载
        self.soft_label_sigma = float(soft_label_sigma)  # ✅ 新增：软标签平滑参数
        self.signal_threshold = 0.0 if signal_threshold is None else float(signal_threshold)
        self.label_mode = str(label_mode).lower()
        self.overlap_threshold = float(loc_overlap_threshold if overlap_threshold is None else overlap_threshold)
        self.always_on_force_target = bool(always_on_force_target)
        self.use_time_pos_encoding = bool(use_time_pos_encoding)
        self.use_observed_mask_feature = bool(use_observed_mask_feature)
        self.observed_nodes_file = observed_nodes_file
        self.use_trend_feature = bool(use_trend_feature)
        self.active_overlap_threshold = float(active_overlap_threshold)
        self.transition_overlap_threshold = float(transition_overlap_threshold)
        self.active_signal_threshold = None if active_signal_threshold is None else float(active_signal_threshold)
        self.transition_signal_threshold = None if transition_signal_threshold is None else float(transition_signal_threshold)
        self.base_feature_cols = None
        self.input_feature_dim = 0
        self.observed_node_ids = set()
        self.observed_node_mask_np = None

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        print("=" * 80)
        print("缺陷检测数据集 - 优化版本 (快速样本创建)")
        print("=" * 80)

        # 构建数据集
        self._build_dataset(
            node_timeseries_file, adjacency_matrix_file,
            node_list_file, defect_matrix_file, feature_names, candidate_nodes_file
        )

    def _build_dataset(self, node_timeseries_file, adjacency_matrix_file,
                       node_list_file, defect_matrix_file, feature_names, candidate_nodes_file=None):
        """构建数据集 - 优化版本"""

        start_time = time.time()

        # . 读取时序数据
        print("\n[/7] 读取时序数据...")
        if node_timeseries_file.endswith('.parquet'):
            try:
                self.node_data = pd.read_parquet(node_timeseries_file)
                print("  [OK] 使用pandas读取parquet文件")
            except AttributeError:
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(node_timeseries_file, use_threads=False)
                    self.node_data = table.to_pandas()
                    print("  [OK] 使用pyarrow读取parquet文件")
                except (AttributeError, ImportError, Exception) as e:
                    raise FileNotFoundError(
                        f"无法读取parquet文件 {node_timeseries_file}\n"
                        "建议解决方案:\n"
                        ". 升级pandas: pip install --upgrade pandas\n"
                        "2. 安装pyarrow: pip install pyarrow\n"
                        "3. 手动将parquet转换为CSV格式"
                    )
        else:
            self.node_data = pd.read_csv(node_timeseries_file)

        self.node_data['datetime'] = pd.to_datetime(self.node_data['datetime'])
        print(f"[OK] 数据: {self.node_data.shape[0]:,} 条记录")

        # 2. 读取缺陷矩阵
        print("\n[2/7] 读取缺陷定义...")
        self.defect_matrix = pd.read_csv(defect_matrix_file)
        self.defect_info = {}
        for _, row in self.defect_matrix.iterrows():
            scenario_id = int(row['defect_id'])
            self.defect_info[scenario_id] = {
                'defect_type': row['defect_type'],
                'node_id': str(row.get('node_id', '')),
                'link_id': str(row.get('link_id', '')),
                # ✅ 时间动态缺陷：若缺陷矩阵提供 start_hour/duration_h，则用于窗口级标签
                'start_hour': row.get('start_hour', None),
                'duration_h': row.get('duration_h', None),
            }
        scenario_ids_in_data = set(pd.to_numeric(self.node_data['scenario_id'], errors='coerce').dropna().astype(int).tolist())
        if 0 in scenario_ids_in_data and 0 not in self.defect_info:
            self.defect_info[0] = {
                'defect_type': 'BASELINE',
                'node_id': '',
                'link_id': '',
                'start_hour': None,
                'duration_h': None,
            }
            print("[OK] 已补充 baseline 场景定义: scenario_id=0")
        print(f"[OK] 缺陷定义: {len(self.defect_info)} 个场景")

        # 3. 加载图结构
        print("\n[3/7] 加载图结构...")
        self.adj_matrix = np.load(adjacency_matrix_file)
        with open(node_list_file, 'r') as f:
            node_info = json.load(f)
            if isinstance(node_info, dict):
                self.node_list = [str(n) for n in node_info.get('node_list', node_info)]
                self.node_to_idx = {str(k): v for k, v in
                                    node_info.get('node_to_idx', {}).items()}
            else:
                self.node_list = [str(n) for n in node_info]
                self.node_to_idx = {str(n): i for i, n in enumerate(self.node_list)}
        print(f"[OK] 节点: {len(self.node_list)} 个")
        print(f"[OK] 邻接矩阵: {self.adj_matrix.shape}")

        observed_node_ids = None
        if self.observed_nodes_file and os.path.exists(self.observed_nodes_file):
            try:
                with open(self.observed_nodes_file, 'r', encoding='utf-8') as f:
                    observed_data = json.load(f)
                if isinstance(observed_data, list):
                    observed_raw = observed_data
                elif isinstance(observed_data, dict):
                    observed_raw = []
                    for key in ("monitor_nodes", "nodes", "observed_nodes"):
                        if key in observed_data and isinstance(observed_data[key], list):
                            observed_raw = observed_data[key]
                            break
                else:
                    observed_raw = []
                observed_node_ids = {str(node_id) for node_id in observed_raw}
                print(f"[OK] 使用外部 observed 节点文件: {self.observed_nodes_file}")
                print(f"     原始 observed 节点数: {len(observed_node_ids)}")
            except Exception as e:
                logger.warning(f"加载 observed_nodes_file 失败: {e}，将回退到数据内 observed 节点")
                observed_node_ids = None

        if observed_node_ids is None:
            observed_node_ids = {
                str(node_id)
                for node_id in self.node_data.get('node_id', pd.Series(dtype=str)).astype(str).unique()
            }
        self.observed_node_ids = {node_id for node_id in observed_node_ids if node_id in self.node_to_idx}
        self.observed_node_mask_np = np.zeros(len(self.node_list), dtype=np.float32)
        for node_id in self.observed_node_ids:
            self.observed_node_mask_np[self.node_to_idx[node_id]] = 1.0
        print(f"[OK] observed_mask: {int(self.observed_node_mask_np.sum())}/{len(self.node_list)}")

        self.soft_labels_cache = {}
        unique_defect_indices = set()
        for _, info in self.defect_info.items():
            node_id = str(info.get('node_id', ''))
            if node_id and (node_id in self.node_to_idx):
                unique_defect_indices.add(int(self.node_to_idx[node_id]))

        for defect_idx in unique_defect_indices:
            self.soft_labels_cache[int(defect_idx)] = get_soft_labels(
                self.adj_matrix, int(defect_idx), sigma=self.soft_label_sigma
            )

        # 4. 加载候选节点（如果提供）
        print("\n[4/7] 加载候选节点...")
        self.candidate_nodes = None  # 保留兼容：候选节点ID集合
        self.candidate_node_list = None  # ✅ 新增：保持顺序的候选节点列表
        self.candidate_node_indices = None  # ✅ 新增：候选节点在全局node_list中的索引（np.ndarray）
        self.candidate_mask_np = None  # ✅ 新增：全局候选mask（np.ndarray）
        self.candidate_nodes_file = candidate_nodes_file

        if candidate_nodes_file and os.path.exists(candidate_nodes_file):
            try:
                with open(candidate_nodes_file, 'r', encoding='utf-8') as f:
                    candidate_data = json.load(f)

                # 兼容三种格式：
                #   1) {"candidate_nodes":[...], ...}
                #   2) {"nodes":[...]} / {"candidates":[...]}
                #   3) 纯list: [...]
                if isinstance(candidate_data, dict):
                    cand_raw = (candidate_data.get('candidate_nodes')
                                or candidate_data.get('candidates')
                                or candidate_data.get('nodes')
                                or [])
                elif isinstance(candidate_data, list):
                    cand_raw = candidate_data
                else:
                    cand_raw = []

                # 统一为字符串，去重但保留顺序
                cand_list = []
                seen = set()
                for n in cand_raw:
                    s = str(n)
                    if s not in seen:
                        seen.add(s)
                        cand_list.append(s)

                # 过滤：只保留确实存在于 node_list 的节点
                missing = [n for n in cand_list if n not in self.node_to_idx]
                if missing:
                    logger.warning(f"候选节点中有 {len(missing)} 个不在 node_list 中，将被忽略。示例: {missing[:5]}")

                cand_list = [n for n in cand_list if n in self.node_to_idx]

                if len(cand_list) == 0:
                    logger.warning("候选节点列表为空，将使用全部节点。")
                    self.candidate_nodes = None
                else:
                    self.candidate_node_list = cand_list
                    self.candidate_nodes = set(cand_list)

                    self.candidate_node_indices = np.array(
                        [self.node_to_idx[n] for n in self.candidate_node_list],
                        dtype=np.int64
                    )
                    self.candidate_mask_np = np.zeros(len(self.node_list), dtype=np.float32)
                    self.candidate_mask_np[self.candidate_node_indices] = 1.0

                    print(f"[OK] 候选节点: {len(self.candidate_node_list)} 个")
                    print(f"    节点列表: {self.candidate_node_list[:10]}...")
            except Exception as e:
                logger.warning(f"加载候选节点失败: {e}，将使用全部节点")
                self.candidate_nodes = None
                self.candidate_node_list = None
                self.candidate_node_indices = None
                self.candidate_mask_np = None
        else:
            if candidate_nodes_file:
                logger.warning(f"候选节点文件不存在: {candidate_nodes_file}，将使用全部节点")
            else:
                print(f"[OK] 未指定候选节点文件，将使用全部 {len(self.node_list)} 个节点")

        # 5. 标签编码
        print("\n[5/7] 创建标签编码...")
        self.defect_type_map = {'BASELINE': 0, 'I': 1, 'E': 2, 'P': 3}
        self.defect_type_names = {
            0: 'BASELINE(无缺陷)', 1: 'I(渗入)', 2: 'E(渗漏)', 3: 'P(点源污染)'
        }
        print(f"[OK] 标签编码: {len(self.defect_type_map)} 种类型")

        # 6. 特征处理
        print("\n[6/7] 特征处理...")

        available_features = set(self.node_data.columns)

        if feature_names is None:
            # 自动选择可用的特征，按优先级依次尝试
            # 优先级1：方案C精简残差特征（删除 flooding/head/volume 对应的残差）
            # 优先级2：全量残差特征
            # 优先级3：方案C精简原始特征
            # 优先级4：全量原始特征
            # 优先级5：所有数值列（兜底）
            candidate_features = [
                # 方案C：6个有效原始特征对应的残差（删掉 flooding/head/volume）
                ('depth_residual', 'lateral_inflow_residual', 'total_inflow_residual',
                 'pollut_BODf_residual', 'pollut_NH4_residual', 'pollut_DO_residual'),
                # 全量残差（9个原始特征 × 2，但实际存在的）
                ('depth_residual', 'head_residual', 'volume_residual',
                 'lateral_inflow_residual', 'total_inflow_residual', 'flooding_residual',
                 'pollut_BODf_residual', 'pollut_NH4_residual', 'pollut_DO_residual'),
                # 方案C：6个有效原始特征（无残差时）
                ('depth', 'lateral_inflow', 'total_inflow',
                 'pollut_BODf', 'pollut_NH4', 'pollut_DO'),
                # 全量原始特征（无残差时）
                ('depth', 'head', 'volume', 'lateral_inflow', 'total_inflow',
                 'flooding', 'pollut_BODf', 'pollut_NH4', 'pollut_DO'),
            ]

            candidate_features = [
                ('depth_residual', 'total_outflow_residual',
                 'pollut_NH4_residual', 'pollut_DO_residual',
                 'pollut_NO3_residual', 'pollut_TSSs_residual',
                 'depth_residual_rel', 'total_outflow_residual_rel',
                 'pollut_NH4_residual_rel', 'pollut_DO_residual_rel',
                 'pollut_NO3_residual_rel', 'pollut_TSSs_residual_rel'),
                ('depth', 'total_outflow', 'pollut_NH4', 'pollut_DO', 'pollut_NO3', 'pollut_TSSs',
                 'depth_residual', 'total_outflow_residual',
                 'pollut_NH4_residual', 'pollut_DO_residual',
                 'pollut_NO3_residual', 'pollut_TSSs_residual',
                 'depth_residual_rel', 'total_outflow_residual_rel',
                 'pollut_NH4_residual_rel', 'pollut_DO_residual_rel',
                 'pollut_NO3_residual_rel', 'pollut_TSSs_residual_rel'),
            ] + list(candidate_features)

            self.feature_cols = None
            for feature_set in candidate_features:
                if all(f in available_features for f in feature_set):
                    self.feature_cols = list(feature_set)
                    break

            if self.feature_cols is None:
                # 兜底：使用所有数值特征（排除 meta 列）
                self.feature_cols = [col for col in available_features
                                     if col not in ['scenario_id', 'datetime', 'node_id',
                                                    'defect_type', 'time_step']]
        else:
            # 用户指定的特征
            self.feature_cols = feature_names

        # 验证特征是否存在
        missing_features = [f for f in self.feature_cols if f not in available_features]

        if missing_features:
            raise ValueError(f"以下特征在数据中不存在: {missing_features}")

        self.base_feature_cols = list(self.feature_cols)
        print(f"[OK] 使用基础特征 ({len(self.feature_cols)}): {self.feature_cols}")

        # ✅ 信号门控：记录含 residual 的特征索引
        self.residual_feature_indices = [
            idx for idx, name in enumerate(self.feature_cols)
            if 'residual' in str(name).lower()
        ]
        if not self.residual_feature_indices:
            # 如果没有残差特征，可退化为使用全部特征评估信号
            self.residual_feature_indices = list(range(len(self.feature_cols)))

        # 7. 核心优化: 快速创建样本
        print("\n[7/7] 创建样本 (优化版)...")
        print("  [优化策略] 预重组数据 -> 向量化提取")

        sample_start = time.time()
        self.samples = []
        self.NO_DEFECT_NODE_IDX = len(self.node_list)

        # [关键优化1] 按scenario_id分组,避免重复过滤
        print("  [步骤1/3] 按场景分组数据...")
        grouped_data = self.node_data.groupby('scenario_id')

        processed_scenarios = 0
        total_scenarios = len(self.defect_info)

        for scenario_id, defect_info in self.defect_info.items():
            if scenario_id not in grouped_data.groups:
                continue

            # 获取该场景的所有数据
            scenario_df = grouped_data.get_group(scenario_id).sort_values('datetime')

            if len(scenario_df) < self.sequence_length:
                continue

            # [关键优化2] 预重组为3D数组 [时间步, 节点, 特征]
            # 这样可以避免在循环中反复查询DataFrame
            feature_array = self._fast_reshape_scenario_data(scenario_df)

            if feature_array is None:
                continue

            # ✅ 0.3 新增：提取 unique_times 用于窗口时间meta信息
            unique_times = scenario_df['datetime'].unique()

            # ✅ 时间动态缺陷：从缺陷矩阵读取缺陷激活窗口（同一天的 start_hour + duration_h）
            defect_start = None
            defect_end = None
            start_hour_raw = defect_info.get('start_hour', None)
            duration_h_raw = defect_info.get('duration_h', None)
            if defect_info.get('defect_type', 'BASELINE') != 'BASELINE':
                try:
                    if pd.notna(start_hour_raw) and pd.notna(duration_h_raw):
                        base_dt = pd.to_datetime(unique_times[0])
                        day0 = base_dt.normalize()
                        defect_start = day0 + timedelta(hours=float(start_hour_raw))
                        defect_end = defect_start + timedelta(hours=float(duration_h_raw))
                except Exception:
                    defect_start = None
                    defect_end = None

            # [关键优化3] 使用滑动窗口批量创建样本
            num_windows = len(feature_array) - self.sequence_length + 1

            for i in range(0, num_windows, self.window_stride):
                # 直接切片提取窗口,无需任何查询
                window_core_features = feature_array[i:i + self.sequence_length]
                window_times = pd.to_datetime(unique_times[i:i + self.sequence_length])
                window_features = self._augment_window_features(window_core_features, window_times)

                # 0.3 改进：窗口时间meta信息
                window_start_time = unique_times[i]
                window_end_time = unique_times[i + self.sequence_length - 1]

                # ✅ overlap_ratio（用于时间门控标签）
                overlap_ratio = 0.0
                if defect_info.get('defect_type',
                                   'BASELINE') != 'BASELINE' and defect_start is not None and defect_end is not None:
                    ws = pd.to_datetime(window_start_time)
                    we = pd.to_datetime(window_end_time)
                    overlap_start = max(ws, defect_start)
                    overlap_end = min(we, defect_end)
                    overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
                    window_seconds = max(1.0, (we - ws).total_seconds())
                    overlap_ratio = float(overlap_seconds / window_seconds)

                # 创建节点级标签
                node_labels = np.zeros(len(self.node_list), dtype=int)
                defect_type = defect_info['defect_type']
                defect_node_id = defect_info['node_id']

                # 获取原始标签（0/1/2/3）
                defect_type_label = self.defect_type_map[defect_type]

                # ✅ 缺陷节点在全局 node_list 中的索引（用于“单标签定位”）
                if defect_node_id in self.node_to_idx:
                    defect_node_idx = self.node_to_idx[defect_node_id]
                    node_labels[defect_node_idx] = defect_type_label
                elif defect_type == 'BASELINE':
                    defect_node_idx = -1
                else:
                    # 缺陷节点不在图中：该样本无法用于定位训练，直接跳过
                    logger.warning(f"场景 {scenario_id}: 缺陷节点 {defect_node_id} 不在 node_list 中，已跳过。")
                    continue

                # ✅ 校验：缺陷节点必须在候选集合中（否则loss/评估会被mask掉）
                if defect_node_idx >= 0 and self.candidate_mask_np is not None and self.candidate_mask_np[defect_node_idx] < 0.5:
                    logger.warning(
                        f"场景 {scenario_id}: 缺陷节点 {defect_node_id} 不在候选集合中（candidate_mask=0）。建议更新候选点集。")

                # ✅ 0. 改进：清晰的标签定义
                # defect_type_raw: 0/1/2/3 (BASELINE/I/E/P)
                defect_type_raw = defect_type_label

                total_hours = (pd.to_datetime(unique_times[-1]) - pd.to_datetime(unique_times[0])).total_seconds() / 3600.0
                label_mode = self._resolve_label_mode(defect_info, total_hours)

                # ✅ 信号门控：缺陷节点残差信号
                signal_score = 0.0
                if defect_node_idx >= 0 and defect_info.get('defect_type', 'BASELINE') != 'BASELINE':
                    residual_slice = window_core_features[:, defect_node_idx, self.residual_feature_indices]
                    signal_score = float(np.mean(np.abs(residual_slice)))

                # 保留旧的映射以兼容
                graph_label_mapped = max(0, defect_type_label - 1)
                node_labels_mapped = (node_labels > 0).astype(int)

                sample = {
                    'features': window_features,
                    'defect_type_label': defect_type_label,  # 原始：0//2/3
                    'defect_type_raw': defect_type_raw,  # 0. 新增：原始标签
                    'has_defect': 0,  # 兼容旧字段，后续由 process label 刷新
                    'defect_type_label_mapped': graph_label_mapped,  # 兼容旧版
                    'node_labels': node_labels,  # 原始：0//2/3
                    'node_labels_mapped': node_labels_mapped,  # 映射后：0/
                    'scenario_id': scenario_id,
                    'defect_type_str': defect_type,
                    'defect_node_id': defect_node_id,
                    'defect_node_idx': int(defect_node_idx),
                    'signal_score': signal_score,
                    'overlap_ratio': overlap_ratio,
                    'label_mode': label_mode,
                    'window_start_time': window_start_time,  # 0.3 新增：窗口起始时间
                    'window_end_time': window_end_time,  # 0.3 新增：窗口结束时间
                }
                self._update_sample_process_labels(sample)
                self.samples.append(sample)

            processed_scenarios += 1
            if processed_scenarios % 5 == 0:
                elapsed = time.time() - sample_start
                progress = processed_scenarios / total_scenarios * 100
                print(f"  进度: {processed_scenarios}/{total_scenarios} "
                      f"({progress:.1f}%) - 已用时 {elapsed:.1f}秒")

        sample_time = time.time() - sample_start
        print(f"[OK] 样本创建完成: {len(self.samples)} 个样本 (用时 {sample_time:.2f}秒)")
        print(f"  [速度] 平均: {len(self.samples) / sample_time:.0f} 样本/秒")

        # 7. 标准化器
        print("\n[7/7] 初始化标准化器...")
        self.scaler = StandardScaler() if self.normalize else None

        total_time = time.time() - start_time
        print(f"\n[OK] 数据集构建完成! 总用时: {total_time:.2f}秒")
        if self.samples:
            self.input_feature_dim = int(self.samples[0]['features'].shape[-1])

        self._print_statistics()

    def _resolve_label_mode(self, defect_info: Dict, total_hours: float) -> str:
        """自动判断标签模式（always_on / time_gated / auto)."""
        mode = self.label_mode
        if mode in {"always_on", "time_gated"}:
            return mode

        start_hour = defect_info.get('start_hour', None)
        duration_h = defect_info.get('duration_h', None)
        if pd.notna(start_hour) and pd.notna(duration_h):
            try:
                start_hour = float(start_hour)
                duration_h = float(duration_h)
                if 0 <= start_hour <= 24 and 0 < duration_h < max(total_hours, 1.0):
                    return "time_gated"
            except Exception:
                pass

        return "always_on"

    def _build_time_position_features(self, window_times: np.ndarray, num_nodes: int) -> np.ndarray:
        """为窗口内每个时间步生成 sin/cos(hour) 编码。"""
        timestamps = pd.to_datetime(window_times)
        hours = (
            timestamps.hour.astype(np.float32)
            + timestamps.minute.astype(np.float32) / 60.0
            + timestamps.second.astype(np.float32) / 3600.0
        )
        angles = (2.0 * np.pi * hours) / 24.0
        time_features = np.stack([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)
        return np.repeat(time_features[:, None, :], num_nodes, axis=1)

    def _build_observed_mask_features(self, num_timesteps: int, num_nodes: int) -> np.ndarray:
        """将监测节点掩码扩展为窗口级静态特征。"""
        if self.observed_node_mask_np is None:
            return np.zeros((num_timesteps, num_nodes, 1), dtype=np.float32)
        observed_mask = self.observed_node_mask_np.astype(np.float32, copy=False).reshape(1, num_nodes, 1)
        return np.repeat(observed_mask, num_timesteps, axis=0)

    def _build_trend_features(self, window_features: np.ndarray) -> np.ndarray:
        """用窗口前后半段差值构造节点级趋势特征。"""
        T, N, _ = window_features.shape
        if T < 2:
            return np.zeros((T, N, len(self.residual_feature_indices)), dtype=np.float32)

        split = max(1, T // 2)
        residual_slice = window_features[:, :, self.residual_feature_indices]
        first_half = residual_slice[:split].mean(axis=0)
        second_half = residual_slice[split:].mean(axis=0)
        trend = (second_half - first_half).astype(np.float32)
        return np.repeat(trend[None, :, :], T, axis=0)

    def _augment_window_features(self, window_features: np.ndarray, window_times: np.ndarray) -> np.ndarray:
        """按配置增量追加过程诊断特征。"""
        augmented = window_features.astype(np.float32, copy=False)
        num_timesteps, num_nodes, _ = augmented.shape

        if self.use_time_pos_encoding:
            time_position = self._build_time_position_features(window_times, num_nodes)
            augmented = np.concatenate([augmented, time_position], axis=-1)

        if self.use_observed_mask_feature:
            observed_mask = self._build_observed_mask_features(num_timesteps, num_nodes)
            augmented = np.concatenate([augmented, observed_mask], axis=-1)

        if self.use_trend_feature:
            trend_features = self._build_trend_features(window_features)
            augmented = np.concatenate([augmented, trend_features], axis=-1)

        return augmented.astype(np.float32, copy=False)

    def _infer_phase_label(self, sample: Dict) -> int:
        defect_type = str(sample.get('defect_type_str', 'BASELINE'))
        if defect_type == 'BASELINE':
            return 0

        label_mode = str(sample.get('label_mode', self.label_mode)).lower()
        overlap_ratio = float(sample.get('overlap_ratio', 0.0))
        signal_score = float(sample.get('signal_score', 0.0))

        if label_mode == 'time_gated':
            if overlap_ratio <= self.transition_overlap_threshold:
                return 0
            if overlap_ratio < self.active_overlap_threshold:
                return 1
            return 2

        active_threshold = self.active_signal_threshold
        transition_threshold = self.transition_signal_threshold

        if active_threshold is None and self.signal_threshold > 0:
            active_threshold = float(self.signal_threshold)
        if transition_threshold is None:
            transition_threshold = active_threshold

        if active_threshold is None:
            if self.always_on_force_target:
                return 2
            return 2 if signal_score > 0 else 0

        if signal_score >= active_threshold:
            return 2
        if transition_threshold is not None and signal_score > transition_threshold:
            return 1
        return 0

    def _update_sample_process_labels(self, sample: Dict) -> None:
        phase_label = int(self._infer_phase_label(sample))
        active_label = int(phase_label == 2)
        defect_node_idx = int(sample.get('defect_node_idx', -1))

        sample['phase_label'] = phase_label
        sample['active_label'] = active_label
        sample['has_defect'] = active_label

        loc_enabled = bool(active_label == 1 and defect_node_idx >= 0)
        if loc_enabled:
            target_node_idx = int(defect_node_idx)
            soft_label = self.soft_labels_cache.get(int(defect_node_idx), None)
            if soft_label is None:
                soft_label = np.zeros(len(self.node_list), dtype=np.float32)
                soft_label[int(defect_node_idx)] = 1.0
        else:
            target_node_idx = -1
            soft_label = np.zeros(len(self.node_list), dtype=np.float32)

        sample['loc_enabled'] = int(loc_enabled)
        sample['target_node_idx'] = int(target_node_idx)
        sample['soft_label'] = soft_label.astype(np.float32, copy=False)

    def configure_process_labels(self,
                                 active_signal_threshold: Optional[float] = None,
                                 transition_signal_threshold: Optional[float] = None) -> None:
        """在划分训练集后回写 always_on 场景的活跃阈值，并刷新样本标签。"""
        if active_signal_threshold is not None:
            self.active_signal_threshold = float(active_signal_threshold)
        if transition_signal_threshold is not None:
            self.transition_signal_threshold = float(transition_signal_threshold)

        for sample in self.samples:
            self._update_sample_process_labels(sample)

    def _fast_reshape_scenario_data(self, scenario_df):
        """
        [核心优化函数] 快速将场景数据重组为3D数组

        原方法: 对每个时间步x节点都查询DataFrame (慢)
        新方法: 先pivot重组,再填充缺失值 (快10-50倍)

        返回: [时间步, 节点, 特征] 的numpy数组
        """
        try:
            # 获取唯一时间步
            unique_times = scenario_df['datetime'].unique()
            n_timesteps = len(unique_times)
            n_nodes = len(self.node_list)
            n_features = len(self.feature_cols)

            # 预分配数组
            feature_array = np.zeros((n_timesteps, n_nodes, n_features), dtype=np.float32)

            # 🔥 关键: 使用pivot_table快速重组
            for feat_idx, feature_col in enumerate(self.feature_cols):
                # 将长格式转为宽格式: 行=时间, 列=节点
                pivot = scenario_df.pivot_table(
                    index='datetime',
                    columns='node_id',
                    values=feature_col,
                    aggfunc='first'  # 如果有重复,取第一个
                )

                # ✅ 改进：确保pivot的行数与n_timesteps一致
                pivot = pivot.reindex(pd.to_datetime(unique_times), fill_value=0.0)

                # 填充缺失节点的值为0
                for node_idx, node_id in enumerate(self.node_list):
                    if node_id in pivot.columns:
                        values = pivot[node_id].fillna(0).values
                        # ✅ 安全检查：确保长度匹配
                        if len(values) == n_timesteps:
                            feature_array[:, node_idx, feat_idx] = values
                        else:
                            # 长度不匹配，用0填充
                            feature_array[:, node_idx, feat_idx] = 0.0
                    else:
                        feature_array[:, node_idx, feat_idx] = 0.0

            return feature_array

        except Exception as e:
            logger.warning(f"场景数据重组失败: {e}")
            logger.warning(f"  预期形状: ({n_timesteps}, {n_nodes}, {n_features})")
            logger.warning(f"  节点列表数: {len(self.node_list)}")
            logger.warning(f"  特征列表: {self.feature_cols}")
            return None

    def fit_scaler(self, train_indices):
        """用训练集拟合标准化器"""
        if self.scaler is not None:
            print("\n[标准化] 用训练集拟合标准化器...")
            all_features = []
            for idx in train_indices:
                sample = self.samples[idx]
                features = sample['features']
                all_features.append(features.reshape(-1, features.shape[-1]))

            all_features = np.concatenate(all_features, axis=0)
            self.scaler.fit(all_features)
            print(f"[OK] 标准化器已拟合")

    def transform_features(self, features):
        """应用标准化"""
        if self.scaler is not None:
            T, N, F = features.shape
            features_flat = features.reshape(-1, F)
            features_normalized = self.scaler.transform(features_flat)
            return features_normalized.reshape(T, N, F)
        return features

    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 80)
        print("数据集统计")
        print("=" * 80)

        type_labels = [s['defect_type_label'] for s in self.samples]
        print(f"\n缺陷类型分布:")
        for label in sorted(set(type_labels)):
            count = type_labels.count(label)
            pct = count / len(type_labels) * 100
            print(f"  {self.defect_type_names[label]}: {count} ({pct:.1f}%)")

        print(f"\n数据集信息:")
        print(f"  样本数: {len(self.samples)}")
        print(f"  节点数: {len(self.node_list)}")
        print(f"  特征数: {len(self.feature_cols)}")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  缺陷类型数: {len(self.defect_type_names)}")

        active_label_list = [int(s.get('active_label', 0)) for s in self.samples]
        n_active = sum(active_label_list)
        n_inactive = len(active_label_list) - n_active

        loc_enabled_list = [int(s.get('loc_enabled', 0)) for s in self.samples]
        n_loc_enabled = sum(loc_enabled_list)
        phase_label_list = [int(s.get('phase_label', 0)) for s in self.samples]
        n_transition = sum(1 for p in phase_label_list if p == 1)

        overlap_list = [float(s.get('overlap_ratio', 0.0)) for s in self.samples]
        b0 = sum(1 for r in overlap_list if r <= 0.0)
        b1 = sum(1 for r in overlap_list if (r > 0.0 and r < 0.1))
        b2 = sum(1 for r in overlap_list if (r >= 0.1 and r < 0.5))
        b3 = sum(1 for r in overlap_list if r >= 0.5)

        print(f"\n窗口级标签统计:")
        print(f"  active_label=1: {n_active} ({(n_active / max(1, len(self.samples)) * 100):.1f}%)")
        print(f"  active_label=0: {n_inactive} ({(n_inactive / max(1, len(self.samples)) * 100):.1f}%)")
        print(f"  phase_label=1(过渡期): {n_transition} ({(n_transition / max(1, len(self.samples)) * 100):.1f}%)")
        print(
            f"  定位启用(target_node_idx>=0): {n_loc_enabled} ({(n_loc_enabled / max(1, len(self.samples)) * 100):.1f}%)")

        if self.active_signal_threshold is not None:
            print(f"  active_signal_threshold: {self.active_signal_threshold:.4f}")
        elif self.signal_threshold > 0:
            print(f"  signal_threshold(兼容): {self.signal_threshold:.4f}")
        if self.transition_signal_threshold is not None:
            print(f"  transition_signal_threshold: {self.transition_signal_threshold:.4f}")

        signal_scores = [float(s.get('signal_score', 0.0)) for s in self.samples if s.get('signal_score') is not None]
        if signal_scores:
            print(f"  signal_score 中位数: {float(np.median(signal_scores)):.4f}")
            print(f"  signal_score 均值: {float(np.mean(signal_scores)):.4f}")

        print(f"\noverlap_ratio 分布:")
        print(f"  overlap=0: {b0} ({(b0 / max(1, len(self.samples)) * 100):.1f}%)")
        print(f"  0<overlap<0.1: {b1} ({(b1 / max(1, len(self.samples)) * 100):.1f}%)")
        print(f"  0.1<=overlap<0.5: {b2} ({(b2 / max(1, len(self.samples)) * 100):.1f}%)")
        print(f"  overlap>=0.5: {b3} ({(b3 / max(1, len(self.samples)) * 100):.1f}%)")

        scenario_counts = {}
        for s in self.samples:
            sid = s.get('scenario_id')
            scenario_counts[sid] = scenario_counts.get(sid, 0) + 1
        if scenario_counts:
            counts = list(scenario_counts.values())
            print(f"\n每场景窗口数(受 stride/阈值影响):")
            print(f"  场景数: {len(scenario_counts)}")
            print(f"  min/mean/max: {min(counts)}/{(sum(counts) / max(1, len(counts))):.1f}/{max(counts)}")
        print("=" * 80)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample['features']
        if self.scaler is not None:
            features = self.transform_features(features)

        # ✅ 生成候选节点mask（全样本共享，避免在 __getitem__ 里循环 O(N)）
        if self.candidate_mask_np is not None:
            candidate_mask = self.candidate_mask_np.copy()
        else:
            candidate_mask = np.ones(len(self.node_list), dtype=np.float32)

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'adj_matrix': torch.tensor(self.adj_matrix, dtype=torch.float32),
            'defect_type': torch.tensor(sample['defect_type_label_mapped'], dtype=torch.long),  # 兼容旧版
            'defect_type_raw': torch.tensor(sample['defect_type_raw'], dtype=torch.long),  # 0. 新增：原始标签 0//2/3
            'has_defect': torch.tensor(sample['has_defect'], dtype=torch.long),  # 0. 新增：图级二分类 0/
            'active_label': torch.tensor(sample.get('active_label', sample['has_defect']), dtype=torch.long),
            'phase_label': torch.tensor(sample.get('phase_label', 0), dtype=torch.long),
            'node_labels': torch.tensor(sample['node_labels_mapped'], dtype=torch.long),  # 兼容旧版
            'candidate_mask': torch.tensor(candidate_mask, dtype=torch.float32),  # ✅ 新增：候选节点mask
            'scenario_id': sample['scenario_id'],
            'defect_type_str': sample['defect_type_str'],
            'defect_node_id': sample['defect_node_id'],
            'target_node_idx': torch.tensor(sample.get('target_node_idx', -1), dtype=torch.long),  # ✅ 单标签定位目标
            'soft_label': torch.tensor(sample.get('soft_label', np.zeros(len(self.node_list), dtype=np.float32)),
                                       dtype=torch.float32),
            'loc_enabled': torch.tensor(sample.get('loc_enabled', 0), dtype=torch.long),
            'observed_mask': torch.tensor(
                self.observed_node_mask_np if self.observed_node_mask_np is not None
                else np.zeros(len(self.node_list), dtype=np.float32),
                dtype=torch.float32
            ),
            'signal_score': torch.tensor(sample.get('signal_score', 0.0), dtype=torch.float32),
            'overlap_ratio': torch.tensor(sample.get('overlap_ratio', 0.0), dtype=torch.float32),
            # ✅ 修复：将 Timestamp 转换为字符串，避免 DataLoader collate 错误
            'window_start_time': str(sample['window_start_time']),  # 0.3 新增：窗口起始时间
            'window_end_time': str(sample['window_end_time']),  # 0.3 新增：窗口结束时间
        }


def create_dataloaders(node_timeseries_file: str,
                       adjacency_matrix_file: str,
                       node_list_file: str,
                       defect_matrix_file: str,
                       batch_size: int = 32,
                       sequence_length: int = 30,
                       window_stride: int = 1,
                       cls_overlap_threshold: float = 0.1,
                       loc_overlap_threshold: float = 0.5,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       normalize: bool = True,
                       random_seed: int = 42,
                       feature_names: Optional[List[str]] = None,
                       candidate_nodes_file: Optional[str] = None,
                       soft_label_sigma: float = 1.0,
                       use_full_graph: bool = False,
                       signal_threshold: Optional[float] = None,
                       loc_ratio_clip: float = 0.0,
                       label_mode: str = "auto",
                       overlap_threshold: Optional[float] = None,
                       always_on_force_target: bool = True,
                       use_time_pos_encoding: bool = True,
                       use_observed_mask_feature: bool = True,
                       observed_nodes_file: Optional[str] = None,
                       use_trend_feature: bool = False,
                       active_overlap_threshold: float = 0.5,
                       transition_overlap_threshold: float = 0.0,
                       active_signal_threshold: Optional[float] = None,
                       transition_signal_threshold: Optional[float] = None,
                       e_class_oversample_ratio: int = 1,
                       split_mode: str = "scenario",
                       n_holdout_nodes: int = 10) -> Tuple[
    DataLoader, DataLoader, DataLoader, DefectDetectionDataset]:
    """
    创建训练、验证和测试数据加载器

    ✅ 使用流程：
    1. 运行 25_baseline_feature_engineering.py 处理原始时序数据
       python 25_baseline_feature_engineering.py
       输出：node_timeseries_with_residuals.parquet

    2. 加载处理后的时序数据到本脚本
       node_timeseries_file = "node_timeseries_with_residuals.parquet"

    3. 本脚本会自动构建数据集并创建DataLoader

    4. 23_anomaly_detection_final_v3_zong.py 调用本脚本的 create_dataloaders()
       train_loader, val_loader, test_loader, dataset = create_dataloaders(...)
    """

    dataset = DefectDetectionDataset(
        node_timeseries_file=node_timeseries_file,
        adjacency_matrix_file=adjacency_matrix_file,
        node_list_file=node_list_file,
        defect_matrix_file=defect_matrix_file,
        sequence_length=sequence_length,
        window_stride=window_stride,
        cls_overlap_threshold=cls_overlap_threshold,
        loc_overlap_threshold=loc_overlap_threshold,
        normalize=normalize,
        random_seed=random_seed,
        feature_names=feature_names,
        candidate_nodes_file=candidate_nodes_file,
        soft_label_sigma=soft_label_sigma,
        signal_threshold=signal_threshold,
        label_mode=label_mode,
        overlap_threshold=overlap_threshold,
        always_on_force_target=always_on_force_target,
        use_time_pos_encoding=use_time_pos_encoding,
        use_observed_mask_feature=use_observed_mask_feature,
        observed_nodes_file=observed_nodes_file,
        use_trend_feature=use_trend_feature,
        active_overlap_threshold=active_overlap_threshold,
        transition_overlap_threshold=transition_overlap_threshold,
        active_signal_threshold=active_signal_threshold,
        transition_signal_threshold=transition_signal_threshold
    )

    # P0 索引一致性验收
    print("\n" + "=" * 80)
    print("[P0] 索引一致性验收")
    print("=" * 80)

    # 1. 验证全局索引定义
    N = len(dataset.node_list)  # 全局节点数
    print(f"全局节点数 N: {N}")
    print(f"节点ID范围: {dataset.node_list[:3]} ... {dataset.node_list[-3:]}")

    # 2. 验证候选集索引
    if candidate_nodes_file:
        C = len(dataset.candidate_nodes)
        candidate_indices = [dataset.node_to_idx[node_id] for node_id in dataset.candidate_nodes]
        print(f"候选节点数 C: {C}")
        print(f"候选全局idx范围: {min(candidate_indices)} .. {max(candidate_indices)}")
        assert max(candidate_indices) < N, f"候选索引超出全局范围: {max(candidate_indices)} >= {N}"

    # 3. 验证训练模式索引一致性
    print(f"训练模式: {'全图' if use_full_graph else '候选集限制'}")

    # 4. 随机抽样验证索引映射
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    print("\n[索引映射验证]:")
    for i, sample_idx in enumerate(sample_indices):
        sample = dataset.samples[sample_idx]
        if sample.get('target_node_idx', -1) >= 0:
            target_idx = sample['target_node_idx']
            defect_node_id = sample['defect_node_id']

            # 验证全局索引
            assert 0 <= target_idx < N, f"目标索引超出范围: {target_idx} not in [0, {N - 1}]"

            # 验证索引映射正确性
            mapped_node_id = dataset.node_list[target_idx]
            assert mapped_node_id == defect_node_id, f"索引映射错误: idx={target_idx}, mapped={mapped_node_id}, expected={defect_node_id}"

            print(f"  样本{i + 1}: idx={target_idx} -> node_id={mapped_node_id} ")

    print("[P0] 索引一致性验收通过 ")
    print("=" * 80 + "\n")

    # 0.2 改进：按 scenario_id 分场景划分，避免同一场景的窗口被分散
    scenario_to_indices = {}
    for idx, sample in enumerate(dataset.samples):
        sid = sample['scenario_id']
        scenario_to_indices.setdefault(sid, []).append(idx)

    # loc_enabled 统计（用于后续抽样加权）
    loc_enabled_mask = np.array([int(s.get('loc_enabled', 0)) for s in dataset.samples], dtype=int)

    scenario_ids = list(scenario_to_indices.keys())
    np.random.seed(random_seed)
    random.seed(random_seed)

    split_mode_norm = (split_mode or "").strip().lower()

    if split_mode_norm == "node_holdout":
        # Node-holdout：先按缺陷节点划分，测试集仅包含“未见过”的缺陷节点
        candidate_node_list = getattr(dataset, "candidate_node_list", None)
        if candidate_node_list is None or len(candidate_node_list) == 0:
            candidate_node_list = list(set(
                str(dataset.defect_info.get(sid, {}).get("node_id", ""))
                for sid in scenario_ids
                if str(dataset.defect_info.get(sid, {}).get("node_id", "")) in dataset.node_to_idx
            ))
        n_holdout = min(n_holdout_nodes, max(1, len(candidate_node_list) - 2))
        holdout_node_ids = set(random.sample(candidate_node_list, n_holdout))
        train_val_node_ids = set(candidate_node_list) - holdout_node_ids

        test_scenario_ids = [
            sid for sid in scenario_ids
            if str(dataset.defect_info.get(sid, {}).get("node_id", "")) in holdout_node_ids
        ]
        train_val_pool = [
            sid for sid in scenario_ids
            if str(dataset.defect_info.get(sid, {}).get("node_id", "")) in train_val_node_ids
        ]
        random.shuffle(train_val_pool)
        n_tv = len(train_val_pool)
        train_s = int(n_tv * train_ratio)
        val_s = int(n_tv * val_ratio)
        train_scenarios = train_val_pool[:train_s]
        val_scenarios = train_val_pool[train_s:train_s + val_s]
        test_scenarios = test_scenario_ids
        if 0 in scenario_ids and 0 not in train_scenarios:
            train_scenarios = [0] + list(train_scenarios)

        print(f"\n[SPLIT] node_holdout: 候选节点数={len(candidate_node_list)}, holdout={n_holdout}")
        print(f"  Holdout节点(仅出现在测试集): {sorted(holdout_node_ids)[:15]}{'...' if len(holdout_node_ids) > 15 else ''}")
        print(f"  测试集场景数(缺陷节点=holdout): {len(test_scenarios)}, 训练+验证场景数: {len(train_val_pool)}")
    elif split_mode_norm in {"scenario_nodecovered", "scenario_seen", "scenario_traincovers"}:
        np.random.shuffle(scenario_ids)
        baseline_scenario = None
        if 0 in scenario_ids:
            baseline_scenario = 0
            scenario_ids = [sid for sid in scenario_ids if sid != 0]

        # 先保证：每个缺陷节点至少有一个场景进入训练集。
        node_to_scenarios = {}
        for sid in scenario_ids:
            node_id = str(dataset.defect_info.get(int(sid), {}).get("node_id", "")).strip()
            if node_id:
                node_to_scenarios.setdefault(node_id, []).append(int(sid))

        covered_train = []
        covered_train_set = set()
        for node_id in sorted(node_to_scenarios.keys()):
            scenario_pool = list(node_to_scenarios[node_id])
            random.shuffle(scenario_pool)
            chosen = scenario_pool[0]
            if chosen not in covered_train_set:
                covered_train.append(chosen)
                covered_train_set.add(chosen)

        remaining = [sid for sid in scenario_ids if sid not in covered_train_set]
        random.shuffle(remaining)

        n_scenarios = len(scenario_ids)
        target_train = max(int(n_scenarios * train_ratio), len(covered_train))
        target_val = int(n_scenarios * val_ratio)
        extra_train_needed = max(target_train - len(covered_train), 0)

        extra_train = remaining[:extra_train_needed]
        after_train = remaining[extra_train_needed:]
        val_scenarios = after_train[:target_val]
        test_scenarios = after_train[target_val:]
        train_scenarios = covered_train + extra_train

        if baseline_scenario is not None:
            train_scenarios = [baseline_scenario] + list(train_scenarios)

        print(f"\n[SPLIT] {split_mode_norm}: 训练集覆盖所有已出现缺陷节点")
        print(f"  覆盖缺陷节点数: {len(node_to_scenarios)}")
        print(f"  训练覆盖场景数(最小覆盖集): {len(covered_train)}")
        print(f"  TRAIN/VAL/TEST 场景数: {len(train_scenarios)}/{len(val_scenarios)}/{len(test_scenarios)}")
    else:
        np.random.shuffle(scenario_ids)
        baseline_scenario = None
        if 0 in scenario_ids:
            baseline_scenario = 0
            scenario_ids = [sid for sid in scenario_ids if sid != 0]
        n_scenarios = len(scenario_ids)
        train_s = int(n_scenarios * train_ratio)
        val_s = int(n_scenarios * val_ratio)
        train_scenarios = scenario_ids[:train_s]
        val_scenarios = scenario_ids[train_s:train_s + val_s]
        test_scenarios = scenario_ids[train_s + val_s:]
        if baseline_scenario is not None:
            train_scenarios = [baseline_scenario] + list(train_scenarios)

    def gather_indices(scenario_list):
        idxs = []
        for sid in scenario_list:
            idxs.extend(scenario_to_indices[sid])
        return np.array(idxs, dtype=int)

    def filter_by_loc_enabled(indices: np.ndarray) -> np.ndarray:
        # loc_ratio_clip 可能为 None（config 未设置时），避免 None <= 0 报错
        if (loc_ratio_clip or 0) <= 0:
            return indices
        if indices.size == 0:
            return indices
        mask = loc_enabled_mask[indices] > 0
        positive_indices = indices[mask]
        negative_indices = indices[~mask]
        desired_pos = int(len(indices) * loc_ratio_clip)
        if desired_pos <= 0:
            return negative_indices
        if len(positive_indices) <= desired_pos:
            return indices
        selected_pos = np.random.choice(positive_indices, desired_pos, replace=False)
        combined = np.concatenate([selected_pos, negative_indices])
        np.random.shuffle(combined)
        return combined

    train_indices = filter_by_loc_enabled(gather_indices(train_scenarios))
    # E 类过采样：使 E 类样本在每 epoch 中多出现，缓解 E 类 Top-1=0
    if e_class_oversample_ratio > 1 and hasattr(dataset, 'samples'):
        extra = []
        for idx in train_indices:
            s = dataset.samples[idx]
            if str(s.get('defect_type_str', '')) == 'E':
                for _ in range(e_class_oversample_ratio - 1):
                    extra.append(idx)
        if extra:
            train_indices = np.concatenate([train_indices, np.array(extra, dtype=train_indices.dtype)])
            np.random.shuffle(train_indices)
            print(f"\n[E类过采样] 比例={e_class_oversample_ratio}x, 训练样本数: {len(train_indices)} (含重复)")
    val_indices = gather_indices(val_scenarios)
    test_indices = gather_indices(test_scenarios)

    def _split_diagnostics(split_name: str, scenario_list: List[int], indices: np.ndarray):
        scenario_set = set(int(s) for s in scenario_list)
        defect_types = []
        defect_nodes = []
        for sid in scenario_set:
            info = dataset.defect_info.get(int(sid), None)
            if info is None:
                continue
            defect_types.append(str(info.get('defect_type', '')))
            defect_nodes.append(str(info.get('node_id', '')))

        type_counts = {}
        for t in defect_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        defect_nodes = [n for n in defect_nodes if n]
        defect_node_set = set(defect_nodes)

        in_candidate = None
        if dataset.candidate_nodes is not None:
            in_candidate = sum(1 for n in defect_node_set if n in dataset.candidate_nodes)

        print(f"\n[SPLIT诊断] {split_name}:")
        print(f"  场景数: {len(scenario_set)}")
        print(f"  窗口数: {int(len(indices))}")
        print(f"  缺陷类型(按场景): {type_counts}")
        print(f"  缺陷节点数(去重): {len(defect_node_set)}")
        if in_candidate is not None:
            cov = (in_candidate / max(len(defect_node_set), 1)) * 100.0
            print(f"  缺陷节点在候选集覆盖率: {in_candidate}/{len(defect_node_set)} ({cov:.1f}%)")

        return defect_node_set

    train_defect_nodes = _split_diagnostics('TRAIN', train_scenarios, train_indices)
    _ = _split_diagnostics('VAL', val_scenarios, val_indices)
    test_defect_nodes = _split_diagnostics('TEST', test_scenarios, test_indices)

    # B 路线：候选集=缺陷节点集，测试缺陷节点均在训练集中出现；不单独报告「未见过节点」
    test_in_train = test_defect_nodes <= train_defect_nodes
    # Windows PowerShell 默认编码可能为 GBK，避免使用 '⊆' 等符号导致 UnicodeEncodeError
    print(f"\n[SPLIT诊断] 测试缺陷节点 subset 训练集: {'是' if test_in_train else '否'} (训练{len(train_defect_nodes)} 测试{len(test_defect_nodes)} 种缺陷节点)")
    if not test_in_train:
        only_in_test = list(sorted(test_defect_nodes - train_defect_nodes))[:10]
        print(f"  仅出现在测试的缺陷节点(示例): {only_in_test}")

    # 各 split 定位启用窗口数（有 target_node_idx>=0 的样本），便于确认 val/test 有足够定位信号
    n_train_loc = int((loc_enabled_mask[train_indices] > 0).sum())
    n_val_loc = int((loc_enabled_mask[val_indices] > 0).sum())
    n_test_loc = int((loc_enabled_mask[test_indices] > 0).sum())
    print(f"\n[SPLIT诊断] 定位启用窗口数(占比): TRAIN {n_train_loc}/{len(train_indices)} ({100.*n_train_loc/max(len(train_indices),1):.1f}%)  "
            f"VAL {n_val_loc}/{len(val_indices)} ({100.*n_val_loc/max(len(val_indices),1):.1f}%)  "
            f"TEST {n_test_loc}/{len(test_indices)} ({100.*n_test_loc/max(len(test_indices),1):.1f}%)")

    if dataset.scaler is not None:
        dataset.fit_scaler(train_indices)

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=batch_size, shuffle=False
    )

    # ✅ Bug 2 修复：定义 n_samples
    n_samples = len(dataset)

    print(f"\n" + "=" * 80)
    print("数据集划分:")
    print(f"  训练集: {len(train_indices)} 样本 ({len(train_indices) / n_samples * 100:.1f}%)")
    print(f"  验证集: {len(val_indices)} 样本 ({len(val_indices) / n_samples * 100:.1f}%)")
    print(f"  测试集: {len(test_indices)} 样本 ({len(test_indices) / n_samples * 100:.1f}%)")
    print("=" * 80)

    return train_loader, val_loader, test_loader, dataset
