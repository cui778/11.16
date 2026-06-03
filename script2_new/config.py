# -*- coding: utf-8 -*-
"""
script2_new 统一配置入口。

所有路径与实验开关尽量由此文件管理，业务脚本通过 config 获取路径，避免硬编码。
第一阶段：仅搭好配置框架，预留 pretest / chapter1 / chapter2 及 V/C/S 相关项。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import os


def _join(root: str, *parts: str) -> str:
    return os.path.normpath(os.path.join(root, *parts))


@dataclass
class Config:
    # ========== 基础路径 ==========
    """项目根目录（含 input_data 等只读数据）"""
    data_root: str = field(default_factory=lambda: os.path.normpath(r"E:\11.16"))
    """本新实验目录根路径"""
    script_root: str = field(default_factory=lambda: os.path.normpath(r"E:\11.16\script2_new"))
    """新路线生成的中间数据目录（候选集、监测集、缺陷矩阵、图、解析结果等），位于 script2_new/input_1"""
    input_dir_new: str = ""

    # ========== 实验阶段 ==========
    # pretest: 输入 V，输出 C，验证框架能力
    # chapter1: 输入 S，输出 C，比较监测点布置规则
    # chapter2: 在 chapter1 最优规则上优化监测点布置
    experiment_stage: str = "chapter1"

    # ========== 三个集合对应文件 ==========
    # 全网节点集 V：通常由 build_graph 产出或从 graph 解析得到，此处预留文件名/路径
    # 候选缺陷节点集 C（输出空间）
    candidate_nodes_file: str = ""
    # 监测节点集 S（输入空间）
    monitor_nodes_file: str = ""
    # 章节实验默认监测节点数量，可改为 15 / 20 / 25 等
    monitor_node_count: int = 25
    # full 策略使用的全节点数量
    full_monitor_node_count: int = 128

    # ========== 输入/输出空间模式 ==========
    # full: 输入为全节点 V（用于 pretest）
    # monitor_only: 输入为监测节点 S（用于 chapter1 / chapter2）
    input_node_mode: str = "monitor_only"

    # ========== 生成策略（预留，第二阶段实现） ==========
    candidate_generation_strategy: str = "from_inp"
    monitor_generation_strategy: str = "fixed"
    defect_generation_mode: str = "realistic"

    # ========== 数据与图 ==========
    inp_file: str = ""
    adjacency_matrix_file: str = ""
    """图路径特征（供 hydraulic_inverse 模型）：shortest_dist, pipe_length_dist, flow_direction, elevation_diff"""
    graph_path_features_file: str = ""
    node_list_file: str = ""
    defect_matrix_file: str = ""
    node_timeseries_file: str = ""
    """新路线时序/训练数据目录（extract_timeseries、residual 输出），位于 script2_new/training_data_new"""
    training_data_dir: str = ""
    """时序数据子目录：'full_injection' | 'time_gated' | ''。用于区分全程注入与时间门控，训练时 node_timeseries_file 指向对应子目录"""
    training_data_subdir: str = "time_gated_node_sensor_v2e_dense_ie"
    """以下由 __post_init__ 根据 input_dir_new 派生，供 prep 脚本使用"""
    parsed_inp_data_file: str = ""
    baseline_flow_stats_file: str = ""

    # ========== 输出目录 ==========
    model_save_dir: str = ""
    log_dir: str = ""
    reports_dir: str = ""

    # ========== 数据处理与训练（与 script2 对齐，供 train/check_labels 使用） ==========
    dataset_processor_path: str = ""
    output_dir: str = ""  # 与 model_save_dir 一致，兼容旧 train 脚本里的 config.output_dir
    output_tag: str = "process_diagnosis"
    sequence_length: int = 36
    window_stride: int = 6
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    random_seed: int = 42
    soft_label_sigma: float = 1.0
    train_with_full_graph: bool = False
    use_time_pos_encoding: bool = True
    use_observed_mask_feature: bool = True
    use_trend_feature: bool = False
    dataset_signal_threshold: float = 0.0
    dataset_loc_ratio_clip: Optional[float] = None
    dataset_label_mode: str = "auto"
    dataset_overlap_threshold: float = 0.5
    dataset_active_overlap_threshold: float = 0.5
    dataset_transition_overlap_threshold: float = 0.0
    dataset_active_signal_quantile: float = 0.95
    dataset_transition_signal_quantile: float = 0.90
    dataset_always_on_force_target: bool = True
    dataset_e_class_oversample_ratio: int = 1
    dataset_persistent_train_sample_ratio: float = 1.0
    """数据划分方式: scenario=按场景随机划分(默认); node_holdout=测试集仅含未见过缺陷节点"""
    dataset_split_mode: str = "scenario"
    dataset_n_holdout_nodes: int = 10
    selected_features: List[str] = field(default_factory=list)
    # 模型与训练（简化，与 script2 一致即可）
    model_type: str = "hydraulic_inverse_deepattn"  # gru_gcn | hydraulic_inverse | hydraulic_inverse_deepattn
    use_flow_direction: bool = True
    use_propagation_delay: bool = False
    propagation_delay_velocity_mps: float = 0.5
    hydraulic_attention_max_hops: int = 0
    time_hidden_dim: int = 128
    spatial_hidden_dim: int = 256
    num_time_layers: int = 1
    num_spatial_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.0003
    weight_decay: float = 5e-4
    use_lr_scheduler: bool = True
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    num_epochs: int = 50
    patience: int = 10
    lambda_type: float = 0.0
    lambda_loc: float = 1.0
    lambda_rank: float = 0.0
    rank_margin: float = 0.1
    lambda_balance: float = 0.0
    lambda_recon: float = 0.0
    recon_mask_prob: float = 0.3
    use_seen_mrr_for_early_stopping: bool = True
    hydraulic_observed_source_only: bool = False
    active_prob_threshold: Optional[float] = None
    scene_active_threshold: Optional[float] = None
    scene_score_mode: str = "topk_mean"
    scene_topk: int = 5

    def __post_init__(self):
        if not self.input_dir_new:
            self.input_dir_new = _join(self.script_root, "input_1")
        if not self.candidate_nodes_file:
            self.candidate_nodes_file = _join(self.input_dir_new, "candidate_nodes_new.json")
        if not self.monitor_nodes_file:
            self.monitor_nodes_file = self.get_monitor_nodes_file("degree")
        if not self.inp_file:
            self.inp_file = _join(self.data_root, "input_data", "2_tuned_v3_merged1.inp")
        if not self.adjacency_matrix_file:
            self.adjacency_matrix_file = _join(self.input_dir_new, "adj_matrix.npy")
        if not self.graph_path_features_file:
            self.graph_path_features_file = _join(self.input_dir_new, "graph_path_features.npz")
        if not self.node_list_file:
            self.node_list_file = _join(self.input_dir_new, "node_list.json")
        if not self.defect_matrix_file:
            self.defect_matrix_file = _join(self.input_dir_new, "defect_matrix_diverse_ie_v2e_dense.csv")
        if not self.training_data_dir:
            self.training_data_dir = _join(self.script_root, "training_data_new")
        if not self.node_timeseries_file:
            subdir = self.training_data_subdir.strip()
            if subdir:
                self.node_timeseries_file = _join(
                    self.training_data_dir, subdir, "node_timeseries_with_residuals.parquet"
                )
            else:
                self.node_timeseries_file = _join(
                    self.training_data_dir, "node_timeseries_with_residuals.parquet"
                )
        self.parsed_inp_data_file = _join(self.input_dir_new, "parsed_inp_data.json")
        self.baseline_flow_stats_file = _join(self.input_dir_new, "baseline_flow_stats.json")
        if not self.model_save_dir:
            self.model_save_dir = _join(self.script_root, "outputs", "model_checkpoints")
        if not self.log_dir:
            self.log_dir = _join(self.script_root, "outputs", "logs")
        if not self.reports_dir:
            self.reports_dir = _join(self.script_root, "outputs", "reports")
        if not self.dataset_processor_path:
            self.dataset_processor_path = _join(self.script_root, "dataset", "processor.py")
        if not self.output_dir:
            self.output_dir = self.model_save_dir
        if not self.selected_features:
            self.selected_features = [
                "depth_residual",
                "total_outflow_residual",
                "pollut_NH4_residual",
                "pollut_TSSs_residual",
                "depth_residual_rel",
                "total_outflow_residual_rel",
                "pollut_NH4_residual_rel",
                "pollut_TSSs_residual_rel",
            ]

    def ensure_output_dirs(self) -> None:
        for d in (self.model_save_dir, self.log_dir, self.reports_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    def get_monitor_nodes_file(self, strategy: str, n: Optional[int] = None) -> str:
        if strategy == "full":
            count = self.full_monitor_node_count if n is None else int(n)
        else:
            count = self.monitor_node_count if n is None else int(n)
        return _join(self.input_dir_new, f"monitor_nodes_{strategy}_N{count}.json")
