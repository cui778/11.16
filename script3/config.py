# -*- coding: utf-8 -*-
"""
Central config for script3.

script3 is the reset Chapter 1 experiment line focused on:
- main task: candidate-node defect localization
- auxiliary task: graph-level defect type classification (I/E/P)
- chapter focus: model innovation under fixed S/C/data protocol
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import os


def _join(root: str, *parts: str) -> str:
    return os.path.normpath(os.path.join(root, *parts))


@dataclass
class Config:
    # Paths
    data_root: str = field(default_factory=lambda: os.path.normpath(r"E:\11.16"))
    script_root: str = field(default_factory=lambda: os.path.normpath(r"E:\11.16\script3"))
    input_dir_new: str = ""
    training_data_dir: str = ""
    training_data_subdir: str = "time_gated_downstream"

    # Chapter 1 framing
    experiment_stage: str = "chapter1_reset"
    primary_task: str = "localization"
    auxiliary_task: str = "type_classification"
    task_mode: str = "node"
    localization_target: str = "node"
    defect_anchor_mode: str = "node_with_upstream_pipe_context"
    defect_type_mode: str = "ie_only"
    allowed_defect_types: List[str] = field(default_factory=lambda: ["I", "E"])
    chapter1_monitor_protocol: str = "downstream_N25"
    chapter1_candidate_protocol: str = "candidate_C50"
    chapter1_defect_protocol: str = "diverse_300_iep"

    # Data protocol files
    candidate_nodes_file: str = ""
    monitor_nodes_file: str = ""
    inp_file: str = ""
    adjacency_matrix_file: str = ""
    graph_path_features_file: str = ""
    node_list_file: str = ""
    defect_matrix_file: str = ""
    node_timeseries_file: str = ""
    link_timeseries_file: str = ""
    node_timeseries_residual_file: str = ""
    link_timeseries_residual_file: str = ""
    edge_timeseries_file: str = ""
    segment_list_file: str = ""
    edge_static_features_file: str = ""
    parsed_inp_data_file: str = ""
    baseline_flow_stats_file: str = ""
    pipe_segment_map_file: str = ""

    # Outputs
    model_save_dir: str = ""
    log_dir: str = ""
    reports_dir: str = ""

    # Dataset/training wiring
    dataset_processor_path: str = ""
    output_dir: str = ""
    input_node_mode: str = "monitor_only"
    candidate_generation_strategy: str = "from_inp"
    monitor_generation_strategy: str = "fixed_downstream"
    defect_generation_mode: str = "diverse"

    sequence_length: int = 36
    window_stride: int = 6
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    random_seed: int = 42
    soft_label_sigma: float = 1.0
    train_with_full_graph: bool = False
    dataset_signal_threshold: float = 0.0
    dataset_loc_ratio_clip: Optional[float] = None
    dataset_label_mode: str = "auto"
    dataset_overlap_threshold: float = 0.5
    dataset_always_on_force_target: bool = True
    dataset_e_class_oversample_ratio: int = 1
    dataset_split_mode: str = "scenario"
    dataset_n_holdout_nodes: int = 10
    selected_features: List[str] = field(default_factory=list)

    # Model defaults aligned to the new Chapter 1 main line
    model_type: str = "hydraulic_inverse"
    use_flow_direction: bool = True
    use_propagation_delay: bool = False
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
    lambda_type: float = 0.2
    use_seen_mrr_for_early_stopping: bool = True

    def __post_init__(self):
        if not self.input_dir_new:
            self.input_dir_new = _join(self.script_root, "input_1")
        if not self.training_data_dir:
            self.training_data_dir = _join(self.script_root, "training_data")

        if not self.candidate_nodes_file:
            self.candidate_nodes_file = _join(self.input_dir_new, "candidate_nodes_new.json")
        if not self.monitor_nodes_file:
            self.monitor_nodes_file = _join(self.input_dir_new, "monitor_nodes_downstream_N25.json")
        if not self.inp_file:
            self.inp_file = _join(self.data_root, "input_data", "2_tuned_v3_merged1.inp")
        if not self.adjacency_matrix_file:
            self.adjacency_matrix_file = _join(self.input_dir_new, "adj_matrix.npy")
        if not self.graph_path_features_file:
            self.graph_path_features_file = _join(self.input_dir_new, "graph_path_features.npz")
        if not self.node_list_file:
            self.node_list_file = _join(self.input_dir_new, "node_list.json")
        if not self.defect_matrix_file:
            self.defect_matrix_file = _join(self.input_dir_new, "defect_matrix_diverse.csv")
        if not self.node_timeseries_file:
            parquet_name = "node_timeseries_with_residuals.parquet"
            if self.training_data_subdir.strip():
                self.node_timeseries_file = _join(self.training_data_dir, self.training_data_subdir, parquet_name)
            else:
                self.node_timeseries_file = _join(self.training_data_dir, parquet_name)
        if not self.link_timeseries_file:
            if self.training_data_subdir.strip():
                self.link_timeseries_file = _join(self.training_data_dir, self.training_data_subdir, "link_timeseries.parquet")
            else:
                self.link_timeseries_file = _join(self.training_data_dir, "link_timeseries.parquet")
        if not self.node_timeseries_residual_file:
            if self.training_data_subdir.strip():
                self.node_timeseries_residual_file = _join(self.training_data_dir, self.training_data_subdir, "node_timeseries_with_residuals.parquet")
            else:
                self.node_timeseries_residual_file = _join(self.training_data_dir, "node_timeseries_with_residuals.parquet")
        if not self.link_timeseries_residual_file:
            if self.training_data_subdir.strip():
                self.link_timeseries_residual_file = _join(self.training_data_dir, self.training_data_subdir, "link_timeseries_with_residuals.parquet")
            else:
                self.link_timeseries_residual_file = _join(self.training_data_dir, "link_timeseries_with_residuals.parquet")
        if not self.edge_timeseries_file:
            if self.training_data_subdir.strip():
                self.edge_timeseries_file = _join(self.training_data_dir, self.training_data_subdir, "edge_timeseries.parquet")
            else:
                self.edge_timeseries_file = _join(self.training_data_dir, "edge_timeseries.parquet")

        self.parsed_inp_data_file = _join(self.input_dir_new, "parsed_inp_data.json")
        self.baseline_flow_stats_file = _join(self.input_dir_new, "baseline_flow_stats.json")
        if not self.segment_list_file:
            self.segment_list_file = _join(self.input_dir_new, "segment_list.json")
        if not self.edge_static_features_file:
            self.edge_static_features_file = _join(self.input_dir_new, "edge_static_features.csv")
        if not self.pipe_segment_map_file:
            self.pipe_segment_map_file = _join(self.input_dir_new, "pipe_segment_map.json")

        if not self.model_save_dir:
            self.model_save_dir = _join(self.script_root, "outputs", "model_checkpoints")
        if not self.log_dir:
            self.log_dir = _join(self.script_root, "outputs", "logs")
        if not self.reports_dir:
            self.reports_dir = _join(self.script_root, "outputs", "reports")
        if not self.dataset_processor_path:
            processor_name = "segment_processor.py" if self.task_mode == "segment" else "processor.py"
            self.dataset_processor_path = _join(self.script_root, "dataset", processor_name)
        if not self.output_dir:
            self.output_dir = self.model_save_dir

        if not self.selected_features:
            self.selected_features = [
                "depth",
                "pollut_BODf",
                "depth_residual",
                "depth_residual_rel",
                "pollut_BODf_residual",
                "pollut_BODf_residual_rel",
                "head",
                "volume",
                "lateral_inflow",
                "total_inflow",
                "head_residual",
                "head_residual_rel",
                "volume_residual",
                "volume_residual_rel",
                "lateral_inflow_residual",
                "lateral_inflow_residual_rel",
                "total_inflow_residual",
                "total_inflow_residual_rel",
                "flooding",
                "flooding_residual",
                "flooding_residual_rel",
            ]

    def ensure_output_dirs(self) -> None:
        for directory in (self.model_save_dir, self.log_dir, self.reports_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
