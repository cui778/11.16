#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-graph sparse-observation training entry.

Two modes are supported:

1. direct sparse-observation student training
   - full 128-node graph is preserved
   - only monitor nodes keep input values
   - non-monitor node features are zero-masked

2. teacher-student distillation
   - same student setting as above
   - optional KD from a full-node teacher

This script is now used not only for KD probes, but also as the main
training entry for full-graph sparse-observation Chapter-1 experiments.
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from dataset.processor import create_dataloaders
from models.anomaly_detection_model import create_model
from train.train import (
    build_artifact_paths,
    compute_split_diagnostics,
    estimate_process_signal_thresholds,
)
from utils.evaluation import evaluate_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


RESIDUAL8 = [
    "depth_residual",
    "total_outflow_residual",
    "pollut_NH4_residual",
    "pollut_TSSs_residual",
    "depth_residual_rel",
    "total_outflow_residual_rel",
    "pollut_NH4_residual_rel",
    "pollut_TSSs_residual_rel",
]

RAW4 = [
    "depth",
    "total_outflow",
    "pollut_NH4",
    "pollut_TSSs",
]


def _features_for_set(feature_set: str) -> List[str]:
    mode = str(feature_set or "residual_only").strip().lower()
    if mode == "raw_only":
        return list(RAW4)
    if mode == "raw_plus_residual":
        return list(RAW4) + list(RESIDUAL8)
    return list(RESIDUAL8)


HYDRAULIC_MODEL_TYPES = {
    "hydraulic_inverse",
    "hydraulic_inverse_ctx",
    "hydraulic_inverse_ctxfix",
    "hydraulic_inverse_typeaware",
    "hydraulic_inverse_lstm",
    "hydraulic_inverse_static",
    "hydraulic_inverse_staticbias",
    "hydraulic_inverse_deepattn_static",
    "hydraulic_inverse_deepattn_staticbias",
    "hydraulic_inverse_deepattn_recon",
    "hydraulic_inverse_deepattn",
}


def _build_config(
    subdir: str,
    seed: int,
    monitor_nodes_file: str,
    split_mode: str = "scenario",
    n_holdout_nodes: int = 10,
    adjacency_matrix_file: str = "",
    graph_path_features_file: str = "",
    defect_matrix_file: str = "",
    use_time_pos_encoding: Optional[bool] = None,
    use_trend_feature: Optional[bool] = None,
    label_mode: Optional[str] = None,
    always_on_force_target: Optional[bool] = None,
    active_signal_quantile: Optional[float] = None,
    transition_signal_quantile: Optional[float] = None,
    feature_set: str = "residual_only",
    sequence_length: Optional[int] = None,
    window_stride: Optional[int] = None,
) -> Config:
    cfg = Config()
    cfg.training_data_subdir = subdir
    cfg.node_timeseries_file = os.path.normpath(
        os.path.join(cfg.training_data_dir, subdir, "node_timeseries_with_residuals.parquet")
    )
    cfg.monitor_nodes_file = monitor_nodes_file
    cfg.selected_features = _features_for_set(feature_set)
    cfg.random_seed = int(seed)
    cfg.dataset_split_mode = str(split_mode)
    cfg.dataset_n_holdout_nodes = int(n_holdout_nodes)
    cfg.model_type = "hydraulic_inverse_deepattn"
    cfg.lambda_type = 0.0
    cfg.lambda_rank = 0.0
    cfg.lambda_balance = 0.0
    cfg.lambda_recon = 0.0
    if sequence_length is not None:
        cfg.sequence_length = int(sequence_length)
    if window_stride is not None:
        cfg.window_stride = int(window_stride)
    if label_mode:
        cfg.dataset_label_mode = str(label_mode)
    if always_on_force_target is not None:
        cfg.dataset_always_on_force_target = bool(always_on_force_target)
    if active_signal_quantile is not None:
        cfg.dataset_active_signal_quantile = float(active_signal_quantile)
    if transition_signal_quantile is not None:
        cfg.dataset_transition_signal_quantile = float(transition_signal_quantile)
    if defect_matrix_file:
        cfg.defect_matrix_file = os.path.normpath(defect_matrix_file)
    else:
        manifest_path = Path(cfg.training_data_dir) / subdir / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_defect_csv = str(manifest.get("defect_csv_path", "")).strip()
                if manifest_defect_csv:
                    cfg.defect_matrix_file = os.path.normpath(manifest_defect_csv)
            except Exception:
                pass
    if use_time_pos_encoding is not None:
        cfg.use_time_pos_encoding = bool(use_time_pos_encoding)
    if use_trend_feature is not None:
        cfg.use_trend_feature = bool(use_trend_feature)
    if adjacency_matrix_file:
        cfg.adjacency_matrix_file = os.path.normpath(adjacency_matrix_file)
    if graph_path_features_file:
        cfg.graph_path_features_file = os.path.normpath(graph_path_features_file)
    cfg.ensure_output_dirs()
    return cfg


def _create_model(cfg: Config, dataset, device: torch.device):
    input_dim = int(dataset.input_feature_dim)
    num_nodes = int(len(dataset.node_list))
    model_kw = dict(
        model_type=cfg.model_type,
        input_dim=input_dim,
        time_hidden_dim=cfg.time_hidden_dim,
        spatial_hidden_dim=cfg.spatial_hidden_dim,
        num_time_layers=cfg.num_time_layers,
        num_spatial_layers=cfg.num_spatial_layers,
        num_nodes=num_nodes,
        dropout=cfg.dropout,
    )
    if cfg.model_type.startswith("hydraulic_inverse_deepattn"):
        model_kw["allow_shallow_deepattn"] = getattr(
            cfg, "allow_shallow_deepattn", False
        )
    if cfg.model_type in HYDRAULIC_MODEL_TYPES:
        npz = np.load(cfg.graph_path_features_file)
        graph_features_dict = {k: npz[k] for k in npz.files}
        model_kw["graph_features_dict"] = graph_features_dict
        model_kw["use_flow_direction"] = getattr(cfg, "use_flow_direction", True)
        model_kw["path_prior_mode"] = getattr(cfg, "path_prior_mode", "full")
        model_kw["use_propagation_delay"] = getattr(cfg, "use_propagation_delay", False)
        model_kw["propagation_delay_velocity_mps"] = getattr(cfg, "propagation_delay_velocity_mps", 0.5)
        model_kw["attention_max_hops"] = getattr(cfg, "hydraulic_attention_max_hops", 0)
        model_kw["observed_source_only"] = getattr(cfg, "hydraulic_observed_source_only", False)
    return create_model(**model_kw).to(device)


def _make_loaders(cfg: Config, observed_nodes_file: str = ""):
    return create_dataloaders(
        node_timeseries_file=cfg.node_timeseries_file,
        adjacency_matrix_file=cfg.adjacency_matrix_file,
        node_list_file=cfg.node_list_file,
        defect_matrix_file=cfg.defect_matrix_file,
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        window_stride=cfg.window_stride,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        normalize=True,
        random_seed=cfg.random_seed,
        feature_names=cfg.selected_features,
        candidate_nodes_file=cfg.candidate_nodes_file,
        soft_label_sigma=cfg.soft_label_sigma,
        use_full_graph=cfg.train_with_full_graph,
        signal_threshold=cfg.dataset_signal_threshold,
        label_mode=cfg.dataset_label_mode,
        overlap_threshold=cfg.dataset_overlap_threshold,
        always_on_force_target=cfg.dataset_always_on_force_target,
        use_time_pos_encoding=cfg.use_time_pos_encoding,
        use_observed_mask_feature=cfg.use_observed_mask_feature,
        observed_nodes_file=observed_nodes_file or None,
        use_trend_feature=cfg.use_trend_feature,
        active_overlap_threshold=cfg.dataset_active_overlap_threshold,
        transition_overlap_threshold=cfg.dataset_transition_overlap_threshold,
        active_signal_threshold=None,
        transition_signal_threshold=None,
        e_class_oversample_ratio=cfg.dataset_e_class_oversample_ratio,
        persistent_train_sample_ratio=cfg.dataset_persistent_train_sample_ratio,
        split_mode=cfg.dataset_split_mode,
        n_holdout_nodes=cfg.dataset_n_holdout_nodes,
    )


def _get_subset_base_and_indices(ds):
    base = getattr(ds, "dataset", ds)
    indices = getattr(ds, "indices", range(len(ds)))
    return base, indices


def _configure_process_thresholds_if_needed(cfg: Config, train_loader: DataLoader) -> None:
    label_mode = str(getattr(cfg, "dataset_label_mode", "auto")).strip().lower()
    if label_mode != "always_on":
        return

    dataset_base, train_indices = _get_subset_base_and_indices(train_loader.dataset)
    active_signal_threshold, transition_signal_threshold = estimate_process_signal_thresholds(
        dataset_base,
        train_indices,
        active_quantile=getattr(cfg, "dataset_active_signal_quantile", 0.95),
        transition_quantile=getattr(cfg, "dataset_transition_signal_quantile", 0.90),
    )
    dataset_base.configure_process_labels(
        active_signal_threshold=active_signal_threshold,
        transition_signal_threshold=transition_signal_threshold,
    )
    if active_signal_threshold is not None:
        logger.info(
            "Process thresholds: active_signal_threshold=%.4f transition_signal_threshold=%.4f",
            active_signal_threshold,
            transition_signal_threshold,
        )
    else:
        logger.info("Process thresholds: no valid always_on threshold estimated; dataset keeps compatible defaults.")


def _iter_loader_no_shuffle(loader: DataLoader) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
    )


def _sample_keys_from_batch(batch) -> List[str]:
    scenario_ids = batch["scenario_id"]
    window_times = batch["window_start_time"]
    if isinstance(scenario_ids, torch.Tensor):
        scenario_ids = scenario_ids.view(-1).cpu().tolist()
    else:
        scenario_ids = list(scenario_ids)
    if not isinstance(window_times, (list, tuple)):
        window_times = [window_times] * len(scenario_ids)
    return [f"{int(sid)}|{str(ts)}" for sid, ts in zip(scenario_ids, window_times)]


def _mask_unobserved_features(x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    mask = observed_mask.bool().unsqueeze(1).unsqueeze(-1)
    return x * mask.float()


class MaskedInputModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        if name == "model":
            return super().__getattr__(name)
        return getattr(self.model, name)

    def forward(self, x, adj, *args, **kwargs):
        observed_mask = x[:, 0, :, -1] > 0.5
        x_masked = _mask_unobserved_features(x, observed_mask)
        return self.model(x_masked, adj, *args, **kwargs)


def _precompute_teacher_cache(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    eval_loader = _iter_loader_no_shuffle(loader)
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["features"].to(device)
            adj = batch["adj_matrix"]
            if isinstance(adj, (list, tuple)):
                adj = adj[0].to(device)
            elif adj.dim() == 3:
                adj = adj[0].to(device)
            else:
                adj = adj.to(device)
            outputs = model(x, adj)
            logits_node = outputs[0] if isinstance(outputs, tuple) else outputs
            logits_active = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            scores = logits_node[:, :, 1].detach().cpu().numpy().astype(np.float32)
            active_logits = None if logits_active is None else logits_active.detach().cpu().numpy().astype(np.float32)
            for idx, key in enumerate(_sample_keys_from_batch(batch)):
                row = {"scores": scores[idx]}
                if active_logits is not None:
                    row["active_logits"] = active_logits[idx]
                cache[key] = row
    return cache


def _gather_teacher_targets(cache: Dict[str, Dict[str, np.ndarray]], batch_keys: List[str], device: torch.device):
    scores = torch.tensor(np.stack([cache[k]["scores"] for k in batch_keys], axis=0), dtype=torch.float32, device=device)
    active_logits = None
    if cache and "active_logits" in next(iter(cache.values())):
        active_logits = torch.tensor(
            np.stack([cache[k]["active_logits"] for k in batch_keys], axis=0),
            dtype=torch.float32,
            device=device,
        )
    return scores, active_logits


def _teacher_enabled(lambda_kd: float, lambda_active_kd: float) -> bool:
    return float(lambda_kd) > 0.0 or float(lambda_active_kd) > 0.0


def _evaluate(
    masked_student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_seen_node_indices: set,
    active_threshold: Optional[float] = None,
    scene_threshold: Optional[float] = None,
    scene_score_mode: str = "topk_mean",
    scene_topk: int = 5,
):
    criterion_loc_eval = nn.KLDivLoss(reduction="batchmean")
    criterion_active_eval = nn.CrossEntropyLoss(reduction="mean")
    return evaluate_model(
        masked_student,
        loader,
        criterion_loc_eval,
        device,
        criterion_active=criterion_active_eval,
        active_prob_threshold=active_threshold,
        scene_active_threshold=scene_threshold,
        scene_score_mode=scene_score_mode,
        scene_topk=scene_topk,
        train_seen_node_indices=train_seen_node_indices,
    )


def main():
    parser = argparse.ArgumentParser(description="Full-graph sparse-observation training / distillation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-monitors", default=r"e:\11.16\script2_new\input_1\monitor_nodes_observability_aware_N25.json")
    parser.add_argument("--teacher-subdir", default="time_gated_full_v2e_dense_ie_truefull")
    parser.add_argument(
        "--teacher-run-tag",
        default="process_diagnosis_hydraulic_inverse_deepattn_time_gated_full_v2e_dense_ie_truefull_residual8_scenario_seed42",
    )
    parser.add_argument("--student-model-type", default="hydraulic_inverse_deepattn")
    parser.add_argument("--num-spatial-layers", type=int, default=None)
    parser.add_argument(
        "--path-prior-mode",
        default="full",
        choices=["full", "distance_only", "none"],
        help=(
            "Path values exposed to hydraulic attention. Reachability masking "
            "is retained for every mode."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--lambda-loc", type=float, default=None)
    parser.add_argument("--lambda-kd", type=float, default=0.7)
    parser.add_argument("--lambda-active-kd", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--split-mode", default="scenario", choices=["scenario", "node_holdout"])
    parser.add_argument("--n-holdout-nodes", type=int, default=10)
    parser.add_argument("--adjacency-matrix-file", default="")
    parser.add_argument("--graph-path-features-file", default="")
    parser.add_argument("--experiment-suffix", default="")
    parser.add_argument("--output-tag", default="", help="Optional short output tag to avoid overly long artifact paths")
    parser.add_argument("--use-time-pos-encoding", dest="use_time_pos_encoding", action="store_true")
    parser.add_argument("--no-time-pos-encoding", dest="use_time_pos_encoding", action="store_false")
    parser.set_defaults(use_time_pos_encoding=None)
    parser.add_argument("--use-trend-feature", dest="use_trend_feature", action="store_true")
    parser.add_argument("--no-trend-feature", dest="use_trend_feature", action="store_false")
    parser.set_defaults(use_trend_feature=None)
    parser.add_argument("--label-mode", default="", choices=["", "auto", "time_gated", "always_on"])
    parser.add_argument("--always-on-force-target", dest="always_on_force_target", action="store_true")
    parser.add_argument("--no-always-on-force-target", dest="always_on_force_target", action="store_false")
    parser.set_defaults(always_on_force_target=None)
    parser.add_argument("--active-signal-quantile", type=float, default=None)
    parser.add_argument("--transition-signal-quantile", type=float, default=None)
    parser.add_argument("--feature-set", default="residual_only", choices=["raw_only", "residual_only", "raw_plus_residual"])
    parser.add_argument("--active-threshold", type=float, default=None)
    parser.add_argument("--scene-threshold", type=float, default=None)
    parser.add_argument("--scene-score-mode", default="topk_mean", choices=["topk_mean", "max"])
    parser.add_argument("--scene-topk", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--persistent-train-sample-ratio", type=float, default=1.0)
    parser.add_argument("--run-persistent-zero-shot", action="store_true")
    parser.add_argument("--persistent-subdir", default="persistent_ie_fullwindow_v1_seed42")
    parser.add_argument(
        "--persistent-defect-csv",
        default=str(SCRIPT_ROOT / "input_1" / "defect_matrix_persistent_ie_v1_seed42.csv"),
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    teacher_cfg = _build_config(
        args.teacher_subdir,
        args.seed,
        monitor_nodes_file=r"e:\11.16\script2_new\input_1\monitor_nodes_full_N128.json",
        split_mode=args.split_mode,
        n_holdout_nodes=args.n_holdout_nodes,
        adjacency_matrix_file=args.adjacency_matrix_file,
        graph_path_features_file=args.graph_path_features_file,
        use_time_pos_encoding=args.use_time_pos_encoding,
        use_trend_feature=args.use_trend_feature,
        label_mode=args.label_mode or None,
        always_on_force_target=args.always_on_force_target,
        active_signal_quantile=args.active_signal_quantile,
        transition_signal_quantile=args.transition_signal_quantile,
        feature_set="residual_only",
        sequence_length=args.sequence_length,
        window_stride=args.window_stride,
    )
    student_cfg = copy.deepcopy(teacher_cfg)
    student_cfg.monitor_nodes_file = os.path.normpath(args.student_monitors)
    student_cfg.model_type = str(args.student_model_type).strip()
    if args.num_spatial_layers is not None:
        if int(args.num_spatial_layers) < 1:
            parser.error("--num-spatial-layers must be >= 1")
        student_cfg.num_spatial_layers = int(args.num_spatial_layers)
        student_cfg.allow_shallow_deepattn = True
    student_cfg.path_prior_mode = str(args.path_prior_mode)
    student_cfg.selected_features = _features_for_set(args.feature_set)
    student_cfg.num_epochs = int(args.num_epochs)
    student_cfg.dataset_persistent_train_sample_ratio = float(args.persistent_train_sample_ratio)
    if args.lambda_loc is not None:
        student_cfg.lambda_loc = float(args.lambda_loc)
    kd_tag = f"kd{str(float(args.lambda_kd)).replace('.', 'p')}"
    active_kd_tag = f"akd{str(float(args.lambda_active_kd)).replace('.', 'p')}"
    time_tag = "tpos1" if bool(student_cfg.use_time_pos_encoding) else "tpos0"
    trend_tag = "trend1" if bool(student_cfg.use_trend_feature) else "trend0"
    loc_tag = f"loc{str(float(student_cfg.lambda_loc)).replace('.', 'p')}"
    feat_tag = str(args.feature_set).strip().lower()
    extra_tag = ""
    if str(args.experiment_suffix).strip():
        extra_tag = f"_{str(args.experiment_suffix).strip()}"
    default_output_tag = (
        f"process_diagnosis_privileged_student_{student_cfg.model_type}_{Path(args.student_monitors).stem}_"
        f"{args.teacher_subdir}_{student_cfg.dataset_split_mode}_{time_tag}_{trend_tag}_{feat_tag}_{loc_tag}_{kd_tag}_{active_kd_tag}"
        f"{extra_tag}_seed{args.seed}"
    )
    student_cfg.output_tag = str(args.output_tag).strip() or default_output_tag
    student_cfg.ensure_output_dirs()
    artifact_paths = build_artifact_paths(student_cfg, student_cfg.output_tag)

    use_teacher = _teacher_enabled(args.lambda_kd, args.lambda_active_kd)

    logger.info("Loading student data...")
    student_train, student_val, student_test, student_dataset = _make_loaders(
        student_cfg,
        observed_nodes_file=student_cfg.monitor_nodes_file,
    )
    _configure_process_thresholds_if_needed(student_cfg, student_train)

    train_cache: Dict[str, Dict[str, np.ndarray]] = {}
    val_cache: Dict[str, Dict[str, np.ndarray]] = {}
    test_cache: Dict[str, Dict[str, np.ndarray]] = {}
    teacher_ckpt_str: Optional[str] = None
    if use_teacher:
        logger.info("Loading teacher data...")
        teacher_train, teacher_val, teacher_test, teacher_dataset = _make_loaders(teacher_cfg)
        _configure_process_thresholds_if_needed(teacher_cfg, teacher_train)
        teacher_model = _create_model(teacher_cfg, teacher_dataset, device)
        teacher_ckpt = Path(teacher_cfg.output_dir) / f"best_model_{args.teacher_run_tag}.pth"
        checkpoint = torch.load(str(teacher_ckpt), map_location=device)
        teacher_model.load_state_dict(checkpoint["model_state_dict"])
        teacher_model.eval()
        teacher_ckpt_str = str(teacher_ckpt)
        logger.info("Teacher checkpoint: %s", teacher_ckpt)

        logger.info("Precomputing teacher cache...")
        train_cache = _precompute_teacher_cache(teacher_model, teacher_train, device)
        val_cache = _precompute_teacher_cache(teacher_model, teacher_val, device)
        test_cache = _precompute_teacher_cache(teacher_model, teacher_test, device)
        logger.info("Teacher cache sizes: train=%d val=%d test=%d", len(train_cache), len(val_cache), len(test_cache))
    else:
        logger.info("Teacher distillation disabled; student will be trained directly.")

    student_model = _create_model(student_cfg, student_dataset, device)
    optimizer = optim.Adam(student_model.parameters(), lr=student_cfg.learning_rate, weight_decay=student_cfg.weight_decay)
    criterion_loc = nn.KLDivLoss(reduction="batchmean")
    criterion_active = nn.CrossEntropyLoss(reduction="mean")

    train_diag = compute_split_diagnostics(student_train, student_val, student_test)
    seen_set = set()
    base = getattr(student_train.dataset, "dataset", student_train.dataset)
    indices = getattr(student_train.dataset, "indices", range(len(student_train.dataset)))
    for i in indices:
        s = base.samples[i]
        seen_set.add(int(s.get("defect_node_idx", -1)))
    seen_set.discard(-1)

    best_metric = -1.0
    best_state = None
    history_rows = []
    patience = 8
    wait = 0

    for epoch in range(1, student_cfg.num_epochs + 1):
        student_model.train()
        total_loss = total_active = total_loc = total_kd = total_active_kd = 0.0
        n_batches = 0

        for batch in student_train:
            x = batch["features"].to(device)
            observed_mask = batch["observed_mask"].to(device).bool()
            x_masked = _mask_unobserved_features(x, observed_mask)

            adj = batch["adj_matrix"]
            if isinstance(adj, (list, tuple)):
                adj = adj[0].to(device)
            elif adj.dim() == 3:
                adj = adj[0].to(device)
            else:
                adj = adj.to(device)

            active_label = batch["active_label"].long().to(device)
            target_node_idx = batch["target_node_idx"].long().to(device)
            soft_labels = batch["soft_label"].to(device)
            candidate_mask = batch["candidate_mask"].to(device).bool()
            if candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.unsqueeze(0).expand(x.shape[0], -1)

            teacher_scores = None
            teacher_active_logits = None
            if use_teacher:
                batch_keys = _sample_keys_from_batch(batch)
                teacher_scores, teacher_active_logits = _gather_teacher_targets(train_cache, batch_keys, device)

            optimizer.zero_grad()
            outputs = student_model(x_masked, adj)
            logits_node = outputs[0] if isinstance(outputs, tuple) else outputs
            logits_active = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None

            loss_active = criterion_active(logits_active, active_label)

            scores = logits_node[:, :, 1]
            loc_mask = (active_label > 0) & (soft_labels.sum(dim=-1) > 1e-8)
            if loc_mask.sum() > 0:
                scores_loc = scores[loc_mask]
                target_loc = target_node_idx[loc_mask]
                soft_loc = soft_labels[loc_mask]
                cand_loc = candidate_mask[loc_mask]
                scores_masked = scores_loc.masked_fill(~cand_loc, -1e9)
                soft_masked = soft_loc.masked_fill(~cand_loc, 0.0)
                soft_norm = soft_masked / soft_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                loss_loc = criterion_loc(torch.log_softmax(scores_masked, dim=-1), soft_norm)

                if use_teacher and teacher_scores is not None:
                    teacher_scores_loc = teacher_scores[loc_mask].masked_fill(~cand_loc, -1e9)
                    t = float(args.temperature)
                    loss_kd = F.kl_div(
                        torch.log_softmax(scores_masked / t, dim=-1),
                        torch.softmax(teacher_scores_loc / t, dim=-1),
                        reduction="batchmean",
                    ) * (t * t)
                else:
                    loss_kd = torch.zeros((), device=device)
            else:
                loss_loc = torch.zeros((), device=device)
                loss_kd = torch.zeros((), device=device)

            loss_active_kd = torch.zeros((), device=device)
            if use_teacher and teacher_active_logits is not None:
                t = float(args.temperature)
                loss_active_kd = F.kl_div(
                    torch.log_softmax(logits_active / t, dim=-1),
                    torch.softmax(teacher_active_logits / t, dim=-1),
                    reduction="batchmean",
                ) * (t * t)

            loss = (
                loss_active
                + float(student_cfg.lambda_loc) * loss_loc
                + float(args.lambda_kd) * loss_kd
                + float(args.lambda_active_kd) * loss_active_kd
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_active += float(loss_active.item())
            total_loc += float(loss_loc.item())
            total_kd += float(loss_kd.item())
            total_active_kd += float(loss_active_kd.item())
            n_batches += 1

        masked_student = MaskedInputModel(student_model)
        val_metrics = _evaluate(
            masked_student,
            student_val,
            device,
            seen_set,
            active_threshold=args.active_threshold,
            scene_threshold=args.scene_threshold,
            scene_score_mode=args.scene_score_mode,
            scene_topk=args.scene_topk,
        )
        val_mrr = float(val_metrics.get("mrr", 0.0))
        history_rows.append({
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "train_active_loss": total_active / max(n_batches, 1),
            "train_loc_loss": total_loc / max(n_batches, 1),
            "train_kd_loss": total_kd / max(n_batches, 1),
            "train_active_kd_loss": total_active_kd / max(n_batches, 1),
            "val_mrr": val_mrr,
            "val_top1": float(val_metrics.get("topk_recall_1", 0.0)),
            "val_top3": float(val_metrics.get("topk_recall_3", 0.0)),
            "val_top5": float(val_metrics.get("topk_recall_5", 0.0)),
            "val_active_acc": float(val_metrics.get("active_acc", 0.0)),
            "val_active_f1": float(val_metrics.get("active_f1", 0.0)),
            "val_normal_window_fpr": float(val_metrics.get("normal_window_fpr", 0.0)),
            "val_scene_acc": float(val_metrics.get("scene_has_defect_acc", 0.0)),
            "val_scene_recall": float(val_metrics.get("scene_recall", 0.0)),
            "val_scene_fpr": float(val_metrics.get("scene_fpr", 0.0)),
            "val_scene_f1": float(val_metrics.get("scene_f1", 0.0)),
        })
        logger.info(
            "Epoch %d/%d loss=%.4f val_mrr=%.4f val_top1=%.4f val_top3=%.4f val_top5=%.4f val_f1=%.4f scene_f1=%.4f scene_fpr=%.4f",
            epoch,
            student_cfg.num_epochs,
            total_loss / max(n_batches, 1),
            val_mrr,
            float(val_metrics.get("topk_recall_1", 0.0)),
            float(val_metrics.get("topk_recall_3", 0.0)),
            float(val_metrics.get("topk_recall_5", 0.0)),
            float(val_metrics.get("active_f1", 0.0)),
            float(val_metrics.get("scene_f1", 0.0)),
            float(val_metrics.get("scene_fpr", 0.0)),
        )

        if val_mrr > best_metric:
            wait = 0
            best_metric = val_mrr
            best_state = copy.deepcopy(student_model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mrr": best_metric,
                    "teacher_checkpoint": teacher_ckpt_str,
                    "student_monitor_nodes_file": student_cfg.monitor_nodes_file,
                    "student_model_type": student_cfg.model_type,
                    "num_spatial_layers": int(student_cfg.num_spatial_layers),
                    "resolved_num_spatial_layers": int(
                        getattr(student_model, "deep_num_layers", student_cfg.num_spatial_layers)
                    ),
                    "allow_shallow_deepattn": bool(student_cfg.allow_shallow_deepattn),
                    "path_prior_mode": str(student_cfg.path_prior_mode),
                    "split_diag": train_diag,
                },
                str(artifact_paths["checkpoint"]),
            )
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is None:
        raise RuntimeError("Student training did not produce a valid checkpoint")

    student_model.load_state_dict(best_state)
    masked_student = MaskedInputModel(student_model)
    test_metrics = _evaluate(
        masked_student,
        student_test,
        device,
        seen_set,
        active_threshold=args.active_threshold,
        scene_threshold=args.scene_threshold,
        scene_score_mode=args.scene_score_mode,
        scene_topk=args.scene_topk,
    )
    test_metrics["teacher_checkpoint"] = teacher_ckpt_str
    test_metrics["student_monitor_nodes_file"] = student_cfg.monitor_nodes_file
    test_metrics["teacher_subdir"] = args.teacher_subdir
    test_metrics["student_subdir"] = args.teacher_subdir
    test_metrics["student_model_type"] = student_cfg.model_type
    test_metrics["lambda_kd"] = float(args.lambda_kd)
    test_metrics["lambda_active_kd"] = float(args.lambda_active_kd)
    test_metrics["lambda_loc"] = float(student_cfg.lambda_loc)
    test_metrics["temperature"] = float(args.temperature)
    test_metrics["feature_set"] = str(args.feature_set)
    test_metrics["num_spatial_layers"] = int(student_cfg.num_spatial_layers)
    test_metrics["resolved_num_spatial_layers"] = int(
        getattr(student_model, "deep_num_layers", student_cfg.num_spatial_layers)
    )
    test_metrics["allow_shallow_deepattn"] = bool(student_cfg.allow_shallow_deepattn)
    test_metrics["path_prior_mode"] = str(student_cfg.path_prior_mode)
    test_metrics["active_threshold"] = None if args.active_threshold is None else float(args.active_threshold)
    test_metrics["scene_threshold_arg"] = None if args.scene_threshold is None else float(args.scene_threshold)
    test_metrics["scene_score_mode_arg"] = str(args.scene_score_mode)
    test_metrics["scene_topk_arg"] = int(args.scene_topk)
    test_metrics["sequence_length"] = int(student_cfg.sequence_length)
    test_metrics["window_stride"] = int(student_cfg.window_stride)
    test_metrics["tag"] = student_cfg.output_tag

    if args.run_persistent_zero_shot:
        persistent_cfg = copy.deepcopy(student_cfg)
        persistent_cfg.training_data_subdir = str(args.persistent_subdir)
        persistent_cfg.node_timeseries_file = os.path.normpath(
            os.path.join(
                persistent_cfg.training_data_dir,
                persistent_cfg.training_data_subdir,
                "node_timeseries_with_residuals.parquet",
            )
        )
        persistent_cfg.defect_matrix_file = os.path.normpath(str(args.persistent_defect_csv))
        persistent_cfg.dataset_label_mode = "time_gated"
        logger.info("Running persistent IE zero-shot evaluation: %s", persistent_cfg.node_timeseries_file)
        _, _, _, _persistent_dataset = _make_loaders(
            persistent_cfg,
            observed_nodes_file=persistent_cfg.monitor_nodes_file,
        )
        persistent_loader = DataLoader(
            _persistent_dataset,
            batch_size=persistent_cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        persistent_metrics = _evaluate(
            masked_student,
            persistent_loader,
            device,
            seen_set,
            active_threshold=args.active_threshold,
            scene_threshold=args.scene_threshold,
            scene_score_mode=args.scene_score_mode,
            scene_topk=args.scene_topk,
        )
        persistent_metrics["persistent_subdir"] = persistent_cfg.training_data_subdir
        persistent_metrics["persistent_defect_csv"] = persistent_cfg.defect_matrix_file
        persistent_metrics["latency_applicable"] = False
        test_metrics["persistent_zero_shot"] = persistent_metrics

    Path(artifact_paths["history"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(artifact_paths["history"], index=False, encoding="utf-8-sig")
    with open(artifact_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    logger.info(
        "Test MRR=%.4f Top1=%.4f Top3=%.4f Top5=%.4f ActiveF1=%.4f NormalFPR=%.4f SceneF1=%.4f SceneFPR=%.4f",
        float(test_metrics.get("mrr", 0.0)),
        float(test_metrics.get("topk_recall_1", 0.0)),
        float(test_metrics.get("topk_recall_3", 0.0)),
        float(test_metrics.get("topk_recall_5", 0.0)),
        float(test_metrics.get("active_f1", 0.0)),
        float(test_metrics.get("normal_window_fpr", 0.0)),
        float(test_metrics.get("scene_f1", 0.0)),
        float(test_metrics.get("scene_fpr", 0.0)),
    )
    if "persistent_zero_shot" in test_metrics:
        pz = test_metrics["persistent_zero_shot"]
        logger.info(
            "Persistent zero-shot Recall=%.4f SceneRecall=%.4f Top1=%.4f MRR=%.4f",
            float(pz.get("active_period_recall", 0.0)),
            float(pz.get("scene_recall", 0.0)),
            float(pz.get("topk_recall_1", 0.0)),
            float(pz.get("mrr", 0.0)),
        )
    logger.info("Saved checkpoint -> %s", artifact_paths["checkpoint"])
    logger.info("Saved metrics -> %s", artifact_paths["metrics"])


if __name__ == "__main__":
    main()
