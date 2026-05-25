#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc candidate prior adjustment sweep.

Goal:
    Reduce over-attraction to a few frequent candidate nodes by adding a
    train-frequency-based logit bias at evaluation time only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config  # noqa: E402
from models.anomaly_detection_model import create_model  # noqa: E402
from train.train import (  # noqa: E402
    build_node_static_features,
    compute_split_diagnostics,
    estimate_process_signal_thresholds,
    load_data_module,
    _get_subset_base_and_indices,
)
from utils.evaluation import evaluate_model  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate prior logit adjustment sweep")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--model-type", default="", help="Override model type when checkpoint has no config snapshot")
    parser.add_argument("--taus", default="0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0", help="Comma-separated tau values")
    parser.add_argument("--metric", default="top1", choices=["top1", "mrr", "event_level_top1"], help="Validation selection metric")
    parser.add_argument("--output-json", default="", help="Optional output json path")
    return parser


def parse_taus(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def metric_value(metrics: Dict, name: str) -> float:
    aliases = {
        "top1": ["top1", "topk_recall_1"],
        "top3": ["top3", "topk_recall_3"],
        "top5": ["top5", "topk_recall_5"],
        "mrr": ["mrr"],
        "event_level_top1": ["event_level_top1"],
    }
    for key in aliases.get(name, [name]):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def compute_candidate_prior(dataset_base, train_indices, candidate_mask: np.ndarray) -> np.ndarray:
    counts = np.ones(len(candidate_mask), dtype=np.float64)  # Laplace smoothing on full graph
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    for idx in train_indices:
        sample = dataset_base.samples[int(idx)]
        if int(sample.get("active_label", 0)) <= 0:
            continue
        if int(sample.get("loc_enabled", 0)) <= 0:
            continue
        target_idx = int(sample.get("target_node_idx", -1))
        if target_idx >= 0 and target_idx < len(counts) and candidate_mask[target_idx]:
            counts[target_idx] += 1.0
    counts[~candidate_mask] = 1.0
    counts /= counts[candidate_mask].sum()
    return counts


def build_model(config: Config, dataset, device: torch.device):
    model_kw = dict(
        model_type=config.model_type,
        input_dim=int(getattr(dataset, "input_feature_dim", 0) or dataset.samples[0]["features"].shape[-1]),
        time_hidden_dim=config.time_hidden_dim,
        spatial_hidden_dim=config.spatial_hidden_dim,
        num_time_layers=config.num_time_layers,
        num_spatial_layers=config.num_spatial_layers,
        num_nodes=len(dataset.node_list),
        dropout=config.dropout,
    )

    if config.model_type in {
        "hydraulic_inverse",
        "hydraulic_inverse_ctx",
        "hydraulic_inverse_ctxfix",
        "hydraulic_inverse_typeaware",
        "hydraulic_inverse_lstm",
        "hydraulic_inverse_static",
        "hydraulic_inverse_staticbias",
    }:
        graph_path_file = getattr(config, "graph_path_features_file", "") or str(SCRIPT_ROOT / "input_1" / "graph_path_features.npz")
        npz = np.load(graph_path_file)
        graph_features_dict = {k: npz[k] for k in npz.files}
        model_kw["graph_features_dict"] = graph_features_dict
        model_kw["use_flow_direction"] = getattr(config, "use_flow_direction", True)
        model_kw["use_propagation_delay"] = getattr(config, "use_propagation_delay", False)
        model_kw["attention_max_hops"] = getattr(config, "hydraulic_attention_max_hops", 0)
        model_kw["observed_source_only"] = getattr(config, "hydraulic_observed_source_only", False)
        if config.model_type in {"hydraulic_inverse_static", "hydraulic_inverse_staticbias"}:
            node_static_features, _ = build_node_static_features(config, dataset, graph_features_dict)
            model_kw["node_static_features"] = node_static_features

    model = create_model(**model_kw).to(device)
    return model


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    snapshot: Dict = checkpoint.get("config_snapshot", {})

    config = Config()
    for key, value in snapshot.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Keep clean-mainline defaults from checkpoint; avoid accidentally reading stale config.
    config.node_timeseries_file = snapshot.get("node_timeseries_file", config.node_timeseries_file)
    config.monitor_nodes_file = snapshot.get("monitor_nodes_file", config.monitor_nodes_file)
    config.model_type = snapshot.get("model_type", "") or args.model_type or config.model_type

    create_dataloaders = load_data_module(config)
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        node_timeseries_file=config.node_timeseries_file,
        adjacency_matrix_file=config.adjacency_matrix_file,
        node_list_file=config.node_list_file,
        defect_matrix_file=config.defect_matrix_file,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        window_stride=config.window_stride,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        normalize=getattr(config, "normalize_features", True),
        random_seed=config.random_seed,
        candidate_nodes_file=config.candidate_nodes_file,
        feature_names=getattr(config, "selected_features", None),
        label_mode=config.dataset_label_mode,
        overlap_threshold=config.dataset_overlap_threshold,
        always_on_force_target=config.dataset_always_on_force_target,
        use_time_pos_encoding=getattr(config, "use_time_pos_encoding", True),
        use_observed_mask_feature=getattr(config, "use_observed_mask_feature", True),
        use_trend_feature=getattr(config, "use_trend_feature", False),
        active_overlap_threshold=getattr(config, "dataset_active_overlap_threshold", 0.5),
        transition_overlap_threshold=getattr(config, "dataset_transition_overlap_threshold", 0.0),
        e_class_oversample_ratio=getattr(config, "dataset_e_class_oversample_ratio", 1),
        split_mode=getattr(config, "dataset_split_mode", "scenario"),
        n_holdout_nodes=getattr(config, "dataset_n_holdout_nodes", 10),
        use_full_graph=getattr(config, "train_with_full_graph", False),
    )

    dataset_base, train_indices = _get_subset_base_and_indices(train_loader.dataset)
    active_signal_threshold, transition_signal_threshold = estimate_process_signal_thresholds(
        dataset_base,
        train_indices,
        active_quantile=getattr(config, "dataset_active_signal_quantile", 0.95),
        transition_quantile=getattr(config, "dataset_transition_signal_quantile", 0.90),
    )
    dataset_base.configure_process_labels(
        active_signal_threshold=active_signal_threshold,
        transition_signal_threshold=transition_signal_threshold,
    )

    train_seen_node_indices = set()
    for i in train_indices:
        sample = dataset_base.samples[i]
        if sample.get("target_node_idx", -1) >= 0:
            train_seen_node_indices.add(sample["target_node_idx"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, dataset, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion_loc = nn.KLDivLoss(reduction="batchmean")
    criterion_active = nn.CrossEntropyLoss(reduction="mean")

    candidate_mask_np = np.asarray(dataset.candidate_mask_np, dtype=bool)
    prior = compute_candidate_prior(dataset_base, train_indices, candidate_mask_np)
    bias_base = np.zeros(len(dataset.node_list), dtype=np.float32)
    candidate_indices = np.where(candidate_mask_np)[0]
    bias_base[candidate_indices] = np.log(prior[candidate_indices] + 1e-12).astype(np.float32)
    bias_base = torch.from_numpy(bias_base)

    taus = parse_taus(args.taus)
    results = []
    best_val = None
    best_tau = None
    best_test = None
    split_diag = compute_split_diagnostics(train_loader, val_loader, test_loader)

    for tau in taus:
        bias = -float(tau) * bias_base
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion_loc,
            device,
            criterion_active=criterion_active,
            active_prob_threshold=None,
            node_logit_bias=bias,
            train_seen_node_indices=train_seen_node_indices,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            criterion_loc,
            device,
            criterion_active=criterion_active,
            active_prob_threshold=None,
            node_logit_bias=bias,
            train_seen_node_indices=train_seen_node_indices,
        )
        row = {
            "tau": tau,
            "val_top1": metric_value(val_metrics, "top1"),
            "val_mrr": metric_value(val_metrics, "mrr"),
            "val_event_top1": float(val_metrics.get("event_level_top1", 0.0)),
            "test_top1": metric_value(test_metrics, "top1"),
            "test_mrr": metric_value(test_metrics, "mrr"),
            "test_top3": metric_value(test_metrics, "top3"),
            "test_top5": metric_value(test_metrics, "top5"),
            "test_event_top1": float(test_metrics.get("event_level_top1", 0.0)),
            "test_event_top3": float(test_metrics.get("event_level_top3", 0.0)),
            "test_event_top5": float(test_metrics.get("event_level_top5", 0.0)),
        }
        results.append(row)
        score = metric_value(val_metrics, args.metric)
        if best_val is None or score > best_val + 1e-12:
            best_val = score
            best_tau = tau
            best_test = test_metrics
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "checkpoint": str(checkpoint_path),
        "metric": args.metric,
        "best_tau": best_tau,
        "best_val_metric": best_val,
        "best_test_metrics": best_test,
        "candidate_prior_top10": [
            {
                "node_id": str(dataset.node_list[idx]),
                "prior": float(prior[idx]),
            }
            for pos, idx in sorted(
                enumerate(candidate_indices),
                key=lambda x: prior[x[1]],
                reverse=True,
            )[:10]
        ],
        "split_diagnostics": split_diag,
        "rows": results,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
