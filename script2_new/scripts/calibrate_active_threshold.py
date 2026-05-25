#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from models.anomaly_detection_model import create_model
from train.train import load_data_module, estimate_process_signal_thresholds
from utils.evaluation import evaluate_model


def build_model(config, input_dim, num_nodes, device):
    model_kw = dict(
        model_type=config.model_type,
        input_dim=input_dim,
        time_hidden_dim=config.time_hidden_dim,
        spatial_hidden_dim=config.spatial_hidden_dim,
        num_time_layers=config.num_time_layers,
        num_spatial_layers=config.num_spatial_layers,
        num_nodes=num_nodes,
        dropout=config.dropout,
    )
    if config.model_type in {"hydraulic_inverse", "hydraulic_inverse_lstm"}:
        graph_path_file = getattr(config, "graph_path_features_file", "") or os.path.join(
            SCRIPT_ROOT, "input_1", "graph_path_features.npz"
        )
        npz = np.load(graph_path_file)
        model_kw["graph_features_dict"] = {k: npz[k] for k in npz.files}
        model_kw["use_flow_direction"] = getattr(config, "use_flow_direction", True)
        model_kw["use_propagation_delay"] = getattr(config, "use_propagation_delay", False)
    return create_model(**model_kw).to(device)


def fbeta_score(precision, recall, beta=2.0):
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def main():
    parser = argparse.ArgumentParser(description="Calibrate active detection threshold on validation split.")
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split-mode", default="scenario", choices=["scenario", "node_holdout"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    args = parser.parse_args()

    config = Config()
    config.training_data_subdir = args.subdir.strip()
    config.model_type = args.model_type.strip()
    config.random_seed = int(args.seed)
    config.dataset_split_mode = args.split_mode.strip()
    config.selected_features = [f.strip() for f in args.features.split(",") if f.strip()]
    config.node_timeseries_file = os.path.normpath(
        os.path.join(config.training_data_dir, config.training_data_subdir, "node_timeseries_with_residuals.parquet")
    )

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
        normalize=True,
        random_seed=config.random_seed,
        feature_names=config.selected_features,
        candidate_nodes_file=config.candidate_nodes_file,
        soft_label_sigma=config.soft_label_sigma,
        use_full_graph=config.train_with_full_graph,
        signal_threshold=config.dataset_signal_threshold,
        loc_ratio_clip=config.dataset_loc_ratio_clip,
        label_mode=config.dataset_label_mode,
        overlap_threshold=config.dataset_overlap_threshold,
        always_on_force_target=config.dataset_always_on_force_target,
        use_time_pos_encoding=getattr(config, "use_time_pos_encoding", True),
        use_observed_mask_feature=getattr(config, "use_observed_mask_feature", True),
        use_trend_feature=getattr(config, "use_trend_feature", False),
        active_overlap_threshold=getattr(config, "dataset_active_overlap_threshold", 0.5),
        transition_overlap_threshold=getattr(config, "dataset_transition_overlap_threshold", 0.0),
        split_mode=config.dataset_split_mode,
        n_holdout_nodes=getattr(config, "dataset_n_holdout_nodes", 10),
    )

    active_signal_threshold, transition_signal_threshold = estimate_process_signal_thresholds(
        dataset,
        getattr(train_loader.dataset, "indices", range(len(train_loader.dataset))),
        active_quantile=getattr(config, "dataset_active_signal_quantile", 0.95),
        transition_quantile=getattr(config, "dataset_transition_signal_quantile", 0.90),
    )
    if hasattr(dataset, "configure_process_labels"):
        dataset.configure_process_labels(
            active_signal_threshold=active_signal_threshold,
            transition_signal_threshold=transition_signal_threshold,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = getattr(dataset, "input_feature_dim", len(config.selected_features))
    num_nodes = getattr(dataset, "num_nodes", 128)

    model = build_model(config, input_dim=input_dim, num_nodes=num_nodes, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion_loc_eval = nn.KLDivLoss(reduction="batchmean")
    criterion_active_eval = nn.CrossEntropyLoss(reduction="mean")

    thresholds = np.arange(args.threshold_start, args.threshold_stop + 1e-9, args.threshold_step)
    rows = []
    best = None

    for thr in thresholds:
        metrics = evaluate_model(
            model,
            val_loader,
            criterion_loc_eval,
            device,
            criterion_active=criterion_active_eval,
            active_prob_threshold=float(thr),
        )
        precision = float(metrics.get("active_precision", 0.0))
        recall = float(metrics.get("active_period_recall", 0.0))
        f2 = fbeta_score(precision, recall, beta=2.0)
        row = {
            "threshold": float(thr),
            "active_acc": float(metrics.get("active_acc", 0.0)),
            "active_precision": precision,
            "active_recall": recall,
            "f2": f2,
            "latency": float(metrics.get("detection_latency_mean", np.nan)),
            "event_top1": float(metrics.get("event_level_top1", 0.0)),
        }
        rows.append(row)
        if best is None or row["f2"] > best["f2"] + 1e-12 or (
            abs(row["f2"] - best["f2"]) <= 1e-12 and row["active_recall"] > best["active_recall"]
        ):
            best = row

    best_threshold = float(best["threshold"])
    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion_loc_eval,
        device,
        criterion_active=criterion_active_eval,
        active_prob_threshold=best_threshold,
    )

    report_dir = Path(config.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"active_threshold_sweep_{args.run_tag}.csv"
    md_path = report_dir / f"active_threshold_calibration_{args.run_tag}.md"
    json_path = report_dir / f"active_threshold_calibration_{args.run_tag}.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "run_tag": args.run_tag,
        "checkpoint": args.checkpoint,
        "best_threshold": best_threshold,
        "validation_best": best,
        "test_metrics": {
            "active_acc": float(test_metrics.get("active_acc", 0.0)),
            "active_precision": float(test_metrics.get("active_precision", 0.0)),
            "active_period_recall": float(test_metrics.get("active_period_recall", 0.0)),
            "mrr": float(test_metrics.get("mrr", 0.0)),
            "top1": float(test_metrics.get("topk_recall_1", 0.0)),
            "top3": float(test_metrics.get("topk_recall_3", 0.0)),
            "top5": float(test_metrics.get("topk_recall_5", 0.0)),
            "event_level_top1": float(test_metrics.get("event_level_top1", 0.0)),
            "event_level_top3": float(test_metrics.get("event_level_top3", 0.0)),
            "event_level_top5": float(test_metrics.get("event_level_top5", 0.0)),
            "detection_latency_mean": float(test_metrics.get("detection_latency_mean", np.nan)),
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Active Threshold Calibration: {args.run_tag}")
    lines.append("")
    lines.append(f"- subdir: `{args.subdir}`")
    lines.append(f"- model_type: `{args.model_type}`")
    lines.append(f"- best_threshold: `{best_threshold:.2f}`")
    lines.append("")
    lines.append("## Validation Best")
    lines.append("")
    for key in ["active_acc", "active_precision", "active_recall", "f2", "latency", "event_top1"]:
        lines.append(f"- {key}: {best[key]:.4f}")
    lines.append("")
    lines.append("## Test Metrics With Best Threshold")
    lines.append("")
    for key, value in summary["test_metrics"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"- sweep_csv: `{csv_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
