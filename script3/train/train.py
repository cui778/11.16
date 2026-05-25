#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from models.anomaly_detection_model import create_model
from utils.evaluation import analyze_predictions, evaluate_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFECT_TYPE_STR_TO_ID = {"I": 0, "E": 1, "P": 2}


def load_data_module(config: Config):
    spec = importlib.util.spec_from_file_location("script3_dataset_processor", config.dataset_processor_path)
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    return dataset_module.create_dataloaders


def _get_subset_base_and_indices(ds):
    base = getattr(ds, "dataset", ds)
    indices = getattr(ds, "indices", range(len(ds)))
    return base, indices


def compute_split_diagnostics(train_loader, val_loader, test_loader):
    def _stats(loader):
        ds = loader.dataset
        base, indices = _get_subset_base_and_indices(ds)
        scenario_set = set()
        defect_nodes = set()
        n_loc_enabled = 0
        n_windows = 0
        samples = getattr(base, "samples", None)
        if samples is None:
            return {
                "n_scenarios": 0,
                "n_windows": 0,
                "n_defect_nodes": 0,
                "loc_enabled": 0,
                "loc_enabled_ratio": 0.0,
                "_defect_node_set": set(),
            }
        for idx in indices:
            sample = samples[idx]
            scenario_set.add(int(sample.get("scenario_id", -1)))
            defect_node_id = str(sample.get("defect_node_id", "")).strip()
            if defect_node_id:
                defect_nodes.add(defect_node_id)
            if int(sample.get("loc_enabled", 0)) > 0:
                n_loc_enabled += 1
            n_windows += 1
        scenario_set.discard(-1)
        return {
            "n_scenarios": int(len(scenario_set)),
            "n_windows": int(n_windows),
            "n_defect_nodes": int(len(defect_nodes)),
            "loc_enabled": int(n_loc_enabled),
            "loc_enabled_ratio": float(n_loc_enabled / max(n_windows, 1)),
            "_defect_node_set": defect_nodes,
        }

    train_stats = _stats(train_loader)
    val_stats = _stats(val_loader)
    test_stats = _stats(test_loader)

    train_nodes = train_stats.pop("_defect_node_set", set())
    val_nodes = val_stats.pop("_defect_node_set", set())
    test_nodes = test_stats.pop("_defect_node_set", set())

    diagnostics = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "val_defect_nodes_subset_train": bool(val_nodes <= train_nodes),
        "test_defect_nodes_subset_train": bool(test_nodes <= train_nodes),
    }
    if not diagnostics["val_defect_nodes_subset_train"]:
        diagnostics["val_only_defect_nodes_sample"] = list(sorted(val_nodes - train_nodes))[:10]
    if not diagnostics["test_defect_nodes_subset_train"]:
        diagnostics["test_only_defect_nodes_sample"] = list(sorted(test_nodes - train_nodes))[:10]
    return diagnostics


def build_feature_list(base_features, feature_mode: str):
    if feature_mode == "raw":
        return [feature for feature in base_features if "residual" not in feature]
    if feature_mode == "residual":
        return [feature for feature in base_features if "residual" in feature]
    return list(base_features)


def resolve_timeseries_file(config: Config, feature_mode: str):
    parquet_name = "node_timeseries.parquet" if feature_mode == "raw" else "node_timeseries_with_residuals.parquet"
    if config.training_data_subdir.strip():
        return os.path.normpath(os.path.join(config.training_data_dir, config.training_data_subdir, parquet_name))
    return os.path.normpath(os.path.join(config.training_data_dir, parquet_name))


def build_train_seen_node_indices(train_loader):
    seen = set()
    ds = train_loader.dataset
    base = getattr(ds, "dataset", ds)
    indices = getattr(ds, "indices", range(len(ds)))
    if hasattr(base, "samples"):
        for idx in indices:
            sample = base.samples[idx]
            if sample.get("target_node_idx", -1) >= 0:
                seen.add(sample["target_node_idx"])
    return seen


def train_epoch(model, train_loader, optimizer, criterion_node, device, use_full_graph=False, lambda_type=0.0, criterion_type=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        features = batch["features"].to(device)
        adj_matrix = batch["adj_matrix"]
        if isinstance(adj_matrix, (list, tuple)):
            adj = adj_matrix[0].to(device)
        elif adj_matrix.dim() == 3:
            adj = adj_matrix[0].to(device)
        else:
            adj = adj_matrix.to(device)

        soft_labels = batch["soft_label"].to(device)
        target_node_idx = batch["target_node_idx"].long().to(device)
        candidate_mask = batch["candidate_mask"].to(device).bool()

        optimizer.zero_grad()
        logits_node, _logits_has_defect, logits_defect_type = model(features, adj)
        scores = logits_node[:, :, 1]

        if candidate_mask.dim() == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand_as(scores)
        if not use_full_graph:
            scores = scores.masked_fill(~candidate_mask, -1e9)

        loc_mask = soft_labels.sum(dim=-1) > 1e-8
        if loc_mask.sum() == 0:
            continue

        scores = scores[loc_mask]
        soft_labels = soft_labels[loc_mask]
        candidate_mask = candidate_mask[loc_mask]
        target_node_idx = target_node_idx[loc_mask]

        scores_masked = scores.masked_fill(~candidate_mask, -1e9)
        soft_labels_masked = soft_labels.masked_fill(~candidate_mask, 0.0)
        soft_sum = soft_labels_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        soft_labels_norm = soft_labels_masked / soft_sum

        log_probs = torch.log_softmax(scores_masked, dim=-1)
        loss = criterion_node(log_probs, soft_labels_norm)

        if lambda_type > 0 and criterion_type is not None:
            has_defect = batch["has_defect"].to(device)
            defect_type_str_batch = batch["defect_type_str"]
            labeled_indices = []
            targets = []
            for i in range(has_defect.shape[0]):
                if has_defect[i].item() != 1:
                    continue
                raw_value = defect_type_str_batch[i] if isinstance(defect_type_str_batch, (list, tuple)) else defect_type_str_batch
                if isinstance(raw_value, torch.Tensor):
                    raw_value = raw_value.item() if raw_value.numel() == 1 else str(raw_value.cpu().tolist()[0])
                defect_type = str(raw_value).strip().upper()
                if defect_type in DEFECT_TYPE_STR_TO_ID:
                    labeled_indices.append(i)
                    targets.append(DEFECT_TYPE_STR_TO_ID[defect_type])
            if labeled_indices:
                logits_type = logits_defect_type[labeled_indices].to(device)
                targets_type = torch.tensor(targets, dtype=torch.long, device=device)
                loss = loss + lambda_type * criterion_type(logits_type, targets_type)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def write_by_scenario_csv(metrics, output_csv: Path):
    by_scenario = metrics.get("by_scenario", {})
    if not by_scenario:
        return
    rows = []
    for defect_id, item in by_scenario.items():
        rows.append(
            {
                "defect_id": defect_id,
                "n_windows": item.get("n_windows", 0),
                "top1": item.get("top1", 0.0),
                "top3": item.get("top3", 0.0),
                "mrr": item.get("mrr", 0.0),
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


def write_last_metrics(config: Config, test_metrics, split_diag, run_name=None):
    payload = {
        "mrr": float(test_metrics.get("mrr", 0.0)),
        "top1": float(test_metrics.get("topk_recall_1", 0.0)),
        "top3": float(test_metrics.get("topk_recall_3", 0.0)),
        "top5": float(test_metrics.get("topk_recall_5", 0.0)),
        "seed": config.random_seed,
        "subdir": getattr(config, "training_data_subdir", ""),
        "defect_csv": str(getattr(config, "defect_matrix_file", "")),
        "split_diag": split_diag,
        "config_snapshot": {
            "model_type": config.model_type,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "patience": config.patience,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "window_stride": config.window_stride,
            "soft_label_sigma": config.soft_label_sigma,
            "lambda_type": config.lambda_type,
            "candidate_nodes_file": os.path.basename(config.candidate_nodes_file),
            "monitor_nodes_file": os.path.basename(config.monitor_nodes_file),
            "node_timeseries_file": config.node_timeseries_file,
            "localization_target": getattr(config, "localization_target", "node"),
        },
    }

    by_type = test_metrics.get("by_defect_type", {})
    if by_type:
        payload["by_type_top1"] = {
            defect_type: float(item.get("top1", 0.0))
            for defect_type, item in by_type.items()
            if item and item.get("n", 0) > 0
        }

    seen_nodes = test_metrics.get("seen_nodes", {})
    unseen_nodes = test_metrics.get("unseen_nodes", {})
    if seen_nodes:
        payload["seen_nodes"] = {
            "n": int(seen_nodes.get("n", 0)),
            "mrr": float(seen_nodes.get("mrr", 0.0)),
            "top1": float(seen_nodes.get("top1", 0.0)),
            "top3": float(seen_nodes.get("top3", 0.0)),
        }
    if unseen_nodes:
        payload["unseen_nodes"] = {
            "n": int(unseen_nodes.get("n", 0)),
            "mrr": float(unseen_nodes.get("mrr", 0.0)),
            "top1": float(unseen_nodes.get("top1", 0.0)),
            "top3": float(unseen_nodes.get("top3", 0.0)),
        }

    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "last_run_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved metrics: %s", metrics_path)

    if run_name:
        backup_path = reports_dir / f"last_run_metrics_{run_name}.json"
        shutil.copy2(metrics_path, backup_path)
        logger.info("Backed up metrics: %s", backup_path)


def backup_run_artifacts(config: Config, run_name: str):
    checkpoint_dir = Path(config.output_dir)
    for src_name, dst_name in [
        ("best_model.pth", f"best_model_{run_name}.pth"),
        ("by_scenario_test_latest.csv", f"by_scenario_test_{run_name}.csv"),
        ("by_scenario_val_best.csv", f"by_scenario_val_{run_name}.csv"),
    ]:
        src = checkpoint_dir / src_name
        dst = checkpoint_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Backed up %s -> %s", src.name, dst.name)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Override config.random_seed.")
    parser.add_argument("--subdir", type=str, default=None, help="Override config.training_data_subdir.")
    parser.add_argument("--training-data-dir", type=str, default=None, help="Override config.training_data_dir.")
    parser.add_argument("--defect-csv", type=str, default=None, help="Override config.defect_matrix_file.")
    parser.add_argument("--run-name", type=str, default=None, help="Run identifier for backups.")
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="raw_residual",
        choices=["raw_residual", "raw", "residual"],
        help="Feature preset for Chapter 1 experiments.",
    )
    parser.add_argument("--raw-features", action="store_true", help="Compatibility alias for --feature-mode raw.")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override config.model_type, e.g. hydraulic_inverse / lstm_graphsage_edge / tcn_graphsage_edge.",
    )
    parser.add_argument("--split-mode", type=str, default=None, choices=["scenario", "node_holdout"], help="Data split mode.")
    parser.add_argument("--n-holdout-nodes", type=int, default=None, help="Held-out defect node count.")
    parser.add_argument("--candidate-file", type=str, default=None, help="Override candidate node file.")
    parser.add_argument("--monitor-file", type=str, default=None, help="Override monitor node file.")
    parser.add_argument("--lambda-type", type=float, default=None, help="Override auxiliary type-head loss weight.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override config.num_epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override config.patience.")
    parser.add_argument("--analyze-samples", type=int, default=0, help="Run qualitative prediction analysis on N samples after testing.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    config = Config()
    config.ensure_output_dirs()

    if args.seed is not None:
        config.random_seed = args.seed
    if args.training_data_dir is not None:
        config.training_data_dir = os.path.normpath(args.training_data_dir)
    if args.subdir is not None:
        config.training_data_subdir = args.subdir.strip()
    if args.defect_csv is not None:
        config.defect_matrix_file = os.path.normpath(args.defect_csv)
    if args.model_type is not None:
        config.model_type = args.model_type.strip()
    if args.split_mode is not None:
        config.dataset_split_mode = args.split_mode.strip()
    if args.n_holdout_nodes is not None:
        config.dataset_n_holdout_nodes = args.n_holdout_nodes
    if args.candidate_file is not None:
        config.candidate_nodes_file = os.path.normpath(args.candidate_file)
    if args.monitor_file is not None:
        config.monitor_nodes_file = os.path.normpath(args.monitor_file)
    if args.lambda_type is not None:
        config.lambda_type = float(args.lambda_type)
    if args.num_epochs is not None:
        config.num_epochs = int(args.num_epochs)
    if args.patience is not None:
        config.patience = int(args.patience)

    feature_mode = "raw" if args.raw_features else args.feature_mode
    config.selected_features = build_feature_list(config.selected_features, feature_mode)
    config.node_timeseries_file = resolve_timeseries_file(config, feature_mode)

    if not Path(config.node_timeseries_file).exists() and feature_mode != "raw":
        fallback_raw = resolve_timeseries_file(config, "raw")
        if Path(fallback_raw).exists():
            logger.warning("Residual parquet not found, falling back to raw parquet: %s", fallback_raw)
            config.node_timeseries_file = fallback_raw
            feature_mode = "raw"
            config.selected_features = build_feature_list(config.selected_features, "raw")

    if not Path(config.node_timeseries_file).exists():
        raise FileNotFoundError(
            f"Missing node timeseries file: {config.node_timeseries_file}\n"
            f"Generate it with script3/prep/extract_timeseries.py and script3/prep/residual_features.py first."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Model: %s | split=%s | feature_mode=%s", config.model_type, config.dataset_split_mode, feature_mode)
    logger.info("Timeseries: %s", config.node_timeseries_file)
    logger.info("Candidate file: %s", config.candidate_nodes_file)
    logger.info("Monitor file: %s", config.monitor_nodes_file)

    create_dataloaders = load_data_module(config)
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        node_timeseries_file=config.node_timeseries_file,
        edge_timeseries_file=config.edge_timeseries_file,
        adjacency_matrix_file=config.adjacency_matrix_file,
        segment_list_file=config.segment_list_file,
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
        e_class_oversample_ratio=config.dataset_e_class_oversample_ratio,
        allowed_defect_types=getattr(config, "allowed_defect_types", None),
        split_mode=config.dataset_split_mode,
        n_holdout_nodes=config.dataset_n_holdout_nodes,
    )

    split_diag = compute_split_diagnostics(train_loader, val_loader, test_loader)
    input_dim = len(dataset.feature_cols)
    num_nodes = len(dataset.node_list)
    train_seen_node_indices = build_train_seen_node_indices(train_loader)

    logger.info("Input dim: %s | nodes: %s", input_dim, num_nodes)
    logger.info("Train/val/test windows: %s / %s / %s", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    model_kwargs = {
        "model_type": config.model_type,
        "input_dim": input_dim,
        "time_hidden_dim": config.time_hidden_dim,
        "spatial_hidden_dim": config.spatial_hidden_dim,
        "num_time_layers": config.num_time_layers,
        "num_spatial_layers": config.num_spatial_layers,
        "num_nodes": num_nodes,
        "dropout": config.dropout,
    }
    if config.model_type == "hydraulic_inverse":
        graph_path_file = config.graph_path_features_file or os.path.join(SCRIPT_ROOT, "input_1", "graph_path_features.npz")
        if not os.path.exists(graph_path_file):
            raise FileNotFoundError(
                f"hydraulic_inverse requires graph path features: {graph_path_file}\n"
                f"Run script3/prep/build_graph_features.py first."
            )
        npz = np.load(graph_path_file)
        model_kwargs["graph_features_dict"] = {key: npz[key] for key in npz.files}
        model_kwargs["use_flow_direction"] = config.use_flow_direction
        model_kwargs["use_propagation_delay"] = config.use_propagation_delay

    model = create_model(**model_kwargs).to(device)
    logger.info("Parameter count: %s", sum(parameter.numel() for parameter in model.parameters()))

    criterion_node = nn.KLDivLoss(reduction="batchmean")
    criterion_type = nn.CrossEntropyLoss(reduction="mean") if config.lambda_type > 0 else None
    criterion_eval = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    primary_metric = "seen_mrr" if config.use_seen_mrr_for_early_stopping else "mrr"
    best_val_score = -float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    model_path = os.path.join(config.output_dir, "best_model.pth")

    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion_node,
            device,
            use_full_graph=config.train_with_full_graph,
            lambda_type=config.lambda_type,
            criterion_type=criterion_type,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion_eval,
            device,
            train_seen_node_indices=train_seen_node_indices,
        )

        if scheduler is not None:
            scheduler.step()

        seen_metrics = val_metrics.get("seen_nodes", {})
        if primary_metric == "seen_mrr" and seen_metrics and seen_metrics.get("n", 0) > 0:
            val_score = float(seen_metrics.get("mrr", 0.0))
        else:
            val_score = float(val_metrics.get("mrr", 0.0))
        val_loss = float(val_metrics.get("loss", float("nan")))

        logger.info(
            "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_mrr=%.4f | top1=%.4f | top3=%.4f | top5=%.4f",
            epoch + 1,
            train_metrics["loss"],
            val_loss,
            val_metrics.get("mrr", 0.0),
            val_metrics.get("topk_recall_1", 0.0),
            val_metrics.get("topk_recall_3", 0.0),
            val_metrics.get("topk_recall_5", 0.0),
        )

        improved = (val_score > best_val_score + 1e-6) or (
            abs(val_score - best_val_score) <= 1e-6 and val_loss < best_val_loss
        )
        if improved:
            best_val_score = val_score
            best_val_loss = val_loss
            patience_counter = 0
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                    "val_loss": best_val_loss,
                    "config": config.__dict__,
                },
                model_path,
            )
            write_by_scenario_csv(val_metrics, Path(config.output_dir) / "by_scenario_val_best.csv")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion_eval,
        device,
        train_seen_node_indices=train_seen_node_indices,
    )

    logger.info(
        "Test | mrr=%.4f | top1=%.4f | top3=%.4f | top5=%.4f",
        test_metrics.get("mrr", 0.0),
        test_metrics.get("topk_recall_1", 0.0),
        test_metrics.get("topk_recall_3", 0.0),
        test_metrics.get("topk_recall_5", 0.0),
    )

    write_by_scenario_csv(test_metrics, Path(config.output_dir) / "by_scenario_test_latest.csv")
    write_last_metrics(config, test_metrics, split_diag, run_name=args.run_name)

    if args.run_name:
        backup_run_artifacts(config, args.run_name)

    if args.analyze_samples > 0:
        try:
            analyze_predictions(model, test_loader, device, num_samples=args.analyze_samples)
        except Exception as exc:
            logger.warning("Prediction analysis skipped: %s", exc)


if __name__ == "__main__":
    main()
