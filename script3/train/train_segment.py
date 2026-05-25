
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

import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from utils.evaluation_segment import summarize_segment_joint_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFECT_TYPE_STR_TO_ID = {"I": 0, "E": 1, "P": 2}


def load_segment_dataloaders():
    module_path = SCRIPT_ROOT / "dataset" / "segment_processor.py"
    spec = importlib.util.spec_from_file_location("script3_segment_processor", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_dataloaders


def resolve_training_data_dir(subdir: str, explicit_dir: str | None) -> str:
    if explicit_dir:
        return os.path.normpath(explicit_dir)
    local_dir = SCRIPT_ROOT / "training_data"
    if (local_dir / subdir).exists():
        return str(local_dir)
    legacy_dir = SCRIPT_ROOT.parent / "script2_new" / "training_data_new"
    if (legacy_dir / subdir).exists():
        return str(legacy_dir)
    return str(local_dir)


def resolve_parquet(training_data_dir: str, subdir: str, filename: str) -> str:
    if subdir.strip():
        return os.path.normpath(os.path.join(training_data_dir, subdir, filename))
    return os.path.normpath(os.path.join(training_data_dir, filename))


def defect_type_tensor_from_batch(batch, device):
    values = batch.get("defect_type_str", [])
    if isinstance(values, str):
        values = [values]
    return torch.tensor([DEFECT_TYPE_STR_TO_ID.get(str(v).upper().strip(), -1) for v in values], dtype=torch.long, device=device)


class SegmentTemporalFusionModel(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.LSTM(edge_input_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.segment_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, 3)

    def forward(self, node_features, edge_features, edge_index):
        batch_size, _, num_nodes, _ = node_features.shape
        _, steps, num_segments, edge_dim = edge_features.shape

        node_repr = torch.relu(self.node_proj(node_features.mean(dim=1)))
        edge_flat = edge_features.permute(0, 2, 1, 3).reshape(batch_size * num_segments, steps, edge_dim)
        _, (edge_hidden, _) = self.edge_encoder(edge_flat)
        edge_repr = edge_hidden[-1].reshape(batch_size, num_segments, -1)

        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        src_idx = edge_index[0].clamp(min=0, max=num_nodes - 1)
        dst_idx = edge_index[1].clamp(min=0, max=num_nodes - 1)
        src_repr = node_repr.index_select(1, src_idx)
        dst_repr = node_repr.index_select(1, dst_idx)

        fused = self.fusion(torch.cat([edge_repr, src_repr, dst_repr], dim=-1))
        segment_scores = self.segment_head(fused).squeeze(-1)
        graph_repr = fused.mean(dim=1)
        type_logits = self.type_head(graph_repr)
        return segment_scores, type_logits

def train_epoch(model, loader, optimizer, device, lambda_type: float):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        edge_features = batch["edge_features"].to(device)
        node_features = batch["features"].to(device)
        edge_index = batch["edge_index"].to(device)
        target_segment_idx = batch["target_segment_idx"].long().to(device)
        candidate_mask = batch["candidate_segment_mask"].to(device).bool()
        type_targets = defect_type_tensor_from_batch(batch, device)

        optimizer.zero_grad()
        segment_scores, type_logits = model(node_features, edge_features, edge_index)
        segment_scores = segment_scores.masked_fill(~candidate_mask, -1e9)
        valid_loc = target_segment_idx >= 0
        if not valid_loc.any():
            continue
        segment_scores = segment_scores[valid_loc]
        target_segment_idx = target_segment_idx[valid_loc]
        type_logits = type_logits[valid_loc]
        type_targets = type_targets[valid_loc]
        loss = nn.functional.cross_entropy(segment_scores, target_segment_idx)

        valid_type = type_targets >= 0
        if lambda_type > 0 and valid_type.any():
            loss = loss + lambda_type * nn.functional.cross_entropy(type_logits[valid_type], type_targets[valid_type])

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def evaluate_segment_model(model, loader, device, lambda_type: float):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    score_rows = []
    target_rows = []
    type_logit_rows = []
    type_target_rows = []
    by_defect_type = {}

    with torch.no_grad():
        for batch in loader:
            edge_features = batch["edge_features"].to(device)
            node_features = batch["features"].to(device)
            edge_index = batch["edge_index"].to(device)
            target_segment_idx = batch["target_segment_idx"].long().to(device)
            candidate_mask = batch["candidate_segment_mask"].to(device).bool()
            type_targets = defect_type_tensor_from_batch(batch, device)

            segment_scores, type_logits = model(node_features, edge_features, edge_index)
            segment_scores = segment_scores.masked_fill(~candidate_mask, -1e9)
            valid_loc = target_segment_idx >= 0
            if not valid_loc.any():
                continue
            segment_scores = segment_scores[valid_loc]
            target_segment_idx = target_segment_idx[valid_loc]
            type_logits = type_logits[valid_loc]
            type_targets = type_targets[valid_loc]
            loss = nn.functional.cross_entropy(segment_scores, target_segment_idx)
            valid_type = type_targets >= 0
            if lambda_type > 0 and valid_type.any():
                loss = loss + lambda_type * nn.functional.cross_entropy(type_logits[valid_type], type_targets[valid_type])
            total_loss += float(loss.item())
            n_batches += 1

            score_rows.append(segment_scores.detach().cpu())
            target_rows.append(target_segment_idx.detach().cpu())
            if valid_type.any():
                type_logit_rows.append(type_logits[valid_type].detach().cpu())
                type_target_rows.append(type_targets[valid_type].detach().cpu())

            pred_idx = torch.argmax(segment_scores, dim=1)
            defect_types = batch.get("defect_type_str", [])
            if isinstance(defect_types, str):
                defect_types = [defect_types]
            elif hasattr(defect_types, "__len__"):
                defect_types = [defect_types[i] for i, ok in enumerate(valid_loc.detach().cpu().tolist()) if ok]
            for i, defect_type in enumerate(defect_types):
                key = str(defect_type).upper().strip()
                stats = by_defect_type.setdefault(key, {"n": 0, "top1": 0})
                stats["n"] += 1
                stats["top1"] += int(pred_idx[i].item() == target_segment_idx[i].item())

    scores = torch.cat(score_rows, dim=0).numpy() if score_rows else []
    targets = torch.cat(target_rows, dim=0).numpy() if target_rows else []
    type_logits = torch.cat(type_logit_rows, dim=0).numpy() if type_logit_rows else None
    type_targets = torch.cat(type_target_rows, dim=0).numpy() if type_target_rows else None
    metrics = summarize_segment_joint_metrics(scores, targets, type_logits=type_logits, type_targets=type_targets)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["by_defect_type"] = {
        k: {"n": v["n"], "top1": float(v["top1"] / max(v["n"], 1))} for k, v in by_defect_type.items()
    }
    return metrics


def write_last_metrics(config: Config, metrics: dict, run_name: str | None):
    payload = {
        "mrr": float(metrics.get("mrr", 0.0)),
        "top1": float(metrics.get("top1", 0.0)),
        "top3": float(metrics.get("top3", 0.0)),
        "top5": float(metrics.get("top5", 0.0)),
        "type_macro_f1": float(metrics.get("type_macro_f1", 0.0)),
        "seed": config.random_seed,
        "subdir": config.training_data_subdir,
        "defect_csv": config.defect_matrix_file,
        "config_snapshot": {
            "task_mode": "segment",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "window_stride": config.window_stride,
            "lambda_type": config.lambda_type,
            "edge_timeseries_file": config.edge_timeseries_file,
            "segment_list_file": config.segment_list_file,
        },
        "by_type_top1": {k: float(v.get("top1", 0.0)) for k, v in metrics.get("by_defect_type", {}).items()},
    }

    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "last_run_metrics_segment.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if run_name:
        shutil.copy2(metrics_path, reports_dir / f"last_run_metrics_segment_{run_name}.json")

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subdir", type=str, default="time_gated_downstream")
    parser.add_argument("--training-data-dir", type=str, default=None)
    parser.add_argument("--defect-csv", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lambda-type", type=float, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = Config()
    config.ensure_output_dirs()
    config.training_data_subdir = args.subdir.strip()
    config.training_data_dir = resolve_training_data_dir(config.training_data_subdir, args.training_data_dir)
    config.task_mode = "segment"
    if args.seed is not None:
        config.random_seed = args.seed
    if args.defect_csv is not None:
        config.defect_matrix_file = os.path.normpath(args.defect_csv)
    if args.num_epochs is not None:
        config.num_epochs = int(args.num_epochs)
    if args.patience is not None:
        config.patience = int(args.patience)
    if args.lambda_type is not None:
        config.lambda_type = float(args.lambda_type)

    config.node_timeseries_file = resolve_parquet(config.training_data_dir, config.training_data_subdir, "node_timeseries_with_residuals.parquet")
    if not Path(config.node_timeseries_file).exists():
        config.node_timeseries_file = resolve_parquet(config.training_data_dir, config.training_data_subdir, "node_timeseries.parquet")
    config.edge_timeseries_file = resolve_parquet(config.training_data_dir, config.training_data_subdir, "edge_timeseries.parquet")

    if not Path(config.node_timeseries_file).exists():
        raise FileNotFoundError(f"Missing node timeseries file: {config.node_timeseries_file}")
    if not Path(config.edge_timeseries_file).exists():
        raise FileNotFoundError(f"Missing edge timeseries file: {config.edge_timeseries_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Node parquet: %s", config.node_timeseries_file)
    logger.info("Edge parquet: %s", config.edge_timeseries_file)

    create_dataloaders = load_segment_dataloaders()
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        node_timeseries_file=config.node_timeseries_file,
        edge_timeseries_file=config.edge_timeseries_file,
        adjacency_matrix_file=config.adjacency_matrix_file,
        node_list_file=config.node_list_file,
        segment_list_file=config.segment_list_file,
        defect_matrix_file=config.defect_matrix_file,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        window_stride=config.window_stride,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        allowed_defect_types=getattr(config, "allowed_defect_types", None),
        random_seed=config.random_seed,
    )

    sample_batch = next(iter(train_loader))
    node_input_dim = sample_batch["features"].shape[-1]
    edge_input_dim = sample_batch["edge_features"].shape[-1]
    model = SegmentTemporalFusionModel(node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, hidden_dim=args.hidden_dim, dropout=config.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_mrr = -1.0
    patience_counter = 0
    model_path = Path(config.output_dir) / "best_model_segment.pth"

    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config.lambda_type)
        val_metrics = evaluate_segment_model(model, val_loader, device, config.lambda_type)
        logger.info(
            "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_mrr=%.4f | top1=%.4f | top3=%.4f | top5=%.4f",
            epoch + 1,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics.get("mrr", 0.0),
            val_metrics.get("top1", 0.0),
            val_metrics.get("top3", 0.0),
            val_metrics.get("top5", 0.0),
        )
        if val_metrics.get("mrr", 0.0) > best_val_mrr + 1e-6:
            best_val_mrr = float(val_metrics.get("mrr", 0.0))
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_segment_model(model, test_loader, device, config.lambda_type)
    logger.info(
        "Test | mrr=%.4f | top1=%.4f | top3=%.4f | top5=%.4f | type_macro_f1=%.4f",
        test_metrics.get("mrr", 0.0),
        test_metrics.get("top1", 0.0),
        test_metrics.get("top3", 0.0),
        test_metrics.get("top5", 0.0),
        test_metrics.get("type_macro_f1", 0.0),
    )
    write_last_metrics(config, test_metrics, args.run_name)


if __name__ == "__main__":
    main()

