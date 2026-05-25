#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a lightweight reranker on top of a frozen base model.

Workflow:
1. Collect base candidate probabilities on train/val/test.
2. Build top-k shortlist examples using base probabilities.
3. Train a logistic-regression reranker on train shortlist candidates.
4. Select k/tau/C by validation Top-1, then report test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from models.anomaly_detection_model import create_model
from train.train import (
    DEFECT_TYPE_STR_TO_ID,
    _get_subset_base_and_indices,
    estimate_process_signal_thresholds,
    load_data_module,
)


def _ensure_adj_2d(adj_matrix: torch.Tensor) -> torch.Tensor:
    if isinstance(adj_matrix, (list, tuple)):
        adj = adj_matrix[0]
    else:
        adj = adj_matrix
    if adj.dim() == 3:
        adj = adj[0]
    return adj


def _make_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.training_data_subdir = args.subdir
    cfg.node_timeseries_file = str(
        Path(cfg.training_data_dir) / args.subdir / "node_timeseries_with_residuals.parquet"
    )
    if getattr(args, "defect_csv", ""):
        cfg.defect_matrix_file = str(Path(args.defect_csv))
    cfg.model_type = args.model_type
    cfg.dataset_split_mode = args.split_mode
    cfg.random_seed = int(args.seed)
    if args.features:
        cfg.selected_features = [x.strip() for x in args.features.split(",") if x.strip()]
    return cfg


def _create_loaders(cfg: Config):
    create_dataloaders = load_data_module(cfg)
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
        label_mode=cfg.dataset_label_mode,
        use_time_pos_encoding=cfg.use_time_pos_encoding,
        use_trend_feature=cfg.use_trend_feature,
        use_observed_mask_feature=getattr(cfg, "use_observed_mask_feature", True),
        active_overlap_threshold=cfg.dataset_active_overlap_threshold,
        transition_overlap_threshold=cfg.dataset_transition_overlap_threshold,
        active_signal_threshold=None,
        transition_signal_threshold=None,
        always_on_force_target=cfg.dataset_always_on_force_target,
        split_mode=cfg.dataset_split_mode,
        n_holdout_nodes=cfg.dataset_n_holdout_nodes,
    )


def _build_model(cfg: Config, dataset, device: torch.device):
    input_dim = int(dataset.input_feature_dim)
    num_nodes = int(len(dataset.node_list))
    model_kw = dict(
        input_dim=input_dim,
        time_hidden_dim=cfg.time_hidden_dim,
        spatial_hidden_dim=cfg.spatial_hidden_dim,
        num_time_layers=cfg.num_time_layers,
        num_spatial_layers=cfg.num_spatial_layers,
        num_nodes=num_nodes,
        dropout=cfg.dropout,
    )
    if cfg.model_type.startswith("hydraulic_inverse"):
        model_kw["graph_features_path"] = cfg.graph_path_features_file
        model_kw["use_flow_direction"] = getattr(cfg, "use_flow_direction", True)
        model_kw["use_propagation_delay"] = getattr(cfg, "use_propagation_delay", False)
        model_kw["attention_max_hops"] = getattr(cfg, "hydraulic_attention_max_hops", 0)
        model_kw["observed_source_only"] = getattr(cfg, "hydraulic_observed_source_only", False)
    return create_model(cfg.model_type, **model_kw).to(device)


def _sym_shortest_dist(path_features_path: str) -> np.ndarray:
    npz = np.load(path_features_path)
    sd = np.array(npz["shortest_dist"], dtype=np.float32)
    return np.minimum(sd, sd.T)


def _load_candidate_tier_features(candidate_tier_csv: str, node_list: List[str]) -> np.ndarray:
    if not candidate_tier_csv:
        return np.zeros((len(node_list), 0), dtype=np.float32)
    import pandas as pd

    df = pd.read_csv(candidate_tier_csv)
    feature_map: Dict[str, np.ndarray] = {}
    for row in df.itertuples(index=False):
        hop = float(getattr(row, "nearest_monitor_hop", 999.0))
        tier = str(getattr(row, "observability_tier", "unknown"))
        hop_clip = min(hop, 20.0)
        feature_map[str(getattr(row, "candidate_node"))] = np.asarray(
            [
                hop_clip / 20.0,
                1.0 / (1.0 + hop_clip),
                1.0 if tier == "direct" else 0.0,
                1.0 if tier == "near" else 0.0,
                1.0 if tier == "far" else 0.0,
            ],
            dtype=np.float32,
        )
    out = np.zeros((len(node_list), 5), dtype=np.float32)
    for i, node in enumerate(node_list):
        feat = feature_map.get(str(node))
        if feat is not None:
            out[i] = feat
    return out


def _collect_split_arrays(model, loader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    prob_rows: List[np.ndarray] = []
    target_rows: List[int] = []
    cand_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    observed_rows: List[np.ndarray] = []
    type_prob_rows: List[np.ndarray] = []
    defect_type_rows: List[str] = []
    node_embed_rows: List[np.ndarray] = []
    graph_embed_rows: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            adj = _ensure_adj_2d(batch["adj_matrix"]).to(device)
            logits = model(x, adj)
            logits_node = logits[0] if isinstance(logits, tuple) else logits
            logits_type = logits[2] if isinstance(logits, tuple) and len(logits) >= 3 else None
            scores = logits_node[:, :, 1]
            node_embeddings = None
            if hasattr(model, "get_node_embeddings"):
                node_embeddings = model.get_node_embeddings(x, adj)

            y_idx = batch["target_node_idx"].long().to(device)
            active_label = batch["active_label"].long().to(device)
            loc_enabled = batch["loc_enabled"].long().to(device)
            cand_mask = batch["candidate_mask"].to(device).bool()
            observed_mask = batch["observed_mask"].to(device)
            defect_types = list(batch["defect_type_str"])

            if cand_mask.dim() == 1:
                cand_mask = cand_mask.unsqueeze(0).expand_as(scores)
            if observed_mask.dim() == 1:
                observed_mask = observed_mask.unsqueeze(0).expand_as(scores)

            valid_mask = (y_idx >= 0) & (active_label > 0) & (loc_enabled > 0)
            valid_mask = valid_mask & cand_mask[torch.arange(cand_mask.size(0), device=device), y_idx]
            if valid_mask.sum().item() == 0:
                continue

            scores = scores[valid_mask]
            y_idx = y_idx[valid_mask]
            cand_mask = cand_mask[valid_mask]
            x = x[valid_mask]
            observed_mask = observed_mask[valid_mask]
            if logits_type is not None:
                logits_type = logits_type[valid_mask]
            if node_embeddings is not None:
                node_embeddings = node_embeddings[valid_mask]

            scores = scores.masked_fill(~cand_mask, -1e9)
            probs = torch.softmax(scores, dim=-1)
            activation = x.abs().mean(dim=1).mean(dim=-1) * observed_mask

            prob_rows.append(probs.detach().cpu().numpy())
            target_rows.extend(y_idx.detach().cpu().tolist())
            cand_rows.append(cand_mask.detach().cpu().numpy().astype(bool))
            act_rows.append(activation.detach().cpu().numpy())
            observed_rows.append(observed_mask.detach().cpu().numpy())
            if logits_type is not None:
                type_prob_rows.append(torch.softmax(logits_type, dim=-1).detach().cpu().numpy())
            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            defect_type_rows.extend([defect_types[i] for i, keep in enumerate(valid_np) if keep])
            if node_embeddings is not None:
                node_embed_rows.append(node_embeddings.detach().cpu().numpy())
                graph_embed_rows.append(node_embeddings.mean(dim=1).detach().cpu().numpy())

    payload = {
        "probs": np.concatenate(prob_rows, axis=0),
        "targets": np.asarray(target_rows, dtype=np.int64),
        "candidate_mask": np.concatenate(cand_rows, axis=0),
        "activation": np.concatenate(act_rows, axis=0),
        "observed_mask": np.concatenate(observed_rows, axis=0),
    }
    if type_prob_rows:
        payload["graph_type_probs"] = np.concatenate(type_prob_rows, axis=0)
        payload["defect_type_str"] = np.asarray(defect_type_rows, dtype=object)
    if node_embed_rows:
        payload["node_embeddings"] = np.concatenate(node_embed_rows, axis=0)
        payload["graph_embeddings"] = np.concatenate(graph_embed_rows, axis=0)
    return payload


def _make_reranker_model(args: argparse.Namespace, c_value: float):
    if args.reranker_kind == "lr":
        return LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
    hidden_sizes = tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip())
    if not hidden_sizes:
        raise ValueError("MLP reranker requires at least one hidden size via --mlp-hidden")
    return MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        alpha=float(args.mlp_alpha),
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=int(args.seed),
    )


def _fit_embedding_projectors(train_arrays: Dict[str, np.ndarray], embedding_dim: int):
    if embedding_dim <= 0:
        return None, None
    if "node_embeddings" not in train_arrays or "graph_embeddings" not in train_arrays:
        return None, None

    node_embeddings = train_arrays["node_embeddings"]
    candidate_mask = train_arrays["candidate_mask"]
    graph_embeddings = train_arrays["graph_embeddings"]
    flat_candidate = node_embeddings[candidate_mask]
    if flat_candidate.size == 0:
        return None, None

    node_dim = min(int(embedding_dim), flat_candidate.shape[-1], flat_candidate.shape[0])
    graph_dim = min(int(embedding_dim), graph_embeddings.shape[-1], graph_embeddings.shape[0])
    if node_dim <= 0 or graph_dim <= 0:
        return None, None

    node_pca = PCA(n_components=node_dim, random_state=0)
    graph_pca = PCA(n_components=graph_dim, random_state=0)
    node_pca.fit(flat_candidate.astype(np.float32))
    graph_pca.fit(graph_embeddings.astype(np.float32))
    return node_pca, graph_pca


def _project_embeddings(arrays: Dict[str, np.ndarray], node_pca, graph_pca) -> Dict[str, np.ndarray]:
    if node_pca is None or graph_pca is None:
        return {}
    if "node_embeddings" not in arrays or "graph_embeddings" not in arrays:
        return {}

    node_embeddings = arrays["node_embeddings"]
    graph_embeddings = arrays["graph_embeddings"]
    B, N, D = node_embeddings.shape
    node_proj = node_pca.transform(node_embeddings.reshape(B * N, D)).reshape(B, N, -1).astype(np.float32)
    graph_proj = graph_pca.transform(graph_embeddings).astype(np.float32)
    return {
        "node_embedding_proj": node_proj,
        "graph_embedding_proj": graph_proj,
    }


def _estimate_candidate_prior(probs: np.ndarray) -> np.ndarray:
    prior = probs.mean(axis=0)
    return np.clip(prior, 1e-8, None)


def _build_path_affinity(activation: np.ndarray, candidate_mask: np.ndarray, sym_dist: np.ndarray, tau: float) -> np.ndarray:
    tau = float(max(tau, 1e-6))
    kernel = np.exp(-sym_dist / tau).astype(np.float32)
    affinity = activation @ kernel.T
    affinity = np.clip(affinity, 1e-8, None)
    affinity = affinity * candidate_mask.astype(np.float32)
    denom = affinity.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return affinity / denom


def _build_shortlist_dataset(
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    topk: int,
    candidate_onehot: bool = False,
    node_embedding_proj: np.ndarray | None = None,
    graph_embedding_proj: np.ndarray | None = None,
    graph_type_probs: np.ndarray | None = None,
    candidate_tier_features: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    labels: List[int] = []
    sample_ids: List[int] = []

    log_probs = np.log(np.clip(probs, 1e-8, None))
    log_path = np.log(np.clip(path_affinity, 1e-8, None))
    log_prior = np.log(np.clip(prior, 1e-8, None))
    monitored_flag = observed_mask.max(axis=0)

    masked_probs = np.where(candidate_mask, probs, -1.0)
    order = np.argsort(-masked_probs, axis=1)
    for row_idx in range(probs.shape[0]):
        shortlist = order[row_idx, :topk]
        target = int(targets[row_idx])
        if target not in shortlist:
            continue
        shortlist_probs = probs[row_idx, shortlist]
        shortlist_ranks = np.arange(1, len(shortlist) + 1, dtype=np.float32)
        prob_gap = shortlist_probs[0] - shortlist_probs
        for local_rank, cand in enumerate(shortlist):
            cand = int(cand)
            row_features = [
                float(log_probs[row_idx, cand]),
                float(probs[row_idx, cand]),
                float(log_path[row_idx, cand]),
                float(path_affinity[row_idx, cand]),
                float(log_prior[cand]),
                float(prior[cand]),
                float(shortlist_ranks[local_rank] / max(1, topk)),
                float(prob_gap[local_rank]),
                float(monitored_flag[cand]),
            ]
            if node_embedding_proj is not None and graph_embedding_proj is not None:
                cand_emb = node_embedding_proj[row_idx, cand]
                graph_emb = graph_embedding_proj[row_idx]
                row_features.extend(cand_emb.tolist())
                row_features.extend(graph_emb.tolist())
                row_features.extend((cand_emb - graph_emb).tolist())
            if graph_type_probs is not None:
                row_features.extend(graph_type_probs[row_idx].tolist())
            if candidate_tier_features is not None and candidate_tier_features.size > 0:
                row_features.extend(candidate_tier_features[cand].tolist())
            if candidate_onehot:
                row_features.extend([1.0 if j == cand else 0.0 for j in range(probs.shape[1])])
            features.append(row_features)
            labels.append(int(cand == target))
            sample_ids.append(row_idx)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(sample_ids, dtype=np.int64)


def _rerank_with_model(
    model_lr: LogisticRegression,
    scaler: StandardScaler,
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    topk: int,
    candidate_onehot: bool = False,
    node_embedding_proj: np.ndarray | None = None,
    graph_embedding_proj: np.ndarray | None = None,
    graph_type_probs: np.ndarray | None = None,
    candidate_tier_features: np.ndarray | None = None,
) -> Dict[str, float]:
    log_probs = np.log(np.clip(probs, 1e-8, None))
    log_path = np.log(np.clip(path_affinity, 1e-8, None))
    log_prior = np.log(np.clip(prior, 1e-8, None))
    monitored_flag = observed_mask.max(axis=0)
    masked_probs = np.where(candidate_mask, probs, -1.0)
    order = np.argsort(-masked_probs, axis=1)

    final_scores = np.where(candidate_mask, log_probs, -1e12).copy()
    for row_idx in range(probs.shape[0]):
        shortlist = order[row_idx, :topk]
        shortlist_probs = probs[row_idx, shortlist]
        shortlist_ranks = np.arange(1, len(shortlist) + 1, dtype=np.float32)
        prob_gap = shortlist_probs[0] - shortlist_probs
        feat = np.asarray([
            [
                float(log_probs[row_idx, cand]),
                float(probs[row_idx, cand]),
                float(log_path[row_idx, cand]),
                float(path_affinity[row_idx, cand]),
                float(log_prior[cand]),
                float(prior[cand]),
                float(shortlist_ranks[i] / max(1, topk)),
                float(prob_gap[i]),
                float(monitored_flag[cand]),
            ]
            + (
                (
                    node_embedding_proj[row_idx, cand].tolist()
                    + graph_embedding_proj[row_idx].tolist()
                    + (node_embedding_proj[row_idx, cand] - graph_embedding_proj[row_idx]).tolist()
                ) if (node_embedding_proj is not None and graph_embedding_proj is not None) else []
            )
            + (graph_type_probs[row_idx].tolist() if graph_type_probs is not None else [])
            + (candidate_tier_features[cand].tolist() if (candidate_tier_features is not None and candidate_tier_features.size > 0) else [])
            + ([1.0 if j == cand else 0.0 for j in range(probs.shape[1])] if candidate_onehot else [])
            for i, cand in enumerate(shortlist)
        ], dtype=np.float32)
        pred = model_lr.predict_proba(scaler.transform(feat))[:, 1]
        final_scores[row_idx, shortlist] = pred

    order_final = np.argsort(-final_scores, axis=1)
    pos = np.where(order_final == targets[:, None])[1] + 1
    n = int(pos.shape[0])
    return {
        "n_samples": n,
        "mrr": float(np.mean(1.0 / pos)) if n else 0.0,
        "top1": float(np.mean(pos <= 1)) if n else 0.0,
        "top3": float(np.mean(pos <= 3)) if n else 0.0,
        "top5": float(np.mean(pos <= 5)) if n else 0.0,
        "top10": float(np.mean(pos <= 10)) if n else 0.0,
        "top20": float(np.mean(pos <= 20)) if n else 0.0,
    }


def _predict_sample_types(graph_type_probs: np.ndarray, available_types: List[str]) -> np.ndarray:
    idxs = [DEFECT_TYPE_STR_TO_ID[t] for t in available_types if t in DEFECT_TYPE_STR_TO_ID]
    if not idxs:
        return np.asarray([""] * graph_type_probs.shape[0], dtype=object)
    sub = graph_type_probs[:, idxs]
    pred_local = np.argmax(sub, axis=1)
    labels = [available_types[int(i)] for i in pred_local]
    return np.asarray(labels, dtype=object)


def _rerank_with_type_models(
    model_dict: Dict[str, object],
    scaler_dict: Dict[str, StandardScaler],
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    topk: int,
    graph_type_probs: np.ndarray,
    candidate_onehot: bool = False,
    node_embedding_proj: np.ndarray | None = None,
    graph_embedding_proj: np.ndarray | None = None,
    candidate_tier_features: np.ndarray | None = None,
) -> Dict[str, float]:
    log_probs = np.log(np.clip(probs, 1e-8, None))
    log_path = np.log(np.clip(path_affinity, 1e-8, None))
    log_prior = np.log(np.clip(prior, 1e-8, None))
    monitored_flag = observed_mask.max(axis=0)
    masked_probs = np.where(candidate_mask, probs, -1.0)
    order = np.argsort(-masked_probs, axis=1)
    final_scores = np.where(candidate_mask, log_probs, -1e12).copy()

    available_types = [t for t in model_dict.keys() if t != "__global__"]
    pred_types = _predict_sample_types(graph_type_probs, available_types)

    for row_idx in range(probs.shape[0]):
        shortlist = order[row_idx, :topk]
        shortlist_probs = probs[row_idx, shortlist]
        shortlist_ranks = np.arange(1, len(shortlist) + 1, dtype=np.float32)
        prob_gap = shortlist_probs[0] - shortlist_probs
        feat = np.asarray([
            [
                float(log_probs[row_idx, cand]),
                float(probs[row_idx, cand]),
                float(log_path[row_idx, cand]),
                float(path_affinity[row_idx, cand]),
                float(log_prior[cand]),
                float(prior[cand]),
                float(shortlist_ranks[i] / max(1, topk)),
                float(prob_gap[i]),
                float(monitored_flag[cand]),
            ]
            + (
                (
                    node_embedding_proj[row_idx, cand].tolist()
                    + graph_embedding_proj[row_idx].tolist()
                    + (node_embedding_proj[row_idx, cand] - graph_embedding_proj[row_idx]).tolist()
                ) if (node_embedding_proj is not None and graph_embedding_proj is not None) else []
            )
            + (graph_type_probs[row_idx].tolist() if graph_type_probs is not None else [])
            + (candidate_tier_features[cand].tolist() if (candidate_tier_features is not None and candidate_tier_features.size > 0) else [])
            + ([1.0 if j == cand else 0.0 for j in range(probs.shape[1])] if candidate_onehot else [])
            for i, cand in enumerate(shortlist)
        ], dtype=np.float32)
        route = pred_types[row_idx] if row_idx < pred_types.shape[0] else ""
        reranker_model = model_dict.get(route, model_dict["__global__"])
        scaler = scaler_dict.get(route, scaler_dict["__global__"])
        pred = reranker_model.predict_proba(scaler.transform(feat))[:, 1]
        final_scores[row_idx, shortlist] = pred

    order_final = np.argsort(-final_scores, axis=1)
    pos = np.where(order_final == targets[:, None])[1] + 1
    n = int(pos.shape[0])
    return {
        "n_samples": n,
        "mrr": float(np.mean(1.0 / pos)) if n else 0.0,
        "top1": float(np.mean(pos <= 1)) if n else 0.0,
        "top3": float(np.mean(pos <= 3)) if n else 0.0,
        "top5": float(np.mean(pos <= 5)) if n else 0.0,
        "top10": float(np.mean(pos <= 10)) if n else 0.0,
        "top20": float(np.mean(pos <= 20)) if n else 0.0,
    }


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--defect-csv", default="")
    parser.add_argument("--model-type", default="hydraulic_inverse")
    parser.add_argument("--split-mode", default="scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", required=True)
    parser.add_argument("--topks", default="5,10,20")
    parser.add_argument("--taus", default="2,5,8")
    parser.add_argument("--c-values", default="0.1,1.0,10.0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--candidate-onehot", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--use-type-probs", action="store_true")
    parser.add_argument("--type-aware-ie", action="store_true")
    parser.add_argument("--candidate-tier-csv", default="")
    parser.add_argument("--reranker-kind", choices=["lr", "mlp"], default="lr")
    parser.add_argument("--mlp-hidden", default="64,32")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = _make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, dataset = _create_loaders(cfg)
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

    model = _build_model(cfg, dataset, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    sym_dist = _sym_shortest_dist(cfg.graph_path_features_file)
    candidate_tier_features = _load_candidate_tier_features(args.candidate_tier_csv, list(dataset.node_list))
    train_arrays = _collect_split_arrays(model, train_loader, device)
    val_arrays = _collect_split_arrays(model, val_loader, device)
    test_arrays = _collect_split_arrays(model, test_loader, device)
    prior = _estimate_candidate_prior(train_arrays["probs"])
    node_pca, graph_pca = _fit_embedding_projectors(train_arrays, args.embedding_dim)
    train_proj = _project_embeddings(train_arrays, node_pca, graph_pca)
    val_proj = _project_embeddings(val_arrays, node_pca, graph_pca)
    test_proj = _project_embeddings(test_arrays, node_pca, graph_pca)

    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]
    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]

    rows: List[Dict[str, float]] = []
    best_val = None
    best_bundle = None
    if args.reranker_kind == "mlp":
        hidden_sizes = tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip())
        if not hidden_sizes:
            raise ValueError("MLP reranker requires at least one hidden size via --mlp-hidden")

    for tau in taus:
        train_path = _build_path_affinity(train_arrays["activation"], train_arrays["candidate_mask"], sym_dist, tau)
        val_path = _build_path_affinity(val_arrays["activation"], val_arrays["candidate_mask"], sym_dist, tau)
        test_path = _build_path_affinity(test_arrays["activation"], test_arrays["candidate_mask"], sym_dist, tau)
        for topk in topks:
            X_train, y_train, train_sample_ids = _build_shortlist_dataset(
                probs=train_arrays["probs"],
                targets=train_arrays["targets"],
                candidate_mask=train_arrays["candidate_mask"],
                path_affinity=train_path,
                prior=prior,
                observed_mask=train_arrays["observed_mask"],
                topk=topk,
                candidate_onehot=args.candidate_onehot,
                node_embedding_proj=train_proj.get("node_embedding_proj"),
                graph_embedding_proj=train_proj.get("graph_embedding_proj"),
                graph_type_probs=train_arrays.get("graph_type_probs") if args.use_type_probs else None,
                candidate_tier_features=candidate_tier_features,
            )
            if X_train.size == 0 or y_train.sum() == 0:
                continue
            X_val, y_val, val_sample_ids = _build_shortlist_dataset(
                probs=val_arrays["probs"],
                targets=val_arrays["targets"],
                candidate_mask=val_arrays["candidate_mask"],
                path_affinity=val_path,
                prior=prior,
                observed_mask=val_arrays["observed_mask"],
                topk=topk,
                candidate_onehot=args.candidate_onehot,
                node_embedding_proj=val_proj.get("node_embedding_proj"),
                graph_embedding_proj=val_proj.get("graph_embedding_proj"),
                graph_type_probs=val_arrays.get("graph_type_probs") if args.use_type_probs else None,
                candidate_tier_features=candidate_tier_features,
            )
            if X_val.size == 0:
                continue

            for c_value in c_values:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                reranker_model = _make_reranker_model(args, c_value)
                reranker_model.fit(X_train_scaled, y_train)

                if args.type_aware_ie:
                    if "defect_type_str" not in train_arrays or "graph_type_probs" not in val_arrays:
                        continue
                    model_dict = {"__global__": reranker_model}
                    scaler_dict = {"__global__": scaler}
                    train_types = np.asarray(train_arrays["defect_type_str"], dtype=object)
                    train_keep = np.unique(train_sample_ids)
                    kept_types = train_types[train_keep]
                    for dtype in ("I", "E"):
                        sample_mask = kept_types == dtype
                        if int(sample_mask.sum()) < 8:
                            continue
                        train_sample_ids_sel = train_keep[sample_mask]
                        row_mask = np.isin(train_sample_ids, train_sample_ids_sel)
                        X_t = X_train[row_mask]
                        y_t = y_train[row_mask]
                        if X_t.size == 0 or int(y_t.sum()) == 0 or int((1 - y_t).sum()) == 0:
                            continue
                        scaler_t = StandardScaler()
                        X_t_scaled = scaler_t.fit_transform(X_t)
                        reranker_t = _make_reranker_model(args, c_value)
                        reranker_t.fit(X_t_scaled, y_t)
                        model_dict[dtype] = reranker_t
                        scaler_dict[dtype] = scaler_t

                    val_metric = _rerank_with_type_models(
                        model_dict=model_dict,
                        scaler_dict=scaler_dict,
                        probs=val_arrays["probs"],
                        targets=val_arrays["targets"],
                        candidate_mask=val_arrays["candidate_mask"],
                        path_affinity=val_path,
                        prior=prior,
                        observed_mask=val_arrays["observed_mask"],
                        topk=topk,
                        graph_type_probs=val_arrays["graph_type_probs"],
                        candidate_onehot=args.candidate_onehot,
                        node_embedding_proj=val_proj.get("node_embedding_proj"),
                        graph_embedding_proj=val_proj.get("graph_embedding_proj"),
                        candidate_tier_features=candidate_tier_features,
                    )
                    bundle = (model_dict, scaler_dict, test_path, topk, True)
                else:
                    val_metric = _rerank_with_model(
                        model_lr=reranker_model,
                        scaler=scaler,
                        probs=val_arrays["probs"],
                        targets=val_arrays["targets"],
                        candidate_mask=val_arrays["candidate_mask"],
                        path_affinity=val_path,
                        prior=prior,
                        observed_mask=val_arrays["observed_mask"],
                        topk=topk,
                        candidate_onehot=args.candidate_onehot,
                        node_embedding_proj=val_proj.get("node_embedding_proj"),
                        graph_embedding_proj=val_proj.get("graph_embedding_proj"),
                        graph_type_probs=val_arrays.get("graph_type_probs") if args.use_type_probs else None,
                        candidate_tier_features=candidate_tier_features,
                    )
                    bundle = (reranker_model, scaler, test_path, topk, False)
                row = dict(val_metric)
                row.update({
                    "tau": float(tau),
                    "topk": int(topk),
                    "c_value": float(c_value),
                    "candidate_onehot": bool(args.candidate_onehot),
                    "embedding_dim": int(args.embedding_dim),
                    "use_type_probs": bool(args.use_type_probs),
                    "type_aware_ie": bool(args.type_aware_ie),
                    "use_candidate_tier_features": bool(args.candidate_tier_csv),
                    "reranker_kind": args.reranker_kind,
                    "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
                    "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
                    "train_shortlist_samples": int(len(np.unique(train_sample_ids))),
                    "val_shortlist_samples": int(len(np.unique(val_sample_ids))),
                })
                rows.append(row)

                if best_val is None or (row["top1"], row["mrr"], row["top3"]) > (
                    best_val["top1"],
                    best_val["mrr"],
                    best_val["top3"],
                ):
                    best_val = row
                    best_bundle = bundle

    if best_val is None or best_bundle is None:
        raise RuntimeError("No valid reranker configuration found.")

    reranker_obj, scaler_obj, test_path, topk, is_type_aware = best_bundle
    if is_type_aware:
        test_metric = _rerank_with_type_models(
            model_dict=reranker_obj,
            scaler_dict=scaler_obj,
            probs=test_arrays["probs"],
            targets=test_arrays["targets"],
            candidate_mask=test_arrays["candidate_mask"],
            path_affinity=test_path,
            prior=prior,
            observed_mask=test_arrays["observed_mask"],
            topk=topk,
            graph_type_probs=test_arrays["graph_type_probs"],
            candidate_onehot=args.candidate_onehot,
            node_embedding_proj=test_proj.get("node_embedding_proj"),
            graph_embedding_proj=test_proj.get("graph_embedding_proj"),
            candidate_tier_features=candidate_tier_features,
        )
    else:
        test_metric = _rerank_with_model(
            model_lr=reranker_obj,
            scaler=scaler_obj,
            probs=test_arrays["probs"],
            targets=test_arrays["targets"],
            candidate_mask=test_arrays["candidate_mask"],
            path_affinity=test_path,
            prior=prior,
            observed_mask=test_arrays["observed_mask"],
            topk=topk,
            candidate_onehot=args.candidate_onehot,
            node_embedding_proj=test_proj.get("node_embedding_proj"),
            graph_embedding_proj=test_proj.get("graph_embedding_proj"),
            graph_type_probs=test_arrays.get("graph_type_probs") if args.use_type_probs else None,
            candidate_tier_features=candidate_tier_features,
        )
    _write_csv(Path(args.output_csv), rows)
    payload = {
        "checkpoint": args.checkpoint,
        "subdir": args.subdir,
        "model_type": args.model_type,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "features": cfg.selected_features,
        "reranker_kind": args.reranker_kind,
        "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
        "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
        "best_on_val": best_val,
        "test_with_best_val_setting": test_metric,
        "embedding_dim": int(args.embedding_dim),
        "use_type_probs": bool(args.use_type_probs),
        "type_aware_ie": bool(args.type_aware_ie),
        "use_candidate_tier_features": bool(args.candidate_tier_csv),
    }
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
