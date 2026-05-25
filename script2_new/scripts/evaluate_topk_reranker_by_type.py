#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained top-k reranker configuration and report by-type metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from train_topk_reranker import (
    _build_model,
    _build_path_affinity,
    _fit_embedding_projectors,
    _build_shortlist_dataset,
    _create_loaders,
    _estimate_candidate_prior,
    _get_subset_base_and_indices,
    _make_config,
    _project_embeddings,
    _sym_shortest_dist,
    estimate_process_signal_thresholds,
)


def _collect_split_arrays_with_type(model, loader, device: torch.device) -> Dict[str, np.ndarray | List[str]]:
    model.eval()
    prob_rows: List[np.ndarray] = []
    target_rows: List[int] = []
    cand_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    observed_rows: List[np.ndarray] = []
    type_rows: List[str] = []
    node_embed_rows: List[np.ndarray] = []
    graph_embed_rows: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            adj = batch["adj_matrix"]
            if isinstance(adj, (list, tuple)):
                adj = adj[0]
            if adj.dim() == 3:
                adj = adj[0]
            adj = adj.to(device)

            logits = model(x, adj)
            logits_node = logits[0] if isinstance(logits, tuple) else logits
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

            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            scores = scores[valid_mask]
            y_idx = y_idx[valid_mask]
            cand_mask = cand_mask[valid_mask]
            x = x[valid_mask]
            observed_mask = observed_mask[valid_mask]
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
            if node_embeddings is not None:
                node_embed_rows.append(node_embeddings.detach().cpu().numpy())
                graph_embed_rows.append(node_embeddings.mean(dim=1).detach().cpu().numpy())
            type_rows.extend([defect_types[i] for i, keep in enumerate(valid_np) if keep])

    payload = {
        "probs": np.concatenate(prob_rows, axis=0),
        "targets": np.asarray(target_rows, dtype=np.int64),
        "candidate_mask": np.concatenate(cand_rows, axis=0),
        "activation": np.concatenate(act_rows, axis=0),
        "observed_mask": np.concatenate(observed_rows, axis=0),
        "defect_type_str": type_rows,
    }
    if node_embed_rows:
        payload["node_embeddings"] = np.concatenate(node_embed_rows, axis=0)
        payload["graph_embeddings"] = np.concatenate(graph_embed_rows, axis=0)
    return payload


def _rerank_positions(
    model_lr,
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
) -> np.ndarray:
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
            + ([1.0 if j == cand else 0.0 for j in range(probs.shape[1])] if candidate_onehot else [])
            for i, cand in enumerate(shortlist)
        ], dtype=np.float32)
        pred = model_lr.predict_proba(scaler.transform(feat))[:, 1]
        final_scores[row_idx, shortlist] = pred

    order_final = np.argsort(-final_scores, axis=1)
    pos = np.where(order_final == targets[:, None])[1] + 1
    return pos.astype(np.int64)


def _metrics_from_positions(pos: np.ndarray) -> Dict[str, float]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--model-type", default="hydraulic_inverse")
    parser.add_argument("--split-mode", default="scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--c-value", type=float, required=True)
    parser.add_argument("--candidate-onehot", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--reranker-kind", choices=["lr", "mlp"], default="lr")
    parser.add_argument("--mlp-hidden", default="64,32")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--output-json", required=True)
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
    train_arrays = _collect_split_arrays_with_type(model, train_loader, device)
    val_arrays = _collect_split_arrays_with_type(model, val_loader, device)
    test_arrays = _collect_split_arrays_with_type(model, test_loader, device)
    prior = _estimate_candidate_prior(train_arrays["probs"])
    node_pca, graph_pca = _fit_embedding_projectors(train_arrays, args.embedding_dim)
    train_proj = _project_embeddings(train_arrays, node_pca, graph_pca)
    val_proj = _project_embeddings(val_arrays, node_pca, graph_pca)
    test_proj = _project_embeddings(test_arrays, node_pca, graph_pca)

    train_path = _build_path_affinity(train_arrays["activation"], train_arrays["candidate_mask"], sym_dist, args.tau)
    val_path = _build_path_affinity(val_arrays["activation"], val_arrays["candidate_mask"], sym_dist, args.tau)
    test_path = _build_path_affinity(test_arrays["activation"], test_arrays["candidate_mask"], sym_dist, args.tau)

    X_train, y_train, _ = _build_shortlist_dataset(
        probs=train_arrays["probs"],
        targets=train_arrays["targets"],
        candidate_mask=train_arrays["candidate_mask"],
        path_affinity=train_path,
        prior=prior,
        observed_mask=train_arrays["observed_mask"],
        topk=args.topk,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=train_proj.get("node_embedding_proj"),
        graph_embedding_proj=train_proj.get("graph_embedding_proj"),
    )
    X_val, y_val, _ = _build_shortlist_dataset(
        probs=val_arrays["probs"],
        targets=val_arrays["targets"],
        candidate_mask=val_arrays["candidate_mask"],
        path_affinity=val_path,
        prior=prior,
        observed_mask=val_arrays["observed_mask"],
        topk=args.topk,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=val_proj.get("node_embedding_proj"),
        graph_embedding_proj=val_proj.get("graph_embedding_proj"),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if args.reranker_kind == "lr":
        reranker = LogisticRegression(
            C=float(args.c_value),
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
    else:
        hidden_sizes = tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip())
        reranker = MLPClassifier(
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
    reranker.fit(X_train_scaled, y_train)

    val_pos = _rerank_positions(
        model_lr=reranker,
        scaler=scaler,
        probs=val_arrays["probs"],
        targets=val_arrays["targets"],
        candidate_mask=val_arrays["candidate_mask"],
        path_affinity=val_path,
        prior=prior,
        observed_mask=val_arrays["observed_mask"],
        topk=args.topk,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=val_proj.get("node_embedding_proj"),
        graph_embedding_proj=val_proj.get("graph_embedding_proj"),
    )
    test_pos = _rerank_positions(
        model_lr=reranker,
        scaler=scaler,
        probs=test_arrays["probs"],
        targets=test_arrays["targets"],
        candidate_mask=test_arrays["candidate_mask"],
        path_affinity=test_path,
        prior=prior,
        observed_mask=test_arrays["observed_mask"],
        topk=args.topk,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=test_proj.get("node_embedding_proj"),
        graph_embedding_proj=test_proj.get("graph_embedding_proj"),
    )

    by_type = {}
    defect_types = np.asarray(test_arrays["defect_type_str"], dtype=object)
    for dtype in sorted(set(defect_types.tolist())):
        if dtype == "BASELINE":
            continue
        mask = defect_types == dtype
        by_type[dtype] = _metrics_from_positions(test_pos[mask])

    payload = {
        "checkpoint": args.checkpoint,
        "subdir": args.subdir,
        "model_type": args.model_type,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "features": cfg.selected_features,
        "selected_setting": {
            "topk": int(args.topk),
            "tau": float(args.tau),
            "c_value": float(args.c_value),
            "candidate_onehot": bool(args.candidate_onehot),
            "embedding_dim": int(args.embedding_dim),
            "reranker_kind": args.reranker_kind,
            "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
            "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
        },
        "val_metrics": _metrics_from_positions(val_pos),
        "test_metrics": _metrics_from_positions(test_pos),
        "test_by_type": by_type,
    }
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
