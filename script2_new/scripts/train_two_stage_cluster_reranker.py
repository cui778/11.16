#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe a true two-stage localization pipeline:
1. coarse cluster routing over candidate nodes
2. cluster-local reranking within selected cluster union

The goal is not to replace the current mainline immediately, but to verify
whether "cluster -> node" routing can outperform the current global reranker.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from train_topk_reranker import (
    _build_model,
    _collect_split_arrays,
    _create_loaders,
    _ensure_adj_2d,  # noqa: F401  # kept for compatibility imports
    _estimate_candidate_prior,
    _fit_embedding_projectors,
    _get_subset_base_and_indices,
    _make_config,
    _make_reranker_model,
    _project_embeddings,
    _sym_shortest_dist,
    _write_csv,
)
from train.train import estimate_process_signal_thresholds


def _build_path_affinity(
    activation: np.ndarray,
    candidate_mask: np.ndarray,
    sym_dist: np.ndarray,
    tau: float,
) -> np.ndarray:
    tau = float(max(tau, 1e-6))
    kernel = np.exp(-sym_dist / tau).astype(np.float32)
    affinity = activation @ kernel.T
    affinity = np.clip(affinity, 1e-8, None)
    affinity = affinity * candidate_mask.astype(np.float32)
    denom = affinity.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return affinity / denom


def _candidate_indices(candidate_mask: np.ndarray) -> np.ndarray:
    if candidate_mask.ndim != 2:
        raise ValueError("candidate_mask must be [B, N]")
    counts = candidate_mask.sum(axis=1)
    if not np.all(counts == counts[0]):
        raise ValueError("candidate_mask varies across rows unexpectedly")
    return np.where(candidate_mask[0])[0].astype(np.int64)


def _cluster_candidates(sym_dist: np.ndarray, candidate_indices: np.ndarray, n_clusters: int) -> np.ndarray:
    cand_dist = sym_dist[np.ix_(candidate_indices, candidate_indices)].astype(np.float32)
    finite = cand_dist[np.isfinite(cand_dist)]
    if finite.size == 0:
        raise RuntimeError("No finite candidate-candidate distances found.")
    max_finite = float(np.max(finite))
    cand_dist = np.where(np.isfinite(cand_dist), cand_dist, max_finite + 1.0)
    cand_dist = np.minimum(cand_dist, cand_dist.T)
    np.fill_diagonal(cand_dist, 0.0)
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            affinity="precomputed",
            linkage="average",
        )
    return clusterer.fit_predict(cand_dist).astype(np.int64)


def _cluster_score_matrix(
    probs: np.ndarray,
    candidate_indices: np.ndarray,
    cluster_labels: np.ndarray,
    score_mode: str,
) -> np.ndarray:
    cluster_ids = sorted(set(cluster_labels.tolist()))
    out = np.zeros((probs.shape[0], len(cluster_ids)), dtype=np.float32)
    cand_probs = probs[:, candidate_indices]
    for col, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        block = cand_probs[:, mask]
        if score_mode == "sum":
            out[:, col] = block.sum(axis=1)
        elif score_mode == "max":
            out[:, col] = block.max(axis=1)
        else:
            raise ValueError(f"Unsupported score_mode: {score_mode}")
    return out


def _build_candidate_features(
    row_idx: int,
    cand_list: np.ndarray,
    probs: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    candidate_onehot: bool,
    node_embedding_proj: np.ndarray | None,
    graph_embedding_proj: np.ndarray | None,
    graph_type_probs: np.ndarray | None,
) -> np.ndarray:
    cand_probs = probs[row_idx, cand_list]
    order = np.argsort(-cand_probs)
    sorted_probs = cand_probs[order]
    local_rank = np.empty_like(order)
    local_rank[order] = np.arange(1, len(cand_list) + 1, dtype=np.int64)
    prob_gap = float(sorted_probs[0]) - cand_probs

    log_probs = np.log(np.clip(probs[row_idx, cand_list], 1e-8, None))
    log_path = np.log(np.clip(path_affinity[row_idx, cand_list], 1e-8, None))
    log_prior = np.log(np.clip(prior[cand_list], 1e-8, None))
    monitored_flag = observed_mask.max(axis=0)[cand_list]

    rows: List[List[float]] = []
    for local_idx, cand in enumerate(cand_list.tolist()):
        feat = [
            float(log_probs[local_idx]),
            float(probs[row_idx, cand]),
            float(log_path[local_idx]),
            float(path_affinity[row_idx, cand]),
            float(log_prior[local_idx]),
            float(prior[cand]),
            float(local_rank[local_idx] / max(1, len(cand_list))),
            float(prob_gap[local_idx]),
            float(monitored_flag[local_idx]),
        ]
        if node_embedding_proj is not None and graph_embedding_proj is not None:
            cand_emb = node_embedding_proj[row_idx, cand]
            graph_emb = graph_embedding_proj[row_idx]
            feat.extend(cand_emb.tolist())
            feat.extend(graph_emb.tolist())
            feat.extend((cand_emb - graph_emb).tolist())
        if graph_type_probs is not None:
            feat.extend(graph_type_probs[row_idx].tolist())
        if candidate_onehot:
            feat.extend([1.0 if j == cand else 0.0 for j in range(probs.shape[1])])
        rows.append(feat)
    return np.asarray(rows, dtype=np.float32)


def _build_train_dataset(
    arrays: Dict[str, np.ndarray],
    path_affinity: np.ndarray,
    prior: np.ndarray,
    candidate_indices: np.ndarray,
    cluster_labels: np.ndarray,
    candidate_onehot: bool,
    node_embedding_proj: np.ndarray | None,
    graph_embedding_proj: np.ndarray | None,
    graph_type_probs: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    kept_samples = 0

    cand_to_cluster = {int(c): int(cluster_labels[i]) for i, c in enumerate(candidate_indices.tolist())}
    cluster_to_cands: Dict[int, np.ndarray] = {}
    for cid in sorted(set(cluster_labels.tolist())):
        cluster_to_cands[int(cid)] = candidate_indices[cluster_labels == cid]

    for row_idx in range(arrays["probs"].shape[0]):
        target = int(arrays["targets"][row_idx])
        cid = cand_to_cluster.get(target, None)
        if cid is None:
            continue
        cand_list = cluster_to_cands[int(cid)]
        if cand_list.size <= 1:
            continue
        feat = _build_candidate_features(
            row_idx=row_idx,
            cand_list=cand_list,
            probs=arrays["probs"],
            path_affinity=path_affinity,
            prior=prior,
            observed_mask=arrays["observed_mask"],
            candidate_onehot=candidate_onehot,
            node_embedding_proj=node_embedding_proj,
            graph_embedding_proj=graph_embedding_proj,
            graph_type_probs=graph_type_probs,
        )
        labels = (cand_list == target).astype(np.int64)
        if int(labels.sum()) != 1:
            continue
        X_rows.append(feat)
        y_rows.extend(labels.tolist())
        kept_samples += 1

    if not X_rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), 0
    X = np.concatenate(X_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.int64)
    return X, y, kept_samples


def _evaluate_two_stage(
    reranker,
    scaler: StandardScaler,
    arrays: Dict[str, np.ndarray],
    path_affinity: np.ndarray,
    prior: np.ndarray,
    candidate_indices: np.ndarray,
    cluster_labels: np.ndarray,
    score_mode: str,
    cluster_topm: int,
    candidate_onehot: bool,
    node_embedding_proj: np.ndarray | None,
    graph_embedding_proj: np.ndarray | None,
    graph_type_probs: np.ndarray | None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    cluster_ids = sorted(set(cluster_labels.tolist()))
    cluster_scores = _cluster_score_matrix(
        arrays["probs"],
        candidate_indices=candidate_indices,
        cluster_labels=cluster_labels,
        score_mode=score_mode,
    )
    order = np.argsort(-cluster_scores, axis=1)
    cluster_to_cands: Dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        cluster_to_cands[int(cid)] = candidate_indices[cluster_labels == cid]

    final_scores = np.where(arrays["candidate_mask"], -1e12, -1e12).astype(np.float32)
    final_scores[arrays["candidate_mask"]] = np.log(np.clip(arrays["probs"][arrays["candidate_mask"]], 1e-8, None))

    for row_idx in range(arrays["probs"].shape[0]):
        chosen_clusters = [cluster_ids[int(i)] for i in order[row_idx, :cluster_topm]]
        cand_list = np.concatenate([cluster_to_cands[int(cid)] for cid in chosen_clusters], axis=0)
        cand_list = np.unique(cand_list.astype(np.int64))
        if cand_list.size == 0:
            continue
        feat = _build_candidate_features(
            row_idx=row_idx,
            cand_list=cand_list,
            probs=arrays["probs"],
            path_affinity=path_affinity,
            prior=prior,
            observed_mask=arrays["observed_mask"],
            candidate_onehot=candidate_onehot,
            node_embedding_proj=node_embedding_proj,
            graph_embedding_proj=graph_embedding_proj,
            graph_type_probs=graph_type_probs,
        )
        pred = reranker.predict_proba(scaler.transform(feat))[:, 1]
        final_scores[row_idx, cand_list] = pred

    order_final = np.argsort(-final_scores, axis=1)
    pos = np.where(order_final == arrays["targets"][:, None])[1] + 1
    n = int(pos.shape[0])
    metric = {
        "n_samples": n,
        "mrr": float(np.mean(1.0 / pos)) if n else 0.0,
        "top1": float(np.mean(pos <= 1)) if n else 0.0,
        "top3": float(np.mean(pos <= 3)) if n else 0.0,
        "top5": float(np.mean(pos <= 5)) if n else 0.0,
        "top10": float(np.mean(pos <= 10)) if n else 0.0,
        "top20": float(np.mean(pos <= 20)) if n else 0.0,
    }

    by_type: Dict[str, Dict[str, float]] = {}
    if "defect_type_str" in arrays:
        dtypes = np.asarray(arrays["defect_type_str"], dtype=object)
        for dtype in sorted(set(dtypes.tolist())):
            if dtype == "BASELINE":
                continue
            mask = dtypes == dtype
            pos_sub = pos[mask]
            by_type[str(dtype)] = {
                "n_samples": int(pos_sub.shape[0]),
                "mrr": float(np.mean(1.0 / pos_sub)) if pos_sub.shape[0] else 0.0,
                "top1": float(np.mean(pos_sub <= 1)) if pos_sub.shape[0] else 0.0,
                "top3": float(np.mean(pos_sub <= 3)) if pos_sub.shape[0] else 0.0,
                "top5": float(np.mean(pos_sub <= 5)) if pos_sub.shape[0] else 0.0,
            }
    return metric, by_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--defect-csv", default="")
    parser.add_argument("--model-type", default="hydraulic_inverse_deepattn")
    parser.add_argument("--split-mode", default="scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", required=True)
    parser.add_argument("--cluster-ks", default="6,8,10")
    parser.add_argument("--cluster-topms", default="1,2")
    parser.add_argument("--score-modes", default="sum,max")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--c-values", default="0.1,1.0,10.0")
    parser.add_argument("--candidate-onehot", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--use-type-probs", action="store_true")
    parser.add_argument("--reranker-kind", choices=["lr", "mlp"], default="mlp")
    parser.add_argument("--mlp-hidden", default="64,32")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
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
    train_arrays = _collect_split_arrays(model, train_loader, device)
    val_arrays = _collect_split_arrays(model, val_loader, device)
    test_arrays = _collect_split_arrays(model, test_loader, device)
    prior = _estimate_candidate_prior(train_arrays["probs"])
    node_pca, graph_pca = _fit_embedding_projectors(train_arrays, args.embedding_dim)
    train_proj = _project_embeddings(train_arrays, node_pca, graph_pca)
    val_proj = _project_embeddings(val_arrays, node_pca, graph_pca)
    test_proj = _project_embeddings(test_arrays, node_pca, graph_pca)

    candidate_indices = _candidate_indices(train_arrays["candidate_mask"])
    train_path = _build_path_affinity(train_arrays["activation"], train_arrays["candidate_mask"], sym_dist, args.tau)
    val_path = _build_path_affinity(val_arrays["activation"], val_arrays["candidate_mask"], sym_dist, args.tau)
    test_path = _build_path_affinity(test_arrays["activation"], test_arrays["candidate_mask"], sym_dist, args.tau)

    cluster_ks = [int(x.strip()) for x in args.cluster_ks.split(",") if x.strip()]
    cluster_topms = [int(x.strip()) for x in args.cluster_topms.split(",") if x.strip()]
    score_modes = [x.strip() for x in args.score_modes.split(",") if x.strip()]
    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]

    rows: List[Dict[str, float]] = []
    best_val = None
    best_bundle = None

    for n_clusters in cluster_ks:
        cluster_labels = _cluster_candidates(sym_dist, candidate_indices, n_clusters)
        cluster_sizes = [int(np.sum(cluster_labels == cid)) for cid in sorted(set(cluster_labels.tolist()))]
        X_train, y_train, kept_train = _build_train_dataset(
            arrays=train_arrays,
            path_affinity=train_path,
            prior=prior,
            candidate_indices=candidate_indices,
            cluster_labels=cluster_labels,
            candidate_onehot=args.candidate_onehot,
            node_embedding_proj=train_proj.get("node_embedding_proj"),
            graph_embedding_proj=train_proj.get("graph_embedding_proj"),
            graph_type_probs=train_arrays.get("graph_type_probs") if args.use_type_probs else None,
        )
        if X_train.size == 0 or int(y_train.sum()) == 0 or int((1 - y_train).sum()) == 0:
            continue
        for c_value in c_values:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            reranker = _make_reranker_model(args, c_value)
            reranker.fit(X_train_scaled, y_train)
            for score_mode in score_modes:
                for cluster_topm in cluster_topms:
                    val_metric, _ = _evaluate_two_stage(
                        reranker=reranker,
                        scaler=scaler,
                        arrays=val_arrays,
                        path_affinity=val_path,
                        prior=prior,
                        candidate_indices=candidate_indices,
                        cluster_labels=cluster_labels,
                        score_mode=score_mode,
                        cluster_topm=cluster_topm,
                        candidate_onehot=args.candidate_onehot,
                        node_embedding_proj=val_proj.get("node_embedding_proj"),
                        graph_embedding_proj=val_proj.get("graph_embedding_proj"),
                        graph_type_probs=val_arrays.get("graph_type_probs") if args.use_type_probs else None,
                    )
                    row = dict(val_metric)
                    row.update(
                        {
                            "n_clusters": int(n_clusters),
                            "cluster_topm": int(cluster_topm),
                            "score_mode": score_mode,
                            "tau": float(args.tau),
                            "c_value": float(c_value),
                            "candidate_onehot": bool(args.candidate_onehot),
                            "embedding_dim": int(args.embedding_dim),
                            "use_type_probs": bool(args.use_type_probs),
                            "reranker_kind": args.reranker_kind,
                            "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
                            "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
                            "train_cluster_samples": int(kept_train),
                            "cluster_sizes": json.dumps(cluster_sizes),
                        }
                    )
                    rows.append(row)
                    if best_val is None or (row["top1"], row["mrr"], row["top3"], row["top5"]) > (
                        best_val["top1"],
                        best_val["mrr"],
                        best_val["top3"],
                        best_val["top5"],
                    ):
                        best_val = row
                        best_bundle = (reranker, scaler, cluster_labels, score_mode, cluster_topm)

    if best_val is None or best_bundle is None:
        raise RuntimeError("No valid two-stage cluster reranker configuration found.")

    reranker, scaler, cluster_labels, score_mode, cluster_topm = best_bundle
    test_metric, test_by_type = _evaluate_two_stage(
        reranker=reranker,
        scaler=scaler,
        arrays=test_arrays,
        path_affinity=test_path,
        prior=prior,
        candidate_indices=candidate_indices,
        cluster_labels=cluster_labels,
        score_mode=score_mode,
        cluster_topm=cluster_topm,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=test_proj.get("node_embedding_proj"),
        graph_embedding_proj=test_proj.get("graph_embedding_proj"),
        graph_type_probs=test_arrays.get("graph_type_probs") if args.use_type_probs else None,
    )

    _write_csv(Path(args.output_csv), rows)
    payload = {
        "checkpoint": args.checkpoint,
        "subdir": args.subdir,
        "model_type": args.model_type,
        "split_mode": args.split_mode,
        "seed": int(args.seed),
        "features": cfg.selected_features,
        "best_on_val": best_val,
        "test_with_best_val_setting": test_metric,
        "test_by_type": test_by_type,
        "candidate_onehot": bool(args.candidate_onehot),
        "embedding_dim": int(args.embedding_dim),
        "use_type_probs": bool(args.use_type_probs),
        "reranker_kind": args.reranker_kind,
        "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
        "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
    }
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
