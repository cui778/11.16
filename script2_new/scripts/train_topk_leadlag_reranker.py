#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a top-k reranker augmented with dynamic lead-lag features derived from
observed time-series responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from train_topk_reranker import (
    _build_model,
    _build_path_affinity,
    _create_loaders,
    _estimate_candidate_prior,
    _fit_embedding_projectors,
    _get_subset_base_and_indices,
    _make_config,
    _make_reranker_model,
    _project_embeddings,
    _sym_shortest_dist,
    _write_csv,
    estimate_process_signal_thresholds,
)


def _collect_split_arrays_with_leadlag(model, loader, device: torch.device) -> Dict[str, np.ndarray]:
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
    centroid_rows: List[np.ndarray] = []
    peak_rows: List[np.ndarray] = []
    spread_rows: List[np.ndarray] = []

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
            logits_type = logits[2] if isinstance(logits, tuple) and len(logits) >= 3 else None
            scores = logits_node[:, :, 1]
            node_embeddings = model.get_node_embeddings(x, adj) if hasattr(model, "get_node_embeddings") else None

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

            dynamic_x = x[..., :-3] if x.shape[-1] > 3 else x
            signal = dynamic_x.abs().mean(dim=-1)  # [B, T, N]
            activation = signal.mean(dim=1) * observed_mask
            t_idx = torch.arange(signal.shape[1], dtype=signal.dtype, device=signal.device).view(1, -1, 1)
            mass = signal.sum(dim=1).clamp_min(1e-8)
            centroid = (signal * t_idx).sum(dim=1) / mass
            peak = signal.argmax(dim=1).float()
            spread = torch.sqrt((((t_idx - centroid.unsqueeze(1)) ** 2) * signal).sum(dim=1) / mass)

            prob_rows.append(probs.detach().cpu().numpy())
            target_rows.extend(y_idx.detach().cpu().tolist())
            cand_rows.append(cand_mask.detach().cpu().numpy().astype(bool))
            act_rows.append(activation.detach().cpu().numpy())
            observed_rows.append(observed_mask.detach().cpu().numpy())
            centroid_rows.append(centroid.detach().cpu().numpy())
            peak_rows.append(peak.detach().cpu().numpy())
            spread_rows.append(spread.detach().cpu().numpy())
            if logits_type is not None:
                type_prob_rows.append(torch.softmax(logits_type, dim=-1).detach().cpu().numpy())
            if node_embeddings is not None:
                node_embed_rows.append(node_embeddings.detach().cpu().numpy())
                graph_embed_rows.append(node_embeddings.mean(dim=1).detach().cpu().numpy())

            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            defect_type_rows.extend([defect_types[i] for i, keep in enumerate(valid_np) if keep])

    payload = {
        "probs": np.concatenate(prob_rows, axis=0),
        "targets": np.asarray(target_rows, dtype=np.int64),
        "candidate_mask": np.concatenate(cand_rows, axis=0),
        "activation": np.concatenate(act_rows, axis=0),
        "observed_mask": np.concatenate(observed_rows, axis=0),
        "temporal_centroid": np.concatenate(centroid_rows, axis=0),
        "temporal_peak": np.concatenate(peak_rows, axis=0),
        "temporal_spread": np.concatenate(spread_rows, axis=0),
    }
    if type_prob_rows:
        payload["graph_type_probs"] = np.concatenate(type_prob_rows, axis=0)
        payload["defect_type_str"] = np.asarray(defect_type_rows, dtype=object)
    if node_embed_rows:
        payload["node_embeddings"] = np.concatenate(node_embed_rows, axis=0)
        payload["graph_embeddings"] = np.concatenate(graph_embed_rows, axis=0)
    return payload


def _leadlag_feature_block(
    row_idx: int,
    cand: int,
    observed_mask: np.ndarray,
    temporal_centroid: np.ndarray,
    temporal_peak: np.ndarray,
    temporal_spread: np.ndarray,
    sym_dist: np.ndarray,
    tau: float,
) -> List[float]:
    obs_idx = np.where(observed_mask[row_idx] > 0.5)[0]
    if obs_idx.size == 0:
        return [0.0] * 9

    d = np.asarray(sym_dist[cand, obs_idx], dtype=np.float32)
    w = np.exp(-d / max(float(tau), 1e-6)).astype(np.float32)
    w_sum = float(np.clip(w.sum(), 1e-8, None))
    w = w / w_sum

    tc = np.asarray(temporal_centroid[row_idx, obs_idx], dtype=np.float32)
    tp = np.asarray(temporal_peak[row_idx, obs_idx], dtype=np.float32)
    ts = np.asarray(temporal_spread[row_idx, obs_idx], dtype=np.float32)

    mean_tc = float((w * tc).sum())
    std_tc = float(np.sqrt(np.clip((w * ((tc - mean_tc) ** 2)).sum(), 0.0, None)))
    mean_tp = float((w * tp).sum())
    mean_ts = float((w * ts).sum())
    min_tc = float(tc.min())
    max_tc = float(tc.max())

    if obs_idx.size >= 2 and float(np.std(d)) > 1e-8 and float(np.std(tc)) > 1e-8:
        corr_d_tc = float(np.corrcoef(d, tc)[0, 1])
    else:
        corr_d_tc = 0.0
    if obs_idx.size >= 2 and float(np.std(d)) > 1e-8 and float(np.std(tp)) > 1e-8:
        corr_d_tp = float(np.corrcoef(d, tp)[0, 1])
    else:
        corr_d_tp = 0.0
    if obs_idx.size >= 2 and float(np.std(tc)) > 1e-8 and float(np.std(tp)) > 1e-8:
        corr_tc_tp = float(np.corrcoef(tc, tp)[0, 1])
    else:
        corr_tc_tp = 0.0

    return [
        mean_tc,
        std_tc,
        mean_tp,
        mean_ts,
        min_tc,
        max_tc,
        corr_d_tc,
        corr_d_tp,
        corr_tc_tp,
    ]


def _build_shortlist_dataset_leadlag(
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    temporal_centroid: np.ndarray,
    temporal_peak: np.ndarray,
    temporal_spread: np.ndarray,
    sym_dist: np.ndarray,
    topk: int,
    tau: float,
    candidate_onehot: bool = False,
    node_embedding_proj: np.ndarray | None = None,
    graph_embedding_proj: np.ndarray | None = None,
    graph_type_probs: np.ndarray | None = None,
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
            row_features.extend(
                _leadlag_feature_block(
                    row_idx=row_idx,
                    cand=cand,
                    observed_mask=observed_mask,
                    temporal_centroid=temporal_centroid,
                    temporal_peak=temporal_peak,
                    temporal_spread=temporal_spread,
                    sym_dist=sym_dist,
                    tau=tau,
                )
            )
            if node_embedding_proj is not None and graph_embedding_proj is not None:
                cand_emb = node_embedding_proj[row_idx, cand]
                graph_emb = graph_embedding_proj[row_idx]
                row_features.extend(cand_emb.tolist())
                row_features.extend(graph_emb.tolist())
                row_features.extend((cand_emb - graph_emb).tolist())
            if graph_type_probs is not None:
                row_features.extend(graph_type_probs[row_idx].tolist())
            if candidate_onehot:
                row_features.extend([1.0 if j == cand else 0.0 for j in range(probs.shape[1])])
            features.append(row_features)
            labels.append(int(cand == target))
            sample_ids.append(row_idx)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(sample_ids, dtype=np.int64)


def _rerank_with_model_leadlag(
    model_lr,
    scaler: StandardScaler,
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    temporal_centroid: np.ndarray,
    temporal_peak: np.ndarray,
    temporal_spread: np.ndarray,
    sym_dist: np.ndarray,
    topk: int,
    tau: float,
    candidate_onehot: bool = False,
    node_embedding_proj: np.ndarray | None = None,
    graph_embedding_proj: np.ndarray | None = None,
    graph_type_probs: np.ndarray | None = None,
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
            + _leadlag_feature_block(
                row_idx=row_idx,
                cand=int(cand),
                observed_mask=observed_mask,
                temporal_centroid=temporal_centroid,
                temporal_peak=temporal_peak,
                temporal_spread=temporal_spread,
                sym_dist=sym_dist,
                tau=tau,
            )
            + (
                (
                    node_embedding_proj[row_idx, cand].tolist()
                    + graph_embedding_proj[row_idx].tolist()
                    + (node_embedding_proj[row_idx, cand] - graph_embedding_proj[row_idx]).tolist()
                ) if (node_embedding_proj is not None and graph_embedding_proj is not None) else []
            )
            + (graph_type_probs[row_idx].tolist() if graph_type_probs is not None else [])
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
    parser.add_argument("--reranker-kind", choices=["lr", "mlp"], default="mlp")
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
    train_arrays = _collect_split_arrays_with_leadlag(model, train_loader, device)
    val_arrays = _collect_split_arrays_with_leadlag(model, val_loader, device)
    test_arrays = _collect_split_arrays_with_leadlag(model, test_loader, device)
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
    for tau in taus:
        train_path = _build_path_affinity(train_arrays["activation"], train_arrays["candidate_mask"], sym_dist, tau)
        val_path = _build_path_affinity(val_arrays["activation"], val_arrays["candidate_mask"], sym_dist, tau)
        test_path = _build_path_affinity(test_arrays["activation"], test_arrays["candidate_mask"], sym_dist, tau)
        for topk in topks:
            X_train, y_train, train_sample_ids = _build_shortlist_dataset_leadlag(
                probs=train_arrays["probs"],
                targets=train_arrays["targets"],
                candidate_mask=train_arrays["candidate_mask"],
                path_affinity=train_path,
                prior=prior,
                observed_mask=train_arrays["observed_mask"],
                temporal_centroid=train_arrays["temporal_centroid"],
                temporal_peak=train_arrays["temporal_peak"],
                temporal_spread=train_arrays["temporal_spread"],
                sym_dist=sym_dist,
                topk=topk,
                tau=tau,
                candidate_onehot=args.candidate_onehot,
                node_embedding_proj=train_proj.get("node_embedding_proj"),
                graph_embedding_proj=train_proj.get("graph_embedding_proj"),
                graph_type_probs=train_arrays.get("graph_type_probs") if args.use_type_probs else None,
            )
            if X_train.size == 0 or y_train.sum() == 0:
                continue
            X_val, y_val, val_sample_ids = _build_shortlist_dataset_leadlag(
                probs=val_arrays["probs"],
                targets=val_arrays["targets"],
                candidate_mask=val_arrays["candidate_mask"],
                path_affinity=val_path,
                prior=prior,
                observed_mask=val_arrays["observed_mask"],
                temporal_centroid=val_arrays["temporal_centroid"],
                temporal_peak=val_arrays["temporal_peak"],
                temporal_spread=val_arrays["temporal_spread"],
                sym_dist=sym_dist,
                topk=topk,
                tau=tau,
                candidate_onehot=args.candidate_onehot,
                node_embedding_proj=val_proj.get("node_embedding_proj"),
                graph_embedding_proj=val_proj.get("graph_embedding_proj"),
                graph_type_probs=val_arrays.get("graph_type_probs") if args.use_type_probs else None,
            )
            if X_val.size == 0:
                continue

            for c_value in c_values:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                reranker_model = _make_reranker_model(args, c_value)
                reranker_model.fit(X_train_scaled, y_train)

                val_metric = _rerank_with_model_leadlag(
                    model_lr=reranker_model,
                    scaler=scaler,
                    probs=val_arrays["probs"],
                    targets=val_arrays["targets"],
                    candidate_mask=val_arrays["candidate_mask"],
                    path_affinity=val_path,
                    prior=prior,
                    observed_mask=val_arrays["observed_mask"],
                    temporal_centroid=val_arrays["temporal_centroid"],
                    temporal_peak=val_arrays["temporal_peak"],
                    temporal_spread=val_arrays["temporal_spread"],
                    sym_dist=sym_dist,
                    topk=topk,
                    tau=tau,
                    candidate_onehot=args.candidate_onehot,
                    node_embedding_proj=val_proj.get("node_embedding_proj"),
                    graph_embedding_proj=val_proj.get("graph_embedding_proj"),
                    graph_type_probs=val_arrays.get("graph_type_probs") if args.use_type_probs else None,
                )
                row = dict(val_metric)
                row.update({
                    "tau": float(tau),
                    "topk": int(topk),
                    "c_value": float(c_value),
                    "candidate_onehot": bool(args.candidate_onehot),
                    "embedding_dim": int(args.embedding_dim),
                    "use_type_probs": bool(args.use_type_probs),
                    "reranker_kind": args.reranker_kind,
                    "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
                    "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
                    "train_shortlist_samples": int(len(np.unique(train_sample_ids))),
                    "val_shortlist_samples": int(len(np.unique(val_sample_ids))),
                })
                rows.append(row)

                if best_val is None or (row["top1"], row["mrr"], row["top3"]) > (
                    best_val["top1"], best_val["mrr"], best_val["top3"]
                ):
                    best_val = row
                    best_bundle = (reranker_model, scaler, test_path, topk, tau)

    if best_val is None or best_bundle is None:
        raise RuntimeError("No valid lead-lag reranker configuration found.")

    reranker_obj, scaler_obj, test_path, topk, tau = best_bundle
    test_metric = _rerank_with_model_leadlag(
        model_lr=reranker_obj,
        scaler=scaler_obj,
        probs=test_arrays["probs"],
        targets=test_arrays["targets"],
        candidate_mask=test_arrays["candidate_mask"],
        path_affinity=test_path,
        prior=prior,
        observed_mask=test_arrays["observed_mask"],
        temporal_centroid=test_arrays["temporal_centroid"],
        temporal_peak=test_arrays["temporal_peak"],
        temporal_spread=test_arrays["temporal_spread"],
        sym_dist=sym_dist,
        topk=topk,
        tau=tau,
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
        "seed": args.seed,
        "features": cfg.selected_features,
        "leadlag_feature_block_dim": 9,
        "best_on_val": best_val,
        "test_with_best_val_setting": test_metric,
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
