#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observability-aware coarse-to-fine reranker prototype.

Idea:
1. Use the existing stage-1 model to get candidate probabilities.
2. Train a global shortlist reranker plus tier-specific rerankers
   (direct / near / far).
3. Route each sample to a preferred tier using stage-1 observability signals.
4. Promote candidates from the routed tier and rerank inside that tier.

This is a prototype for testing whether observability-aware task organization
helps beyond the current single global reranker.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.train_topk_reranker import (  # type: ignore
    _build_model,
    _build_path_affinity,
    _collect_split_arrays,
    _create_loaders,
    _fit_embedding_projectors,
    _load_candidate_tier_features,
    _make_config,
    _make_reranker_model,
    _project_embeddings,
    _sym_shortest_dist,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--model-type", default="hydraulic_inverse_deepattn")
    parser.add_argument("--split-mode", default="scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", required=True)
    parser.add_argument("--candidate-tier-csv", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--candidate-onehot", action="store_true")
    parser.add_argument("--reranker-kind", choices=["lr", "mlp"], default="mlp")
    parser.add_argument("--mlp-hidden", default="64,32")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _load_candidate_tiers(candidate_tier_csv: str, node_list: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_csv(candidate_tier_csv)
    tier_map = dict(zip(df["candidate_node"].astype(str), df["observability_tier"].astype(str)))
    tiers = np.asarray([tier_map.get(str(node), "non_candidate") for node in node_list], dtype=object)
    masks = {
        "direct": tiers == "direct",
        "near": tiers == "near",
        "far": tiers == "far",
    }
    return tiers, masks


def _build_rows(
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    path_affinity: np.ndarray,
    prior: np.ndarray,
    observed_mask: np.ndarray,
    topk: int,
    candidate_tiers: np.ndarray,
    candidate_onehot: bool = False,
    node_embedding_proj: np.ndarray | None = None,
    graph_embedding_proj: np.ndarray | None = None,
    candidate_tier_features: np.ndarray | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    log_probs = np.log(np.clip(probs, 1e-8, None))
    log_path = np.log(np.clip(path_affinity, 1e-8, None))
    log_prior = np.log(np.clip(prior, 1e-8, None))
    monitored_flag = observed_mask.max(axis=0)
    masked_probs = np.where(candidate_mask, probs, -1.0)
    order = np.argsort(-masked_probs, axis=1)

    for sample_idx in range(probs.shape[0]):
        shortlist = order[sample_idx, :topk]
        target = int(targets[sample_idx])
        shortlist_probs = probs[sample_idx, shortlist]
        shortlist_ranks = np.arange(1, len(shortlist) + 1, dtype=np.float32)
        prob_gap = shortlist_probs[0] - shortlist_probs
        target_tier = str(candidate_tiers[target])
        for local_rank, cand in enumerate(shortlist):
            cand = int(cand)
            feat = [
                float(log_probs[sample_idx, cand]),
                float(probs[sample_idx, cand]),
                float(log_path[sample_idx, cand]),
                float(path_affinity[sample_idx, cand]),
                float(log_prior[cand]),
                float(prior[cand]),
                float(shortlist_ranks[local_rank] / max(1, topk)),
                float(prob_gap[local_rank]),
                float(monitored_flag[cand]),
            ]
            if node_embedding_proj is not None and graph_embedding_proj is not None:
                cand_emb = node_embedding_proj[sample_idx, cand]
                graph_emb = graph_embedding_proj[sample_idx]
                feat.extend(cand_emb.tolist())
                feat.extend(graph_emb.tolist())
                feat.extend((cand_emb - graph_emb).tolist())
            if candidate_tier_features is not None and candidate_tier_features.size > 0:
                feat.extend(candidate_tier_features[cand].tolist())
            if candidate_onehot:
                feat.extend([1.0 if j == cand else 0.0 for j in range(probs.shape[1])])
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "candidate_idx": cand,
                    "candidate_tier": str(candidate_tiers[cand]),
                    "target_idx": target,
                    "target_tier": target_tier,
                    "label": int(cand == target),
                    "feat": np.asarray(feat, dtype=np.float32),
                }
            )
    return rows


def _stack_features(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack([r["feat"] for r in rows], axis=0).astype(np.float32)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    return X, y


def _fit_model(rows: List[Dict[str, object]], args: argparse.Namespace):
    X, y = _stack_features(rows)
    if X.size == 0 or int(y.sum()) == 0 or int((1 - y).sum()) == 0:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = _make_reranker_model(args, args.c_value)
    model.fit(X_scaled, y)
    return model, scaler


def _route_tier(sample_rows: List[Dict[str, object]], mode: str) -> str:
    if not sample_rows:
        return "global"
    if mode == "top1_tier":
        top_row = max(sample_rows, key=lambda r: float(r["feat"][1]))
        return str(top_row["candidate_tier"])
    masses = {"direct": 0.0, "near": 0.0, "far": 0.0}
    for r in sample_rows:
        masses[str(r["candidate_tier"])] += float(r["feat"][1])
    return max(masses.items(), key=lambda kv: kv[1])[0]


def _evaluate_rows(
    rows: List[Dict[str, object]],
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    global_model,
    global_scaler,
    tier_models: Dict[str, object],
    tier_scalers: Dict[str, StandardScaler],
    route_mode: str,
    tier_bonus: float,
) -> Dict[str, float]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_idx"]), []).append(row)

    log_probs = np.log(np.clip(probs, 1e-8, None))
    final_scores = np.where(candidate_mask, log_probs, -1e12).copy()

    for sample_idx in range(probs.shape[0]):
        sample_rows = grouped.get(sample_idx, [])
        if not sample_rows:
            continue
        route = _route_tier(sample_rows, route_mode)
        for r in sample_rows:
            feat = r["feat"][None, :]
            cand_tier = str(r["candidate_tier"])
            model = tier_models.get(cand_tier, global_model)
            scaler = tier_scalers.get(cand_tier, global_scaler)
            score = float(model.predict_proba(scaler.transform(feat))[0, 1])
            if cand_tier == route:
                score += float(tier_bonus)
            final_scores[sample_idx, int(r["candidate_idx"])] = score

    order_final = np.argsort(-final_scores, axis=1)
    pos = np.where(order_final == targets[:, None])[1] + 1
    if pos.size == 0:
        return {"n_samples": 0, "mrr": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0}
    return {
        "n_samples": int(pos.size),
        "mrr": float(np.mean(1.0 / pos)),
        "top1": float(np.mean(pos <= 1)),
        "top3": float(np.mean(pos <= 3)),
        "top5": float(np.mean(pos <= 5)),
        "top10": float(np.mean(pos <= 10)),
        "top20": float(np.mean(pos <= 20)),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    cfg = _make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, dataset = _create_loaders(cfg)
    model = _build_model(cfg, dataset, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    sym_dist = _sym_shortest_dist(cfg.graph_path_features_file)
    train_arrays = _collect_split_arrays(model, train_loader, device)
    val_arrays = _collect_split_arrays(model, val_loader, device)
    test_arrays = _collect_split_arrays(model, test_loader, device)
    prior = np.clip(train_arrays["probs"].mean(axis=0), 1e-8, None)
    node_pca, graph_pca = _fit_embedding_projectors(train_arrays, args.embedding_dim)
    train_proj = _project_embeddings(train_arrays, node_pca, graph_pca)
    val_proj = _project_embeddings(val_arrays, node_pca, graph_pca)
    test_proj = _project_embeddings(test_arrays, node_pca, graph_pca)
    candidate_tiers, _ = _load_candidate_tiers(args.candidate_tier_csv, list(dataset.node_list))
    candidate_tier_features = _load_candidate_tier_features(args.candidate_tier_csv, list(dataset.node_list))

    train_path = _build_path_affinity(train_arrays["activation"], train_arrays["candidate_mask"], sym_dist, args.tau)
    val_path = _build_path_affinity(val_arrays["activation"], val_arrays["candidate_mask"], sym_dist, args.tau)
    test_path = _build_path_affinity(test_arrays["activation"], test_arrays["candidate_mask"], sym_dist, args.tau)

    train_rows = _build_rows(
        train_arrays["probs"], train_arrays["targets"], train_arrays["candidate_mask"],
        train_path, prior, train_arrays["observed_mask"], args.topk, candidate_tiers,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=train_proj.get("node_embedding_proj"),
        graph_embedding_proj=train_proj.get("graph_embedding_proj"),
        candidate_tier_features=candidate_tier_features,
    )
    val_rows = _build_rows(
        val_arrays["probs"], val_arrays["targets"], val_arrays["candidate_mask"],
        val_path, prior, val_arrays["observed_mask"], args.topk, candidate_tiers,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=val_proj.get("node_embedding_proj"),
        graph_embedding_proj=val_proj.get("graph_embedding_proj"),
        candidate_tier_features=candidate_tier_features,
    )
    test_rows = _build_rows(
        test_arrays["probs"], test_arrays["targets"], test_arrays["candidate_mask"],
        test_path, prior, test_arrays["observed_mask"], args.topk, candidate_tiers,
        candidate_onehot=args.candidate_onehot,
        node_embedding_proj=test_proj.get("node_embedding_proj"),
        graph_embedding_proj=test_proj.get("graph_embedding_proj"),
        candidate_tier_features=candidate_tier_features,
    )

    global_model, global_scaler = _fit_model(train_rows, args)
    if global_model is None or global_scaler is None:
        raise RuntimeError("Failed to fit global reranker")

    tier_models: Dict[str, object] = {}
    tier_scalers: Dict[str, StandardScaler] = {}
    for tier in ("direct", "near", "far"):
        tier_rows = [r for r in train_rows if r["target_tier"] == tier and r["candidate_tier"] == tier]
        model_t, scaler_t = _fit_model(tier_rows, args)
        if model_t is not None and scaler_t is not None:
            tier_models[tier] = model_t
            tier_scalers[tier] = scaler_t

    rows: List[Dict[str, object]] = []
    best_val = None
    best_setting = None
    for route_mode in ("top1_tier", "mass_tier"):
        for tier_bonus in (0.0, 0.25, 0.5, 1.0):
            val_metric = _evaluate_rows(
                val_rows,
                val_arrays["probs"],
                val_arrays["targets"],
                val_arrays["candidate_mask"],
                global_model,
                global_scaler,
                tier_models,
                tier_scalers,
                route_mode,
                tier_bonus,
            )
            row = dict(val_metric)
            row.update(
                {
                    "route_mode": route_mode,
                    "tier_bonus": float(tier_bonus),
                    "topk": int(args.topk),
                    "tau": float(args.tau),
                    "c_value": float(args.c_value),
                    "candidate_onehot": bool(args.candidate_onehot),
                    "embedding_dim": int(args.embedding_dim),
                    "reranker_kind": args.reranker_kind,
                    "mlp_hidden": args.mlp_hidden if args.reranker_kind == "mlp" else "",
                    "mlp_alpha": float(args.mlp_alpha) if args.reranker_kind == "mlp" else 0.0,
                }
            )
            rows.append(row)
            if best_val is None or (row["top1"], row["mrr"], row["top3"]) > (
                best_val["top1"],
                best_val["mrr"],
                best_val["top3"],
            ):
                best_val = row
                best_setting = (route_mode, tier_bonus)

    if best_val is None or best_setting is None:
        raise RuntimeError("No valid route setting found")

    route_mode, tier_bonus = best_setting
    test_metric = _evaluate_rows(
        test_rows,
        test_arrays["probs"],
        test_arrays["targets"],
        test_arrays["candidate_mask"],
        global_model,
        global_scaler,
        tier_models,
        tier_scalers,
        route_mode,
        tier_bonus,
    )

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_csv, rows)
    payload = {
        "checkpoint": args.checkpoint,
        "subdir": args.subdir,
        "model_type": args.model_type,
        "split_mode": args.split_mode,
        "seed": int(args.seed),
        "features": cfg.selected_features,
        "candidate_tier_csv": args.candidate_tier_csv,
        "best_on_val": best_val,
        "test_with_best_val_setting": test_metric,
        "fitted_tiers": sorted(tier_models.keys()),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
