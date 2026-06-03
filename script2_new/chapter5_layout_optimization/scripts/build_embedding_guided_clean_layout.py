#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Chapter-5 embedding-guided clean layout.

This script is intentionally isolated from existing layout builders. It loads a
fixed diagnosis checkpoint, extracts node embeddings on the scenario-split train
windows only, and selects monitor nodes by max-min diversity in embedding space.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.train_privileged_teacher_student import (  # noqa: E402
    _build_config,
    _configure_process_thresholds_if_needed,
    _create_model,
    _make_loaders,
)


ROOT = Path(r"E:\11.16")
CH5_DIR = ROOT / "script2_new" / "chapter5_layout_optimization"
INPUT_DIR = ROOT / "script2_new" / "input_1"
CHECKPOINT = ROOT / "script2_new" / "outputs" / "model_checkpoints" / "best_model_ch4_timewin_seq24_s42.pth"
FULL_MONITORS = INPUT_DIR / "monitor_nodes_full_N128.json"
GRAPH_FEATURES = INPUT_DIR / "graph_path_features.npz"
NODE_LIST = INPUT_DIR / "node_list.json"
CANDIDATE_NODES = INPUT_DIR / "candidate_nodes_new.json"

DEFAULT_OUT_DIR = CH5_DIR / "outputs" / "embedding_guided_clean" / "seed42"
DEFAULT_LAYOUT_DIR = CH5_DIR / "outputs" / "layouts" / "embedding_guided_clean"

INF_HOP = 999


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_list(path: Path, keys: Iterable[str]) -> list[str]:
    data = read_json(path)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in keys:
            values = data.get(key)
            if isinstance(values, list):
                return [str(x) for x in values]
    raise ValueError(f"Unsupported JSON structure: {path}")


def infer_model_type(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("layer_query_projs.") for k in keys):
        return "hydraulic_inverse_deepattn"
    if any(k.startswith("query_proj.") for k in keys):
        return "hydraulic_inverse"
    raise ValueError("Cannot infer hydraulic model type from checkpoint keys")


def sym_hop(shortest: np.ndarray, i: int, j: int) -> int:
    a = int(shortest[i, j])
    b = int(shortest[j, i])
    vals = [x for x in (a, b) if x < INF_HOP]
    return min(vals) if vals else INF_HOP


def summarize_candidate_hops(
    selected_nodes: list[str],
    node_list: list[str],
    candidate_nodes: list[str],
    shortest: np.ndarray,
) -> dict[str, float]:
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    selected_idx = [node_to_idx[node] for node in selected_nodes]
    best_hops = []
    for candidate in candidate_nodes:
        ci = node_to_idx[candidate]
        best_hops.append(min(sym_hop(shortest, ci, si) for si in selected_idx))
    best_hops_arr = np.asarray(best_hops, dtype=np.int16)
    finite = best_hops_arr[best_hops_arr < INF_HOP]
    return {
        "direct": int((best_hops_arr == 0).sum()),
        "near": int(((best_hops_arr >= 1) & (best_hops_arr <= 2)).sum()),
        "far": int((best_hops_arr > 2).sum()),
        "mean_hop": float(finite.mean()) if finite.size else float("nan"),
        "max_hop": float(finite.max()) if finite.size else float("nan"),
        "overlap_count": int(sum(node in set(candidate_nodes) for node in selected_nodes)),
    }


def monitor_pair_metrics(selected_nodes: list[str], node_list: list[str], shortest: np.ndarray) -> dict[str, float]:
    if len(selected_nodes) <= 1:
        return {"monitor_dispersion_mean_hop": 0.0, "monitor_redundancy_mean_jaccard": 0.0}
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx = [node_to_idx[node] for node in selected_nodes]
    pair_hops = []
    redundancy_proxy = []
    for a_pos in range(len(idx)):
        for b_pos in range(a_pos + 1, len(idx)):
            hop = sym_hop(shortest, idx[a_pos], idx[b_pos])
            if hop < INF_HOP:
                pair_hops.append(float(hop))
                redundancy_proxy.append(1.0 / (1.0 + float(hop)))
    return {
        "monitor_dispersion_mean_hop": float(np.mean(pair_hops)) if pair_hops else float("nan"),
        "monitor_redundancy_mean_jaccard": float(np.mean(redundancy_proxy)) if redundancy_proxy else 0.0,
    }


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def max_min_select(embeddings: np.ndarray, node_list: list[str], budget: int, candidate_set: set[str]) -> tuple[list[str], pd.DataFrame]:
    if embeddings.shape[0] != len(node_list):
        raise ValueError(f"embedding rows {embeddings.shape[0]} != node_list length {len(node_list)}")
    if budget > len(node_list):
        raise ValueError(f"budget {budget} > node count {len(node_list)}")

    center = embeddings.mean(axis=0, keepdims=True)
    center_dist = np.linalg.norm(embeddings - center, axis=1)
    first_idx = int(np.argmax(center_dist))
    selected_idx = [first_idx]
    remaining = set(range(len(node_list)))
    remaining.remove(first_idx)
    min_dist = np.linalg.norm(embeddings - embeddings[first_idx:first_idx + 1], axis=1)

    rows = [
        {
            "step": 1,
            "node_id": node_list[first_idx],
            "node_index": first_idx,
            "is_candidate": node_list[first_idx] in candidate_set,
            "selection_score": float(center_dist[first_idx]),
            "score_type": "distance_to_embedding_mean",
            "min_distance_to_previous_selected": math.nan,
        }
    ]

    while len(selected_idx) < budget:
        best_idx = max(remaining, key=lambda idx: (float(min_dist[idx]), -idx))
        selected_idx.append(best_idx)
        remaining.remove(best_idx)
        rows.append(
            {
                "step": len(selected_idx),
                "node_id": node_list[best_idx],
                "node_index": best_idx,
                "is_candidate": node_list[best_idx] in candidate_set,
                "selection_score": float(min_dist[best_idx]),
                "score_type": "max_min_embedding_distance",
                "min_distance_to_previous_selected": float(min_dist[best_idx]),
            }
        )
        new_dist = np.linalg.norm(embeddings - embeddings[best_idx:best_idx + 1], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return [node_list[idx] for idx in selected_idx], pd.DataFrame(rows)


def get_adj_from_batch(batch, device: torch.device):
    adj = batch["adj_matrix"]
    if isinstance(adj, (list, tuple)):
        return adj[0].to(device)
    if adj.dim() == 3:
        return adj[0].to(device)
    return adj.to(device)


def extract_embeddings(args: argparse.Namespace, out_dir: Path) -> tuple[np.ndarray, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model_type = infer_model_type(state_dict)

    cfg = _build_config(
        args.teacher_subdir,
        args.seed,
        monitor_nodes_file=str(args.full_monitors),
        split_mode="scenario",
        graph_path_features_file=str(args.graph_features),
        feature_set="raw_plus_residual",
    )
    cfg.sequence_length = int(checkpoint.get("sequence_length", 24))
    cfg.window_stride = int(checkpoint.get("window_stride", 6))
    cfg.model_type = model_type

    train_loader, _val_loader, _test_loader, dataset = _make_loaders(cfg, observed_nodes_file=str(args.full_monitors))
    _configure_process_thresholds_if_needed(cfg, train_loader)

    model = _create_model(cfg, dataset, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    node_sum = None
    total_windows = 0
    with torch.no_grad():
        for batch in train_loader:
            x = batch["features"].to(device)
            adj = get_adj_from_batch(batch, device)
            emb = model.get_node_embeddings(x, adj)
            emb_np = emb.detach().cpu().numpy().astype(np.float64)
            batch_sum = emb_np.sum(axis=0)
            node_sum = batch_sum if node_sum is None else node_sum + batch_sum
            total_windows += int(emb_np.shape[0])

    if node_sum is None or total_windows <= 0:
        raise RuntimeError("No embeddings extracted from train loader")

    pooled = (node_sum / float(total_windows)).astype(np.float32)
    pooled = l2_normalize(pooled).astype(np.float32)
    np.save(out_dir / "node_embeddings_mean_l2.npy", pooled)

    meta = {
        "checkpoint": str(args.checkpoint),
        "encoder_model_type_resolved": model_type,
        "device": str(device),
        "teacher_subdir": args.teacher_subdir,
        "seed": int(args.seed),
        "sequence_length": int(cfg.sequence_length),
        "window_stride": int(cfg.window_stride),
        "train_windows_used": int(total_windows),
        "embedding_shape": list(pooled.shape),
        "full_monitors": str(args.full_monitors),
    }
    return pooled, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding-guided clean N25 layout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--teacher-subdir", default="time_gated_full_ie_v4_formal_conservative420_seed42")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--full-monitors", type=Path, default=FULL_MONITORS)
    parser.add_argument("--graph-features", type=Path, default=GRAPH_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    required = [args.checkpoint, args.full_monitors, args.graph_features, NODE_LIST, CANDIDATE_NODES]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layout_dir.mkdir(parents=True, exist_ok=True)

    node_list = read_json_list(NODE_LIST, ("node_list", "nodes"))
    candidate_nodes = read_json_list(CANDIDATE_NODES, ("candidate_nodes", "nodes", "candidates"))
    candidate_set = set(candidate_nodes)

    embeddings, meta = extract_embeddings(args, args.out_dir)
    if embeddings.shape[0] != len(node_list):
        raise AssertionError(f"Expected {len(node_list)} node embeddings, got {embeddings.shape[0]}")

    selected_nodes, trace = max_min_select(embeddings, node_list, int(args.budget), candidate_set)
    if len(selected_nodes) != int(args.budget) or len(set(selected_nodes)) != int(args.budget):
        raise AssertionError("Selected monitor nodes are not unique or budget-sized")

    graph_npz = np.load(args.graph_features)
    shortest = graph_npz["shortest_dist"]
    layout_metrics = summarize_candidate_hops(selected_nodes, node_list, candidate_nodes, shortest)
    layout_metrics.update(monitor_pair_metrics(selected_nodes, node_list, shortest))

    layout = {
        "monitor_nodes": selected_nodes,
        "strategy": "embedding_guided_clean",
        "n": int(args.budget),
        "frozen_protocol": {
            "source_data": args.teacher_subdir,
            "seed": int(args.seed),
            "selection_uses_train_split_only": True,
            "selection_rule": "max_min_diversity_on_l2_normalized_node_embeddings",
            "evaluation_protocol": "chapter5_sparse_observation_scenario_seed42",
        },
        "objective_meta": meta,
        "layout_metrics": layout_metrics,
    }

    layout_path = args.layout_dir / f"monitor_nodes_embedding_guided_clean_N{int(args.budget)}.json"
    layout_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")

    trace.to_csv(args.out_dir / "selection_trace.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"method_short": "embedding-guided clean", "method": "embedding_guided_clean", **layout_metrics}]).to_csv(
        args.out_dir / "layout_structure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.out_dir / "metadata.json").write_text(json.dumps({**meta, "layout_file": str(layout_path)}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote layout: {layout_path}")
    print(json.dumps(layout_metrics, ensure_ascii=False, indent=2))
    print(f"Resolved encoder model type: {meta['encoder_model_type_resolved']}")


if __name__ == "__main__":
    main()
