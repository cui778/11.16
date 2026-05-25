#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a lightweight learnable layout network for Chapter 5.

Design goal:
- do not handcraft a final ranking directly
- learn node selection preference from historical layout-quality records
- then decode a budget-constrained monitor layout with spacing / overlap control

This is the first learnable structural-innovation version:
- learnable_layout_network_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn

from build_layout_suite import INF_HOP, load_json_list
from build_two_stage_balanced_layout import (
    build_directed_graph,
    layout_summary,
    save_json,
    sym_hop,
)


ROOT = Path(r"E:\11.16\script2_new")
CH5_ROOT = ROOT / "chapter5_layout_optimization"
INPUT_DIR = ROOT / "input_1"
OUTPUT_DIR = CH5_ROOT / "outputs"
STRUCT_DIR = OUTPUT_DIR / "structural_innovation"
SURROGATE_DIR = STRUCT_DIR / "surrogate"


def safe_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def load_static_graph_assets() -> Dict[str, object]:
    candidate_path = INPUT_DIR / "candidate_nodes_new.json"
    node_list_path = INPUT_DIR / "node_list.json"
    adj_path = INPUT_DIR / "adj_matrix.npy"
    graph_features_path = INPUT_DIR / "graph_path_features.npz"
    response_path = OUTPUT_DIR / "cache" / "response_signature_ie_energy.npz"
    sensitivity_path = OUTPUT_DIR / "cache" / "evaluator_node_sensitivity_ch5_tsbal25_s42.npz"

    node_list = load_json_list(node_list_path, ("node_list", "nodes"))
    candidate_nodes = load_json_list(candidate_path, ("candidate_nodes", "nodes"))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    graph = build_directed_graph(node_list, adj_path)
    shortest = np.load(str(graph_features_path))["shortest_dist"]

    response_npz = np.load(str(response_path), allow_pickle=True)
    response_tensor = response_npz["tensor"].astype(np.float32)

    sensitivity = np.zeros(len(node_list), dtype=np.float32)
    if sensitivity_path.exists():
        sens_npz = np.load(str(sensitivity_path), allow_pickle=True)
        if "sensitivity" in sens_npz.files:
            sensitivity = sens_npz["sensitivity"].astype(np.float32)

    return {
        "node_list": node_list,
        "candidate_nodes": candidate_nodes,
        "node_to_idx": node_to_idx,
        "graph": graph,
        "shortest": shortest,
        "response_tensor": response_tensor,
        "sensitivity": sensitivity,
    }


def compute_node_features(assets: Dict[str, object]) -> Tuple[pd.DataFrame, np.ndarray]:
    node_list: List[str] = assets["node_list"]  # type: ignore[assignment]
    candidate_nodes: List[str] = assets["candidate_nodes"]  # type: ignore[assignment]
    node_to_idx: Dict[str, int] = assets["node_to_idx"]  # type: ignore[assignment]
    graph: nx.DiGraph = assets["graph"]  # type: ignore[assignment]
    shortest: np.ndarray = assets["shortest"]  # type: ignore[assignment]
    response_tensor: np.ndarray = assets["response_tensor"]  # type: ignore[assignment]
    sensitivity: np.ndarray = assets["sensitivity"]  # type: ignore[assignment]

    graph_u = graph.to_undirected()
    betweenness = nx.betweenness_centrality(graph_u, normalized=True)
    closeness = nx.closeness_centrality(graph_u)
    g_rev = graph.reverse(copy=True)
    candidate_indices = [node_to_idx[node] for node in candidate_nodes]
    candidate_set = set(candidate_nodes)

    rows = []
    feature_matrix = []
    for node in node_list:
        idx = node_to_idx[node]
        in_degree = float(graph.in_degree(node))
        out_degree = float(graph.out_degree(node))
        total_degree = in_degree + out_degree
        downstream_reach = float(len(nx.descendants(g_rev, node)))

        hops = np.asarray([sym_hop(shortest, idx, cand_idx) for cand_idx in candidate_indices], dtype=np.float32)
        finite_hops = hops[hops < INF_HOP]
        min_hop = float(finite_hops.min()) if finite_hops.size else float(INF_HOP)
        mean_hop = float(finite_hops.mean()) if finite_hops.size else float(INF_HOP)
        within2 = float(np.sum(finite_hops <= 2)) if finite_hops.size else 0.0

        response_vec = response_tensor[idx].reshape(-1)
        response_abs = np.abs(response_vec)
        response_mean = float(response_abs.mean())
        response_std = float(response_abs.std())
        response_max = float(response_abs.max())

        row = {
            "node": node,
            "candidate_flag": 1.0 if node in candidate_set else 0.0,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree,
            "betweenness": float(betweenness.get(node, 0.0)),
            "closeness": float(closeness.get(node, 0.0)),
            "downstream_reach": downstream_reach,
            "min_hop_to_candidate": min_hop,
            "mean_hop_to_candidate": mean_hop,
            "within2_count": within2,
            "response_mean": response_mean,
            "response_std": response_std,
            "response_max": response_max,
            "sensitivity": float(sensitivity[idx]),
        }
        rows.append(row)
        feature_matrix.append(list(row.values())[1:])

    features_df = pd.DataFrame(rows)
    feature_cols = [column for column in features_df.columns if column != "node"]
    x = features_df[feature_cols].to_numpy(dtype=np.float32)
    x_norm = np.column_stack([safe_minmax(x[:, col]) for col in range(x.shape[1])]).astype(np.float32)
    return features_df, x_norm


def quality_score(row: pd.Series, preset: str) -> float:
    scenario_mrr = float(row.get("scenario_mrr", 0.0) or 0.0)
    scenario_top1 = float(row.get("scenario_top1", 0.0) or 0.0)
    nh_mrr = float(row.get("node_holdout_mrr", 0.0) or 0.0)
    nh_top1 = float(row.get("node_holdout_top1", 0.0) or 0.0)

    if preset == "scenario":
        return 0.70 * scenario_mrr + 0.30 * scenario_top1
    if preset == "generalization":
        return 0.25 * scenario_mrr + 0.15 * scenario_top1 + 0.35 * nh_mrr + 0.25 * nh_top1
    return 0.45 * scenario_mrr + 0.20 * scenario_top1 + 0.20 * nh_mrr + 0.15 * nh_top1


def build_node_targets(features_df: pd.DataFrame, preset: str) -> Tuple[np.ndarray, pd.DataFrame]:
    quality_df = pd.read_csv(STRUCT_DIR / "layout_quality_dataset_wide.csv")
    quality_df = quality_df[quality_df["budget"] == 25].copy()

    weights = np.asarray([quality_score(row, preset) for _, row in quality_df.iterrows()], dtype=np.float32)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float32)

    target = np.zeros(len(features_df), dtype=np.float32)
    contribution_rows = []
    node_to_idx = {node: i for i, node in enumerate(features_df["node"].tolist())}

    for (_, row), weight in zip(quality_df.iterrows(), weights):
        layout_file = Path(str(row["layout_file"]))
        if not layout_file.exists():
            continue
        payload = json.loads(layout_file.read_text(encoding="utf-8"))
        selected_nodes = [str(x) for x in payload.get("monitor_nodes", [])]
        for node in selected_nodes:
            if node in node_to_idx:
                target[node_to_idx[node]] += float(weight)
        contribution_rows.append(
            {
                "method": str(row["method"]),
                "preset": preset,
                "quality_score": float(weight),
                "selected_count": int(len(selected_nodes)),
            }
        )

    target = target / float(weights.sum())
    return target.astype(np.float32), pd.DataFrame(contribution_rows)


class NodeScoringMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_node_model(x: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[NodeScoringMLP, Dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    model = NodeScoringMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.MSELoss()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    best_state = None
    best_loss = None
    for epoch in range(800):
        model.train()
        pred = model(x_tensor)
        loss = criterion(pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().astype(np.float32)

    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return model, {"train_mse": float(best_loss), "train_mae": mae, "train_rmse": rmse}, pred


def overlap_anchor_degree() -> int:
    summary = pd.read_csv(OUTPUT_DIR / "layout_summary.csv")
    row = summary[(summary["strategy"] == "degree") & (summary["n"] == 25)]
    if row.empty:
        return 12
    return int(row.iloc[0]["overlap_count"])


def select_layout(
    preset: str,
    budget: int,
    scores: np.ndarray,
    features_df: pd.DataFrame,
    assets: Dict[str, object],
) -> Tuple[List[str], Dict[str, float]]:
    node_list: List[str] = assets["node_list"]  # type: ignore[assignment]
    candidate_nodes: List[str] = assets["candidate_nodes"]  # type: ignore[assignment]
    node_to_idx: Dict[str, int] = assets["node_to_idx"]  # type: ignore[assignment]
    shortest: np.ndarray = assets["shortest"]  # type: ignore[assignment]

    candidate_set = set(candidate_nodes)
    anchor = overlap_anchor_degree()
    if preset == "scenario":
        overlap_cap = anchor + 3
        backbone_bonus = 0.02
    elif preset == "generalization":
        overlap_cap = anchor + 1
        backbone_bonus = 0.08
    else:
        overlap_cap = anchor + 2
        backbone_bonus = 0.05

    selected: List[str] = []
    selected_idx: List[int] = []
    feature_lookup = features_df.set_index("node")

    for _ in range(budget):
        best_node = None
        best_gain = None
        overlap_now = sum(1 for node in selected if node in candidate_set)
        for node, base_score in zip(node_list, scores):
            if node in selected:
                continue
            if node in candidate_set and overlap_now >= overlap_cap:
                continue

            row = feature_lookup.loc[node]
            gain = float(base_score)
            gain += 0.03 * float(row["response_max"])
            gain += 0.04 * float(row["sensitivity"])
            if node not in candidate_set:
                gain += backbone_bonus * float(row["betweenness"])

            if selected_idx:
                idx = node_to_idx[node]
                hops = [
                    sym_hop(shortest, idx, chosen_idx)
                    for chosen_idx in selected_idx
                ]
                finite_hops = [hop for hop in hops if hop < INF_HOP]
                min_hop = min(finite_hops) if finite_hops else INF_HOP
                if min_hop <= 1:
                    gain -= 0.20
                elif min_hop == 2:
                    gain -= 0.08
                elif min_hop >= 5:
                    gain += 0.03

            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None:
            break
        selected.append(best_node)
        selected_idx.append(node_to_idx[best_node])

    pair_score = np.zeros((len(candidate_nodes), len(candidate_nodes)), dtype=np.float32)
    metrics = layout_summary(selected, candidate_nodes, node_to_idx, shortest, pair_score)
    metrics["overlap_anchor_target"] = int(anchor)
    metrics["overlap_cap"] = int(overlap_cap)
    metrics["score_mean_selected"] = float(np.mean([scores[node_to_idx[node]] for node in selected])) if selected else 0.0
    metrics["score_mean_all"] = float(np.mean(scores))
    return selected, metrics


def preset_model_dir(preset: str) -> Path:
    return STRUCT_DIR / "learnable_layout_network_v0" / preset


def save_training_outputs(
    preset: str,
    model: nn.Module,
    features_df: pd.DataFrame,
    target: np.ndarray,
    pred: np.ndarray,
    fit_metrics: Dict[str, float],
    contribution_df: pd.DataFrame,
) -> None:
    out_dir = preset_model_dir(preset)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "node_scorer_v0.pt"
    torch.save(model.state_dict(), state_path)

    node_score_df = features_df.copy()
    node_score_df["target_score"] = target
    node_score_df["pred_score"] = pred
    node_score_df = node_score_df.sort_values("pred_score", ascending=False).reset_index(drop=True)
    node_score_df.to_csv(out_dir / "node_scores.csv", index=False, encoding="utf-8-sig")

    contribution_df.to_csv(out_dir / "layout_contributions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "training_report.json").write_text(
        json.dumps({"preset": preset, **fit_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learnable layout network v0.")
    parser.add_argument("--budgets", default="25")
    parser.add_argument("--presets", default="balanced,scenario,generalization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assets = load_static_graph_assets()
    features_df, x = compute_node_features(assets)
    node_list: List[str] = assets["node_list"]  # type: ignore[assignment]
    node_to_idx: Dict[str, int] = assets["node_to_idx"]  # type: ignore[assignment]

    summary_rows = []
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    presets = [x.strip() for x in args.presets.split(",") if x.strip()]

    for preset in presets:
        target, contribution_df = build_node_targets(features_df, preset=preset)
        model, fit_metrics, pred = train_node_model(x, target, seed=args.seed)
        save_training_outputs(preset, model, features_df, target, pred, fit_metrics, contribution_df)

        for budget in budgets:
            selected_nodes, metrics = select_layout(
                preset=preset,
                budget=budget,
                scores=pred,
                features_df=features_df,
                assets=assets,
            )

            strategy_name = f"learnable_layout_network_v0_{preset}"
            layout_dir = OUTPUT_DIR / "layouts" / strategy_name
            layout_path = layout_dir / f"monitor_nodes_{strategy_name}_N{budget}.json"
            payload = {
                "monitor_nodes": selected_nodes,
                "strategy": strategy_name,
                "n": int(budget),
                "protocol": {
                    "note": "Chapter-5 structural innovation: learned node scorer plus constrained decoder",
                    "candidate_nodes_file": str(INPUT_DIR / "candidate_nodes_new.json"),
                    "formal_data_dir": str(ROOT / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42"),
                },
                "learning_meta": {
                    "preset": preset,
                    "seed": int(args.seed),
                    "fit_metrics": fit_metrics,
                    "node_score_file": str(preset_model_dir(preset) / "node_scores.csv"),
                },
                "layout_metrics": metrics,
            }
            save_json(layout_path, payload)

            row = {
                "strategy": strategy_name,
                "preset": preset,
                "budget": int(budget),
                "layout_file": str(layout_path),
                "train_mse": float(fit_metrics["train_mse"]),
                "train_mae": float(fit_metrics["train_mae"]),
                "train_rmse": float(fit_metrics["train_rmse"]),
            }
            row.update(metrics)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = STRUCT_DIR / "learnable_layout_network_v0_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] summary -> {summary_path}")
    for row in summary_rows:
        print(
            f"[{row['strategy']} N{row['budget']}] "
            f"direct={row['direct']} near={row['near']} far={row['far']} "
            f"mean_hop={row['mean_hop']:.2f} overlap={row['overlap_count']}"
        )


if __name__ == "__main__":
    main()
