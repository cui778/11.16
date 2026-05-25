#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 overlap-controlled probe layouts.

Goal:
- keep budget fixed
- keep candidate-monitor overlap fixed
- compare whether layout structure still matters

This script is intended for the first Chapter-5 mainline probe before
launching a full budget sweep.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


INF_HOP = 999


def load_json_list(path: Path, keys: Tuple[str, ...]) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in keys:
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]
    raise ValueError(f"Unsupported JSON structure: {path}")


def build_directed_graph(node_list: List[str], adj_path: Path) -> nx.DiGraph:
    adj = np.load(str(adj_path))
    graph = nx.DiGraph()
    graph.add_nodes_from(node_list)
    for i, src in enumerate(node_list):
        for j, dst in enumerate(node_list):
            if float(adj[i, j]) > 0:
                graph.add_edge(src, dst, weight=float(adj[i, j]))
    return graph


def shortest_array(graph: nx.DiGraph, node_list: Sequence[str]) -> np.ndarray:
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    shortest = np.full((len(node_list), len(node_list)), INF_HOP, dtype=np.int16)
    np.fill_diagonal(shortest, 0)
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for src, mapping in lengths.items():
        i = node_to_idx[src]
        for dst, dist in mapping.items():
            shortest[i, node_to_idx[dst]] = min(int(dist), INF_HOP)
    return shortest


def sym_hop(shortest: np.ndarray, i: int, j: int) -> int:
    a = int(shortest[i, j])
    b = int(shortest[j, i])
    vals = [x for x in (a, b) if x < INF_HOP]
    return min(vals) if vals else INF_HOP


def hop_utility_candidate(hop: int) -> float:
    if hop == 0:
        return 8.0
    if hop == 1:
        return 5.0
    if hop == 2:
        return 3.0
    if hop == 3:
        return 1.0
    return 0.0


def hop_utility_global(hop: int) -> float:
    if hop == 0:
        return 4.0
    if hop == 1:
        return 3.0
    if hop == 2:
        return 2.0
    if hop == 3:
        return 1.0
    return 0.0


def summarize_candidate_hops(best_hops: np.ndarray) -> Dict[str, float]:
    finite = best_hops[best_hops < INF_HOP]
    return {
        "direct": int((best_hops == 0).sum()),
        "near": int(((best_hops >= 1) & (best_hops <= 2)).sum()),
        "far": int((best_hops > 2).sum()),
        "mean_hop": float(finite.mean()) if finite.size else float("nan"),
        "max_hop": float(finite.max()) if finite.size else float("nan"),
    }


def monitor_pair_metrics(shortest: np.ndarray, selected_indices: List[int]) -> Dict[str, float]:
    if len(selected_indices) <= 1:
        return {
            "monitor_dispersion_mean_hop": 0.0,
            "monitor_redundancy_mean_jaccard": 0.0,
        }
    pair_hops = []
    redundancy_proxy = []
    for a, b in combinations(selected_indices, 2):
        hop = sym_hop(shortest, a, b)
        if hop < INF_HOP:
            pair_hops.append(float(hop))
            redundancy_proxy.append(1.0 / (1.0 + float(hop)))
    return {
        "monitor_dispersion_mean_hop": float(np.mean(pair_hops)) if pair_hops else float("nan"),
        "monitor_redundancy_mean_jaccard": float(np.mean(redundancy_proxy)) if redundancy_proxy else 0.0,
    }


def degree_score(graph: nx.DiGraph, nodes: Sequence[str]) -> Dict[str, float]:
    graph_u = graph.to_undirected()
    bc = nx.betweenness_centrality(graph_u, normalized=True)
    degree = {node: graph.in_degree(node) + graph.out_degree(node) for node in graph.nodes()}
    out = {}
    for node in nodes:
        out[str(node)] = 0.6 * float(degree.get(node, 0.0)) + 0.4 * float(bc.get(node, 0.0))
    return out


def build_hop_views(
    shortest: np.ndarray,
    node_list: Sequence[str],
    candidate_nodes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    candidate_idx = [node_to_idx[node] for node in candidate_nodes]
    hop_matrix = np.full((len(candidate_nodes), len(node_list)), INF_HOP, dtype=np.int16)
    for ci, src_idx in enumerate(candidate_idx):
        for j in range(len(node_list)):
            hop_matrix[ci, j] = sym_hop(shortest, src_idx, j)
    return hop_matrix, np.asarray(candidate_idx, dtype=np.int64), node_to_idx


def greedy_select_exact_overlap(
    pool_nodes: List[str],
    pool_indices: List[int],
    candidate_set: set[str],
    hop_matrix: np.ndarray,
    shortest: np.ndarray,
    budget: int,
    target_overlap: int,
    mode: str,
    graph: nx.DiGraph,
) -> Tuple[List[str], Dict[str, float]]:
    if target_overlap > budget:
        raise ValueError(f"target_overlap {target_overlap} > budget {budget}")

    selected: List[str] = []
    selected_indices: List[int] = []
    selected_set = set()
    overlap_count = 0
    best_candidate_hops = np.full(hop_matrix.shape[0], INF_HOP, dtype=np.int16)
    best_global_hops = np.full(len(pool_nodes), INF_HOP, dtype=np.int16)
    centrality = degree_score(graph, pool_nodes)

    while len(selected) < budget:
        remaining_slots = budget - len(selected)
        remaining_overlap = target_overlap - overlap_count
        best_score = None
        best_node = None
        best_pool_pos = None
        best_new_candidate_hops = None
        best_new_global_hops = None

        for local_pos, node in enumerate(pool_nodes):
            if node in selected_set:
                continue
            is_candidate = node in candidate_set
            if is_candidate and remaining_overlap <= 0:
                continue
            if (not is_candidate) and remaining_slots <= remaining_overlap:
                continue

            new_candidate_hops = np.minimum(best_candidate_hops, hop_matrix[:, pool_indices[local_pos]])
            new_global_hops = np.minimum(
                best_global_hops,
                np.asarray([sym_hop(shortest, pool_indices[local_pos], idx) for idx in pool_indices], dtype=np.int16),
            )

            cand_util_gain = float(
                np.sum([hop_utility_candidate(int(h)) for h in new_candidate_hops])
                - np.sum([hop_utility_candidate(int(h)) for h in best_candidate_hops])
            )
            global_util_gain = float(
                np.sum([hop_utility_global(int(h)) for h in new_global_hops])
                - np.sum([hop_utility_global(int(h)) for h in best_global_hops])
            )
            direct_gain = int((new_candidate_hops == 0).sum()) - int((best_candidate_hops == 0).sum())
            nonfar_gain = int((new_candidate_hops <= 2).sum()) - int((best_candidate_hops <= 2).sum())
            dispersion_gain = 0.0
            if selected_indices:
                hops = [sym_hop(shortest, pool_indices[local_pos], idx) for idx in selected_indices]
                finite = [float(h) for h in hops if h < INF_HOP]
                if finite:
                    dispersion_gain = float(np.mean(finite))

            centrality_bonus = float(centrality.get(node, 0.0))
            candidate_bonus = 1.0 if is_candidate else 0.0

            if mode == "candidate_focus":
                score = (
                    1400.0 * nonfar_gain
                    + 100.0 * direct_gain
                    + 12.0 * cand_util_gain
                    + 0.25 * global_util_gain
                    + 0.03 * centrality_bonus
                    + 0.2 * candidate_bonus
                )
            elif mode == "balanced":
                score = (
                    850.0 * nonfar_gain
                    + 45.0 * direct_gain
                    + 8.0 * cand_util_gain
                    + 3.5 * global_util_gain
                    + 1.25 * dispersion_gain
                    + 0.05 * centrality_bonus
                    + (0.15 if not is_candidate else 0.0)
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if best_score is None or score > best_score:
                best_score = score
                best_node = node
                best_pool_pos = local_pos
                best_new_candidate_hops = new_candidate_hops
                best_new_global_hops = new_global_hops

        if best_node is None or best_pool_pos is None:
            raise RuntimeError(
                f"Could not satisfy exact overlap constraint for mode={mode}, budget={budget}, target_overlap={target_overlap}"
            )

        selected.append(best_node)
        selected_indices.append(pool_indices[best_pool_pos])
        selected_set.add(best_node)
        overlap_count += int(best_node in candidate_set)
        best_candidate_hops = best_new_candidate_hops
        best_global_hops = best_new_global_hops

    metrics = summarize_candidate_hops(best_candidate_hops)
    metrics["overlap_count"] = int(overlap_count)
    metrics.update(monitor_pair_metrics(shortest, selected_indices))
    metrics["global_mean_hop"] = float(best_global_hops[best_global_hops < INF_HOP].mean())
    return selected, metrics


def infer_target_overlap(layout_summary: pd.DataFrame, budget: int, anchor_strategy: str = "degree") -> int:
    sub = layout_summary[(layout_summary["strategy"] == anchor_strategy) & (layout_summary["n"] == budget)]
    if sub.empty:
        raise ValueError(f"Missing overlap anchor for strategy={anchor_strategy}, budget={budget}")
    return int(sub.iloc[0]["overlap_count"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build overlap-controlled Chapter-5 probe layouts.")
    repo_root = Path(__file__).resolve().parents[3]
    ch5_root = repo_root / "script2_new" / "chapter5_layout_optimization"
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25])
    parser.add_argument(
        "--protocol-note",
        type=str,
        default="control-variable Chapter-5 probe; fixed budget and fixed candidate-monitor overlap",
    )
    parser.add_argument(
        "--anchor-summary",
        type=str,
        default=str(ch5_root / "outputs" / "layout_summary.csv"),
    )
    args = parser.parse_args()

    input_dir = repo_root / "script2_new" / "input_1"
    node_list = load_json_list(input_dir / "node_list.json", ("node_list",))
    candidate_nodes = load_json_list(input_dir / "candidate_nodes_new.json", ("candidate_nodes",))
    candidate_set = set(candidate_nodes)
    graph = build_directed_graph(node_list, input_dir / "adj_matrix.npy")
    shortest = shortest_array(graph, node_list)
    hop_matrix, _, node_to_idx = build_hop_views(shortest, node_list, candidate_nodes)

    exclude = {node for node in graph.nodes() if graph.in_degree(node) + graph.out_degree(node) == 0}
    pool_nodes = [node for node in node_list if node not in exclude]
    pool_indices = [node_to_idx[node] for node in pool_nodes]

    layout_summary = pd.read_csv(args.anchor_summary)
    out_base = ch5_root / "outputs" / "layouts"
    summary_rows: List[Dict[str, object]] = []

    method_specs = [
        ("overlap_controlled_candidate_focus", "candidate_focus"),
        ("overlap_controlled_balanced", "balanced"),
    ]

    for budget in args.budgets:
        target_overlap = infer_target_overlap(layout_summary, budget, anchor_strategy="degree")
        for strategy_name, mode in method_specs:
            selected, metrics = greedy_select_exact_overlap(
                pool_nodes=pool_nodes,
                pool_indices=pool_indices,
                candidate_set=candidate_set,
                hop_matrix=hop_matrix,
                shortest=shortest,
                budget=budget,
                target_overlap=target_overlap,
                mode=mode,
                graph=graph,
            )
            out_dir = out_base / strategy_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"monitor_nodes_{strategy_name}_N{budget}.json"
            payload = {
                "monitor_nodes": selected,
                "strategy": strategy_name,
                "n": int(budget),
                "protocol": {
                    "candidate_nodes_file": str(input_dir / "candidate_nodes_new.json"),
                    "defect_matrix_file": str(input_dir / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"),
                    "formal_data_dir": str(
                        repo_root
                        / "script2_new"
                        / "training_data_new"
                        / "time_gated_full_ie_v4_formal_conservative420_seed42"
                    ),
                    "note": args.protocol_note,
                },
                "objective_meta": {
                    "mode": mode,
                    "fixed_overlap_target": int(target_overlap),
                    "overlap_anchor_strategy": "degree",
                    "probe_budget_role": "mainline discrimination before full budget sweep",
                },
                "layout_metrics": metrics,
            }
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            summary_rows.append(
                {
                    "strategy": strategy_name,
                    "n": int(budget),
                    "layout_file": str(out_file),
                    "fixed_overlap_target": int(target_overlap),
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ch5_root / "outputs" / "overlap_controlled_probe_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
