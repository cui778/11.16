#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 monitor layout suites under the frozen Chapter-1 protocol.

Key rule:
- defect matrix stays fixed
- candidate set stays fixed
- only the monitor layout changes

Outputs:
- layout JSON files for each strategy / budget
- one static summary CSV for quick comparison
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


INF_HOP = 999
HOP_CLIP = 6


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


def sym_hop(shortest: np.ndarray, i: int, j: int) -> int:
    a = int(shortest[i, j])
    b = int(shortest[j, i])
    vals = [x for x in (a, b) if x < INF_HOP]
    return min(vals) if vals else INF_HOP


def hop_utility(hop: int) -> float:
    if hop == 0:
        return 6.0
    if hop == 1:
        return 4.0
    if hop == 2:
        return 3.0
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


def strategy_degree(graph: nx.DiGraph, n: int, exclude: set[str]) -> List[str]:
    degree_dict = {
        node: graph.in_degree(node) + graph.out_degree(node)
        for node in graph.nodes()
        if node not in exclude
    }
    return sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:n]


def strategy_betweenness(graph: nx.DiGraph, n: int, exclude: set[str]) -> List[str]:
    graph_u = graph.to_undirected()
    bc = nx.betweenness_centrality(graph_u, normalized=True)
    bc = {k: v for k, v in bc.items() if k not in exclude}
    return sorted(bc, key=lambda x: bc[x], reverse=True)[:n]


def strategy_downstream(graph: nx.DiGraph, n: int, exclude: set[str]) -> List[str]:
    g_rev = graph.reverse(copy=True)
    upstream_count = {}
    for node in graph.nodes():
        if node in exclude:
            continue
        upstream_count[node] = len(nx.descendants(g_rev, node))
    return sorted(upstream_count, key=lambda x: upstream_count[x], reverse=True)[:n]


def strategy_random(nodes: List[str], n: int, exclude: set[str], seed: int) -> List[str]:
    pool = [node for node in nodes if node not in exclude]
    rng = np.random.default_rng(seed)
    if len(pool) <= n:
        return list(pool)
    return list(rng.choice(pool, size=n, replace=False))


def candidate_overlap_cap(n: int) -> int:
    return max(1, int(np.floor(0.6 * n)))


def build_candidate_observability_layout(
    all_nodes: List[str],
    candidate_nodes: List[str],
    hop_matrix: np.ndarray,
    n: int,
) -> Tuple[List[str], Dict[str, float]]:
    candidate_set = set(candidate_nodes)
    selected: List[str] = []
    best_hops = np.full(len(candidate_nodes), INF_HOP, dtype=np.int16)
    max_overlap = candidate_overlap_cap(n)

    for _ in range(n):
        best_score = None
        best_idx = None
        best_new_hops = None

        overlap_count = sum(1 for x in selected if x in candidate_set)
        for j, node in enumerate(all_nodes):
            if node in selected:
                continue
            if node in candidate_set and overlap_count >= max_overlap:
                continue

            new_hops = np.minimum(best_hops, hop_matrix[:, j])
            old_nonfar = int((best_hops <= 2).sum())
            new_nonfar = int((new_hops <= 2).sum())
            old_direct = int((best_hops == 0).sum())
            new_direct = int((new_hops == 0).sum())
            old_util = float(np.sum([hop_utility(int(h)) for h in best_hops]))
            new_util = float(np.sum([hop_utility(int(h)) for h in new_hops]))
            hop_reduction = (
                float(best_hops[best_hops < INF_HOP].mean() - new_hops.mean())
                if (best_hops < INF_HOP).any()
                else 0.0
            )
            noncandidate_bonus = 2.0 if node not in candidate_set else 0.0
            score = (
                1000.0 * (new_nonfar - old_nonfar)
                + 60.0 * (new_direct - old_direct)
                + 10.0 * (new_util - old_util)
                + 4.0 * hop_reduction
                + noncandidate_bonus
            )

            if best_score is None or score > best_score:
                best_score = score
                best_idx = j
                best_new_hops = new_hops

        if best_idx is None or best_new_hops is None:
            break
        selected.append(all_nodes[best_idx])
        best_hops = best_new_hops

    meta = summarize_candidate_hops(best_hops)
    meta["max_candidate_overlap"] = int(max_overlap)
    return selected, meta


def precompute_pair_contrib(candidate_to_node_hops: np.ndarray) -> np.ndarray:
    n_candidates, n_nodes = candidate_to_node_hops.shape
    contrib = np.zeros((n_nodes, n_candidates, n_candidates), dtype=np.float32)
    for node_idx in range(n_nodes):
        hops = np.clip(candidate_to_node_hops[:, node_idx].astype(np.float32), 0, HOP_CLIP)
        for a in range(n_candidates):
            for b in range(a + 1, n_candidates):
                ha = hops[a]
                hb = hops[b]
                diff = abs(float(ha) - float(hb))
                base = diff / (1.0 + min(float(ha), float(hb)))
                contrib[node_idx, a, b] = base
                contrib[node_idx, b, a] = base
    return contrib


def signature_nn_stats(signature_scores: np.ndarray) -> Dict[str, float]:
    masked = signature_scores.copy()
    np.fill_diagonal(masked, np.inf)
    nn = masked.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return {"mean_nn": 0.0, "min_nn": 0.0, "p10_nn": 0.0}
    return {
        "mean_nn": float(np.mean(nn)),
        "min_nn": float(np.min(nn)),
        "p10_nn": float(np.quantile(nn, 0.10)),
    }


def aggregate_pair_score(selected_local_indices: List[int], pair_contrib: np.ndarray, n_candidates: int) -> np.ndarray:
    if not selected_local_indices:
        return np.zeros((n_candidates, n_candidates), dtype=np.float32)
    pair_score = np.zeros((n_candidates, n_candidates), dtype=np.float32)
    for idx in selected_local_indices:
        pair_score += pair_contrib[idx]
    return pair_score


def build_identifiability_layout(
    all_nodes: List[str],
    candidate_nodes: List[str],
    hop_matrix: np.ndarray,
    pair_contrib: np.ndarray,
    n: int,
) -> Tuple[List[str], Dict[str, float], np.ndarray]:
    candidate_set = set(candidate_nodes)
    selected: List[str] = []
    best_hops = np.full(len(candidate_nodes), INF_HOP, dtype=np.int16)
    pair_score = np.zeros((len(candidate_nodes), len(candidate_nodes)), dtype=np.float32)
    max_overlap = candidate_overlap_cap(n)

    for _ in range(n):
        best_score = None
        best_idx = None
        best_new_hops = None
        best_pair_score = None

        overlap_count = sum(1 for x in selected if x in candidate_set)
        for j, node in enumerate(all_nodes):
            if node in selected:
                continue
            if node in candidate_set and overlap_count >= max_overlap:
                continue

            new_hops = np.minimum(best_hops, hop_matrix[:, j])
            new_pair_score = pair_score + pair_contrib[j]
            nn_stats = signature_nn_stats(new_pair_score)

            old_nonfar = int((best_hops <= 2).sum())
            new_nonfar = int((new_hops <= 2).sum())
            old_direct = int((best_hops == 0).sum())
            new_direct = int((new_hops == 0).sum())
            old_util = float(np.sum([hop_utility(int(h)) for h in best_hops]))
            new_util = float(np.sum([hop_utility(int(h)) for h in new_hops]))
            noncandidate_bonus = 2.0 if node not in candidate_set else 0.0

            score = (
                800.0 * (new_nonfar - old_nonfar)
                + 50.0 * (new_direct - old_direct)
                + 8.0 * (new_util - old_util)
                + 200.0 * nn_stats["mean_nn"]
                + 400.0 * nn_stats["min_nn"]
                + 200.0 * nn_stats["p10_nn"]
                + noncandidate_bonus
            )

            if best_score is None or score > best_score:
                best_score = score
                best_idx = j
                best_new_hops = new_hops
                best_pair_score = new_pair_score

        if best_idx is None or best_new_hops is None or best_pair_score is None:
            break
        selected.append(all_nodes[best_idx])
        best_hops = best_new_hops
        pair_score = best_pair_score

    meta = summarize_candidate_hops(best_hops)
    meta["max_candidate_overlap"] = int(max_overlap)
    meta.update({f"signature_{k}": v for k, v in signature_nn_stats(pair_score).items()})
    return selected, meta, pair_score


def layout_summary(
    selected: List[str],
    candidate_nodes: List[str],
    node_to_idx: Dict[str, int],
    shortest: np.ndarray,
    pair_score: np.ndarray | None = None,
) -> Dict[str, float]:
    selected_idx = [node_to_idx[n] for n in selected]
    candidate_idx = [node_to_idx[n] for n in candidate_nodes]

    best_hops = []
    for cand_i in candidate_idx:
        if not selected_idx:
            best_hops.append(INF_HOP)
            continue
        best_hops.append(min(sym_hop(shortest, cand_i, mon_i) for mon_i in selected_idx))
    best_hops_arr = np.asarray(best_hops, dtype=np.int16)
    out = summarize_candidate_hops(best_hops_arr)
    out["overlap_count"] = int(len(set(selected) & set(candidate_nodes)))

    disp = []
    for a, b in combinations(selected_idx, 2):
        disp.append(sym_hop(shortest, a, b))
    out["monitor_dispersion_mean_hop"] = float(np.mean(disp)) if disp else 0.0

    cover_sets = []
    for mon_i in selected_idx:
        cover = set()
        for cand_j, cand_i in enumerate(candidate_idx):
            hop = sym_hop(shortest, cand_i, mon_i)
            if hop <= 2:
                cover.add(cand_j)
        cover_sets.append(cover)

    jaccards = []
    for a, b in combinations(range(len(cover_sets)), 2):
        u = cover_sets[a] | cover_sets[b]
        inter = cover_sets[a] & cover_sets[b]
        jaccards.append((len(inter) / len(u)) if u else 0.0)
    out["monitor_redundancy_mean_jaccard"] = float(np.mean(jaccards)) if jaccards else 0.0

    if pair_score is not None:
        out.update({f"signature_{k}": v for k, v in signature_nn_stats(pair_score).items()})
    return out


def save_layout(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter-5 layout suite.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument(
        "--strategies",
        default="random,degree,betweenness,downstream,candidate_observability,identifiability_driven",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[2]
    chapter5_root = script_root / "chapter5_layout_optimization"
    input_dir = script_root / "input_1"
    outputs_dir = chapter5_root / "outputs"
    layouts_dir = outputs_dir / "layouts"
    summary_csv = outputs_dir / "layout_summary.csv"

    candidate_path = input_dir / "candidate_nodes_new.json"
    node_list_path = input_dir / "node_list.json"
    adj_path = input_dir / "adj_matrix.npy"
    graph_features_path = input_dir / "graph_path_features.npz"

    candidate_nodes = load_json_list(candidate_path, ("candidate_nodes", "nodes"))
    node_list = load_json_list(node_list_path, ("node_list", "nodes"))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    graph = build_directed_graph(node_list, adj_path)
    shortest = np.load(str(graph_features_path))["shortest_dist"]

    exclude = {node for node in graph.nodes() if graph.in_degree(node) + graph.out_degree(node) == 0}
    all_nodes = [node for node in node_list if node not in exclude]
    candidate_idx = [node_to_idx[node] for node in candidate_nodes]
    all_idx = [node_to_idx[node] for node in all_nodes]

    hop_matrix = np.full((len(candidate_nodes), len(all_nodes)), INF_HOP, dtype=np.int16)
    for ci, cand_i in enumerate(candidate_idx):
        for nj, node_j in enumerate(all_idx):
            hop_matrix[ci, nj] = sym_hop(shortest, cand_i, node_j)

    pair_contrib = precompute_pair_contrib(hop_matrix)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    rows = []

    for n in budgets:
        for strategy in strategies:
            if strategy == "random":
                selected = strategy_random(all_nodes, n, set(), seed=args.random_seed)
                objective_meta = {"random_seed": int(args.random_seed)}
            elif strategy == "degree":
                selected = strategy_degree(graph, n, exclude)
                objective_meta = {"rule": "top degree centrality"}
            elif strategy == "betweenness":
                selected = strategy_betweenness(graph, n, exclude)
                objective_meta = {"rule": "top betweenness centrality"}
            elif strategy == "downstream":
                selected = strategy_downstream(graph, n, exclude)
                objective_meta = {"rule": "maximize upstream reach"}
            elif strategy == "candidate_observability":
                selected, objective_meta = build_candidate_observability_layout(all_nodes, candidate_nodes, hop_matrix, n)
            elif strategy == "identifiability_driven":
                selected, objective_meta, pair_score = build_identifiability_layout(
                    all_nodes, candidate_nodes, hop_matrix, pair_contrib, n
                )
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            if strategy != "identifiability_driven":
                selected_local_indices = [all_nodes.index(node) for node in selected]
                pair_score = aggregate_pair_score(selected_local_indices, pair_contrib, len(candidate_nodes))

            metrics = layout_summary(selected, candidate_nodes, node_to_idx, shortest, pair_score=pair_score)
            rel_path = Path("layouts") / strategy / f"monitor_nodes_{strategy}_N{n}.json"
            abs_path = outputs_dir / rel_path
            payload = {
                "monitor_nodes": selected,
                "strategy": strategy,
                "n": int(n),
                "frozen_protocol": {
                    "candidate_nodes_file": str(candidate_path),
                    "defect_matrix_file": str(input_dir / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"),
                    "formal_data_dir": str(script_root / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42"),
                    "note": "defect matrix fixed; only monitor layout changes",
                },
                "objective_meta": objective_meta,
                "layout_metrics": metrics,
            }
            save_layout(abs_path, payload)

            rows.append(
                {
                    "strategy": strategy,
                    "n": int(n),
                    "layout_file": str(abs_path),
                    **metrics,
                }
            )

    df = pd.DataFrame(rows).sort_values(["n", "strategy"]).reset_index(drop=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(str(summary_csv))


if __name__ == "__main__":
    main()
