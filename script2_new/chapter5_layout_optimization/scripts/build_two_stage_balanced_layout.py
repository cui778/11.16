#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 two-stage balanced layouts.

Stage 1:
- compress the feasible monitor pool using joint topology/response similarity
- keep roughly compression_ratio * budget representatives

Stage 2:
- optimize the final layout within the representative pool
- objective balances task observability, candidate separability,
  whole-graph coverage, key-structure coverage, and redundancy penalty

Outputs:
- representative pool JSONs
- layout JSONs
- one summary CSV
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


INF_HOP = 999
FEATURE_COLS = [
    "depth_residual",
    "total_outflow_residual",
    "pollut_NH4_residual",
    "pollut_TSSs_residual",
]
DEFECT_TYPES = ("I", "E")


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


def hop_utility_key(hop: int) -> float:
    if hop == 0:
        return 3.0
    if hop == 1:
        return 2.0
    if hop == 2:
        return 1.0
    return 0.0


def _row_l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return mat / norms


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    normed = _row_l2_normalize(mat.astype(np.float32))
    sim = normed @ normed.T
    sim = np.clip(sim, -1.0, 1.0)
    return ((sim + 1.0) / 2.0).astype(np.float32)


def build_topology_similarity(shortest: np.ndarray, pool_indices: List[int], tau: float) -> np.ndarray:
    n = len(pool_indices)
    sim = np.zeros((n, n), dtype=np.float32)
    for a, ia in enumerate(pool_indices):
        sim[a, a] = 1.0
        for b in range(a + 1, n):
            ib = pool_indices[b]
            hop = sym_hop(shortest, ia, ib)
            val = 0.0 if hop >= INF_HOP else float(math.exp(-float(hop) / max(tau, 1e-6)))
            sim[a, b] = val
            sim[b, a] = val
    return sim


def compute_response_tensor(
    parquet_path: Path,
    defect_csv: Path,
    node_order: List[str],
    candidate_nodes: List[str],
    cache_path: Path,
    refresh_cache: bool,
) -> np.ndarray:
    if cache_path.exists() and not refresh_cache:
        cached = np.load(str(cache_path), allow_pickle=True)
        cached_nodes = [str(x) for x in cached["node_order"].tolist()]
        cached_candidates = [str(x) for x in cached["candidate_nodes"].tolist()]
        cached_types = [str(x) for x in cached["defect_types"].tolist()]
        if cached_nodes == node_order and cached_candidates == candidate_nodes and cached_types == list(DEFECT_TYPES):
            return cached["tensor"].astype(np.float32)

    defect_df = pd.read_csv(defect_csv)
    defect_df = defect_df.rename(columns={"defect_id": "scenario_id", "node_id": "defect_node"})
    defect_df = defect_df[["scenario_id", "defect_node", "defect_type"]]

    df = pd.read_parquet(parquet_path, columns=["scenario_id", "node_id"] + FEATURE_COLS)
    df = df[df["scenario_id"] > 0].copy()
    for col in FEATURE_COLS:
        df[col] = df[col].abs()
    df["response_energy"] = df[FEATURE_COLS].mean(axis=1)

    per_scenario = df.groupby(["scenario_id", "node_id"], as_index=False)["response_energy"].mean()
    merged = per_scenario.merge(defect_df, on="scenario_id", how="inner")
    grouped = (
        merged.groupby(["node_id", "defect_node", "defect_type"], as_index=False)["response_energy"]
        .mean()
        .rename(columns={"node_id": "monitor_node"})
    )

    row_lookup = {node: i for i, node in enumerate(node_order)}
    cand_lookup = {node: i for i, node in enumerate(candidate_nodes)}
    type_lookup = {dtype: i for i, dtype in enumerate(DEFECT_TYPES)}

    tensor = np.zeros((len(node_order), len(candidate_nodes), len(DEFECT_TYPES)), dtype=np.float32)
    for row in grouped.itertuples(index=False):
        mon = str(row.monitor_node)
        defect_node = str(row.defect_node)
        defect_type = str(row.defect_type)
        if mon not in row_lookup or defect_node not in cand_lookup or defect_type not in type_lookup:
            continue
        tensor[row_lookup[mon], cand_lookup[defect_node], type_lookup[defect_type]] = float(row.response_energy)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        tensor=tensor,
        node_order=np.asarray(node_order, dtype=object),
        candidate_nodes=np.asarray(candidate_nodes, dtype=object),
        defect_types=np.asarray(DEFECT_TYPES, dtype=object),
    )
    return tensor


def compute_importance(
    graph: nx.DiGraph,
    pool_nodes: List[str],
    pool_response_tensor: np.ndarray,
    candidate_set: set[str],
) -> np.ndarray:
    graph_u = graph.to_undirected()
    bc = nx.betweenness_centrality(graph_u, normalized=True)
    degree_vals = np.asarray([graph.in_degree(node) + graph.out_degree(node) for node in pool_nodes], dtype=np.float32)
    bc_vals = np.asarray([float(bc.get(node, 0.0)) for node in pool_nodes], dtype=np.float32)
    response_energy = pool_response_tensor.mean(axis=(1, 2)).astype(np.float32)
    candidate_bonus = np.asarray([1.0 if node in candidate_set else 0.0 for node in pool_nodes], dtype=np.float32)
    importance = (
        0.40 * _safe_minmax(degree_vals)
        + 0.25 * _safe_minmax(bc_vals)
        + 0.25 * _safe_minmax(response_energy)
        + 0.10 * candidate_bonus
    )
    return importance.astype(np.float32)


def init_medoids(distance: np.ndarray, importance: np.ndarray, k: int) -> List[int]:
    n = distance.shape[0]
    first = int(np.argmax(importance))
    medoids = [first]
    while len(medoids) < min(k, n):
        min_dist = np.min(distance[:, medoids], axis=1)
        min_dist[medoids] = -1.0
        score = min_dist * (0.75 + 0.25 * importance)
        nxt = int(np.argmax(score))
        if nxt in medoids:
            break
        medoids.append(nxt)
    return medoids


def assign_clusters(distance: np.ndarray, medoids: List[int]) -> np.ndarray:
    d = distance[:, medoids]
    return np.argmin(d, axis=1)


def recompute_medoids(distance: np.ndarray, labels: np.ndarray, medoids: List[int], importance: np.ndarray) -> List[int]:
    new_medoids: List[int] = []
    for cluster_id, current in enumerate(medoids):
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            new_medoids.append(current)
            continue
        intra = distance[np.ix_(members, members)]
        weighted = intra.sum(axis=1) - 0.05 * importance[members]
        best_local = int(members[int(np.argmin(weighted))])
        new_medoids.append(best_local)
    return new_medoids


def run_kmedoids(distance: np.ndarray, importance: np.ndarray, k: int, max_iter: int = 8) -> Tuple[List[int], np.ndarray]:
    medoids = init_medoids(distance, importance, k)
    for _ in range(max_iter):
        labels = assign_clusters(distance, medoids)
        new_medoids = recompute_medoids(distance, labels, medoids, importance)
        if new_medoids == medoids:
            break
        medoids = new_medoids
    labels = assign_clusters(distance, medoids)
    return medoids, labels


def precompute_data_driven_pair_contrib(pool_response_tensor: np.ndarray) -> np.ndarray:
    # pool_response_tensor: [n_pool, n_candidate, n_type]
    n_pool, n_candidate, _ = pool_response_tensor.shape
    contrib = np.zeros((n_pool, n_candidate, n_candidate), dtype=np.float32)
    for i in range(n_pool):
        resp = pool_response_tensor[i]
        diffs = np.abs(resp[:, None, :] - resp[None, :, :]).mean(axis=2)
        contrib[i] = diffs.astype(np.float32)
    scale = float(np.quantile(contrib, 0.95)) if contrib.size else 1.0
    scale = max(scale, 1e-6)
    return contrib / scale


def candidate_observability_score(candidate_to_pool_hops: np.ndarray, selected_local: List[int]) -> float:
    if not selected_local:
        return 0.0
    best = np.min(candidate_to_pool_hops[:, selected_local], axis=1)
    utilities = np.asarray([hop_utility_candidate(int(h)) for h in best], dtype=np.float32)
    return float(utilities.mean() / 8.0)


def global_coverage_score(all_to_pool_hops: np.ndarray, selected_local: List[int], all_weights: np.ndarray) -> float:
    if not selected_local:
        return 0.0
    best = np.min(all_to_pool_hops[:, selected_local], axis=1)
    utilities = np.asarray([hop_utility_global(int(h)) for h in best], dtype=np.float32) / 4.0
    return float(np.average(utilities, weights=all_weights))


def key_structure_score(key_to_pool_hops: np.ndarray, selected_local: List[int], key_weights: np.ndarray) -> float:
    if not selected_local or key_to_pool_hops.size == 0:
        return 0.0
    best = np.min(key_to_pool_hops[:, selected_local], axis=1)
    utilities = np.asarray([hop_utility_key(int(h)) for h in best], dtype=np.float32) / 3.0
    return float(np.average(utilities, weights=key_weights))


def separability_score(pair_contrib: np.ndarray, selected_local: List[int]) -> float:
    if not selected_local:
        return 0.0
    score = pair_contrib[selected_local].sum(axis=0)
    masked = score.copy()
    np.fill_diagonal(masked, np.inf)
    nn = masked.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0
    mean_nn = float(np.mean(nn))
    p10_nn = float(np.quantile(nn, 0.10))
    combined = 0.55 * mean_nn + 0.45 * p10_nn
    return float(1.0 - math.exp(-combined))


def redundancy_penalty(pool_similarity: np.ndarray, pool_hops: np.ndarray, selected_local: List[int]) -> float:
    if len(selected_local) <= 1:
        return 0.0
    sims = []
    close_pairs = 0
    total_pairs = 0
    for a, b in combinations(selected_local, 2):
        sims.append(float(pool_similarity[a, b]))
        total_pairs += 1
        if int(pool_hops[a, b]) <= 1:
            close_pairs += 1
    mean_sim = float(np.mean(sims)) if sims else 0.0
    close_frac = float(close_pairs / max(total_pairs, 1))
    return 0.7 * mean_sim + 0.3 * close_frac


def objective_breakdown(
    selected_local: List[int],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    pool_similarity: np.ndarray,
    pool_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
) -> Dict[str, float]:
    obs = candidate_observability_score(candidate_to_pool_hops, selected_local)
    sep = separability_score(pair_contrib, selected_local)
    glob = global_coverage_score(all_to_pool_hops, selected_local, all_weights)
    key = key_structure_score(key_to_pool_hops, selected_local, key_weights)
    red = redundancy_penalty(pool_similarity, pool_hops, selected_local)
    total = 0.42 * obs + 0.23 * sep + 0.20 * glob + 0.15 * key - 0.18 * red
    return {
        "objective_total": float(total),
        "objective_candidate_observability": float(obs),
        "objective_candidate_separability": float(sep),
        "objective_global_coverage": float(glob),
        "objective_key_structure": float(key),
        "objective_redundancy_penalty": float(red),
    }


def greedy_plus_local_swap(
    pool_nodes: List[str],
    budget: int,
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    pool_similarity: np.ndarray,
    pool_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
) -> Tuple[List[int], Dict[str, float]]:
    selected: List[int] = []
    available = list(range(len(pool_nodes)))

    for _ in range(min(budget, len(pool_nodes))):
        best_idx = None
        best_total = None
        for cand in available:
            trial = selected + [cand]
            total = objective_breakdown(
                trial,
                candidate_to_pool_hops,
                all_to_pool_hops,
                key_to_pool_hops,
                pair_contrib,
                pool_similarity,
                pool_hops,
                all_weights,
                key_weights,
            )["objective_total"]
            if best_total is None or total > best_total:
                best_total = total
                best_idx = cand
        if best_idx is None:
            break
        selected.append(best_idx)
        available.remove(best_idx)

    improved = True
    while improved and selected:
        improved = False
        current_breakdown = objective_breakdown(
            selected,
            candidate_to_pool_hops,
            all_to_pool_hops,
            key_to_pool_hops,
            pair_contrib,
            pool_similarity,
            pool_hops,
            all_weights,
            key_weights,
        )
        current_total = current_breakdown["objective_total"]
        for out_idx in list(selected):
            for in_idx in available:
                trial = [x for x in selected if x != out_idx] + [in_idx]
                trial_breakdown = objective_breakdown(
                    trial,
                    candidate_to_pool_hops,
                    all_to_pool_hops,
                    key_to_pool_hops,
                    pair_contrib,
                    pool_similarity,
                    pool_hops,
                    all_weights,
                    key_weights,
                )
                if trial_breakdown["objective_total"] > current_total + 1e-8:
                    selected = sorted(trial)
                    available = [x for x in range(len(pool_nodes)) if x not in selected]
                    current_total = trial_breakdown["objective_total"]
                    improved = True
                    break
            if improved:
                break

    final_breakdown = objective_breakdown(
        selected,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        pair_contrib,
        pool_similarity,
        pool_hops,
        all_weights,
        key_weights,
    )
    return selected, final_breakdown


def summarize_candidate_hops(best_hops: np.ndarray) -> Dict[str, float]:
    finite = best_hops[best_hops < INF_HOP]
    return {
        "direct": int((best_hops == 0).sum()),
        "near": int(((best_hops >= 1) & (best_hops <= 2)).sum()),
        "far": int((best_hops > 2).sum()),
        "mean_hop": float(finite.mean()) if finite.size else float("nan"),
        "max_hop": float(finite.max()) if finite.size else float("nan"),
    }


def signature_nn_stats(pair_score: np.ndarray) -> Dict[str, float]:
    masked = pair_score.copy()
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


def layout_summary(
    selected_nodes: List[str],
    candidate_nodes: List[str],
    node_to_idx: Dict[str, int],
    shortest: np.ndarray,
    pair_score: np.ndarray,
) -> Dict[str, float]:
    selected_idx = [node_to_idx[n] for n in selected_nodes]
    candidate_idx = [node_to_idx[n] for n in candidate_nodes]
    best_hops = []
    for cand_i in candidate_idx:
        if not selected_idx:
            best_hops.append(INF_HOP)
        else:
            best_hops.append(min(sym_hop(shortest, cand_i, mon_i) for mon_i in selected_idx))
    best_hops_arr = np.asarray(best_hops, dtype=np.int16)
    out = summarize_candidate_hops(best_hops_arr)
    out["overlap_count"] = int(len(set(selected_nodes) & set(candidate_nodes)))

    dispersion = []
    for a, b in combinations(selected_idx, 2):
        dispersion.append(sym_hop(shortest, a, b))
    out["monitor_dispersion_mean_hop"] = float(np.mean(dispersion)) if dispersion else 0.0

    cover_sets = []
    for mon_i in selected_idx:
        cover = set()
        for cand_j, cand_i in enumerate(candidate_idx):
            if sym_hop(shortest, cand_i, mon_i) <= 2:
                cover.add(cand_j)
        cover_sets.append(cover)
    jaccards = []
    for a, b in combinations(range(len(cover_sets)), 2):
        union = cover_sets[a] | cover_sets[b]
        inter = cover_sets[a] & cover_sets[b]
        jaccards.append((len(inter) / len(union)) if union else 0.0)
    out["monitor_redundancy_mean_jaccard"] = float(np.mean(jaccards)) if jaccards else 0.0
    out.update({f"signature_{k}": v for k, v in signature_nn_stats(pair_score).items()})
    return out


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_key_nodes(graph: nx.DiGraph, pool_nodes: List[str], candidate_set: set[str], top_k: int) -> Tuple[List[str], np.ndarray]:
    graph_u = graph.to_undirected()
    bc = nx.betweenness_centrality(graph_u, normalized=True)
    degree_vals = np.asarray([graph.in_degree(n) + graph.out_degree(n) for n in pool_nodes], dtype=np.float32)
    bc_vals = np.asarray([bc.get(n, 0.0) for n in pool_nodes], dtype=np.float32)
    candidate_bonus = np.asarray([1.0 if n in candidate_set else 0.0 for n in pool_nodes], dtype=np.float32)
    score = 0.55 * _safe_minmax(degree_vals) + 0.30 * _safe_minmax(bc_vals) + 0.15 * candidate_bonus
    order = np.argsort(-score)[: min(top_k, len(pool_nodes))]
    key_nodes = [pool_nodes[i] for i in order]
    key_weights = score[order]
    if float(key_weights.sum()) <= 1e-12:
        key_weights = np.ones_like(key_weights, dtype=np.float32)
    return key_nodes, key_weights.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-stage balanced Chapter-5 layouts.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument("--compression-ratio", type=float, default=1.4)
    parser.add_argument("--similarity-topology-weight", type=float, default=0.55)
    parser.add_argument("--similarity-response-weight", type=float, default=0.45)
    parser.add_argument("--topology-tau", type=float, default=3.0)
    parser.add_argument("--strategy-name", default="two_stage_balanced_layout_v1")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[2]
    chapter5_root = script_root / "chapter5_layout_optimization"
    input_dir = script_root / "input_1"
    outputs_dir = chapter5_root / "outputs"
    cache_dir = outputs_dir / "cache"
    layouts_dir = outputs_dir / "layouts" / args.strategy_name
    rep_dir = outputs_dir / "representative_pools" / args.strategy_name
    summary_csv = outputs_dir / f"{args.strategy_name}_summary.csv"

    candidate_path = input_dir / "candidate_nodes_new.json"
    node_list_path = input_dir / "node_list.json"
    adj_path = input_dir / "adj_matrix.npy"
    graph_features_path = input_dir / "graph_path_features.npz"
    defect_csv = input_dir / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"
    parquet_path = script_root / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42" / "node_timeseries_with_residuals.parquet"

    candidate_nodes = load_json_list(candidate_path, ("candidate_nodes", "nodes"))
    node_list = load_json_list(node_list_path, ("node_list", "nodes"))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    graph = build_directed_graph(node_list, adj_path)
    shortest = np.load(str(graph_features_path))["shortest_dist"]

    exclude = {node for node in graph.nodes() if graph.in_degree(node) + graph.out_degree(node) == 0}
    pool_nodes = [node for node in node_list if node not in exclude]
    pool_indices = [node_to_idx[node] for node in pool_nodes]
    candidate_indices = [node_to_idx[node] for node in candidate_nodes]
    candidate_set = set(candidate_nodes)

    response_tensor_full = compute_response_tensor(
        parquet_path=parquet_path,
        defect_csv=defect_csv,
        node_order=node_list,
        candidate_nodes=candidate_nodes,
        cache_path=cache_dir / "response_signature_ie_energy.npz",
        refresh_cache=args.refresh_cache,
    )
    response_tensor_pool = response_tensor_full[pool_indices]
    response_matrix_pool = response_tensor_pool.reshape(len(pool_nodes), -1)

    topology_sim = build_topology_similarity(shortest, pool_indices, tau=args.topology_tau)
    response_sim = cosine_similarity_matrix(response_matrix_pool)
    joint_similarity = (
        float(args.similarity_topology_weight) * topology_sim
        + float(args.similarity_response_weight) * response_sim
    ).astype(np.float32)
    joint_similarity = np.clip(joint_similarity, 0.0, 1.0)
    joint_distance = 1.0 - joint_similarity

    importance = compute_importance(graph, pool_nodes, response_tensor_pool, candidate_set)
    key_nodes, key_weights = pick_key_nodes(
        graph=graph,
        pool_nodes=pool_nodes,
        candidate_set=candidate_set,
        top_k=max(12, int(round(0.15 * len(pool_nodes)))),
    )
    key_indices = [node_to_idx[n] for n in key_nodes]

    candidate_to_pool_hops = np.full((len(candidate_nodes), len(pool_nodes)), INF_HOP, dtype=np.int16)
    for ci, cand_idx in enumerate(candidate_indices):
        for pj, pool_idx in enumerate(pool_indices):
            candidate_to_pool_hops[ci, pj] = sym_hop(shortest, cand_idx, pool_idx)

    all_to_pool_hops = np.full((len(pool_nodes), len(pool_nodes)), INF_HOP, dtype=np.int16)
    for ai, a_idx in enumerate(pool_indices):
        for pj, pool_idx in enumerate(pool_indices):
            all_to_pool_hops[ai, pj] = sym_hop(shortest, a_idx, pool_idx)

    key_to_pool_hops = np.full((len(key_nodes), len(pool_nodes)), INF_HOP, dtype=np.int16)
    for ki, key_idx in enumerate(key_indices):
        for pj, pool_idx in enumerate(pool_indices):
            key_to_pool_hops[ki, pj] = sym_hop(shortest, key_idx, pool_idx)

    all_weights = 0.60 * _safe_minmax(importance) + 0.40
    pair_contrib_full = precompute_data_driven_pair_contrib(response_tensor_pool)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    rows: List[Dict[str, float]] = []

    for budget in budgets:
        rep_budget = min(len(pool_nodes), max(budget, int(math.ceil(float(args.compression_ratio) * budget))))
        medoids, labels = run_kmedoids(joint_distance, importance, rep_budget)
        rep_nodes = [pool_nodes[i] for i in medoids]

        rep_hops = all_to_pool_hops[np.ix_(medoids, medoids)]
        rep_similarity = joint_similarity[np.ix_(medoids, medoids)]
        rep_pair_contrib = pair_contrib_full[medoids]
        rep_candidate_hops = candidate_to_pool_hops[:, medoids]
        rep_all_hops = all_to_pool_hops[:, medoids]
        rep_key_hops = key_to_pool_hops[:, medoids]
        rep_all_weights = all_weights

        selected_local, breakdown = greedy_plus_local_swap(
            pool_nodes=rep_nodes,
            budget=budget,
            candidate_to_pool_hops=rep_candidate_hops,
            all_to_pool_hops=rep_all_hops,
            key_to_pool_hops=rep_key_hops,
            pair_contrib=rep_pair_contrib,
            pool_similarity=rep_similarity,
            pool_hops=rep_hops,
            all_weights=rep_all_weights,
            key_weights=key_weights,
        )
        selected_nodes = [rep_nodes[i] for i in selected_local]
        pair_score = rep_pair_contrib[selected_local].sum(axis=0) if selected_local else np.zeros(
            (len(candidate_nodes), len(candidate_nodes)), dtype=np.float32
        )
        metrics = layout_summary(selected_nodes, candidate_nodes, node_to_idx, shortest, pair_score)
        metrics.update(breakdown)
        metrics["representative_pool_size"] = int(len(rep_nodes))
        metrics["representative_pool_candidate_overlap"] = int(len(set(rep_nodes) & candidate_set))
        metrics["key_node_count"] = int(len(key_nodes))

        rep_payload = {
            "strategy": args.strategy_name,
            "budget": int(budget),
            "compression_ratio": float(args.compression_ratio),
            "representative_pool_nodes": rep_nodes,
            "cluster_labels": {pool_nodes[i]: int(labels[i]) for i in range(len(pool_nodes))},
            "similarity_weights": {
                "topology": float(args.similarity_topology_weight),
                "response": float(args.similarity_response_weight),
            },
            "key_nodes": key_nodes,
        }
        save_json(rep_dir / f"representative_pool_{args.strategy_name}_N{budget}.json", rep_payload)

        layout_payload = {
            "monitor_nodes": selected_nodes,
            "strategy": args.strategy_name,
            "n": int(budget),
            "representative_pool_file": str(rep_dir / f"representative_pool_{args.strategy_name}_N{budget}.json"),
            "protocol": {
                "candidate_nodes_file": str(candidate_path),
                "defect_matrix_file": str(defect_csv),
                "formal_data_dir": str(parquet_path.parent),
                "note": "control-variable Chapter-5 layout study; only monitor layout changes",
            },
            "objective_meta": {
                "stage1": "joint topology-response k-medoids compression",
                "stage2": "greedy plus local swap",
                "compression_ratio": float(args.compression_ratio),
                "similarity_topology_weight": float(args.similarity_topology_weight),
                "similarity_response_weight": float(args.similarity_response_weight),
                "topology_tau": float(args.topology_tau),
                "selected_representative_pool_size": int(len(rep_nodes)),
            },
            "layout_metrics": metrics,
        }
        layout_path = layouts_dir / f"monitor_nodes_{args.strategy_name}_N{budget}.json"
        save_json(layout_path, layout_payload)

        row = {
            "strategy": args.strategy_name,
            "budget": int(budget),
            "layout_file": str(layout_path),
            "representative_pool_file": str(rep_dir / f"representative_pool_{args.strategy_name}_N{budget}.json"),
        }
        row.update(metrics)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {summary_csv}")
    for row in rows:
        print(
            f"[{row['strategy']} N{row['budget']}] "
            f"obj={row['objective_total']:.4f} "
            f"direct={row['direct']} near={row['near']} far={row['far']} "
            f"mean_hop={row['mean_hop']:.2f}"
        )


if __name__ == "__main__":
    main()
