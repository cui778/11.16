#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 strengthened main-method layouts.

This variant is designed after the overlap-controlled probe:

- keep the two-stage framework
- keep evaluator sensitivity / reconstruction / robustness
- explicitly strengthen global/generalization constraints
- explicitly regularize excessive candidate-monitor overlap

Main additions over v2:
- non-candidate backbone coverage term
- coverage balance term between candidate and non-candidate support
- monitor dispersion term
- overlap-regularization term anchored by the degree baseline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_two_stage_balanced_layout import (
    INF_HOP,
    _safe_minmax,
    build_directed_graph,
    build_topology_similarity,
    compute_importance,
    compute_response_tensor,
    cosine_similarity_matrix,
    global_coverage_score,
    key_structure_score,
    candidate_observability_score,
    redundancy_penalty,
    separability_score,
    layout_summary,
    load_json_list,
    pick_key_nodes,
    precompute_data_driven_pair_contrib,
    run_kmedoids,
    save_json,
    sym_hop,
)
from build_two_stage_balanced_layout_v2 import (
    SCRIPT_ROOT,
    blended_importance,
    blended_key_nodes,
    compute_evaluator_sensitivity,
    reconstruction_score,
)


def noncandidate_backbone_nodes(
    graph,
    pool_nodes: List[str],
    candidate_set: set[str],
    sensitivity_scores: np.ndarray,
    node_to_idx: Dict[str, int],
    top_k: int,
) -> Tuple[List[str], np.ndarray]:
    graph_u = graph.to_undirected()
    bc = nx.betweenness_centrality(graph_u, normalized=True)
    degree_vals = np.asarray([graph.in_degree(n) + graph.out_degree(n) for n in pool_nodes], dtype=np.float32)
    bc_vals = np.asarray([bc.get(n, 0.0) for n in pool_nodes], dtype=np.float32)
    sens_vals = np.asarray([float(sensitivity_scores[node_to_idx[n]]) for n in pool_nodes], dtype=np.float32)
    noncandidate_mask = np.asarray([0.0 if n in candidate_set else 1.0 for n in pool_nodes], dtype=np.float32)
    score = (
        0.45 * _safe_minmax(degree_vals)
        + 0.35 * _safe_minmax(bc_vals)
        + 0.20 * _safe_minmax(sens_vals)
    ) * noncandidate_mask
    order = np.argsort(-score)
    order = [int(i) for i in order if score[i] > 0][: min(top_k, int((noncandidate_mask > 0).sum()))]
    nodes = [pool_nodes[i] for i in order]
    weights = score[np.asarray(order, dtype=np.int64)] if order else np.zeros((0,), dtype=np.float32)
    if weights.size and float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float32)
    return nodes, weights.astype(np.float32)


def overlap_anchor(layout_summary_csv: Path, budget: int, strategy: str = "degree") -> int:
    df = pd.read_csv(layout_summary_csv)
    sub = df[(df["strategy"] == strategy) & (df["n"] == budget)]
    if sub.empty:
        raise ValueError(f"Missing overlap anchor for strategy={strategy}, budget={budget}")
    return int(sub.iloc[0]["overlap_count"])


def noncandidate_backbone_score(
    noncandidate_to_pool_hops: np.ndarray,
    selected_local: List[int],
    noncandidate_weights: np.ndarray,
) -> float:
    if not selected_local or noncandidate_to_pool_hops.size == 0:
        return 0.0
    best = np.min(noncandidate_to_pool_hops[:, selected_local], axis=1)
    utilities = np.asarray([1.0 if int(h) == 0 else 0.8 if int(h) == 1 else 0.55 if int(h) == 2 else 0.25 if int(h) == 3 else 0.0 for h in best], dtype=np.float32)
    return float(np.average(utilities, weights=noncandidate_weights))


def coverage_balance_score(candidate_score: float, noncandidate_score: float) -> float:
    return float(max(0.0, 1.0 - abs(candidate_score - noncandidate_score)))


def monitor_dispersion_score(rep_hops: np.ndarray, selected_local: List[int]) -> float:
    if len(selected_local) <= 1:
        return 0.0
    hops = []
    for a, b in combinations(selected_local, 2):
        hop = int(rep_hops[a, b])
        if hop < INF_HOP:
            hops.append(float(hop))
    if not hops:
        return 0.0
    mean_hop = float(np.mean(hops))
    return float(min(mean_hop / 6.0, 1.0))


def overlap_regularization(
    selected_local: List[int],
    pool_nodes: List[str],
    candidate_set: set[str],
    overlap_anchor_target: int,
    budget: int,
) -> float:
    overlap = sum(1 for idx in selected_local if pool_nodes[idx] in candidate_set)
    excess = max(0.0, float(overlap - overlap_anchor_target))
    return float(excess / max(float(budget), 1.0))


def overlap_upper_bound(overlap_anchor_target: int, budget: int) -> int:
    return min(int(budget), int(overlap_anchor_target) + max(1, int(round(0.04 * budget))))


def objective_weight_config(preset: str) -> Dict[str, float]:
    if preset == "v3_1":
        return {
            "obs": 0.19,
            "sep": 0.15,
            "glob": 0.12,
            "key": 0.10,
            "noncand": 0.10,
            "balance": 0.07,
            "disp": 0.05,
            "recon": 0.10,
            "red_pen": 0.08,
            "overlap_pen": 0.05,
            "robust": 0.10,
        }
    if preset == "v3_2":
        return {
            "obs": 0.18,
            "sep": 0.17,
            "glob": 0.11,
            "key": 0.10,
            "noncand": 0.10,
            "balance": 0.06,
            "disp": 0.04,
            "recon": 0.11,
            "red_pen": 0.08,
            "overlap_pen": 0.05,
            "robust": 0.10,
        }
    return {
        "obs": 0.14,
        "sep": 0.10,
        "glob": 0.16,
        "key": 0.12,
        "noncand": 0.14,
        "balance": 0.10,
        "disp": 0.08,
        "recon": 0.12,
        "red_pen": 0.10,
        "overlap_pen": 0.10,
        "robust": 0.12,
    }


def objective_core_v3(
    selected_local: List[int],
    pool_nodes: List[str],
    candidate_set: set[str],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    noncandidate_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    noncandidate_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
    overlap_anchor_target: int,
    budget: int,
    weights: Dict[str, float],
) -> Dict[str, float]:
    obs = candidate_observability_score(candidate_to_pool_hops, selected_local)
    sep = separability_score(pair_contrib, selected_local)
    glob = global_coverage_score(all_to_pool_hops, selected_local, all_weights)
    key = key_structure_score(key_to_pool_hops, selected_local, key_weights)
    noncand = noncandidate_backbone_score(noncandidate_to_pool_hops, selected_local, noncandidate_weights)
    balance = coverage_balance_score(obs, noncand)
    disp = monitor_dispersion_score(rep_hops, selected_local)
    red = redundancy_penalty(rep_similarity, rep_hops, selected_local)
    recon, recon_rmse, recon_r2 = reconstruction_score(response_matrix_pool, rep_similarity, rep_hops, selected_local)
    overlap_pen = overlap_regularization(selected_local, pool_nodes, candidate_set, overlap_anchor_target, budget)

    total = (
        float(weights["obs"]) * obs
        + float(weights["sep"]) * sep
        + float(weights["glob"]) * glob
        + float(weights["key"]) * key
        + float(weights["noncand"]) * noncand
        + float(weights["balance"]) * balance
        + float(weights["disp"]) * disp
        + float(weights["recon"]) * recon
        - float(weights["red_pen"]) * red
        - float(weights["overlap_pen"]) * overlap_pen
    )
    return {
        "core_total": float(total),
        "objective_candidate_observability": float(obs),
        "objective_candidate_separability": float(sep),
        "objective_global_coverage": float(glob),
        "objective_key_structure": float(key),
        "objective_noncandidate_backbone": float(noncand),
        "objective_coverage_balance": float(balance),
        "objective_monitor_dispersion": float(disp),
        "objective_redundancy_penalty": float(red),
        "objective_reconstruction": float(recon),
        "objective_overlap_penalty": float(overlap_pen),
        "reconstruction_norm_rmse": float(recon_rmse),
        "reconstruction_r2": float(recon_r2),
    }


def _sample_failure_cases(selected_local: List[int], rng: np.random.Generator) -> List[Tuple[int, ...]]:
    cases: List[Tuple[int, ...]] = [tuple()]
    if not selected_local:
        return cases
    singles = [(x,) for x in selected_local]
    if len(singles) > 6:
        pick = rng.choice(len(singles), size=6, replace=False)
        singles = [singles[i] for i in pick]
    cases.extend(singles)
    pairs = list(combinations(selected_local, 2))
    if len(pairs) > 6:
        pick = rng.choice(len(pairs), size=6, replace=False)
        pairs = [pairs[i] for i in pick]
    cases.extend([tuple(p) for p in pairs])
    return cases


def robustness_score_v3(
    selected_local: List[int],
    pool_nodes: List[str],
    candidate_set: set[str],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    noncandidate_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    noncandidate_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
    overlap_anchor_target: int,
    budget: int,
    weights: Dict[str, float],
) -> Dict[str, float]:
    seed = 7919 + int(sum((i + 1) * x for i, x in enumerate(sorted(selected_local))))
    rng = np.random.default_rng(seed)
    scores = []
    for failed in _sample_failure_cases(selected_local, rng):
        remain = [x for x in selected_local if x not in set(failed)]
        if not remain:
            scores.append(0.0)
            continue
        scores.append(
            objective_core_v3(
                remain,
                pool_nodes,
                candidate_set,
                candidate_to_pool_hops,
                all_to_pool_hops,
                key_to_pool_hops,
                noncandidate_to_pool_hops,
                pair_contrib,
                rep_similarity,
                rep_hops,
                all_weights,
                key_weights,
                noncandidate_weights,
                response_matrix_pool,
                overlap_anchor_target,
                budget,
                weights,
            )["core_total"]
        )
    return {
        "objective_failure_robustness": float(np.mean(scores)) if scores else 0.0,
        "robustness_case_count": int(len(scores)),
    }


def objective_breakdown_v3(
    selected_local: List[int],
    pool_nodes: List[str],
    candidate_set: set[str],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    noncandidate_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    noncandidate_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
    overlap_anchor_target: int,
    budget: int,
    weights: Dict[str, float],
) -> Dict[str, float]:
    core = objective_core_v3(
        selected_local,
        pool_nodes,
        candidate_set,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        noncandidate_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        noncandidate_weights,
        response_matrix_pool,
        overlap_anchor_target,
        budget,
        weights,
    )
    robust = robustness_score_v3(
        selected_local,
        pool_nodes,
        candidate_set,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        noncandidate_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        noncandidate_weights,
        response_matrix_pool,
        overlap_anchor_target,
        budget,
        weights,
    )
    out = dict(core)
    out.update(robust)
    out["objective_total"] = float(core["core_total"] + float(weights["robust"]) * robust["objective_failure_robustness"])
    return out


def greedy_local_v3(
    pool_nodes: List[str],
    budget: int,
    candidate_set: set[str],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    noncandidate_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    noncandidate_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
    overlap_anchor_target: int,
    overlap_cap: int,
    weights: Dict[str, float],
) -> Tuple[List[int], Dict[str, float]]:
    selected: List[int] = []
    available = list(range(len(pool_nodes)))
    for _ in range(min(budget, len(pool_nodes))):
        best_idx = None
        best_total = None
        for cand in available:
            if pool_nodes[cand] in candidate_set:
                current_overlap = sum(1 for idx in selected if pool_nodes[idx] in candidate_set)
                if current_overlap + 1 > overlap_cap:
                    continue
            trial = selected + [cand]
            total = objective_breakdown_v3(
                trial, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
                noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
                noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
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
        current_total = objective_breakdown_v3(
            selected, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
            noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
            noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
        )["objective_total"]
        for out_idx in list(selected):
            for in_idx in available:
                if pool_nodes[in_idx] in candidate_set:
                    trial_overlap = sum(1 for idx in selected if pool_nodes[idx] in candidate_set and idx != out_idx)
                    if trial_overlap + 1 > overlap_cap:
                        continue
                trial = [x for x in selected if x != out_idx] + [in_idx]
                trial_total = objective_breakdown_v3(
                    trial, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
                    noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
                    noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
                )["objective_total"]
                if trial_total > current_total + 1e-8:
                    selected = sorted(trial)
                    available = [x for x in range(len(pool_nodes)) if x not in selected]
                    current_total = trial_total
                    improved = True
                    break
            if improved:
                break

    breakdown = objective_breakdown_v3(
        selected, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
        noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
        noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
    )
    return selected, breakdown


def simulated_annealing_finetune_v3(
    initial_selected: List[int],
    pool_nodes: List[str],
    candidate_set: set[str],
    budget: int,
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    noncandidate_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    noncandidate_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
    overlap_anchor_target: int,
    overlap_cap: int,
    weights: Dict[str, float],
    steps: int,
    seed: int,
) -> Tuple[List[int], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    current = sorted(initial_selected)
    current_score = objective_breakdown_v3(
        current, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
        noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
        noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
    )["objective_total"]
    best = list(current)
    best_score = float(current_score)
    pool_size = len(pool_nodes)

    for step in range(max(1, steps)):
        if not current:
            break
        temperature = max(0.02, 1.0 - step / max(steps, 1))
        out_idx = int(rng.choice(current))
        candidates = [x for x in range(pool_size) if x not in current]
        if not candidates:
            break
        in_idx = int(rng.choice(candidates))
        if pool_nodes[in_idx] in candidate_set:
            trial_overlap = sum(1 for idx in current if pool_nodes[idx] in candidate_set and idx != out_idx)
            if trial_overlap + 1 > overlap_cap:
                continue
        trial = sorted([x for x in current if x != out_idx] + [in_idx])
        trial_breakdown = objective_breakdown_v3(
            trial, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
            noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
            noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
        )
        delta = float(trial_breakdown["objective_total"] - current_score)
        accept = delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-6))
        if accept:
            current = trial
            current_score = float(trial_breakdown["objective_total"])
            if current_score > best_score:
                best = list(current)
                best_score = float(current_score)

    best_breakdown = objective_breakdown_v3(
        best, pool_nodes, candidate_set, candidate_to_pool_hops, all_to_pool_hops, key_to_pool_hops,
        noncandidate_to_pool_hops, pair_contrib, rep_similarity, rep_hops, all_weights, key_weights,
        noncandidate_weights, response_matrix_pool, overlap_anchor_target, budget, weights,
    )
    best_breakdown["anneal_steps"] = int(steps)
    return best, best_breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strengthened Chapter-5 generalization-balanced layouts.")
    parser.add_argument("--budgets", default="25")
    parser.add_argument("--compression-ratio", type=float, default=1.45)
    parser.add_argument("--similarity-topology-weight", type=float, default=0.45)
    parser.add_argument("--similarity-response-weight", type=float, default=0.30)
    parser.add_argument("--similarity-sensitivity-weight", type=float, default=0.25)
    parser.add_argument("--topology-tau", type=float, default=3.2)
    parser.add_argument("--strategy-name", default="two_stage_generalization_balanced_layout_v3")
    parser.add_argument("--weight-preset", default="v3", choices=["v3", "v3_1", "v3_2"])
    parser.add_argument("--anneal-steps", type=int, default=180)
    parser.add_argument("--overlap-cap-slack", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-sensitivity-batches", type=int, default=10)
    parser.add_argument(
        "--reference-layout-file",
        default=str(
            SCRIPT_ROOT
            / "chapter5_layout_optimization"
            / "outputs"
            / "layouts"
            / "two_stage_balanced_layout_v1"
            / "monitor_nodes_two_stage_balanced_layout_v1_N25.json"
        ),
    )
    parser.add_argument(
        "--reference-checkpoint",
        default=str(SCRIPT_ROOT / "outputs" / "model_checkpoints" / "best_model_ch5_tsbal25_s42.pth"),
    )
    parser.add_argument(
        "--layout-summary-anchor",
        default=str(SCRIPT_ROOT / "chapter5_layout_optimization" / "outputs" / "layout_summary.csv"),
    )
    args = parser.parse_args()

    chapter5_root = SCRIPT_ROOT / "chapter5_layout_optimization"
    input_dir = SCRIPT_ROOT / "input_1"
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
    parquet_path = SCRIPT_ROOT / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42" / "node_timeseries_with_residuals.parquet"

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

    sensitivity_scores = compute_evaluator_sensitivity(
        checkpoint_path=Path(args.reference_checkpoint),
        reference_layout_file=Path(args.reference_layout_file),
        cache_path=cache_dir / "evaluator_node_sensitivity_ch5_tsbal25_s42.npz",
        refresh_cache=args.refresh_cache,
        max_batches=int(args.max_sensitivity_batches),
    )

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
    sens_pool = sensitivity_scores[np.asarray(pool_indices, dtype=np.int64)]
    sens_diff = np.abs(sens_pool[:, None] - sens_pool[None, :]).astype(np.float32)
    sens_sim = 1.0 - sens_diff
    joint_similarity = (
        float(args.similarity_topology_weight) * topology_sim
        + float(args.similarity_response_weight) * response_sim
        + float(args.similarity_sensitivity_weight) * sens_sim
    ).astype(np.float32)
    joint_similarity = np.clip(joint_similarity, 0.0, 1.0)
    joint_distance = 1.0 - joint_similarity

    base_importance = compute_importance(graph, pool_nodes, response_tensor_pool, candidate_set)
    importance = blended_importance(base_importance, sensitivity_scores, pool_indices)

    structural_key_nodes, structural_key_weights = pick_key_nodes(
        graph=graph,
        pool_nodes=pool_nodes,
        candidate_set=candidate_set,
        top_k=max(12, int(round(0.15 * len(pool_nodes)))),
    )
    key_nodes, key_weights = blended_key_nodes(
        pool_nodes=pool_nodes,
        structural_key_nodes=structural_key_nodes,
        structural_key_weights=structural_key_weights,
        sensitivity_scores=sensitivity_scores,
        node_to_idx=node_to_idx,
        top_k=max(12, int(round(0.15 * len(pool_nodes)))),
    )
    noncandidate_nodes, noncandidate_weights = noncandidate_backbone_nodes(
        graph=graph,
        pool_nodes=pool_nodes,
        candidate_set=candidate_set,
        sensitivity_scores=sensitivity_scores,
        node_to_idx=node_to_idx,
        top_k=max(14, int(round(0.18 * len(pool_nodes)))),
    )
    key_indices = [node_to_idx[n] for n in key_nodes]
    noncandidate_indices = [node_to_idx[n] for n in noncandidate_nodes]

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

    noncandidate_to_pool_hops = np.full((len(noncandidate_nodes), len(pool_nodes)), INF_HOP, dtype=np.int16)
    for ni, non_idx in enumerate(noncandidate_indices):
        for pj, pool_idx in enumerate(pool_indices):
            noncandidate_to_pool_hops[ni, pj] = sym_hop(shortest, non_idx, pool_idx)

    all_weights = 0.52 * _safe_minmax(importance) + 0.48
    pair_contrib_full = precompute_data_driven_pair_contrib(response_tensor_pool)
    weight_cfg = objective_weight_config(args.weight_preset)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    rows: List[Dict[str, float]] = []
    anchor_csv = Path(args.layout_summary_anchor)

    for budget in budgets:
        overlap_target = overlap_anchor(anchor_csv, budget, strategy="degree")
        overlap_cap = overlap_upper_bound(overlap_target, budget)
        if args.overlap_cap_slack is not None:
            overlap_cap = min(int(budget), int(overlap_target) + int(args.overlap_cap_slack))
        rep_budget = min(len(pool_nodes), max(budget, int(math.ceil(float(args.compression_ratio) * budget))))
        medoids, labels = run_kmedoids(joint_distance, importance, rep_budget)
        rep_nodes = [pool_nodes[i] for i in medoids]
        rep_hops = all_to_pool_hops[np.ix_(medoids, medoids)]
        rep_similarity = joint_similarity[np.ix_(medoids, medoids)]
        rep_pair_contrib = pair_contrib_full[medoids]
        rep_candidate_hops = candidate_to_pool_hops[:, medoids]
        rep_all_hops = all_to_pool_hops[:, medoids]
        rep_key_hops = key_to_pool_hops[:, medoids]
        rep_noncandidate_hops = noncandidate_to_pool_hops[:, medoids]
        rep_response_matrix = response_matrix_pool[medoids]

        greedy_selected, _ = greedy_local_v3(
            pool_nodes=rep_nodes,
            budget=budget,
            candidate_set=candidate_set,
            candidate_to_pool_hops=rep_candidate_hops,
            all_to_pool_hops=rep_all_hops,
            key_to_pool_hops=rep_key_hops,
            noncandidate_to_pool_hops=rep_noncandidate_hops,
            pair_contrib=rep_pair_contrib,
            rep_similarity=rep_similarity,
            rep_hops=rep_hops,
            all_weights=all_weights,
            key_weights=key_weights,
            noncandidate_weights=noncandidate_weights,
            response_matrix_pool=rep_response_matrix,
            overlap_anchor_target=overlap_target,
            overlap_cap=overlap_cap,
            weights=weight_cfg,
        )
        selected_local, breakdown = simulated_annealing_finetune_v3(
            initial_selected=greedy_selected,
            pool_nodes=rep_nodes,
            candidate_set=candidate_set,
            budget=budget,
            candidate_to_pool_hops=rep_candidate_hops,
            all_to_pool_hops=rep_all_hops,
            key_to_pool_hops=rep_key_hops,
            noncandidate_to_pool_hops=rep_noncandidate_hops,
            pair_contrib=rep_pair_contrib,
            rep_similarity=rep_similarity,
            rep_hops=rep_hops,
            all_weights=all_weights,
            key_weights=key_weights,
            noncandidate_weights=noncandidate_weights,
            response_matrix_pool=rep_response_matrix,
            overlap_anchor_target=overlap_target,
            overlap_cap=overlap_cap,
            weights=weight_cfg,
            steps=int(args.anneal_steps),
            seed=4200 + budget,
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
        metrics["noncandidate_backbone_count"] = int(len(noncandidate_nodes))
        metrics["overlap_anchor_target"] = int(overlap_target)
        metrics["overlap_cap"] = int(overlap_cap)
        metrics["selected_overlap_count"] = int(len(set(selected_nodes) & candidate_set))
        metrics["sensitivity_mean_selected"] = float(np.mean([sensitivity_scores[node_to_idx[n]] for n in selected_nodes])) if selected_nodes else 0.0
        metrics["weight_preset"] = args.weight_preset

        rep_path = rep_dir / f"representative_pool_{args.strategy_name}_N{budget}.json"
        save_json(
            rep_path,
            {
                "strategy": args.strategy_name,
                "budget": int(budget),
                "compression_ratio": float(args.compression_ratio),
                "representative_pool_nodes": rep_nodes,
                "cluster_labels": {pool_nodes[i]: int(labels[i]) for i in range(len(pool_nodes))},
                "key_nodes": key_nodes,
                "noncandidate_backbone_nodes": noncandidate_nodes,
                "overlap_anchor_target": int(overlap_target),
                "overlap_cap": int(overlap_cap),
                "weight_preset": args.weight_preset,
            },
        )

        layout_path = layouts_dir / f"monitor_nodes_{args.strategy_name}_N{budget}.json"
        save_json(
            layout_path,
            {
                "monitor_nodes": selected_nodes,
                "strategy": args.strategy_name,
                "n": int(budget),
                "representative_pool_file": str(rep_path),
                "protocol": {
                    "candidate_nodes_file": str(candidate_path),
                    "defect_matrix_file": str(defect_csv),
                    "formal_data_dir": str(parquet_path.parent),
                    "note": "control-variable Chapter-5 strengthened main method; monitor layout only",
                },
                "objective_meta": {
                    "stage1": "joint topology-response-sensitivity k-medoids compression",
                    "stage2": "greedy plus local swap plus simulated annealing",
                    "main_additions": [
                        "noncandidate backbone coverage",
                        "candidate-noncandidate coverage balance",
                        "monitor dispersion",
                        "overlap regularization anchored by degree baseline",
                    ],
                    "weight_preset": args.weight_preset,
                    "overlap_anchor_target": int(overlap_target),
                    "overlap_cap": int(overlap_cap),
                    "anneal_steps": int(args.anneal_steps),
                    "selected_representative_pool_size": int(len(rep_nodes)),
                },
                "layout_metrics": metrics,
            },
        )

        row = {
            "strategy": args.strategy_name,
            "budget": int(budget),
            "layout_file": str(layout_path),
            "representative_pool_file": str(rep_path),
        }
        row.update(metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {summary_csv}")
    for row in rows:
        print(
            f"[{row['strategy']} N{row['budget']}] "
            f"obj={row['objective_total']:.4f} "
            f"obs={row['objective_candidate_observability']:.4f} "
            f"noncand={row['objective_noncandidate_backbone']:.4f} "
            f"overlap={row['selected_overlap_count']}/{row['overlap_anchor_target']} "
            f"far={row['far']}"
        )


if __name__ == "__main__":
    main()
