#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 two-stage balanced layouts v2.

Enhancements over v1:
- evaluator-derived node sensitivity prior
- response-signature reconstruction score
- random sensor failure robustness term
- simulated annealing fine-tuning
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_two_stage_balanced_layout import (
    DEFECT_TYPES,
    INF_HOP,
    _safe_minmax,
    build_directed_graph,
    build_topology_similarity,
    compute_importance,
    compute_response_tensor,
    cosine_similarity_matrix,
    greedy_plus_local_swap,
    layout_summary,
    load_json_list,
    pick_key_nodes,
    precompute_data_driven_pair_contrib,
    run_kmedoids,
    save_json,
    signature_nn_stats,
    sym_hop,
)

SCRIPT_ROOT = SCRIPT_PATH.parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.train_privileged_teacher_student import _build_config, _create_model, _make_loaders


def _ensure_adj_2d(adj_matrix: torch.Tensor) -> torch.Tensor:
    if isinstance(adj_matrix, (list, tuple)):
        adj = adj_matrix[0]
    else:
        adj = adj_matrix
    if adj.dim() == 3:
        adj = adj[0]
    return adj


def compute_evaluator_sensitivity(
    checkpoint_path: Path,
    reference_layout_file: Path,
    cache_path: Path,
    refresh_cache: bool,
    max_batches: int,
) -> np.ndarray:
    if cache_path.exists() and not refresh_cache:
        cached = np.load(str(cache_path), allow_pickle=True)
        return cached["sensitivity"].astype(np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _build_config(
        subdir="time_gated_full_ie_v4_formal_conservative420_seed42",
        seed=42,
        monitor_nodes_file=str(reference_layout_file),
        split_mode="scenario",
        defect_matrix_file=str(SCRIPT_ROOT / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"),
    )
    train_loader, val_loader, _test_loader, _dataset = _make_loaders(
        cfg,
        observed_nodes_file=str(reference_layout_file),
    )
    base_ds = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    model = _create_model(cfg, base_ds, device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sensitivity = torch.zeros(len(base_ds.node_list), dtype=torch.float32, device=device)
    used_batches = 0

    for batch in val_loader:
        if used_batches >= max_batches:
            break

        x = batch["features"].to(device)
        x.requires_grad_(True)
        adj = _ensure_adj_2d(batch["adj_matrix"]).to(device)
        with torch.backends.cudnn.flags(enabled=False):
            outputs = model(x, adj)
            logits_node = outputs[0] if isinstance(outputs, tuple) else outputs

            if "target_node_idx" not in batch or "candidate_mask" not in batch:
                continue

            target = batch["target_node_idx"].long().to(device)
            valid_mask = target >= 0
            if "active_label" in batch:
                valid_mask = valid_mask & (batch["active_label"].long().to(device) > 0)
            if "loc_enabled" in batch:
                valid_mask = valid_mask & (batch["loc_enabled"].long().to(device) > 0)

            if valid_mask.sum().item() == 0:
                continue

            batch_idx = torch.arange(target.size(0), device=device)[valid_mask]
            target_score = logits_node[batch_idx, target[valid_mask], 1].sum()

            model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad.zero_()
            target_score.backward()

        grad = x.grad.detach().abs().sum(dim=(0, 1, 3))
        sensitivity += grad
        used_batches += 1

    sensitivity_np = sensitivity.detach().cpu().numpy()
    sensitivity_np = _safe_minmax(sensitivity_np)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), sensitivity=sensitivity_np)
    return sensitivity_np.astype(np.float32)


def blended_importance(
    base_importance: np.ndarray,
    sensitivity_scores: np.ndarray,
    pool_indices: List[int],
) -> np.ndarray:
    sens_pool = sensitivity_scores[np.asarray(pool_indices, dtype=np.int64)]
    return (0.78 * base_importance + 0.22 * sens_pool).astype(np.float32)


def blended_key_nodes(
    pool_nodes: List[str],
    structural_key_nodes: List[str],
    structural_key_weights: np.ndarray,
    sensitivity_scores: np.ndarray,
    node_to_idx: Dict[str, int],
    top_k: int,
) -> Tuple[List[str], np.ndarray]:
    structural_map = {node: float(w) for node, w in zip(structural_key_nodes, structural_key_weights)}
    blended = []
    for node in pool_nodes:
        s_w = structural_map.get(node, 0.0)
        sen = float(sensitivity_scores[node_to_idx[node]])
        blended.append(0.65 * s_w + 0.35 * sen)
    blended = np.asarray(blended, dtype=np.float32)
    order = np.argsort(-blended)[: min(top_k, len(pool_nodes))]
    nodes = [pool_nodes[i] for i in order]
    weights = blended[order]
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float32)
    return nodes, weights.astype(np.float32)


def reconstruction_score(
    response_matrix_pool: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    selected_local: List[int],
) -> Tuple[float, float, float]:
    if not selected_local:
        return 0.0, 1.0, -1.0

    truth = response_matrix_pool.astype(np.float32)
    pred = np.zeros_like(truth)
    selected_set = set(selected_local)

    for i in range(truth.shape[0]):
        if i in selected_set:
            pred[i] = truth[i]
            continue
        weights = []
        vectors = []
        for j in selected_local:
            hop = int(rep_hops[i, j])
            hop_factor = 0.0 if hop >= INF_HOP else math.exp(-float(hop) / 2.5)
            sim_factor = 0.5 + 0.5 * float(rep_similarity[i, j])
            w = hop_factor * sim_factor
            if w <= 1e-8:
                continue
            weights.append(w)
            vectors.append(truth[j])
        if not weights:
            pred[i] = 0.0
        else:
            w_arr = np.asarray(weights, dtype=np.float32)
            v_arr = np.asarray(vectors, dtype=np.float32)
            pred[i] = (w_arr[:, None] * v_arr).sum(axis=0) / max(float(w_arr.sum()), 1e-8)

    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    denom = max(float(np.std(truth)), 1e-8)
    norm_rmse = rmse / denom

    ss_res = float(np.sum((pred - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    recon_score = 0.60 * max(0.0, 1.0 - min(norm_rmse, 1.0)) + 0.40 * max(0.0, min(r2, 1.0))
    return float(recon_score), float(norm_rmse), float(r2)


def objective_core_v2(
    selected_local: List[int],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
) -> Dict[str, float]:
    from build_two_stage_balanced_layout import (
        candidate_observability_score,
        global_coverage_score,
        key_structure_score,
        redundancy_penalty,
        separability_score,
    )

    obs = candidate_observability_score(candidate_to_pool_hops, selected_local)
    sep = separability_score(pair_contrib, selected_local)
    glob = global_coverage_score(all_to_pool_hops, selected_local, all_weights)
    key = key_structure_score(key_to_pool_hops, selected_local, key_weights)
    red = redundancy_penalty(rep_similarity, rep_hops, selected_local)
    recon, recon_rmse, recon_r2 = reconstruction_score(response_matrix_pool, rep_similarity, rep_hops, selected_local)

    total = 0.28 * obs + 0.16 * sep + 0.14 * glob + 0.14 * key + 0.20 * recon - 0.14 * red
    return {
        "core_total": float(total),
        "objective_candidate_observability": float(obs),
        "objective_candidate_separability": float(sep),
        "objective_global_coverage": float(glob),
        "objective_key_structure": float(key),
        "objective_redundancy_penalty": float(red),
        "objective_reconstruction": float(recon),
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


def robustness_score_v2(
    selected_local: List[int],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
) -> Dict[str, float]:
    seed = 7919 + int(sum((i + 1) * x for i, x in enumerate(sorted(selected_local))))
    rng = np.random.default_rng(seed)
    cases = _sample_failure_cases(selected_local, rng)
    scores = []
    for failed in cases:
        remain = [x for x in selected_local if x not in set(failed)]
        if not remain:
            scores.append(0.0)
            continue
        scores.append(
            objective_core_v2(
                remain,
                candidate_to_pool_hops,
                all_to_pool_hops,
                key_to_pool_hops,
                pair_contrib,
                rep_similarity,
                rep_hops,
                all_weights,
                key_weights,
                response_matrix_pool,
            )["core_total"]
        )
    robustness = float(np.mean(scores)) if scores else 0.0
    return {
        "objective_failure_robustness": robustness,
        "robustness_case_count": int(len(cases)),
    }


def objective_breakdown_v2(
    selected_local: List[int],
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
) -> Dict[str, float]:
    core = objective_core_v2(
        selected_local,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        response_matrix_pool,
    )
    robust = robustness_score_v2(
        selected_local,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        response_matrix_pool,
    )
    total = core["core_total"] + 0.22 * robust["objective_failure_robustness"]
    out = dict(core)
    out.update(robust)
    out["objective_total"] = float(total)
    return out


def greedy_local_v2(
    pool_nodes: List[str],
    budget: int,
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
) -> Tuple[List[int], Dict[str, float]]:
    selected: List[int] = []
    available = list(range(len(pool_nodes)))

    for _ in range(min(budget, len(pool_nodes))):
        best_idx = None
        best_total = None
        for cand in available:
            trial = selected + [cand]
            total = objective_breakdown_v2(
                trial,
                candidate_to_pool_hops,
                all_to_pool_hops,
                key_to_pool_hops,
                pair_contrib,
                rep_similarity,
                rep_hops,
                all_weights,
                key_weights,
                response_matrix_pool,
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
        current_total = objective_breakdown_v2(
            selected,
            candidate_to_pool_hops,
            all_to_pool_hops,
            key_to_pool_hops,
            pair_contrib,
            rep_similarity,
            rep_hops,
            all_weights,
            key_weights,
            response_matrix_pool,
        )["objective_total"]
        for out_idx in list(selected):
            for in_idx in available:
                trial = [x for x in selected if x != out_idx] + [in_idx]
                trial_total = objective_breakdown_v2(
                    trial,
                    candidate_to_pool_hops,
                    all_to_pool_hops,
                    key_to_pool_hops,
                    pair_contrib,
                    rep_similarity,
                    rep_hops,
                    all_weights,
                    key_weights,
                    response_matrix_pool,
                )["objective_total"]
                if trial_total > current_total + 1e-8:
                    selected = sorted(trial)
                    available = [x for x in range(len(pool_nodes)) if x not in selected]
                    current_total = trial_total
                    improved = True
                    break
            if improved:
                break

    final_breakdown = objective_breakdown_v2(
        selected,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        response_matrix_pool,
    )
    return selected, final_breakdown


def simulated_annealing_finetune(
    initial_selected: List[int],
    budget: int,
    pool_size: int,
    candidate_to_pool_hops: np.ndarray,
    all_to_pool_hops: np.ndarray,
    key_to_pool_hops: np.ndarray,
    pair_contrib: np.ndarray,
    rep_similarity: np.ndarray,
    rep_hops: np.ndarray,
    all_weights: np.ndarray,
    key_weights: np.ndarray,
    response_matrix_pool: np.ndarray,
    steps: int,
    seed: int,
) -> Tuple[List[int], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    current = sorted(initial_selected)
    current_score = objective_breakdown_v2(
        current,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        response_matrix_pool,
    )["objective_total"]
    best = list(current)
    best_score = current_score

    full_set = list(range(pool_size))
    for step in range(max(steps, 1)):
        temp = 0.08 * (0.985 ** step) + 0.003
        out_idx = int(rng.choice(current))
        available = [x for x in full_set if x not in current]
        if not available:
            break
        in_idx = int(rng.choice(available))
        trial = sorted([x for x in current if x != out_idx] + [in_idx])
        trial_breakdown = objective_breakdown_v2(
            trial,
            candidate_to_pool_hops,
            all_to_pool_hops,
            key_to_pool_hops,
            pair_contrib,
            rep_similarity,
            rep_hops,
            all_weights,
            key_weights,
            response_matrix_pool,
        )
        delta = float(trial_breakdown["objective_total"] - current_score)
        accept = delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-6))
        if accept:
            current = trial
            current_score = float(trial_breakdown["objective_total"])
            if current_score > best_score:
                best = list(current)
                best_score = current_score

    best_breakdown = objective_breakdown_v2(
        best,
        candidate_to_pool_hops,
        all_to_pool_hops,
        key_to_pool_hops,
        pair_contrib,
        rep_similarity,
        rep_hops,
        all_weights,
        key_weights,
        response_matrix_pool,
    )
    best_breakdown["anneal_steps"] = int(steps)
    return best, best_breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-stage balanced layouts v2.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument("--compression-ratio", type=float, default=1.4)
    parser.add_argument("--similarity-topology-weight", type=float, default=0.50)
    parser.add_argument("--similarity-response-weight", type=float, default=0.40)
    parser.add_argument("--similarity-sensitivity-weight", type=float, default=0.10)
    parser.add_argument("--topology-tau", type=float, default=3.0)
    parser.add_argument("--strategy-name", default="two_stage_balanced_layout_v2")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-sensitivity-batches", type=int, default=8)
    parser.add_argument("--anneal-steps", type=int, default=240)
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

    all_weights = 0.55 * _safe_minmax(importance) + 0.45
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
        rep_response_matrix = response_matrix_pool[medoids]

        greedy_selected, _greedy_breakdown = greedy_local_v2(
            pool_nodes=rep_nodes,
            budget=budget,
            candidate_to_pool_hops=rep_candidate_hops,
            all_to_pool_hops=rep_all_hops,
            key_to_pool_hops=rep_key_hops,
            pair_contrib=rep_pair_contrib,
            rep_similarity=rep_similarity,
            rep_hops=rep_hops,
            all_weights=rep_all_weights,
            key_weights=key_weights,
            response_matrix_pool=rep_response_matrix,
        )
        selected_local, breakdown = simulated_annealing_finetune(
            initial_selected=greedy_selected,
            budget=budget,
            pool_size=len(rep_nodes),
            candidate_to_pool_hops=rep_candidate_hops,
            all_to_pool_hops=rep_all_hops,
            key_to_pool_hops=rep_key_hops,
            pair_contrib=rep_pair_contrib,
            rep_similarity=rep_similarity,
            rep_hops=rep_hops,
            all_weights=rep_all_weights,
            key_weights=key_weights,
            response_matrix_pool=rep_response_matrix,
            steps=int(args.anneal_steps),
            seed=42 + budget,
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
        metrics["sensitivity_mean_selected"] = float(np.mean([sensitivity_scores[node_to_idx[n]] for n in selected_nodes])) if selected_nodes else 0.0

        rep_payload = {
            "strategy": args.strategy_name,
            "budget": int(budget),
            "compression_ratio": float(args.compression_ratio),
            "representative_pool_nodes": rep_nodes,
            "cluster_labels": {pool_nodes[i]: int(labels[i]) for i in range(len(pool_nodes))},
            "similarity_weights": {
                "topology": float(args.similarity_topology_weight),
                "response": float(args.similarity_response_weight),
                "sensitivity": float(args.similarity_sensitivity_weight),
            },
            "key_nodes": key_nodes,
            "reference_checkpoint": str(args.reference_checkpoint),
            "reference_layout_file": str(args.reference_layout_file),
        }
        rep_path = rep_dir / f"representative_pool_{args.strategy_name}_N{budget}.json"
        save_json(rep_path, rep_payload)

        layout_payload = {
            "monitor_nodes": selected_nodes,
            "strategy": args.strategy_name,
            "n": int(budget),
            "representative_pool_file": str(rep_path),
            "protocol": {
                "candidate_nodes_file": str(candidate_path),
                "defect_matrix_file": str(defect_csv),
                "formal_data_dir": str(parquet_path.parent),
                "note": "control-variable Chapter-5 layout study; only monitor layout changes",
            },
            "objective_meta": {
                "stage1": "joint topology-response-sensitivity k-medoids compression",
                "stage2": "greedy plus local swap plus simulated annealing",
                "compression_ratio": float(args.compression_ratio),
                "similarity_topology_weight": float(args.similarity_topology_weight),
                "similarity_response_weight": float(args.similarity_response_weight),
                "similarity_sensitivity_weight": float(args.similarity_sensitivity_weight),
                "topology_tau": float(args.topology_tau),
                "anneal_steps": int(args.anneal_steps),
                "selected_representative_pool_size": int(len(rep_nodes)),
                "reference_checkpoint": str(args.reference_checkpoint),
            },
            "layout_metrics": metrics,
        }
        layout_path = layouts_dir / f"monitor_nodes_{args.strategy_name}_N{budget}.json"
        save_json(layout_path, layout_payload)

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
            f"recon={row['objective_reconstruction']:.4f} "
            f"robust={row['objective_failure_robustness']:.4f} "
            f"far={row['far']}"
        )


if __name__ == "__main__":
    main()
