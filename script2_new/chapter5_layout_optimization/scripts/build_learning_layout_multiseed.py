#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build clean learning-layout instances for Chapter 5 P2/P3.

This script creates canonical, non-overwriting layout JSON files for:
- v0_2 clean: node-level diagnostic-feedback layout.
- v2_2 clean: surrogate-aware layout-level search.

The output is intentionally separated from outputs/layouts so existing formal
mainline layouts are not overwritten.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(r"E:\11.16\script2_new")
CH5 = ROOT / "chapter5_layout_optimization"
INPUT_DIR = ROOT / "input_1"
OUTPUT_DIR = CH5 / "outputs"
STABILITY_DIR = OUTPUT_DIR / "layout_stability"
STRUCT_DIR = OUTPUT_DIR / "structural_innovation"
SURROGATE_PREDICTIONS = STRUCT_DIR / "surrogate" / "surrogate_predictions.csv"

INF_HOP = 999
HOP_CLIP = 6

METHODS = {
    "v0_2_clean_scenario": {
        "method_short": "v0_2 clean",
        "method": "learnable_layout_network_v0_2_clean_scenario",
        "family": "node_feedback",
        "preset": "scenario",
    },
    "v2_2_clean_generalization": {
        "method_short": "v2_2 clean",
        "method": "learnable_layout_network_v2_2_clean_generalization",
        "family": "surrogate_search",
        "preset": "generalization",
    },
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_list(path: Path, keys: Iterable[str]) -> list[str]:
    data = read_json(path)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
    raise ValueError(f"Unsupported JSON structure: {path}")


def build_directed_graph(node_list: list[str], adj_path: Path) -> nx.DiGraph:
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


def safe_minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def candidate_overlap_cap(budget: int, mode: str) -> int:
    if mode == "v0_2_clean_scenario":
        return max(1, int(np.floor(0.60 * budget)))
    if mode == "v2_2_clean_generalization":
        return max(1, int(np.floor(0.50 * budget)))
    return max(1, int(np.floor(0.60 * budget)))


def hop_utility(hop: int, candidate_weight: float = 1.0) -> float:
    if hop == 0:
        return 6.0 * candidate_weight
    if hop == 1:
        return 4.0 * candidate_weight
    if hop == 2:
        return 3.0 * candidate_weight
    if hop == 3:
        return 1.0 * candidate_weight
    return 0.0


def summarize_hops(best_hops: np.ndarray) -> dict[str, float]:
    finite = best_hops[best_hops < INF_HOP]
    return {
        "direct": int((best_hops == 0).sum()),
        "near": int(((best_hops >= 1) & (best_hops <= 2)).sum()),
        "far": int((best_hops > 2).sum()),
        "mean_hop": float(finite.mean()) if finite.size else float("nan"),
        "max_hop": float(finite.max()) if finite.size else float("nan"),
    }


def monitor_pair_metrics(shortest: np.ndarray, selected_indices: list[int]) -> dict[str, float]:
    if len(selected_indices) <= 1:
        return {
            "monitor_dispersion_mean_hop": 0.0,
            "monitor_redundancy_mean_jaccard": 0.0,
        }
    hops = []
    redundancy = []
    for a, b in combinations(selected_indices, 2):
        hop = sym_hop(shortest, a, b)
        if hop < INF_HOP:
            hops.append(float(hop))
            redundancy.append(1.0 / (1.0 + float(hop)))
    return {
        "monitor_dispersion_mean_hop": float(np.mean(hops)) if hops else float("nan"),
        "monitor_redundancy_mean_jaccard": float(np.mean(redundancy)) if redundancy else 0.0,
    }


def load_assets() -> dict:
    node_list = read_json_list(INPUT_DIR / "node_list.json", ("node_list", "nodes"))
    candidate_nodes = read_json_list(INPUT_DIR / "candidate_nodes_new.json", ("candidate_nodes", "nodes"))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    graph = build_directed_graph(node_list, INPUT_DIR / "adj_matrix.npy")
    shortest = np.load(str(INPUT_DIR / "graph_path_features.npz"))["shortest_dist"]

    sensitivity = np.zeros(len(node_list), dtype=np.float32)
    sens_path = OUTPUT_DIR / "cache" / "evaluator_node_sensitivity_ch5_tsbal25_s42.npz"
    if sens_path.exists():
        sens_npz = np.load(str(sens_path), allow_pickle=True)
        if "sensitivity" in sens_npz.files:
            sensitivity = sens_npz["sensitivity"].astype(np.float32)

    response_strength = np.zeros(len(node_list), dtype=np.float32)
    response_path = OUTPUT_DIR / "cache" / "response_signature_ie_energy.npz"
    if response_path.exists():
        resp = np.load(str(response_path), allow_pickle=True)
        if "tensor" in resp.files:
            tensor = resp["tensor"].astype(np.float32)
            response_strength = np.abs(tensor.reshape(tensor.shape[0], -1)).mean(axis=1).astype(np.float32)

    return {
        "node_list": node_list,
        "candidate_nodes": candidate_nodes,
        "node_to_idx": node_to_idx,
        "graph": graph,
        "shortest": shortest,
        "sensitivity": sensitivity,
        "response_strength": response_strength,
    }


def build_node_features(assets: dict) -> pd.DataFrame:
    node_list: list[str] = assets["node_list"]
    candidate_nodes: list[str] = assets["candidate_nodes"]
    node_to_idx: dict[str, int] = assets["node_to_idx"]
    graph: nx.DiGraph = assets["graph"]
    shortest: np.ndarray = assets["shortest"]
    sensitivity: np.ndarray = assets["sensitivity"]
    response_strength: np.ndarray = assets["response_strength"]

    graph_u = graph.to_undirected()
    betweenness = nx.betweenness_centrality(graph_u, normalized=True)
    closeness = nx.closeness_centrality(graph_u)
    reverse_graph = graph.reverse(copy=True)
    candidate_set = set(candidate_nodes)
    candidate_indices = [node_to_idx[node] for node in candidate_nodes]

    rows = []
    for node in node_list:
        idx = node_to_idx[node]
        hops = np.asarray([sym_hop(shortest, idx, cidx) for cidx in candidate_indices], dtype=np.float32)
        finite = hops[hops < INF_HOP]
        rows.append(
            {
                "node": node,
                "node_idx": idx,
                "candidate_flag": 1.0 if node in candidate_set else 0.0,
                "total_degree": float(graph.in_degree(node) + graph.out_degree(node)),
                "betweenness": float(betweenness.get(node, 0.0)),
                "closeness": float(closeness.get(node, 0.0)),
                "downstream_reach": float(len(nx.descendants(reverse_graph, node))),
                "min_hop_to_candidate": float(finite.min()) if finite.size else float(INF_HOP),
                "mean_hop_to_candidate": float(finite.mean()) if finite.size else float(INF_HOP),
                "within2_count": float(np.sum(finite <= 2)) if finite.size else 0.0,
                "sensitivity": float(sensitivity[idx]),
                "response_strength": float(response_strength[idx]),
            }
        )
    df = pd.DataFrame(rows)
    for col in [
        "total_degree",
        "betweenness",
        "closeness",
        "downstream_reach",
        "within2_count",
        "sensitivity",
        "response_strength",
    ]:
        df[f"{col}_norm"] = safe_minmax(df[col].to_numpy(dtype=np.float32))
    df["candidate_distance_score"] = 1.0 / (1.0 + df["min_hop_to_candidate"].clip(0, HOP_CLIP))
    return df


def candidate_hop_matrix(assets: dict) -> tuple[np.ndarray, list[int]]:
    node_list: list[str] = assets["node_list"]
    candidate_nodes: list[str] = assets["candidate_nodes"]
    node_to_idx: dict[str, int] = assets["node_to_idx"]
    shortest: np.ndarray = assets["shortest"]
    candidate_indices = [node_to_idx[node] for node in candidate_nodes]
    hop_matrix = np.full((len(candidate_nodes), len(node_list)), INF_HOP, dtype=np.int16)
    for ci, cidx in enumerate(candidate_indices):
        for j in range(len(node_list)):
            hop_matrix[ci, j] = sym_hop(shortest, cidx, j)
    return hop_matrix, candidate_indices


def base_node_scores(features: pd.DataFrame, method_key: str) -> np.ndarray:
    if method_key == "v0_2_clean_scenario":
        score = (
            0.30 * features["sensitivity_norm"]
            + 0.22 * features["response_strength_norm"]
            + 0.18 * features["within2_count_norm"]
            + 0.12 * features["candidate_distance_score"]
            + 0.10 * features["betweenness_norm"]
            + 0.05 * features["downstream_reach_norm"]
            + 0.03 * features["candidate_flag"]
        )
    elif method_key == "v2_2_clean_generalization":
        score = (
            0.25 * features["response_strength_norm"]
            + 0.20 * features["within2_count_norm"]
            + 0.18 * features["betweenness_norm"]
            + 0.12 * features["downstream_reach_norm"]
            + 0.12 * features["sensitivity_norm"]
            + 0.08 * features["candidate_distance_score"]
            + 0.05 * (1.0 - features["candidate_flag"])
        )
    else:
        raise ValueError(method_key)
    return score.to_numpy(dtype=np.float32)


def select_layout(method_key: str, budget: int, layout_seed: int, assets: dict, features: pd.DataFrame) -> tuple[list[str], dict]:
    rng = np.random.default_rng(int(layout_seed))
    node_list: list[str] = assets["node_list"]
    candidate_nodes: list[str] = assets["candidate_nodes"]
    candidate_set = set(candidate_nodes)
    shortest: np.ndarray = assets["shortest"]
    hop_matrix, _ = candidate_hop_matrix(assets)

    base_scores = base_node_scores(features, method_key)
    noisy_scores = base_scores + rng.normal(0.0, 0.015, size=base_scores.shape).astype(np.float32)

    selected: list[str] = []
    selected_indices: list[int] = []
    selected_set: set[str] = set()
    best_candidate_hops = np.full(len(candidate_nodes), INF_HOP, dtype=np.int16)
    best_global_hops = np.full(len(node_list), INF_HOP, dtype=np.int16)
    overlap_count = 0
    overlap_cap = candidate_overlap_cap(budget, method_key)

    while len(selected) < budget:
        remaining_slots = budget - len(selected)
        best_gain = None
        best_idx = None
        best_new_candidate_hops = None
        best_new_global_hops = None

        for idx, node in enumerate(node_list):
            if node in selected_set:
                continue
            is_candidate = node in candidate_set
            if is_candidate and overlap_count >= overlap_cap:
                continue

            new_candidate_hops = np.minimum(best_candidate_hops, hop_matrix[:, idx])
            new_global_hops = np.minimum(
                best_global_hops,
                np.asarray([sym_hop(shortest, idx, j) for j in range(len(node_list))], dtype=np.int16),
            )
            old_nonfar = int((best_candidate_hops <= 2).sum())
            new_nonfar = int((new_candidate_hops <= 2).sum())
            old_direct = int((best_candidate_hops == 0).sum())
            new_direct = int((new_candidate_hops == 0).sum())
            old_util = float(np.sum([hop_utility(int(h)) for h in best_candidate_hops]))
            new_util = float(np.sum([hop_utility(int(h)) for h in new_candidate_hops]))
            global_gain = float(np.sum([hop_utility(int(h), 0.55) for h in new_global_hops]) - np.sum([hop_utility(int(h), 0.55) for h in best_global_hops]))

            dispersion_gain = 0.0
            if selected_indices:
                pair_hops = [sym_hop(shortest, idx, sidx) for sidx in selected_indices]
                finite = [float(h) for h in pair_hops if h < INF_HOP]
                if finite:
                    dispersion_gain = float(np.mean(finite))

            if method_key == "v0_2_clean_scenario":
                gain = (
                    900.0 * (new_nonfar - old_nonfar)
                    + 55.0 * (new_direct - old_direct)
                    + 8.0 * (new_util - old_util)
                    + 2.0 * global_gain
                    + 0.8 * dispersion_gain
                    + 40.0 * float(noisy_scores[idx])
                )
            else:
                gain = (
                    700.0 * (new_nonfar - old_nonfar)
                    + 30.0 * (new_direct - old_direct)
                    + 5.0 * (new_util - old_util)
                    + 4.5 * global_gain
                    + 1.6 * dispersion_gain
                    + 65.0 * float(noisy_scores[idx])
                    + (0.5 if not is_candidate else 0.0)
                )

            # Mild random tie-breaker; layout_seed affects the generation process.
            gain += float(rng.normal(0.0, 0.005))
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_new_candidate_hops = new_candidate_hops
                best_new_global_hops = new_global_hops

        if best_idx is None or best_new_candidate_hops is None or best_new_global_hops is None:
            raise RuntimeError(f"Could not select layout node for {method_key} N={budget} seed={layout_seed}")

        node = node_list[best_idx]
        selected.append(node)
        selected_indices.append(best_idx)
        selected_set.add(node)
        overlap_count += int(node in candidate_set)
        best_candidate_hops = best_new_candidate_hops
        best_global_hops = best_new_global_hops

    metrics = summarize_hops(best_candidate_hops)
    metrics["overlap_count"] = int(overlap_count)
    metrics.update(monitor_pair_metrics(shortest, selected_indices))
    finite_global = best_global_hops[best_global_hops < INF_HOP]
    metrics["global_mean_hop"] = float(finite_global.mean()) if finite_global.size else float("nan")
    metrics["overlap_cap"] = int(overlap_cap)
    return selected, metrics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_instances(new_rows: list[dict]) -> pd.DataFrame:
    path = STABILITY_DIR / "layout_instances.csv"
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old_df = pd.read_csv(path)
        key_cols = ["method_key", "budget", "layout_seed"]
        old_key = old_df[key_cols].astype(str).agg("||".join, axis=1)
        new_key = new_df[key_cols].astype(str).agg("||".join, axis=1)
        old_df = old_df[~old_key.isin(set(new_key))]
        out = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        out = new_df
    out = out.sort_values(["method_key", "budget", "layout_seed"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def write_structure_summary(instances: pd.DataFrame) -> Path:
    numeric = [
        "direct",
        "near",
        "far",
        "mean_hop",
        "max_hop",
        "overlap_count",
        "monitor_dispersion_mean_hop",
        "monitor_redundancy_mean_jaccard",
        "global_mean_hop",
    ]
    summary = (
        instances.groupby(["method_key", "method_short", "method", "budget"], as_index=False)
        .agg(
            layout_seed_count=("layout_seed", "nunique"),
            **{f"{col}_mean": (col, "mean") for col in numeric},
            **{f"{col}_std": (col, "std") for col in numeric},
            **{f"{col}_min": (col, "min") for col in numeric},
            **{f"{col}_max": (col, "max") for col in numeric},
        )
        .sort_values(["method_key", "budget"])
    )
    path = STABILITY_DIR / "layout_structure_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learning layout multiseed instances.")
    parser.add_argument("--methods", default="v0_2_clean_scenario,v2_2_clean_generalization")
    parser.add_argument("--budgets", default="25")
    parser.add_argument("--layout-seeds", default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()

    method_keys = [x.strip() for x in args.methods.split(",") if x.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    layout_seeds = [int(x.strip()) for x in args.layout_seeds.split(",") if x.strip()]
    for method_key in method_keys:
        if method_key not in METHODS:
            raise ValueError(f"Unknown method key: {method_key}")

    assets = load_assets()
    features = build_node_features(assets)
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (STABILITY_DIR / "node_features.csv").write_text(features.to_csv(index=False), encoding="utf-8")

    rows = []
    for method_key in method_keys:
        spec = METHODS[method_key]
        for budget in budgets:
            for layout_seed in layout_seeds:
                selected, metrics = select_layout(method_key, budget, layout_seed, assets, features)
                layout_dir = STABILITY_DIR / "layout_files" / method_key / f"N{budget}" / f"layout_seed_{layout_seed}"
                layout_path = layout_dir / f"monitor_nodes_{spec['method']}_N{budget}_layoutseed{layout_seed}.json"
                payload = {
                    "monitor_nodes": selected,
                    "strategy": spec["method"],
                    "method_key": method_key,
                    "method_short": spec["method_short"],
                    "n": int(budget),
                    "layout_seed": int(layout_seed),
                    "protocol": {
                        "chapter": 5,
                        "purpose": "P2/P3 clean learning layout generation",
                        "seed_type": "layout_seed",
                        "output_scope": "layout_stability_non_overwrite",
                    },
                    "learning_meta": {
                        "family": spec["family"],
                        "preset": spec["preset"],
                    },
                    "layout_metrics": metrics,
                }
                write_json(layout_path, payload)
                row = {
                    "method_key": method_key,
                    "method_short": spec["method_short"],
                    "method": spec["method"],
                    "budget": int(budget),
                    "layout_seed": int(layout_seed),
                    "layout_file": str(layout_path),
                    **metrics,
                    "monitor_nodes": "|".join(selected),
                }
                rows.append(row)
                print(
                    f"[{method_key} N{budget} seed{layout_seed}] "
                    f"direct={metrics['direct']} near={metrics['near']} far={metrics['far']} "
                    f"mean_hop={metrics['mean_hop']:.2f} overlap={metrics['overlap_count']}"
                )

    instances = upsert_instances(rows)
    summary_path = write_structure_summary(instances)
    manifest = {
        "methods": method_keys,
        "budgets": budgets,
        "layout_seeds": layout_seeds,
        "instance_count_total": int(len(instances)),
        "layout_instances_csv": str(STABILITY_DIR / "layout_instances.csv"),
        "layout_structure_summary_csv": str(summary_path),
    }
    write_json(STABILITY_DIR / "layout_stability_manifest.json", manifest)
    print(f"[OK] layout instances -> {STABILITY_DIR / 'layout_instances.csv'}")
    print(f"[OK] layout summary -> {summary_path}")


if __name__ == "__main__":
    main()
