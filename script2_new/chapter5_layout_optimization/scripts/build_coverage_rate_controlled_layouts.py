#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 coverage-rate controlled monitoring layouts.

Scientific question:
    If the diagnosis model and formal evaluation protocol are fixed, does
    localization performance change monotonically with defect-node structural
    coverage?

This script only creates monitor layouts and an evaluation manifest. It does
not train models. Run run_coverage_rate_controlled_eval.py after reviewing the
generated manifest.
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
LEVEL_TARGETS = {
    "low": 0.35,
    "mid": 0.55,
    "high": 0.75,
    "very_high": 0.90,
}


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


def build_hop_matrix(
    shortest: np.ndarray,
    node_list: Sequence[str],
    candidate_nodes: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    hop_matrix = np.full((len(candidate_nodes), len(node_list)), INF_HOP, dtype=np.int16)
    for ci, candidate in enumerate(candidate_nodes):
        src_idx = node_to_idx[candidate]
        for j in range(len(node_list)):
            hop_matrix[ci, j] = sym_hop(shortest, src_idx, j)
    return hop_matrix, node_to_idx


def degree_score(graph: nx.DiGraph, nodes: Sequence[str]) -> Dict[str, float]:
    graph_u = graph.to_undirected()
    bc = nx.betweenness_centrality(graph_u, normalized=True)
    degree = {node: graph.in_degree(node) + graph.out_degree(node) for node in graph.nodes()}
    raw = {
        node: 0.6 * float(degree.get(node, 0.0)) + 0.4 * float(bc.get(node, 0.0))
        for node in nodes
    }
    max_score = max(raw.values()) if raw else 1.0
    return {node: value / max_score for node, value in raw.items()}


def summarize_candidate_hops(best_hops: np.ndarray) -> Dict[str, float]:
    finite = best_hops[best_hops < INF_HOP]
    direct = int((best_hops == 0).sum())
    near = int(((best_hops >= 1) & (best_hops <= 2)).sum())
    far = int((best_hops > 2).sum())
    total = int(best_hops.size)
    return {
        "direct": direct,
        "near": near,
        "far": far,
        "coverage_rate": float((direct + near) / total) if total else float("nan"),
        "direct_rate": float(direct / total) if total else float("nan"),
        "far_ratio": float(far / total) if total else float("nan"),
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


def select_layout_for_target(
    pool_nodes: List[str],
    pool_indices: List[int],
    hop_matrix: np.ndarray,
    shortest: np.ndarray,
    centrality: Dict[str, float],
    budget: int,
    target_coverage: float,
    rng: np.random.Generator,
    attempts: int,
    forbidden_signatures: set[tuple[str, ...]] | None = None,
) -> Tuple[List[str], Dict[str, float]]:
    candidates = []
    forbidden_signatures = forbidden_signatures or set()

    for _ in range(attempts):
        selected: List[str] = []
        selected_indices: List[int] = []
        selected_set = set()
        best_hops = np.full(hop_matrix.shape[0], INF_HOP, dtype=np.int16)

        while len(selected) < budget:
            best_score = -1e18
            best_node = None
            best_idx = None
            best_new_hops = None

            order = rng.permutation(len(pool_nodes))
            for local_pos in order:
                node = pool_nodes[int(local_pos)]
                if node in selected_set:
                    continue
                node_idx = pool_indices[int(local_pos)]
                new_hops = np.minimum(best_hops, hop_matrix[:, node_idx])
                metrics = summarize_candidate_hops(new_hops)
                coverage_error = abs(metrics["coverage_rate"] - target_coverage)
                nonfar_gain = int((new_hops <= 2).sum()) - int((best_hops <= 2).sum())
                direct_gain = int((new_hops == 0).sum()) - int((best_hops == 0).sum())

                dispersion_gain = 0.0
                if selected_indices:
                    hops = [sym_hop(shortest, node_idx, idx) for idx in selected_indices]
                    finite = [float(hop) for hop in hops if hop < INF_HOP]
                    dispersion_gain = float(np.mean(finite)) if finite else 0.0

                if target_coverage <= 0.45:
                    coverage_direction = -1.0
                elif target_coverage >= 0.70:
                    coverage_direction = 1.0
                else:
                    coverage_direction = 0.0

                # Target coverage dominates; centrality and dispersion avoid odd isolated layouts.
                # Low-coverage probes intentionally avoid direct/near candidate coverage.
                score = (
                    -1400.0 * coverage_error
                    + coverage_direction * 18.0 * nonfar_gain
                    + coverage_direction * 2.0 * direct_gain
                    + 3.5 * float(centrality.get(node, 0.0))
                    + 0.45 * dispersion_gain
                    + float(rng.normal(0.0, 0.15))
                )
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_idx = node_idx
                    best_new_hops = new_hops

            if best_node is None or best_idx is None or best_new_hops is None:
                raise RuntimeError("Could not select a valid monitor node.")
            selected.append(best_node)
            selected_indices.append(best_idx)
            selected_set.add(best_node)
            best_hops = best_new_hops

        metrics = summarize_candidate_hops(best_hops)
        metrics.update(monitor_pair_metrics(shortest, selected_indices))
        final_error = abs(metrics["coverage_rate"] - target_coverage)
        objective = final_error - 0.001 * metrics["monitor_dispersion_mean_hop"]
        signature = tuple(sorted(selected))
        candidates.append((objective, signature, selected, metrics))

    candidates.sort(key=lambda item: item[0])
    for _, signature, selected, metrics in candidates:
        if signature not in forbidden_signatures:
            return selected, metrics
    _, _, selected, metrics = candidates[0]
    return selected, metrics


def build_command(layout_file: Path, tag: str) -> str:
    parts = [
        "D:/conda3/envs/swmm_gpu/python.exe",
        "E:/11.16/script2_new/scripts/train_privileged_teacher_student.py",
        "--student-monitors",
        str(layout_file).replace("\\", "/"),
        "--output-tag",
        tag,
        "--seed",
        "42",
        "--teacher-subdir",
        "ie420_plus_normal20_v1",
        "--split-mode",
        "scenario",
        "--feature-set",
        "raw_plus_residual",
        "--lambda-loc",
        "0.5",
        "--student-model-type",
        "hydraulic_inverse_deepattn",
        "--lambda-kd",
        "0",
        "--lambda-active-kd",
        "0",
    ]
    return " ".join(parts)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ch5_root = repo_root / "script2_new" / "chapter5_layout_optimization"
    input_dir = repo_root / "script2_new" / "input_1"

    parser = argparse.ArgumentParser(description="Build coverage-rate controlled Chapter-5 layouts.")
    parser.add_argument("--budgets", nargs="+", type=int, default=[25])
    parser.add_argument("--levels", nargs="+", choices=list(LEVEL_TARGETS), default=list(LEVEL_TARGETS))
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    node_list = load_json_list(input_dir / "node_list.json", ("node_list",))
    candidate_nodes = load_json_list(input_dir / "candidate_nodes_new.json", ("candidate_nodes",))
    graph = build_directed_graph(node_list, input_dir / "adj_matrix.npy")
    shortest = shortest_array(graph, node_list)
    hop_matrix, node_to_idx = build_hop_matrix(shortest, node_list, candidate_nodes)

    exclude = {node for node in graph.nodes() if graph.in_degree(node) + graph.out_degree(node) == 0}
    pool_nodes = [node for node in node_list if node not in exclude]
    pool_indices = [node_to_idx[node] for node in pool_nodes]
    centrality = degree_score(graph, pool_nodes)

    out_base = ch5_root / "outputs" / "layouts" / "coverage_rate_controlled"
    summary_dir = ch5_root / "outputs" / "coverage_rate_controlled"
    summary_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = repo_root / "script2_new" / "outputs" / "reports"

    rng_master = np.random.default_rng(args.seed)
    summary_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    used_signatures: Dict[tuple[int, str], set[tuple[str, ...]]] = {}

    for budget in args.budgets:
        for level in args.levels:
            target = LEVEL_TARGETS[level]
            for variant in range(1, args.variants + 1):
                variant_seed = int(rng_master.integers(1, 2_000_000_000))
                rng = np.random.default_rng(variant_seed)
                selected, metrics = select_layout_for_target(
                    pool_nodes=pool_nodes,
                    pool_indices=pool_indices,
                    hop_matrix=hop_matrix,
                    shortest=shortest,
                    centrality=centrality,
                    budget=budget,
                    target_coverage=target,
                    rng=rng,
                    attempts=args.attempts,
                    forbidden_signatures=used_signatures.setdefault((budget, level), set()),
                )
                used_signatures[(budget, level)].add(tuple(sorted(selected)))

                tag = (
                    f"ch5_coverage_rate_controlled_N{budget}_{level}_v{variant}"
                    f"_normal20_rawres_loc0p5_s42"
                )
                out_dir = out_base / f"N{budget}" / level
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"monitor_nodes_coverage_rate_N{budget}_{level}_v{variant}.json"

                payload = {
                    "monitor_nodes": selected,
                    "strategy": "coverage_rate_controlled",
                    "n": int(budget),
                    "coverage_level": level,
                    "variant": int(variant),
                    "target_coverage_rate": float(target),
                    "selection_seed": int(variant_seed),
                    "protocol": {
                        "teacher_subdir": "ie420_plus_normal20_v1",
                        "split_mode": "scenario",
                        "feature_set": "raw_plus_residual",
                        "student_model_type": "hydraulic_inverse_deepattn",
                        "lambda_loc": 0.5,
                        "candidate_nodes_file": str(input_dir / "candidate_nodes_new.json"),
                        "defect_matrix_file": str(
                            input_dir / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"
                        ),
                    },
                    "layout_metrics": metrics,
                }
                out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                row = {
                    "strategy": "coverage_rate_controlled",
                    "budget": int(budget),
                    "coverage_level": level,
                    "variant": int(variant),
                    "target_coverage_rate": float(target),
                    "layout_file": str(out_file),
                    "output_tag": tag,
                    "metrics_file": str(reports_dir / f"last_run_metrics_{tag}.json"),
                }
                row.update(metrics)
                summary_rows.append(row)

                manifest_row = dict(row)
                manifest_row["command"] = build_command(out_file, tag)
                manifest_rows.append(manifest_row)

    summary_df = pd.DataFrame(summary_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    summary_path = summary_dir / "coverage_rate_controlled_layout_summary.csv"
    manifest_path = summary_dir / "coverage_rate_controlled_run_manifest.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {manifest_path}")
    print(summary_df[["budget", "coverage_level", "variant", "coverage_rate", "far_ratio", "mean_hop"]])


if __name__ == "__main__":
    main()
