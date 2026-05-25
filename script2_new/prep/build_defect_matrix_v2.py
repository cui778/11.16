#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a denser defect matrix for process-diagnosis retrials.

Design goals:
1. Keep the overall scenario count comparable to the current matrix.
2. Reduce the number of defect nodes per type.
3. Increase the number of scenarios per node so the model sees richer
   variations for each defect location.
4. Emit both an all-type matrix (I/E/P) and an I/E-only matrix.

Current default design:
    I: 12 nodes x 10 scenarios = 120
    E: 12 nodes x 10 scenarios = 120
    P:  6 nodes x 10 scenarios =  60

This script intentionally does NOT attempt to fix true SWMM-native water-quality
injection semantics for P. It only rebuilds a denser, cleaner defect matrix that
matches the currently executable pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class MatrixDesign:
    type_name: str
    n_nodes: int
    scenes_per_node: int
    intensity_choices: Sequence[int]
    start_hour_choices: Sequence[int]
    duration_choices: Sequence[int]
    flow_factor: float


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def _make_scene_templates(
    scenes_per_node: int,
    intensity_choices: Sequence[int],
    start_hour_choices: Sequence[int],
    duration_choices: Sequence[int],
    seed: int,
) -> List[dict]:
    combos = []
    for intensity in intensity_choices:
        for start_hour in start_hour_choices:
            for duration_h in duration_choices:
                combos.append(
                    {
                        "intensity_pct": int(intensity),
                        "start_hour": int(start_hour),
                        "duration_h": int(duration_h),
                    }
                )
    rng = np.random.default_rng(seed)
    rng.shuffle(combos)
    if scenes_per_node > len(combos):
        raise ValueError(
            f"scenes_per_node={scenes_per_node} exceeds available combinations={len(combos)}"
        )
    return combos[:scenes_per_node]


def _greedy_select_nodes(
    candidate_nodes: Sequence[str],
    node_scores: Dict[str, float],
    shortest_dist: np.ndarray,
    node_to_idx: Dict[str, int],
    n_pick: int,
    excluded: set[str],
) -> List[str]:
    available = [str(n) for n in candidate_nodes if str(n) not in excluded and str(n) in node_to_idx]
    if len(available) < n_pick:
        raise ValueError(f"Not enough available nodes to pick {n_pick}; only {len(available)} remain.")

    selected: List[str] = []
    while len(selected) < n_pick:
        best_node = None
        best_value = None
        for node_id in available:
            if node_id in selected:
                continue
            base_score = float(node_scores.get(node_id, 0.0))
            if not selected:
                value = base_score
            else:
                dists = []
                for other in selected:
                    i = node_to_idx[node_id]
                    j = node_to_idx[other]
                    d = float(shortest_dist[i, j])
                    if not np.isfinite(d) or d >= 999:
                        d = 15.0
                    dists.append(d)
                min_dist = min(dists) if dists else 0.0
                value = base_score + 0.12 * min(float(min_dist), 10.0)
            if best_value is None or value > best_value:
                best_value = value
                best_node = node_id
        selected.append(best_node)
    return selected


def _build_node_scores(
    candidate_nodes: Sequence[str],
    baseline_stats: dict,
    monitor_nodes: Sequence[str],
    node_list: Sequence[str],
    shortest_dist: np.ndarray,
    adj: np.ndarray,
) -> Dict[str, float]:
    node_to_idx = {str(n): i for i, n in enumerate(node_list)}
    monitor_indices = [node_to_idx[n] for n in monitor_nodes if n in node_to_idx]
    degrees = _safe_norm(adj.sum(axis=0) + adj.sum(axis=1))

    flows = []
    closeness = []
    for node_id in candidate_nodes:
        node_stat = baseline_stats["nodes"].get(str(node_id), {})
        flows.append(float(node_stat.get("mean", 0.0)))
        if monitor_indices and node_id in node_to_idx:
            idx = node_to_idx[node_id]
            hops = shortest_dist[idx, monitor_indices]
            hops = np.where(np.isfinite(hops), hops, 15.0)
            hop = float(np.min(np.clip(hops, 0, 15)))
            closeness.append(1.0 - min(hop, 10.0) / 10.0)
        else:
            closeness.append(0.0)

    flow_norm = _safe_norm(np.asarray(flows, dtype=np.float32))
    close_norm = _safe_norm(np.asarray(closeness, dtype=np.float32))

    node_scores = {}
    for pos, node_id in enumerate(candidate_nodes):
        deg = float(degrees[node_to_idx[node_id]]) if node_id in node_to_idx else 0.0
        # Conservative scoring: prefer nodes with enough baseline signal,
        # decent observability, and moderate structural centrality.
        node_scores[str(node_id)] = 0.45 * float(flow_norm[pos]) + 0.35 * float(close_norm[pos]) + 0.20 * deg
    return node_scores


def _build_rows_for_type(
    type_name: str,
    node_ids: Sequence[str],
    baseline_stats: dict,
    scenes_per_node: int,
    scene_templates: Sequence[dict],
    flow_factor: float,
    rng_seed: int,
) -> List[dict]:
    rng = np.random.default_rng(rng_seed)
    rows: List[dict] = []
    defect_id = 1
    for node_idx, node_id in enumerate(node_ids):
        baseline_flow = float(baseline_stats["nodes"].get(str(node_id), {}).get("mean", 0.0))
        if baseline_flow <= 0:
            baseline_flow = 1.0
        templates = list(scene_templates)
        rng.shuffle(templates)
        for tpl in templates[:scenes_per_node]:
            intensity_pct = int(tpl["intensity_pct"])
            start_hour = int(tpl["start_hour"])
            duration_h = int(tpl["duration_h"])
            sign = -1.0 if type_name == "E" else 1.0
            flow = sign * baseline_flow * flow_factor * intensity_pct
            # Keep existing schema. P remains executable but not yet a true SWMM-native
            # quality injection scenario.
            if type_name == "P":
                bodf = float(rng.uniform(140.0, 280.0))
                nh4 = float(rng.uniform(10.0, 20.0))
                do = float(rng.uniform(2.0, 6.0))
            else:
                bodf = float(rng.uniform(120.0, 260.0))
                nh4 = float(rng.uniform(10.0, 18.0))
                do = float(rng.uniform(2.0, 7.0))
            rows.append(
                {
                    "defect_id": defect_id,
                    "defect_type": type_name,
                    "node_id": str(node_id),
                    "link_id": "",
                    "intensity_pct": intensity_pct,
                    "start_hour": start_hour,
                    "duration_h": duration_h,
                    "flow": round(float(flow), 4),
                    "BODf": round(bodf, 2),
                    "NH4": round(nh4, 2),
                    "DO": round(do, 2),
                    "baseline_flow": round(float(baseline_flow), 4),
                }
            )
            defect_id += 1
    return rows


def build_matrix(
    baseline_stats_path: Path,
    candidate_nodes_path: Path,
    node_list_path: Path,
    graph_path_features_path: Path,
    monitor_nodes_path: Path,
    output_path: Path,
    designs: Sequence[MatrixDesign],
    seed: int = 42,
) -> pd.DataFrame:
    baseline_stats = _load_json(baseline_stats_path)
    candidate_nodes = _load_json(candidate_nodes_path)["candidate_nodes"]
    node_list = _load_json(node_list_path)
    graph_npz = np.load(graph_path_features_path)
    shortest_dist = graph_npz["shortest_dist"]
    monitor_nodes = _load_json(monitor_nodes_path)["monitor_nodes"]
    adj = np.load(output_path.parent / "adj_matrix.npy")

    node_to_idx = {str(n): i for i, n in enumerate(node_list)}
    node_scores = _build_node_scores(candidate_nodes, baseline_stats, monitor_nodes, node_list, shortest_dist, adj)

    all_rows: List[dict] = []
    used_nodes: set[str] = set()
    current_defect_id = 1
    for type_offset, design in enumerate(designs):
        selected_nodes = _greedy_select_nodes(
            candidate_nodes=candidate_nodes,
            node_scores=node_scores,
            shortest_dist=shortest_dist,
            node_to_idx=node_to_idx,
            n_pick=design.n_nodes,
            excluded=used_nodes,
        )
        used_nodes.update(selected_nodes)
        templates = _make_scene_templates(
            scenes_per_node=design.scenes_per_node,
            intensity_choices=design.intensity_choices,
            start_hour_choices=design.start_hour_choices,
            duration_choices=design.duration_choices,
            seed=seed + type_offset * 17,
        )
        rows = _build_rows_for_type(
            type_name=design.type_name,
            node_ids=selected_nodes,
            baseline_stats=baseline_stats,
            scenes_per_node=design.scenes_per_node,
            scene_templates=templates,
            flow_factor=design.flow_factor,
            rng_seed=seed + type_offset * 97,
        )
        for row in rows:
            row["defect_id"] = current_defect_id
            current_defect_id += 1
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def summarize_matrix(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": int(len(df)),
        "type_counts": df["defect_type"].value_counts().to_dict(),
        "type_unique_nodes": df.groupby("defect_type")["node_id"].nunique().to_dict(),
        "type_scenes_per_node_mean": (
            df.groupby(["defect_type", "node_id"]).size().groupby(level=0).mean().round(3).to_dict()
        ),
        "type_flow_abs_mean": (
            df.assign(flow_abs=df["flow"].abs()).groupby("defect_type")["flow_abs"].mean().round(4).to_dict()
        ),
    }
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    input_dir = repo_root / "input_1"
    parser = argparse.ArgumentParser(description="Build denser defect_matrix v2 for clean retrials.")
    parser.add_argument("--baseline-stats", default=str(input_dir / "baseline_flow_stats.json"))
    parser.add_argument("--candidate-nodes", default=str(input_dir / "candidate_nodes_new.json"))
    parser.add_argument("--node-list", default=str(input_dir / "node_list.json"))
    parser.add_argument("--graph-path-features", default=str(input_dir / "graph_path_features.npz"))
    parser.add_argument("--monitor-nodes", default=str(input_dir / "monitor_nodes_degree_N25.json"))
    parser.add_argument("--output-all", default=str(input_dir / "defect_matrix_diverse_v2.csv"))
    parser.add_argument("--output-ie", default=str(input_dir / "defect_matrix_diverse_ie_v2.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--i-scenes-per-node", type=int, default=10)
    parser.add_argument("--e-scenes-per-node", type=int, default=10)
    parser.add_argument("--p-scenes-per-node", type=int, default=10)
    parser.add_argument("--i-nodes", type=int, default=12)
    parser.add_argument("--e-nodes", type=int, default=12)
    parser.add_argument("--p-nodes", type=int, default=6)
    parser.add_argument("--i-flow-factor", type=float, default=0.01)
    parser.add_argument("--e-flow-factor", type=float, default=0.01)
    parser.add_argument("--p-flow-factor", type=float, default=0.002)
    parser.add_argument("--i-intensities", default="40,50,60")
    parser.add_argument("--e-intensities", default="30,40,50")
    parser.add_argument("--p-intensities", default="30,40,50")
    args = parser.parse_args()

    def _parse_int_list(text: str) -> list[int]:
        values = []
        for token in str(text).split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token))
        if not values:
            raise ValueError(f"Invalid empty intensity list: {text!r}")
        return values

    all_designs = [
        MatrixDesign("I", n_nodes=args.i_nodes, scenes_per_node=args.i_scenes_per_node, intensity_choices=_parse_int_list(args.i_intensities), start_hour_choices=[2, 6, 10, 14, 18], duration_choices=[6, 12, 18, 24], flow_factor=args.i_flow_factor),
        MatrixDesign("E", n_nodes=args.e_nodes, scenes_per_node=args.e_scenes_per_node, intensity_choices=_parse_int_list(args.e_intensities), start_hour_choices=[0, 4, 8, 12, 16], duration_choices=[6, 12, 18, 24], flow_factor=args.e_flow_factor),
        MatrixDesign("P", n_nodes=args.p_nodes, scenes_per_node=args.p_scenes_per_node, intensity_choices=_parse_int_list(args.p_intensities), start_hour_choices=[3, 9, 15, 21, 23], duration_choices=[6, 12, 18, 24], flow_factor=args.p_flow_factor),
    ]
    ie_designs = [d for d in all_designs if d.type_name in {"I", "E"}]

    output_all = Path(args.output_all)
    df_all = build_matrix(
        baseline_stats_path=Path(args.baseline_stats),
        candidate_nodes_path=Path(args.candidate_nodes),
        node_list_path=Path(args.node_list),
        graph_path_features_path=Path(args.graph_path_features),
        monitor_nodes_path=Path(args.monitor_nodes),
        output_path=output_all,
        designs=all_designs,
        seed=args.seed,
    )
    summary_all = summarize_matrix(df_all)

    output_ie = Path(args.output_ie)
    df_ie = build_matrix(
        baseline_stats_path=Path(args.baseline_stats),
        candidate_nodes_path=Path(args.candidate_nodes),
        node_list_path=Path(args.node_list),
        graph_path_features_path=Path(args.graph_path_features),
        monitor_nodes_path=Path(args.monitor_nodes),
        output_path=output_ie,
        designs=ie_designs,
        seed=args.seed,
    )
    summary_ie = summarize_matrix(df_ie)

    manifest = {
        "seed": args.seed,
        "all_design_summary": summary_all,
        "ie_design_summary": summary_ie,
        "all_output": str(output_all),
        "ie_output": str(output_ie),
    }
    manifest_path = output_all.parent / f"{output_all.stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
