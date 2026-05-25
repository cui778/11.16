#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a balanced observability-aware monitor layout under a fixed budget.

Compared with the unconstrained observability-aware layout, this builder:
- keeps the monitor budget fixed
- caps candidate-monitor overlap
- forces a minimum number of non-candidate monitors
- preserves monitor dispersion / global backbone coverage
- still optimizes candidate observability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


INF_HOP = 999


def load_candidate_nodes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "candidate_nodes" in data:
        return [str(x) for x in data["candidate_nodes"]]
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError(f"Unsupported candidate node file: {path}")


def load_node_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "node_list" in data:
        return [str(x) for x in data["node_list"]]
    raise ValueError(f"Unsupported node list file: {path}")


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
        return 0.5
    return 0.0


def summarize_hops(best_hops: np.ndarray) -> dict[str, float]:
    direct = int((best_hops == 0).sum())
    near = int(((best_hops >= 1) & (best_hops <= 2)).sum())
    far = int((best_hops > 2).sum())
    finite = best_hops[best_hops < INF_HOP]
    mean_hop = float(finite.mean()) if finite.size else float("nan")
    return {
        "direct": direct,
        "near": near,
        "far": far,
        "mean_hop": mean_hop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced observability-aware N25 layout.")
    parser.add_argument("--candidate-nodes-file", required=True)
    parser.add_argument("--node-list-file", required=True)
    parser.add_argument("--graph-features-file", required=True)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--max-overlap", type=int, default=12)
    parser.add_argument("--min-noncandidate", type=int, default=13)
    parser.add_argument("--exclude-nodes", default="1,MH0100819")
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    candidate_nodes = load_candidate_nodes(Path(args.candidate_nodes_file))
    node_list = load_node_list(Path(args.node_list_file))
    shortest = np.load(args.graph_features_file)["shortest_dist"]
    node_to_idx = {str(node): i for i, node in enumerate(node_list)}
    exclude_nodes = {x.strip() for x in args.exclude_nodes.split(",") if x.strip()}

    all_nodes = [str(node) for node in node_list if str(node) not in exclude_nodes]
    candidate_set = set(candidate_nodes)
    noncandidate_nodes = [n for n in all_nodes if n not in candidate_set]

    candidate_idx = [node_to_idx[str(c)] for c in candidate_nodes]
    all_idx = [node_to_idx[str(n)] for n in all_nodes]

    hop_matrix = np.full((len(candidate_nodes), len(all_nodes)), INF_HOP, dtype=np.int16)
    for ci, cand_i in enumerate(candidate_idx):
        for nj, node_j in enumerate(all_idx):
            hop_matrix[ci, nj] = sym_hop(shortest, cand_i, node_j)

    node_hop = np.full((len(all_nodes), len(all_nodes)), INF_HOP, dtype=np.int16)
    for i, node_i in enumerate(all_idx):
        for j, node_j in enumerate(all_idx):
            node_hop[i, j] = sym_hop(shortest, node_i, node_j)

    # Smaller mean hop to all nodes => more globally central / backbone-like.
    global_score = np.zeros(len(all_nodes), dtype=np.float32)
    for j in range(len(all_nodes)):
        finite = node_hop[j][node_hop[j] < INF_HOP]
        global_score[j] = 0.0 if finite.size == 0 else 1.0 / float(finite.mean())

    best_hops = np.full(len(candidate_nodes), INF_HOP, dtype=np.int16)
    selected_nodes: list[str] = []

    def score_candidate(j: int) -> tuple[float, np.ndarray] | None:
        node = all_nodes[j]
        overlap_count = sum(1 for x in selected_nodes if x in candidate_set)
        noncandidate_count = len(selected_nodes) - overlap_count
        remaining_slots = args.n - len(selected_nodes)
        remaining_noncandidate_needed = max(args.min_noncandidate - noncandidate_count, 0)

        is_candidate = node in candidate_set
        if is_candidate and overlap_count >= args.max_overlap:
            return None

        # Preserve enough room to still satisfy the minimum number of non-candidate monitors.
        if is_candidate and (remaining_slots - 1) < remaining_noncandidate_needed:
            return None

        new_hops = np.minimum(best_hops, hop_matrix[:, j])

        old_util = np.array([hop_utility(int(h)) for h in best_hops], dtype=np.float32)
        new_util = np.array([hop_utility(int(h)) for h in new_hops], dtype=np.float32)
        util_gain = float(new_util.sum() - old_util.sum())

        old_nonfar = int((best_hops <= 2).sum())
        new_nonfar = int((new_hops <= 2).sum())
        old_direct = int((best_hops == 0).sum())
        new_direct = int((new_hops == 0).sum())
        hop_reduction = (
            float(best_hops[best_hops < INF_HOP].mean() - new_hops.mean())
            if (best_hops < INF_HOP).any()
            else 0.0
        )

        if selected_nodes:
            selected_idx = [all_nodes.index(x) for x in selected_nodes]
            min_disp_hop = min(int(node_hop[j, sj]) for sj in selected_idx)
        else:
            min_disp_hop = 0
        dispersion_bonus = float(min(min_disp_hop, 4))

        noncandidate_bonus = 2.0 if node not in candidate_set else 0.0

        score = (
            1000.0 * (new_nonfar - old_nonfar)
            + 60.0 * (new_direct - old_direct)
            + 12.0 * util_gain
            + 4.0 * hop_reduction
            + 120.0 * float(global_score[j])
            + 10.0 * dispersion_bonus
            + noncandidate_bonus
        )
        return score, new_hops

    # Phase 1: guarantee enough non-candidate monitors.
    while len(selected_nodes) < args.min_noncandidate:
        best_score = None
        best_j = None
        best_new_hops = None
        for node in noncandidate_nodes:
            if node in selected_nodes:
                continue
            j = all_nodes.index(node)
            scored = score_candidate(j)
            if scored is None:
                continue
            score, new_hops = scored
            if best_score is None or score > best_score:
                best_score = score
                best_j = j
                best_new_hops = new_hops
        if best_j is None or best_new_hops is None:
            break
        selected_nodes.append(all_nodes[best_j])
        best_hops = best_new_hops

    # Phase 2: fill the remaining slots under overlap cap.
    while len(selected_nodes) < args.n:
        best_score = None
        best_j = None
        best_new_hops = None
        for j, node in enumerate(all_nodes):
            if node in selected_nodes:
                continue
            scored = score_candidate(j)
            if scored is None:
                continue
            score, new_hops = scored
            if best_score is None or score > best_score:
                best_score = score
                best_j = j
                best_new_hops = new_hops
        if best_j is None or best_new_hops is None:
            break
        selected_nodes.append(all_nodes[best_j])
        best_hops = best_new_hops

    summary = summarize_hops(best_hops)
    overlap_nodes = [x for x in selected_nodes if x in candidate_set]
    out = {
        "monitor_nodes": selected_nodes,
        "strategy": "balanced_observability",
        "n": int(args.n),
        "candidate_nodes_file": str(Path(args.candidate_nodes_file).resolve()),
        "node_list_file": str(Path(args.node_list_file).resolve()),
        "graph_features_file": str(Path(args.graph_features_file).resolve()),
        "exclude_nodes": sorted(exclude_nodes),
        "constraints": {
            "max_overlap": int(args.max_overlap),
            "min_noncandidate": int(args.min_noncandidate),
        },
        "objective": {
            "primary": "maximize candidate nodes within 2 hops while respecting overlap cap",
            "secondary": "increase direct coverage, preserve non-candidate backbone coverage, and keep monitor dispersion",
        },
        "overlap_count": len(overlap_nodes),
        "noncandidate_count": len(selected_nodes) - len(overlap_nodes),
        "overlap_nodes": overlap_nodes,
        "observability_summary": summary,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out["observability_summary"], ensure_ascii=False, indent=2))
    print(f"overlap_count={out['overlap_count']}, noncandidate_count={out['noncandidate_count']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
