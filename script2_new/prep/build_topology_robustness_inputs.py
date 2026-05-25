#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build topology-perturbed adjacency/path-prior inputs for Chapter 1 robustness probes.

Unlike the default graph-feature builder, this script uses the perturbed
adjacency matrix itself as the graph structure so that adjacency and path
features are perturbed consistently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np


INF_HOP = 999
INF_LEN = 1e6
MAX_HOP = 15
MAX_LEN = 5000.0


def _load_node_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "node_list" in data:
        return [str(x) for x in data["node_list"]]
    return [str(x) for x in data]


def _load_parsed_inp(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _edge_length_map(parsed: dict, node_to_idx: dict[str, int]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for _, info in parsed.get("conduits", {}).items():
        u_id = str(info.get("from_node", ""))
        v_id = str(info.get("to_node", ""))
        if u_id in node_to_idx and v_id in node_to_idx:
            out[(node_to_idx[u_id], node_to_idx[v_id])] = float(info.get("length", 0.0))
    return out


def _elevation_array(parsed: dict, node_list: list[str]) -> np.ndarray:
    elevation = np.zeros(len(node_list), dtype=np.float32)
    junctions = parsed.get("junctions", {})
    for i, nid in enumerate(node_list):
        if nid in junctions:
            elevation[i] = float(junctions[nid].get("elevation", 0.0))
    return elevation


def _drop_edges(adj: np.ndarray, drop_ratio: float, seed: int) -> np.ndarray:
    if drop_ratio <= 0:
        return adj.copy()
    rng = np.random.default_rng(seed)
    edge_idx = np.argwhere(adj > 0)
    n_drop = max(1, int(round(edge_idx.shape[0] * drop_ratio)))
    chosen = rng.choice(edge_idx.shape[0], size=min(n_drop, edge_idx.shape[0]), replace=False)
    out = adj.copy()
    for pos in chosen:
        i, j = edge_idx[pos]
        out[i, j] = 0.0
    return out


def _build_graph_features_from_adj(
    adj: np.ndarray,
    node_list: list[str],
    parsed: dict,
) -> dict[str, np.ndarray]:
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edge_lengths = _edge_length_map(parsed, node_to_idx)
    elevation = _elevation_array(parsed, node_list)

    graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                graph.add_edge(i, j, weight=1.0)

    shortest_dist = np.zeros((n, n), dtype=np.float32)
    pipe_length_dist = np.zeros((n, n), dtype=np.float32)
    flow_direction = np.zeros((n, n), dtype=np.float32)
    elevation_diff = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            elevation_diff[i, j] = elevation[j] - elevation[i]
            if i == j:
                continue
            try:
                path = nx.shortest_path(graph, source=i, target=j, weight="weight")
                shortest_dist[i, j] = float(len(path) - 1)
                total_len = 0.0
                for k in range(len(path) - 1):
                    total_len += edge_lengths.get((path[k], path[k + 1]), 0.0)
                pipe_length_dist[i, j] = total_len
                flow_direction[i, j] = 1.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    nx.shortest_path(graph, source=j, target=i, weight="weight")
                    flow_direction[i, j] = -1.0
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                shortest_dist[i, j] = INF_HOP
                pipe_length_dist[i, j] = INF_LEN

    return {
        "shortest_dist": np.clip(shortest_dist, 0, MAX_HOP).astype(np.float32),
        "pipe_length_dist": np.clip(pipe_length_dist, 0, MAX_LEN).astype(np.float32),
        "flow_direction": flow_direction.astype(np.float32),
        "elevation_diff": elevation_diff.astype(np.float32),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "input_1"

    parser = argparse.ArgumentParser(description="Build perturbed topology inputs for robustness tests.")
    parser.add_argument("--adj-file", default=str(input_dir / "adj_matrix.npy"))
    parser.add_argument("--node-list-file", default=str(input_dir / "node_list.json"))
    parser.add_argument("--parsed-inp-file", default=str(input_dir / "parsed_inp_data.json"))
    parser.add_argument("--output-dir", default=str(input_dir / "topology_robustness"))
    parser.add_argument("--drop-ratios", nargs="+", type=float, default=[0.05, 0.10, 0.15])
    parser.add_argument("--seed", type=int, default=20260402)
    args = parser.parse_args()

    adj = np.load(args.adj_file).astype(np.float32)
    node_list = _load_node_list(args.node_list_file)
    parsed = _load_parsed_inp(args.parsed_inp_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_edges = int((adj > 0).sum())
    print(f"baseline edges: {baseline_edges}")

    for ratio in args.drop_ratios:
        pct = int(round(ratio * 100))
        tag = f"edge_drop{pct:02d}"
        adj_out = output_dir / f"adj_matrix_{tag}.npy"
        graph_out = output_dir / f"graph_path_features_{tag}.npz"

        perturbed = _drop_edges(adj, ratio, args.seed + pct)
        np.save(adj_out, perturbed)
        feats = _build_graph_features_from_adj(perturbed, node_list, parsed)
        np.savez(graph_out, **feats)

        finite_hops = feats["shortest_dist"][feats["shortest_dist"] < MAX_HOP]
        hop_max = float(finite_hops.max()) if finite_hops.size else 0.0
        print(
            f"{tag}: edges {int((perturbed > 0).sum())} "
            f"(dropped {baseline_edges - int((perturbed > 0).sum())}), "
            f"finite hop max {hop_max:.1f}"
        )


if __name__ == "__main__":
    main()
