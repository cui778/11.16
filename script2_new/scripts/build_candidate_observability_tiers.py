#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build observability tiers for candidate defect nodes under a given monitor layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INF_HOP = 999


def load_monitor_nodes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(x) for x in data["monitor_nodes"]]


def load_candidate_nodes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "candidate_nodes" in data:
        return [str(x) for x in data["candidate_nodes"]]
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError(f"Unsupported candidate node file format: {path}")


def sym_hop(shortest: np.ndarray, i: int, j: int) -> int:
    a = int(shortest[i, j])
    b = int(shortest[j, i])
    vals = [x for x in (a, b) if x < INF_HOP]
    return min(vals) if vals else INF_HOP


def hop_to_tier(hop: int, is_monitored: bool) -> str:
    if is_monitored or hop == 0:
        return "direct"
    if hop <= 2:
        return "near"
    return "far"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate observability tiers.")
    parser.add_argument("--candidate-nodes-file", required=True)
    parser.add_argument("--monitor-nodes-file", required=True)
    parser.add_argument("--node-list-file", required=True)
    parser.add_argument("--graph-features-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    candidate_nodes = load_candidate_nodes(Path(args.candidate_nodes_file))
    monitor_nodes = load_monitor_nodes(Path(args.monitor_nodes_file))
    node_list = json.loads(Path(args.node_list_file).read_text(encoding="utf-8"))
    node_to_idx = {str(n): i for i, n in enumerate(node_list)}
    shortest = np.load(args.graph_features_file)["shortest_dist"]

    rows = []
    for node_id in candidate_nodes:
        idx = node_to_idx[str(node_id)]
        best_monitor = None
        best_hop = INF_HOP
        for mon in monitor_nodes:
            mon_idx = node_to_idx[str(mon)]
            hop = sym_hop(shortest, idx, mon_idx)
            if hop < best_hop:
                best_hop = hop
                best_monitor = str(mon)
        is_monitored = str(node_id) in set(monitor_nodes)
        rows.append(
            {
                "candidate_node": str(node_id),
                "is_monitored": bool(is_monitored),
                "nearest_monitor": best_monitor,
                "nearest_monitor_hop": np.nan if best_hop >= INF_HOP else best_hop,
                "observability_tier": hop_to_tier(best_hop, is_monitored),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["observability_tier", "nearest_monitor_hop", "candidate_node"]).reset_index(drop=True)

    summary = (
        out_df.groupby("observability_tier")
        .agg(
            n_candidates=("candidate_node", "count"),
            monitored_share=("is_monitored", "mean"),
            mean_hop=("nearest_monitor_hop", "mean"),
        )
        .reset_index()
    )

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_name(out_prefix.name + "_candidate_tiers.csv")
    json_path = out_prefix.with_name(out_prefix.name + "_candidate_tiers_summary.json")
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "candidate_nodes_file": str(args.candidate_nodes_file),
        "monitor_nodes_file": str(args.monitor_nodes_file),
        "graph_features_file": str(args.graph_features_file),
        "tier_rule": {
            "direct": "candidate is monitored or nearest_monitor_hop == 0",
            "near": "1 <= nearest_monitor_hop <= 2",
            "far": "nearest_monitor_hop > 2",
        },
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nsaved:\n{csv_path}\n{json_path}")


if __name__ == "__main__":
    main()
