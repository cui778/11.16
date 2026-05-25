#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze per-defect-node and per-region localization difficulty from by-scenario outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd


INF_HOP = 999


def load_monitor_nodes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(x) for x in data["monitor_nodes"]]


def symmetric_hop_distance(shortest: np.ndarray, i: int, j: int) -> int:
    a = int(shortest[i, j])
    b = int(shortest[j, i])
    vals = [x for x in (a, b) if x < INF_HOP]
    return min(vals) if vals else INF_HOP


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze localization difficulty by defect node and region.")
    parser.add_argument("--by-scenario-csv", required=True)
    parser.add_argument("--defect-csv", required=True)
    parser.add_argument("--monitor-nodes-file", required=True)
    parser.add_argument("--node-list-file", required=True)
    parser.add_argument("--graph-features-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    by_scenario = pd.read_csv(args.by_scenario_csv)
    defect_df = pd.read_csv(args.defect_csv)
    monitor_nodes = load_monitor_nodes(Path(args.monitor_nodes_file))
    node_list = json.loads(Path(args.node_list_file).read_text(encoding="utf-8"))
    node_to_idx = {str(n): i for i, n in enumerate(node_list)}
    shortest = np.load(args.graph_features_file)["shortest_dist"]

    merged = by_scenario.merge(
        defect_df[["defect_id", "defect_type", "node_id", "intensity_pct", "start_hour", "duration_h"]],
        on="defect_id",
        how="left",
        validate="one_to_one",
    )
    merged["node_id"] = merged["node_id"].astype(str)
    merged["is_monitored"] = merged["node_id"].isin(monitor_nodes)

    nearest_monitor = []
    nearest_hop = []
    for node_id in merged["node_id"]:
        idx = node_to_idx[node_id]
        best_monitor = None
        best_hop = INF_HOP
        for mon in monitor_nodes:
            mon_idx = node_to_idx[str(mon)]
            hop = symmetric_hop_distance(shortest, idx, mon_idx)
            if hop < best_hop:
                best_hop = hop
                best_monitor = str(mon)
        nearest_monitor.append(best_monitor)
        nearest_hop.append(best_hop if best_hop < INF_HOP else np.nan)

    merged["nearest_monitor"] = nearest_monitor
    merged["nearest_monitor_hop"] = nearest_hop

    node_summary = (
        merged.groupby(["defect_type", "node_id", "is_monitored", "nearest_monitor", "nearest_monitor_hop"], dropna=False)
        .agg(
            n_scenarios=("defect_id", "count"),
            mean_windows=("n_windows", "mean"),
            top1_mean=("top1", "mean"),
            top3_mean=("top3", "mean"),
            mrr_mean=("mrr", "mean"),
            intensity_mean=("intensity_pct", "mean"),
            duration_mean=("duration_h", "mean"),
        )
        .reset_index()
        .sort_values(["top1_mean", "mrr_mean", "n_scenarios"], ascending=[False, False, False])
    )

    region_summary = (
        merged.groupby(["nearest_monitor", "nearest_monitor_hop"], dropna=False)
        .agg(
            n_scenarios=("defect_id", "count"),
            unique_defect_nodes=("node_id", "nunique"),
            top1_mean=("top1", "mean"),
            top3_mean=("top3", "mean"),
            mrr_mean=("mrr", "mean"),
            monitored_share=("is_monitored", "mean"),
        )
        .reset_index()
        .sort_values(["top1_mean", "mrr_mean"], ascending=[False, False])
    )

    hop_summary = (
        merged.groupby("nearest_monitor_hop", dropna=False)
        .agg(
            n_scenarios=("defect_id", "count"),
            unique_defect_nodes=("node_id", "nunique"),
            top1_mean=("top1", "mean"),
            top3_mean=("top3", "mean"),
            mrr_mean=("mrr", "mean"),
        )
        .reset_index()
        .sort_values("nearest_monitor_hop")
    )

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    node_path = out_prefix.with_name(out_prefix.name + "_node_summary.csv")
    region_path = out_prefix.with_name(out_prefix.name + "_region_summary.csv")
    hop_path = out_prefix.with_name(out_prefix.name + "_hop_summary.csv")
    merged_path = out_prefix.with_name(out_prefix.name + "_scenario_merged.csv")

    node_summary.to_csv(node_path, index=False, encoding="utf-8-sig")
    region_summary.to_csv(region_path, index=False, encoding="utf-8-sig")
    hop_summary.to_csv(hop_path, index=False, encoding="utf-8-sig")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    print("saved:")
    print(node_path)
    print(region_path)
    print(hop_path)
    print(merged_path)
    print("\nTop difficult nodes:")
    print(node_summary.sort_values(["top1_mean", "mrr_mean"]).head(10).to_string(index=False))
    print("\nTop easy nodes:")
    print(node_summary.head(10).to_string(index=False))
    print("\nHop summary:")
    print(hop_summary.to_string(index=False))


if __name__ == "__main__":
    main()
