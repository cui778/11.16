#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_segment_artifacts(segment_list_file, edge_static_features_file):
    with open(segment_list_file, "r", encoding="utf-8") as f:
        segment_payload = json.load(f)
    segments = pd.DataFrame(segment_payload.get("segments", []))
    static_df = pd.read_csv(edge_static_features_file)
    return segments, static_df


def build_edge_timeseries(node_file, link_file, segment_list_file, edge_static_features_file, output_file):
    if str(node_file).endswith(".parquet"):
        node_df = pd.read_parquet(node_file)
    else:
        node_df = pd.read_csv(node_file)
    if str(link_file).endswith(".parquet"):
        link_df = pd.read_parquet(link_file)
    else:
        link_df = pd.read_csv(link_file)

    segments, static_df = load_segment_artifacts(segment_list_file, edge_static_features_file)
    if segments.empty:
        raise ValueError("segment list is empty")

    node_df["datetime"] = pd.to_datetime(node_df["datetime"])
    link_df["datetime"] = pd.to_datetime(link_df["datetime"])

    node_base_cols = [
        "scenario_id",
        "datetime",
        "node_id",
        "depth",
        "head",
        "total_inflow",
        "lateral_inflow",
        "depth_residual",
        "depth_residual_rel",
        "head_residual",
        "head_residual_rel",
        "total_inflow_residual",
        "total_inflow_residual_rel",
    ]
    node_base_cols = [col for col in node_base_cols if col in node_df.columns]

    upstream_df = node_df[node_base_cols].rename(
        columns={
            "node_id": "from_node",
            "depth": "depth_up",
            "head": "head_up",
            "total_inflow": "total_inflow_up",
            "lateral_inflow": "lateral_inflow_up",
            "depth_residual": "depth_up_residual",
            "depth_residual_rel": "depth_up_residual_rel",
            "head_residual": "head_up_residual",
            "head_residual_rel": "head_up_residual_rel",
            "total_inflow_residual": "total_inflow_up_residual",
            "total_inflow_residual_rel": "total_inflow_up_residual_rel",
        }
    )
    downstream_df = node_df[node_base_cols].rename(
        columns={
            "node_id": "to_node",
            "depth": "depth_down",
            "head": "head_down",
            "total_inflow": "total_inflow_down",
            "lateral_inflow": "lateral_inflow_down",
            "depth_residual": "depth_down_residual",
            "depth_residual_rel": "depth_down_residual_rel",
            "head_residual": "head_down_residual",
            "head_residual_rel": "head_down_residual_rel",
            "total_inflow_residual": "total_inflow_down_residual",
            "total_inflow_residual_rel": "total_inflow_down_residual_rel",
        }
    )

    merged = link_df.merge(segments, on="link_id", how="inner")
    merged = merged.merge(static_df, on=["segment_id", "segment_idx", "link_id", "from_node", "to_node"], how="left")
    merged = merged.merge(upstream_df, on=["scenario_id", "datetime", "from_node"], how="left")
    merged = merged.merge(downstream_df, on=["scenario_id", "datetime", "to_node"], how="left")

    # Dynamic edge features aligned with the revised opening report.
    if "depth_up" in merged.columns and "depth_down" in merged.columns:
        merged["depth_diff_ud"] = merged["depth_up"] - merged["depth_down"]
    if "head_up" in merged.columns and "head_down" in merged.columns:
        merged["head_diff_ud"] = merged["head_up"] - merged["head_down"]
    if "total_inflow_up" in merged.columns and "total_inflow_down" in merged.columns:
        merged["inflow_diff_ud"] = merged["total_inflow_up"] - merged["total_inflow_down"]
    if "flow" in merged.columns and "diameter" in merged.columns:
        merged["flux_proxy"] = merged["flow"] * merged["diameter"].fillna(0.0)
    if "velocity" in merged.columns and "diameter" in merged.columns:
        merged["velocity_area_proxy"] = merged["velocity"] * np.square(merged["diameter"].fillna(0.0))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False, compression="gzip")
    return merged


def main():
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import Config

    cfg = Config()
    node_file = cfg.node_timeseries_residual_file if Path(cfg.node_timeseries_residual_file).exists() else cfg.node_timeseries_file
    link_file = cfg.link_timeseries_residual_file if Path(cfg.link_timeseries_residual_file).exists() else cfg.link_timeseries_file

    build_edge_timeseries(
        node_file=node_file,
        link_file=link_file,
        segment_list_file=cfg.segment_list_file,
        edge_static_features_file=cfg.edge_static_features_file,
        output_file=cfg.edge_timeseries_file,
    )
    print(f"[OK] wrote {cfg.edge_timeseries_file}")


if __name__ == "__main__":
    main()
