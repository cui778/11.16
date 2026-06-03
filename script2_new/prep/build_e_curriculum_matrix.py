#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an IE-only matrix variant with a structured E-type curriculum.

Design:
- keep I rows unchanged from the input matrix
- regenerate E rows per node using the same number of scenes per node
- use an explicit easy/medium/hard template set instead of random-like mixes

This is a controlled B-scheme probe:
- same IE task
- same total E scene count
- different within-node E scene distribution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


E_TEMPLATES = [
    # easy
    {"intensity_pct": 50.0, "duration_h": 24.0, "start_hour": 4.0},
    {"intensity_pct": 50.0, "duration_h": 18.0, "start_hour": 8.0},
    {"intensity_pct": 50.0, "duration_h": 12.0, "start_hour": 4.0},
    {"intensity_pct": 50.0, "duration_h": 18.0, "start_hour": 12.0},
    # medium
    {"intensity_pct": 40.0, "duration_h": 24.0, "start_hour": 4.0},
    {"intensity_pct": 40.0, "duration_h": 18.0, "start_hour": 8.0},
    {"intensity_pct": 40.0, "duration_h": 12.0, "start_hour": 12.0},
    {"intensity_pct": 40.0, "duration_h": 6.0, "start_hour": 16.0},
    # hard
    {"intensity_pct": 30.0, "duration_h": 18.0, "start_hour": 4.0},
    {"intensity_pct": 30.0, "duration_h": 12.0, "start_hour": 8.0},
    {"intensity_pct": 30.0, "duration_h": 6.0, "start_hour": 12.0},
    {"intensity_pct": 30.0, "duration_h": 6.0, "start_hour": 16.0},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if not {"defect_id", "defect_type", "node_id", "baseline_flow"}.issubset(df.columns):
        raise RuntimeError("Input matrix does not have the required IE matrix columns.")

    df_i = df[df["defect_type"].astype(str) == "I"].copy()
    df_e = df[df["defect_type"].astype(str) == "E"].copy()
    if df_i.empty or df_e.empty:
        raise RuntimeError("Input matrix must contain both I and E rows.")

    e_counts = df_e.groupby("node_id").size()
    unique_scene_counts = sorted(set(int(x) for x in e_counts.tolist()))
    if len(unique_scene_counts) != 1:
        raise RuntimeError(f"E rows per node are not uniform: {unique_scene_counts}")
    scenes_per_e_node = unique_scene_counts[0]
    if scenes_per_e_node != len(E_TEMPLATES):
        raise RuntimeError(
            f"E template count {len(E_TEMPLATES)} must match scenes_per_e_node {scenes_per_e_node}"
        )

    rebuilt_e_rows = []
    next_defect_id = int(df["defect_id"].max()) + 1
    for node_id, group in df_e.groupby("node_id", sort=True):
        base_row = group.iloc[0].copy()
        baseline_flow = float(base_row["baseline_flow"])
        link_id = base_row.get("link_id", "")
        bodf = float(base_row.get("BODf", 0.0))
        nh4 = float(base_row.get("NH4", 0.0))
        do = float(base_row.get("DO", 0.0))
        for tpl in E_TEMPLATES:
            row = base_row.copy()
            row["defect_id"] = next_defect_id
            row["defect_type"] = "E"
            row["node_id"] = node_id
            row["link_id"] = link_id
            row["intensity_pct"] = float(tpl["intensity_pct"])
            row["start_hour"] = float(tpl["start_hour"])
            row["duration_h"] = float(tpl["duration_h"])
            row["flow"] = -round(float(tpl["intensity_pct"]) * baseline_flow * 0.01, 4)
            row["BODf"] = bodf
            row["NH4"] = nh4
            row["DO"] = do
            row["baseline_flow"] = baseline_flow
            rebuilt_e_rows.append(row)
            next_defect_id += 1

    df_out = pd.concat([df_i, pd.DataFrame(rebuilt_e_rows)], ignore_index=True)
    df_out = df_out.sort_values(["defect_type", "node_id", "defect_id"]).reset_index(drop=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    manifest = {
        "source_csv": str(in_path.resolve()),
        "output_csv": str(out_path.resolve()),
        "rule": "E-curriculum",
        "scene_count": int(len(df_out)),
        "type_counts": df_out.groupby("defect_type").size().to_dict(),
        "type_unique_nodes": df_out.groupby("defect_type")["node_id"].nunique().to_dict(),
        "e_templates": E_TEMPLATES,
    }
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
