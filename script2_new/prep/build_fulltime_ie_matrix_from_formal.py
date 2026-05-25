#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a true full-time injection matrix from the formal IE420 matrix.

This does not change node/type/intensity coverage. It only changes the temporal
protocol to represent a real full-time injection dataset:
- start_hour = 0
- duration_h = scenario duration

The output is intentionally independent from the formal time-gated matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = SCRIPT_ROOT / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"
DEFAULT_OUTPUT = SCRIPT_ROOT / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative420_fulltime_seed42.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-time IE matrix from formal time-gated matrix")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--duration-h",
        type=float,
        default=48.0,
        help="Full scenario duration in hours. 48h covers the current 287x10min simulation horizon.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    if "start_hour" not in df.columns or "duration_h" not in df.columns:
        raise ValueError("Input matrix must contain start_hour and duration_h columns")

    out = df.copy()
    out["source_start_hour"] = out["start_hour"]
    out["source_duration_h"] = out["duration_h"]
    out["start_hour"] = 0.0
    out["duration_h"] = float(args.duration_h)
    out["temporal_protocol"] = "fulltime_injection"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    candidate_nodes = set(out["node_id"].astype(str))
    i_nodes = set(out.loc[out["defect_type"] == "I", "node_id"].astype(str))
    e_nodes = set(out.loc[out["defect_type"] == "E", "node_id"].astype(str))
    summary = {
        "source_csv": str(input_csv),
        "output_csv": str(output_csv),
        "temporal_protocol": "fulltime_injection",
        "start_hour": 0.0,
        "duration_h": float(args.duration_h),
        "n_rows": int(len(out)),
        "i_rows": int((out["defect_type"] == "I").sum()),
        "e_rows": int((out["defect_type"] == "E").sum()),
        "active_unique_nodes": int(len(candidate_nodes)),
        "i_unique_nodes": int(len(i_nodes)),
        "e_unique_nodes": int(len(e_nodes)),
        "note": "This is a true full-time injection matrix. It should be used with extract_timeseries.py without --time-gated.",
    }
    summary_path = output_csv.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] fulltime matrix -> {output_csv}")
    print(f"[OK] summary -> {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
