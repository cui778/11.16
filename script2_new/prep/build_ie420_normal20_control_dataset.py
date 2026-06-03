#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the IE420 + normal20 control dataset.

This control keeps the time-gated IE420 protocol unchanged and only replaces
normal10 with normal20. It is used to isolate whether the mixed-v1 performance
drop came from adding persistent IE windows or from expanding normal scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_48h_restructured_datasets import (
    _write_manifest,
    _write_summary,
    build_normal_dataset,
)
from dataset_build_utils import _infer_ie_defect_csv


ROOT = Path(r"E:\11.16\script2_new")
DEFAULT_IE_DIR = ROOT / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42"
DEFAULT_NORMAL20_DIR = ROOT / "training_data_new" / "normal_multibaseline_v2_seedset20"
DEFAULT_OUTPUT_DIR = ROOT / "training_data_new" / "ie420_plus_normal20_v1"


def build_control_dataset(ie_dir: Path, normal_dir: Path, output_dir: Path, defect_csv: Path) -> dict:
    ie_raw = pd.read_parquet(ie_dir / "node_timeseries.parquet")
    ie_res = pd.read_parquet(ie_dir / "node_timeseries_with_residuals.parquet")
    normal_raw = pd.read_parquet(normal_dir / "node_timeseries.parquet")
    normal_res = pd.read_parquet(normal_dir / "node_timeseries_with_residuals.parquet")

    combined_raw = pd.concat([ie_raw, normal_raw], ignore_index=True)
    combined_res = pd.concat([ie_res, normal_res], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_raw.to_parquet(output_dir / "node_timeseries.parquet", index=False, compression="gzip")
    combined_res.to_parquet(output_dir / "node_timeseries_with_residuals.parquet", index=False, compression="gzip")
    _write_summary(combined_res, output_dir / "scenario_summary.csv")

    scenario_type = combined_res.groupby("scenario_id")["defect_type"].first().astype(str).str.upper()
    manifest = {
        "dataset_type": "ie420_plus_normal20",
        "ie_dir": str(ie_dir),
        "normal_dir": str(normal_dir),
        "output_dir": str(output_dir),
        "defect_csv_path": str(defect_csv),
        "scenario_count": int(combined_res["scenario_id"].nunique()),
        "baseline_count": int((scenario_type == "BASELINE").sum()),
        "time_gated_count": int(ie_res["scenario_id"].nunique()),
        "persistent_count": 0,
        "normal_count": int(normal_res["scenario_id"].nunique()),
        "i_count_total": int((scenario_type == "I").sum()),
        "e_count_total": int((scenario_type == "E").sum()),
        "unique_node_count": int(combined_res["node_id"].nunique()),
    }
    _write_manifest(output_dir / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IE420 + normal20 control dataset.")
    parser.add_argument("--ie-dir", default=str(DEFAULT_IE_DIR))
    parser.add_argument("--normal-dir", default=str(DEFAULT_NORMAL20_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--normal-seeds",
        default="42,101,202,303,404,505,606,707,808,909,1001,1102,1203,1304,1405,1506,1607,1708,1809,1910",
    )
    parser.add_argument("--normal-start-id", type=int, default=800001)
    parser.add_argument("--flow-noise-pct", type=float, default=6.0)
    parser.add_argument("--quality-noise-pct", type=float, default=4.0)
    parser.add_argument("--daily-variation-pct", type=float, default=3.0)
    args = parser.parse_args()

    ie_dir = Path(args.ie_dir)
    normal_dir = Path(args.normal_dir)
    output_dir = Path(args.output_dir)
    defect_csv = _infer_ie_defect_csv(ie_dir)

    seeds = [int(x.strip()) for x in str(args.normal_seeds).split(",") if x.strip()]
    normal_manifest = build_normal_dataset(
        ie_dir=ie_dir,
        normal_dir=normal_dir,
        seeds=seeds,
        start_scenario_id=int(args.normal_start_id),
        flow_noise_pct=float(args.flow_noise_pct),
        quality_noise_pct=float(args.quality_noise_pct),
        daily_variation_pct=float(args.daily_variation_pct),
    )
    control_manifest = build_control_dataset(
        ie_dir=ie_dir,
        normal_dir=normal_dir,
        output_dir=output_dir,
        defect_csv=defect_csv,
    )
    print(json.dumps({"normal": normal_manifest, "control": control_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
