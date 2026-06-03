#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build data directories for the 48 h IE diagnosis restructuring.

Outputs:
1. normal_multibaseline_v1_seedset10
2. persistent_ie_fullwindow_v1_seed42 (finalized as 84 defect scenarios, no baseline)
3. ie420_plus_normal_multibaseline_v1_seedset10

The normal scenes are generated from the verified seed42 baseline response with
controlled temporal perturbations. This script keeps the graph/topology fixed
and creates normal operating variability for detection/FPR training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


RAW_FEATURES = [
    "depth",
    "head",
    "volume",
    "lateral_inflow",
    "total_inflow",
    "total_outflow",
    "flooding",
    "pollut_BODf",
    "pollut_BODs",
    "pollut_NH4",
    "pollut_NO3",
    "pollut_DO",
    "pollut_TSSs",
]


NOISE_FEATURES = [
    "depth",
    "lateral_inflow",
    "total_inflow",
    "total_outflow",
    "pollut_BODf",
    "pollut_BODs",
    "pollut_NH4",
    "pollut_NO3",
    "pollut_DO",
    "pollut_TSSs",
]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _ensure_time_step(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values(["scenario_id", "datetime", "node_id"]).reset_index(drop=True)
    out["time_step"] = (
        out.groupby("scenario_id")["datetime"].rank(method="dense").astype(int) - 1
    )
    return out


def _baseline_frame(df: pd.DataFrame, baseline_scenario_id: int = 0) -> pd.DataFrame:
    baseline = df.loc[df["scenario_id"] == baseline_scenario_id].copy()
    if baseline.empty:
        raise ValueError(f"Baseline scenario {baseline_scenario_id} not found.")
    return _ensure_time_step(baseline)


def _ar1_noise(n: int, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    eps = np.zeros(n, dtype=np.float32)
    innovation = float(sigma) * np.sqrt(max(1.0 - float(rho) ** 2, 1e-8))
    for i in range(n):
        if i == 0:
            eps[i] = rng.normal(0.0, sigma)
        else:
            eps[i] = rho * eps[i - 1] + rng.normal(0.0, innovation)
    return eps


def _generate_one_normal_scene(
    baseline: pd.DataFrame,
    scenario_id: int,
    seed: int,
    flow_noise_pct: float,
    quality_noise_pct: float,
    daily_variation_pct: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    out = baseline.copy()
    out["scenario_id"] = int(scenario_id)
    out["defect_type"] = "NORMAL"

    # Preserve a coherent water-level relation where possible.
    original_depth = out["depth"].astype(float).to_numpy(copy=True)
    original_head = out["head"].astype(float).to_numpy(copy=True)

    for node_id, idx in out.groupby("node_id", sort=False).groups.items():
        idx_list = np.asarray(list(idx), dtype=np.int64)
        n = len(idx_list)
        node_shift = rng.normal(0.0, daily_variation_pct / 100.0)
        for col in NOISE_FEATURES:
            values = out.loc[idx_list, col].astype(float).to_numpy(copy=True)
            mean_abs = float(np.mean(np.abs(values)))
            if mean_abs <= 1e-12:
                continue
            pct = quality_noise_pct if str(col).startswith("pollut_") else flow_noise_pct
            rel = _ar1_noise(n, sigma=pct / 100.0, rho=0.45, rng=rng)
            perturbed = values * (1.0 + node_shift + rel)
            if col != "pollut_DO":
                perturbed = np.maximum(perturbed, 0.0)
            out.loc[idx_list, col] = perturbed

    depth_delta = out["depth"].astype(float).to_numpy() - original_depth
    out["head"] = original_head + depth_delta
    return out


def _add_residuals(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_time_step(df)
    base = baseline[["time_step", "node_id"] + RAW_FEATURES].copy()
    out = work.copy()
    rel_eps = 1e-6
    rel_clip = 10.0
    for col in RAW_FEATURES:
        merged = out[["time_step", "node_id", col]].merge(
            base[["time_step", "node_id", col]].rename(columns={col: f"{col}_baseline"}),
            on=["time_step", "node_id"],
            how="left",
        )
        residual = out[col].astype(float).to_numpy() - merged[f"{col}_baseline"].astype(float).to_numpy()
        out[f"{col}_residual"] = residual
        denom = np.abs(merged[f"{col}_baseline"].astype(float).to_numpy()) + rel_eps
        out[f"{col}_residual_rel"] = np.clip(residual / denom, -rel_clip, rel_clip)
    return out


def _write_manifest(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_summary(df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for scenario_id, grp in df.groupby("scenario_id", sort=True):
        rows.append(
            {
                "scenario_id": int(scenario_id),
                "defect_type": str(grp["defect_type"].iloc[0]),
                "record_count": int(len(grp)),
                "node_count": int(grp["node_id"].nunique()),
                "time_points": int(grp["datetime"].nunique()),
                "start_time": str(pd.to_datetime(grp["datetime"]).min()),
                "end_time": str(pd.to_datetime(grp["datetime"]).max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def build_normal_dataset(
    ie_dir: Path,
    normal_dir: Path,
    seeds: Iterable[int],
    start_scenario_id: int,
    flow_noise_pct: float,
    quality_noise_pct: float,
    daily_variation_pct: float,
) -> dict:
    raw = pd.read_parquet(ie_dir / "node_timeseries.parquet")
    baseline = _baseline_frame(raw, baseline_scenario_id=0)
    scenes = []
    for offset, seed in enumerate(seeds):
        scenes.append(
            _generate_one_normal_scene(
                baseline,
                scenario_id=int(start_scenario_id) + offset,
                seed=int(seed),
                flow_noise_pct=flow_noise_pct,
                quality_noise_pct=quality_noise_pct,
                daily_variation_pct=daily_variation_pct,
            )
        )
    normal_raw = pd.concat(scenes, ignore_index=True)
    normal_with_residuals = _add_residuals(normal_raw, baseline)

    normal_dir.mkdir(parents=True, exist_ok=True)
    normal_raw.drop(columns=["time_step"], errors="ignore").to_parquet(
        normal_dir / "node_timeseries.parquet", index=False, compression="gzip"
    )
    normal_with_residuals.to_parquet(
        normal_dir / "node_timeseries_with_residuals.parquet", index=False, compression="gzip"
    )
    _write_summary(normal_with_residuals, normal_dir / "scenario_summary.csv")
    manifest = {
        "dataset_type": "normal_multibaseline",
        "source_ie_dir": str(ie_dir),
        "output_dir": str(normal_dir),
        "normal_seeds": list(map(int, seeds)),
        "scenario_count": int(normal_with_residuals["scenario_id"].nunique()),
        "defect_count": 0,
        "unique_node_count": int(normal_with_residuals["node_id"].nunique()),
        "flow_noise_pct": float(flow_noise_pct),
        "quality_noise_pct": float(quality_noise_pct),
        "daily_variation_pct": float(daily_variation_pct),
    }
    _write_manifest(normal_dir / "dataset_manifest.json", manifest)
    return manifest


def finalize_persistent_dataset(ie_dir: Path, persistent_dir: Path) -> dict:
    raw_ie = pd.read_parquet(ie_dir / "node_timeseries.parquet")
    baseline = _baseline_frame(raw_ie, baseline_scenario_id=0)
    raw_persistent = pd.read_parquet(persistent_dir / "node_timeseries.parquet")
    raw_persistent = raw_persistent.loc[raw_persistent["scenario_id"] != 0].copy()
    raw_persistent["defect_type"] = raw_persistent["defect_type"].astype(str).str.upper()
    persistent_with_residuals = _add_residuals(raw_persistent, baseline)

    raw_persistent.drop(columns=["time_step"], errors="ignore").to_parquet(
        persistent_dir / "node_timeseries.parquet", index=False, compression="gzip"
    )
    persistent_with_residuals.to_parquet(
        persistent_dir / "node_timeseries_with_residuals.parquet", index=False, compression="gzip"
    )
    _write_summary(persistent_with_residuals, persistent_dir / "scenario_summary.csv")
    manifest = {
        "dataset_type": "persistent_ie_zero_shot",
        "source_ie_dir": str(ie_dir),
        "output_dir": str(persistent_dir),
        "scenario_count": int(persistent_with_residuals["scenario_id"].nunique()),
        "baseline_included": False,
        "i_count": int((persistent_with_residuals["defect_type"] == "I").groupby(persistent_with_residuals["scenario_id"]).first().sum()),
        "e_count": int((persistent_with_residuals["defect_type"] == "E").groupby(persistent_with_residuals["scenario_id"]).first().sum()),
        "unique_node_count": int(persistent_with_residuals["node_id"].nunique()),
        "residual_reference": "source IE scenario_id=0 baseline",
    }
    _write_manifest(persistent_dir / "dataset_manifest.json", manifest)
    return manifest


def build_combined_dataset(ie_dir: Path, normal_dir: Path, combined_dir: Path) -> dict:
    ie_raw = pd.read_parquet(ie_dir / "node_timeseries.parquet")
    normal_raw = pd.read_parquet(normal_dir / "node_timeseries.parquet")
    ie_res = pd.read_parquet(ie_dir / "node_timeseries_with_residuals.parquet")
    normal_res = pd.read_parquet(normal_dir / "node_timeseries_with_residuals.parquet")

    combined_raw = pd.concat([ie_raw, normal_raw], ignore_index=True)
    combined_res = pd.concat([ie_res, normal_res], ignore_index=True)

    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_raw.to_parquet(combined_dir / "node_timeseries.parquet", index=False, compression="gzip")
    combined_res.to_parquet(combined_dir / "node_timeseries_with_residuals.parquet", index=False, compression="gzip")
    _write_summary(combined_res, combined_dir / "scenario_summary.csv")

    ie_manifest_path = ie_dir / "dataset_manifest.json"
    ie_manifest = {}
    if ie_manifest_path.exists():
        with open(ie_manifest_path, "r", encoding="utf-8") as f:
            ie_manifest = json.load(f)
    manifest = {
        "dataset_type": "ie420_plus_normal_multibaseline",
        "ie_dir": str(ie_dir),
        "normal_dir": str(normal_dir),
        "output_dir": str(combined_dir),
        "defect_csv_path": ie_manifest.get("defect_csv_path", ""),
        "scenario_count": int(combined_res["scenario_id"].nunique()),
        "normal_count": int(normal_res["scenario_id"].nunique()),
        "ie_count": int(ie_res["scenario_id"].nunique()),
        "unique_node_count": int(combined_res["node_id"].nunique()),
    }
    _write_manifest(combined_dir / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 48 h restructured IE diagnosis datasets.")
    root = Path(r"E:\11.16\script2_new")
    parser.add_argument("--ie-dir", default=str(root / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42"))
    parser.add_argument("--normal-dir", default=str(root / "training_data_new" / "normal_multibaseline_v1_seedset10"))
    parser.add_argument("--persistent-dir", default=str(root / "training_data_new" / "persistent_ie_fullwindow_v1_seed42"))
    parser.add_argument("--combined-dir", default=str(root / "training_data_new" / "ie420_plus_normal_multibaseline_v1_seedset10"))
    parser.add_argument("--normal-seeds", default="42,101,202,303,404,505,606,707,808,909")
    parser.add_argument("--normal-start-id", type=int, default=800001)
    parser.add_argument("--flow-noise-pct", type=float, default=6.0)
    parser.add_argument("--quality-noise-pct", type=float, default=4.0)
    parser.add_argument("--daily-variation-pct", type=float, default=3.0)
    parser.add_argument("--skip-persistent", action="store_true")
    args = parser.parse_args()

    ie_dir = Path(args.ie_dir)
    normal_dir = Path(args.normal_dir)
    persistent_dir = Path(args.persistent_dir)
    combined_dir = Path(args.combined_dir)
    seeds = _parse_int_list(args.normal_seeds)

    normal_manifest = build_normal_dataset(
        ie_dir=ie_dir,
        normal_dir=normal_dir,
        seeds=seeds,
        start_scenario_id=int(args.normal_start_id),
        flow_noise_pct=float(args.flow_noise_pct),
        quality_noise_pct=float(args.quality_noise_pct),
        daily_variation_pct=float(args.daily_variation_pct),
    )
    persistent_manifest = None
    if not args.skip_persistent and (persistent_dir / "node_timeseries.parquet").exists():
        persistent_manifest = finalize_persistent_dataset(ie_dir=ie_dir, persistent_dir=persistent_dir)
    combined_manifest = build_combined_dataset(ie_dir=ie_dir, normal_dir=normal_dir, combined_dir=combined_dir)

    print(json.dumps(
        {
            "normal": normal_manifest,
            "persistent": persistent_manifest,
            "combined": combined_manifest,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
