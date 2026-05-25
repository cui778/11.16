from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(r"E:\11.16\script2_new\chapter5_layout_optimization")
OUTPUT_DIR = ROOT / "outputs"
STRUCT_DIR = OUTPUT_DIR / "structural_innovation"

SUMMARY_FILES = [
    OUTPUT_DIR / "layout_summary.csv",
    OUTPUT_DIR / "two_stage_balanced_layout_v1_summary.csv",
    OUTPUT_DIR / "two_stage_balanced_layout_v2_summary.csv",
    OUTPUT_DIR / "two_stage_generalization_balanced_layout_v3_summary.csv",
    OUTPUT_DIR / "two_stage_generalization_balanced_layout_v3_1_summary.csv",
    OUTPUT_DIR / "two_stage_generalization_balanced_layout_v3_1a_summary.csv",
    OUTPUT_DIR / "two_stage_generalization_balanced_layout_v3_2_summary.csv",
    OUTPUT_DIR / "overlap_controlled_probe_summary.csv",
]

PERFORMANCE_FILES = [
    {"path": OUTPUT_DIR / "ch5_first_comparison_N25_seed42.csv", "split_mode": "scenario"},
    {"path": OUTPUT_DIR / "ch5_node_holdout_comparison_N25_seed42.csv", "split_mode": "node_holdout"},
    {"path": OUTPUT_DIR / "ch5_two_stage_framework_compare_N25_seed42.csv", "split_mode": "scenario"},
    {"path": OUTPUT_DIR / "ch5_two_stage_framework_node_holdout_compare_N25_seed42.csv", "split_mode": "node_holdout"},
    {"path": OUTPUT_DIR / "ch5_strengthened_main_method_v3_compare_N25_seed42.csv", "split_mode": None},
    {"path": OUTPUT_DIR / "ch5_overlap_controlled_probe_scenario_N25_seed42.csv", "split_mode": "scenario"},
    {"path": OUTPUT_DIR / "ch5_overlap_controlled_probe_node_holdout_N25_seed42.csv", "split_mode": "node_holdout"},
]

RESERVED_COLUMNS = {
    "method",
    "split_mode",
    "budget",
    "layout_file",
    "representative_pool_file",
    "metrics_file",
    "weight_preset",
}


def _first_present(row: pd.Series, names: Iterable[str], default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def load_summary_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.copy()
    if "strategy" in df.columns:
        df["method"] = df["strategy"]
    elif "layout" in df.columns:
        df["method"] = df["layout"]
    elif "method" not in df.columns:
        df["method"] = path.stem.replace("_summary", "")

    if "budget" not in df.columns:
        if "n" in df.columns:
            df["budget"] = pd.to_numeric(df["n"], errors="coerce")
        else:
            df["budget"] = None

    if "weight_preset" not in df.columns:
        df["weight_preset"] = None

    if "layout_file" not in df.columns:
        df["layout_file"] = None

    return df


def load_performance_table(spec: dict) -> pd.DataFrame:
    path = spec["path"]
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.copy()
    if "layout" in df.columns:
        df["method"] = df["layout"]
    if "split_mode" not in df.columns:
        df["split_mode"] = spec["split_mode"]
    df["budget"] = 25
    return df


def build_static_table() -> pd.DataFrame:
    frames = [load_summary_table(path) for path in SUMMARY_FILES]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    static_df = pd.concat(frames, ignore_index=True, sort=False)
    static_df = static_df.drop_duplicates(subset=["method", "budget"], keep="last")
    return static_df


def build_performance_table() -> pd.DataFrame:
    frames = [load_performance_table(spec) for spec in PERFORMANCE_FILES]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    perf_df = pd.concat(frames, ignore_index=True, sort=False)
    perf_df = perf_df.drop_duplicates(subset=["method", "budget", "split_mode"], keep="last")
    return perf_df


def build_long_table(static_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    if static_df.empty or perf_df.empty:
        return pd.DataFrame()

    long_df = perf_df.merge(static_df, on=["method", "budget"], how="left", suffixes=("", "_static"))
    long_df = long_df.sort_values(["budget", "method", "split_mode"]).reset_index(drop=True)
    return long_df


def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    metrics = [
        "mrr",
        "top1",
        "top3",
        "top5",
        "active_period_recall",
        "event_level_top1",
        "event_level_top3",
        "event_level_top5",
    ]

    static_cols = [column for column in long_df.columns if column not in {"split_mode", *metrics}]
    wide_df = long_df[static_cols].drop_duplicates(subset=["method", "budget"], keep="last").copy()

    for split_mode in sorted(long_df["split_mode"].dropna().unique().tolist()):
        split_df = long_df[long_df["split_mode"] == split_mode][["method", "budget", *metrics]].copy()
        split_df = split_df.rename(columns={metric: f"{split_mode}_{metric}" for metric in metrics})
        wide_df = wide_df.merge(split_df, on=["method", "budget"], how="left")

    wide_df = wide_df.sort_values(["budget", "method"]).reset_index(drop=True)
    return wide_df


def build_manifest(static_df: pd.DataFrame, perf_df: pd.DataFrame, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict:
    numeric_feature_columns = []
    if not wide_df.empty:
        numeric_feature_columns = [
            column
            for column in wide_df.columns
            if column not in RESERVED_COLUMNS
            and not column.startswith("scenario_")
            and not column.startswith("node_holdout_")
            and pd.api.types.is_numeric_dtype(wide_df[column])
        ]

    return {
        "static_layout_count": int(len(static_df)),
        "performance_record_count": int(len(perf_df)),
        "long_row_count": int(len(long_df)),
        "wide_row_count": int(len(wide_df)),
        "methods": sorted(long_df["method"].dropna().unique().tolist()) if not long_df.empty else [],
        "budgets": sorted([int(value) for value in long_df["budget"].dropna().unique().tolist()]) if not long_df.empty else [],
        "split_modes": sorted(long_df["split_mode"].dropna().unique().tolist()) if not long_df.empty else [],
        "numeric_feature_columns": numeric_feature_columns,
    }


def main() -> None:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    static_df = build_static_table()
    perf_df = build_performance_table()
    long_df = build_long_table(static_df, perf_df)
    wide_df = build_wide_table(long_df)
    manifest = build_manifest(static_df, perf_df, long_df, wide_df)

    long_path = STRUCT_DIR / "layout_quality_dataset_long.csv"
    wide_path = STRUCT_DIR / "layout_quality_dataset_wide.csv"
    manifest_path = STRUCT_DIR / "layout_quality_dataset_manifest.json"

    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] long dataset -> {long_path}")
    print(f"[OK] wide dataset -> {wide_path}")
    print(f"[OK] manifest -> {manifest_path}")
    print(f"[INFO] methods={manifest['methods']}")
    print(f"[INFO] splits={manifest['split_modes']}")


if __name__ == "__main__":
    main()
