#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect formal Chapter-5 budget experiment metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


METRICS = {
    "mrr": "mrr",
    "top1": "topk_recall_1",
    "top3": "topk_recall_3",
    "top5": "topk_recall_5",
    "event_top1": "event_level_top1",
    "event_top3": "event_level_top3",
    "event_top5": "event_level_top5",
    "active_f1": "active_f1",
    "normal_fpr": "normal_window_fpr",
    "scene_f1": "scene_f1",
}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect formal Ch5 budget results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail when any metrics file is missing.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    records = []
    missing = []

    for row in rows:
        metrics_path = Path(row["metrics_file"])
        if not metrics_path.exists():
            missing.append(metrics_path)
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        record = {
            "method_key": row["method_key"],
            "method": row["method"],
            "budget": int(row["budget"]),
            "diagnosis_seed": int(row["diagnosis_seed"]),
            "output_tag": row["output_tag"],
            "metrics_file": str(metrics_path),
        }
        for output_name, json_key in METRICS.items():
            record[output_name] = data.get(json_key)
        records.append(record)

    if args.strict and missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} metrics files:\n{preview}")
    if not records:
        raise FileNotFoundError("No completed metrics files found for this manifest.")

    out_dir = manifest_path.parent
    stem = manifest_path.stem.replace("manifest", "results")
    per_seed_path = out_dir / f"{stem}_per_seed.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    status_path = out_dir / f"{stem}_status.csv"

    per_seed = pd.DataFrame(records).sort_values(
        ["method", "budget", "diagnosis_seed"]
    )
    per_seed.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    metric_columns = list(METRICS)
    summary = (
        per_seed.groupby(["method_key", "method", "budget"], as_index=False)[metric_columns]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in summary.columns.to_flat_index()
    ]
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    status_records = []
    for row in rows:
        status_records.append(
            {
                "method": row["method"],
                "budget": int(row["budget"]),
                "diagnosis_seed": int(row["diagnosis_seed"]),
                "status": "done" if Path(row["metrics_file"]).exists() else "pending",
                "metrics_file": row["metrics_file"],
            }
        )
    pd.DataFrame(status_records).to_csv(status_path, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {per_seed_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {status_path}")
    print(f"completed={len(records)} missing={len(missing)}")


if __name__ == "__main__":
    main()
