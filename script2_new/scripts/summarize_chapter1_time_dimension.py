#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports"

RUNS = [
    {
        "setting": "time_gated_baseline",
        "label_mode": "time_gated",
        "time_pos": 1,
        "trend": 0,
        "metrics_file": REPORT_DIR / "last_run_metrics_ch1_fullgraph_degree_ie420_s42_fix1.json",
    },
    {
        "setting": "time_gated_no_time_pos",
        "label_mode": "time_gated",
        "time_pos": 0,
        "trend": 0,
        "metrics_file": REPORT_DIR / "last_run_metrics_ch1_fullgraph_degree_ie420_tpos0_s42.json",
    },
    {
        "setting": "time_gated_with_trend",
        "label_mode": "time_gated",
        "time_pos": 1,
        "trend": 1,
        "metrics_file": REPORT_DIR / "last_run_metrics_ch1_fullgraph_degree_ie420_trend1_s42.json",
    },
    {
        "setting": "always_on",
        "label_mode": "always_on",
        "time_pos": 1,
        "trend": 0,
        "metrics_file": REPORT_DIR / "last_run_metrics_ch1_fullgraph_degree_ie420_alwayson_s42.json",
    },
]

OUT_CSV = REPORT_DIR / "chapter1_restart_time_dimension_seed42.csv"


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []
    for run in RUNS:
        metrics = load_metrics(run["metrics_file"])
        rows.append(
            {
                "setting": run["setting"],
                "label_mode": run["label_mode"],
                "time_pos": run["time_pos"],
                "trend": run["trend"],
                "mrr": float(metrics.get("mrr", 0.0)),
                "top1": float(metrics.get("topk_recall_1", 0.0)),
                "top3": float(metrics.get("topk_recall_3", 0.0)),
                "top5": float(metrics.get("topk_recall_5", 0.0)),
                "active_period_recall": float(metrics.get("active_period_recall", 0.0)),
                "detection_latency_mean": float(metrics.get("detection_latency_mean", 0.0)),
                "event_level_top1": float(metrics.get("event_level_top1", 0.0)),
                "event_level_top3": float(metrics.get("event_level_top3", 0.0)),
                "event_level_top5": float(metrics.get("event_level_top5", 0.0)),
                "metrics_file": run["metrics_file"].name,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved -> {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
