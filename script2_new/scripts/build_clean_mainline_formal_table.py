#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import statistics
from pathlib import Path


REPORT_DIR = Path(r"E:\11.16\script2_new\outputs\reports")

FILES = [
    REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario.json",
    REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
    REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
    REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed42.json",
    REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
    REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
]

METRICS = [
    "active_period_recall",
    "mrr",
    "top1",
    "top3",
    "top5",
    "event_level_top1",
    "event_level_top3",
    "event_level_top5",
    "detection_latency_mean",
]


def load_rows():
    rows = []
    for path in FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "model": data["config_snapshot"]["model_type"],
            "seed": int(data["seed"]),
        }
        for metric in METRICS:
            row[metric] = float(data[metric])
        rows.append(row)
    return rows


def build_summary(rows):
    summary = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        item = {
            "model": model,
            "seeds": "/".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def write_csv(rows, summary):
    with (REPORT_DIR / "clean_mainline_formal_table_per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed"] + METRICS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["model"], x["seed"])))

    with (REPORT_DIR / "clean_mainline_formal_table_multiseed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fieldnames = ["model", "seeds"]
        for metric in METRICS:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main():
    rows = load_rows()
    summary = build_summary(rows)
    write_csv(rows, summary)
    print("clean_mainline_formal_table_per_seed.csv")
    print("clean_mainline_formal_table_multiseed.csv")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
