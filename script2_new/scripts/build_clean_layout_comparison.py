#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import statistics
from pathlib import Path


REPORT_DIR = Path(r"E:\11.16\script2_new\outputs\reports")

FILE_GROUPS = {
    "degree_N25": [
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed42.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
    ],
    "downstream_N25": [
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_downstream_node_sensor_v2_rebuild_residual12_scenario.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_downstream_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_downstream_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_downstream_node_sensor_v2_rebuild_residual12_scenario.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_downstream_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_downstream_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
    ],
}

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

TYPE_ORDER = ["I", "E", "P"]


def load_rows():
    rows = []
    for layout, files in FILE_GROUPS.items():
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "layout": layout,
                "model": data["config_snapshot"]["model_type"],
                "seed": int(data["seed"]),
            }
            for metric in METRICS:
                row[metric] = float(data[metric])
            for defect_type in TYPE_ORDER:
                row[f"type_top1_{defect_type}"] = float(data.get("by_type_top1", {}).get(defect_type, 0.0))
            rows.append(row)
    return rows


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def build_summary(rows):
    summary = []
    for layout in sorted({row["layout"] for row in rows}):
        for model in sorted({row["model"] for row in rows if row["layout"] == layout}):
            group = [row for row in rows if row["layout"] == layout and row["model"] == model]
            item = {
                "layout": layout,
                "model": model,
                "seeds": "/".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
            }
            for metric in METRICS:
                values = [row[metric] for row in group]
                item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(values)
            for defect_type in TYPE_ORDER:
                values = [row[f"type_top1_{defect_type}"] for row in group]
                item[f"type_top1_{defect_type}_mean"], item[f"type_top1_{defect_type}_std"] = mean_std(values)
            summary.append(item)
    return summary


def write_csv(rows, summary):
    per_seed_path = REPORT_DIR / "clean_layout_comparison_per_seed.csv"
    summary_path = REPORT_DIR / "clean_layout_comparison_multiseed.csv"
    type_path = REPORT_DIR / "clean_layout_type_top1_multiseed.csv"

    with per_seed_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["layout", "model", "seed"] + METRICS + [f"type_top1_{t}" for t in TYPE_ORDER]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["layout"], x["model"], x["seed"])))

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["layout", "model", "seeds"]
        for metric in METRICS:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{key: item[key] for key in fieldnames} for item in summary]
        )

    with type_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["layout", "model", "seeds"]
        for defect_type in TYPE_ORDER:
            fieldnames.extend([f"type_top1_{defect_type}_mean", f"type_top1_{defect_type}_std"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{key: item[key] for key in fieldnames} for item in summary]
        )


def main():
    rows = load_rows()
    summary = build_summary(rows)
    write_csv(rows, summary)
    print("clean_layout_comparison_per_seed.csv")
    print("clean_layout_comparison_multiseed.csv")
    print("clean_layout_type_top1_multiseed.csv")


if __name__ == "__main__":
    main()
