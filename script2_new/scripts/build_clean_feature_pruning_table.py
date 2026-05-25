#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import statistics
from pathlib import Path


REPORT_DIR = Path(r"E:\11.16\script2_new\outputs\reports")

FEATURE_GROUPS = {
    "residual12": [
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario_seed123.json",
    ],
    "residual10": [
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual10_scenario.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual10_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual10_scenario_seed123.json",
    ],
    "residual8": [
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual8_scenario.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual8_scenario_seed7.json",
        REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual8_scenario_seed123.json",
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

TYPE_KEYS = ["I", "E", "P"]


def load_rows():
    rows = []
    for feature_scope, paths in FEATURE_GROUPS.items():
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "feature_scope": feature_scope,
                "seed": int(data["seed"]),
            }
            for metric in METRICS:
                row[metric] = float(data[metric])
            by_type = data.get("by_type_top1", {})
            for defect_type in TYPE_KEYS:
                row[f"top1_{defect_type}"] = float(by_type.get(defect_type, 0.0))
            rows.append(row)
    return rows


def build_summary(rows):
    summary = []
    for feature_scope in FEATURE_GROUPS:
        group = [row for row in rows if row["feature_scope"] == feature_scope]
        item = {
            "feature_scope": feature_scope,
            "seeds": "/".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for defect_type in TYPE_KEYS:
            values = [row[f"top1_{defect_type}"] for row in group]
            item[f"top1_{defect_type}_mean"] = statistics.mean(values)
            item[f"top1_{defect_type}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def write_csv(rows, summary):
    with (REPORT_DIR / "clean_feature_pruning_per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fieldnames = ["feature_scope", "seed"] + METRICS + [f"top1_{k}" for k in TYPE_KEYS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["feature_scope"], x["seed"])))

    with (REPORT_DIR / "clean_feature_pruning_multiseed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fieldnames = ["feature_scope", "seeds"]
        for metric in METRICS:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        for defect_type in TYPE_KEYS:
            fieldnames.extend([f"top1_{defect_type}_mean", f"top1_{defect_type}_std"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main():
    rows = load_rows()
    summary = build_summary(rows)
    write_csv(rows, summary)
    print("clean_feature_pruning_per_seed.csv")
    print("clean_feature_pruning_multiseed.csv")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
