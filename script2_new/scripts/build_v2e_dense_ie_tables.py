#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


REPORT_DIR = Path(r"e:\11.16\script2_new\outputs\reports")

STAGE_FILES = [
    "last_run_metrics_process_diagnosis_hydraulic_inverse_deepattn_time_gated_node_sensor_v2e_dense_ie_residual8_scenario_seed42.json",
    "last_run_metrics_process_diagnosis_hydraulic_inverse_deepattn_time_gated_node_sensor_v2e_dense_ie_residual8_scenario_seed7.json",
    "last_run_metrics_process_diagnosis_hydraulic_inverse_deepattn_time_gated_node_sensor_v2e_dense_ie_residual8_scenario_seed123.json",
]

RERANK_FILES = [
    "topk_reranker_v2e_dense_ie_seed42_emb16_onehot_mlp.json",
    "topk_reranker_v2e_dense_ie_seed7_emb16_onehot_mlp.json",
    "topk_reranker_v2e_dense_ie_seed123_emb16_onehot_mlp.json",
]


def _load_json(name: str):
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def _mean_std(values):
    values = [float(v) for v in values]
    return statistics.mean(values), statistics.pstdev(values)


def build():
    stage_rows = [_load_json(name) for name in STAGE_FILES]
    rerank_rows = [_load_json(name)["test_with_best_val_setting"] for name in RERANK_FILES]

    stage_summary = {"config": "v2e_dense_ie_stage1"}
    for key in ["mrr", "top1", "top3", "top5", "active_period_recall", "event_level_top1"]:
        mean, std = _mean_std([row[key] for row in stage_rows])
        stage_summary[f"{key}_mean"] = mean
        stage_summary[f"{key}_std"] = std

    rerank_summary = {"config": "v2e_dense_ie_reranker"}
    for key in ["mrr", "top1", "top3", "top5"]:
        mean, std = _mean_std([row[key] for row in rerank_rows])
        rerank_summary[f"{key}_mean"] = mean
        rerank_summary[f"{key}_std"] = std

    with (REPORT_DIR / "defect_matrix_v2e_dense_ie_multiseed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_summary.keys()))
        writer.writeheader()
        writer.writerow(stage_summary)

    compare_rows = [
        {
            "config": "v2e_dense_ie_stage1",
            "mrr_mean": stage_summary["mrr_mean"],
            "mrr_std": stage_summary["mrr_std"],
            "top1_mean": stage_summary["top1_mean"],
            "top1_std": stage_summary["top1_std"],
            "top3_mean": stage_summary["top3_mean"],
            "top3_std": stage_summary["top3_std"],
            "top5_mean": stage_summary["top5_mean"],
            "top5_std": stage_summary["top5_std"],
        },
        rerank_summary,
    ]
    with (REPORT_DIR / "defect_matrix_v2e_dense_ie_with_reranker_multiseed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    by_type_rows = []
    for defect_type in ["I", "E"]:
        mean, std = _mean_std([(row.get("by_type_top1") or {}).get(defect_type, 0.0) for row in stage_rows])
        by_type_rows.append({"type": defect_type, "top1_mean": mean, "top1_std": std})

    with (REPORT_DIR / "defect_matrix_v2e_dense_ie_by_type_top1_multiseed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["type", "top1_mean", "top1_std"])
        writer.writeheader()
        writer.writerows(by_type_rows)

    with (REPORT_DIR / "ie_only_v2e_dense_mainline_formal_table_multiseed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "mrr_mean",
                "mrr_std",
                "top1_mean",
                "top1_std",
                "top3_mean",
                "top3_std",
                "top5_mean",
                "top5_std",
                "active_period_recall_mean",
                "active_period_recall_std",
                "event_level_top1_mean",
                "event_level_top1_std",
            ],
        )
        writer.writeheader()
        writer.writerow(stage_summary)

    per_seed_rows = []
    for seed_name, row in zip(["42", "7", "123"], stage_rows):
        per_seed_rows.append(
            {
                "seed": seed_name,
                "config": "v2e_dense_ie_stage1",
                "mrr": float(row["mrr"]),
                "top1": float(row["top1"]),
                "top3": float(row["top3"]),
                "top5": float(row["top5"]),
                "active_period_recall": float(row["active_period_recall"]),
                "event_level_top1": float(row["event_level_top1"]),
                "type_top1_I": float((row.get("by_type_top1") or {}).get("I", 0.0)),
                "type_top1_E": float((row.get("by_type_top1") or {}).get("E", 0.0)),
            }
        )
    with (REPORT_DIR / "ie_only_v2e_dense_mainline_formal_table_per_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    print("defect_matrix_v2e_dense_ie_multiseed.csv")
    print("defect_matrix_v2e_dense_ie_with_reranker_multiseed.csv")
    print("defect_matrix_v2e_dense_ie_by_type_top1_multiseed.csv")
    print("ie_only_v2e_dense_mainline_formal_table_multiseed.csv")
    print("ie_only_v2e_dense_mainline_formal_table_per_seed.csv")
    print(stage_summary)
    print(rerank_summary)
    print(by_type_rows)


if __name__ == "__main__":
    build()
