#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build formal tables for the reranked clean mainline.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


REPORT_DIR = Path(r"e:\11.16\script2_new\outputs\reports")
INPUT_FILES = [
    REPORT_DIR / "topk_reranker_by_type_refresh42.json",
    REPORT_DIR / "topk_reranker_by_type_seed7.json",
    REPORT_DIR / "topk_reranker_by_type_seed123.json",
]


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def main():
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in INPUT_FILES]

    overall_rows = []
    for payload in payloads:
        m = payload["test_metrics"]
        overall_rows.append(
            {
                "seed": payload["seed"],
                "topk": payload["selected_setting"]["topk"],
                "tau": payload["selected_setting"]["tau"],
                "c_value": payload["selected_setting"]["c_value"],
                "mrr": m["mrr"],
                "top1": m["top1"],
                "top3": m["top3"],
                "top5": m["top5"],
                "top10": m["top10"],
                "top20": m["top20"],
            }
        )

    with (REPORT_DIR / "reranked_clean_mainline_formal_table_per_seed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overall_rows)

    metric_names = ["mrr", "top1", "top3", "top5", "top10", "top20"]
    overall_summary = {"model": "hydraulic_inverse + topk_reranker", "seeds": "42/7/123"}
    for metric in metric_names:
        values = [row[metric] for row in overall_rows]
        overall_summary[f"{metric}_mean"] = _mean(values)
        overall_summary[f"{metric}_std"] = _std(values)

    with (REPORT_DIR / "reranked_clean_mainline_formal_table_multiseed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_summary.keys()))
        writer.writeheader()
        writer.writerow(overall_summary)

    by_type_rows = []
    for defect_type in ["I", "E", "P"]:
        row = {"defect_type": defect_type}
        for metric in ["mrr", "top1", "top3", "top5"]:
            values = [payload["test_by_type"][defect_type][metric] for payload in payloads]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
        row["n_samples_mean"] = _mean([payload["test_by_type"][defect_type]["n_samples"] for payload in payloads])
        by_type_rows.append(row)

    with (REPORT_DIR / "reranked_clean_mainline_by_type_multiseed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_type_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_type_rows)


if __name__ == "__main__":
    main()
