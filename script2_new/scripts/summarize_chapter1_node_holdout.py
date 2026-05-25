from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path


ROOT = Path(r"e:\11.16")
REPORTS = ROOT / "script2_new" / "outputs" / "reports"


SCENARIO_ITEMS = [
    (
        42,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
    ),
    (
        7,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed7.json",
    ),
    (
        123,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed123.json",
    ),
]


NODE_HOLDOUT_ITEMS = [
    (
        42,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_node_holdout_kd0p0_akd0p0_seed42.json",
    ),
    (
        7,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_node_holdout_kd0p0_akd0p0_seed7.json",
    ),
    (
        123,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_node_holdout_kd0p0_akd0p0_seed123.json",
    ),
]


def load_rows(split_name: str, items: list[tuple[int, Path]]) -> list[dict[str, float]]:
    rows = []
    for seed, path in items:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "split": split_name,
                "seed": seed,
                "mrr": data["mrr"],
                "top1": data["topk_recall_1"],
                "top3": data["topk_recall_3"],
                "top5": data["topk_recall_5"],
                "active_period_recall": data.get("active_period_recall"),
                "detection_latency_mean": data.get("detection_latency_mean"),
                "event_level_top1": data.get("event_level_top1"),
                "event_level_top3": data.get("event_level_top3"),
                "event_level_top5": data.get("event_level_top5"),
            }
        )
    return rows


def summarize(rows: list[dict[str, float]], split_name: str) -> dict[str, float]:
    vals = [r for r in rows if r["split"] == split_name]
    return {
        "split": split_name,
        "n_seeds": len(vals),
        "mrr_mean": sum(v["mrr"] for v in vals) / len(vals),
        "mrr_std": st.pstdev([v["mrr"] for v in vals]) if len(vals) > 1 else 0.0,
        "top1_mean": sum(v["top1"] for v in vals) / len(vals),
        "top1_std": st.pstdev([v["top1"] for v in vals]) if len(vals) > 1 else 0.0,
        "top3_mean": sum(v["top3"] for v in vals) / len(vals),
        "top3_std": st.pstdev([v["top3"] for v in vals]) if len(vals) > 1 else 0.0,
        "top5_mean": sum(v["top5"] for v in vals) / len(vals),
        "top5_std": st.pstdev([v["top5"] for v in vals]) if len(vals) > 1 else 0.0,
        "active_period_recall_mean": sum(v["active_period_recall"] for v in vals) / len(vals),
        "detection_latency_mean_mean": sum(v["detection_latency_mean"] for v in vals) / len(vals),
        "event_level_top1_mean": sum(v["event_level_top1"] for v in vals) / len(vals),
        "event_level_top3_mean": sum(v["event_level_top3"] for v in vals) / len(vals),
        "event_level_top5_mean": sum(v["event_level_top5"] for v in vals) / len(vals),
    }


def main() -> None:
    rows = load_rows("scenario", SCENARIO_ITEMS) + load_rows("node_holdout", NODE_HOLDOUT_ITEMS)

    per_seed_path = REPORTS / "chapter1_fullgraph_node_holdout_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "seed",
                "mrr",
                "top1",
                "top3",
                "top5",
                "active_period_recall",
                "detection_latency_mean",
                "event_level_top1",
                "event_level_top3",
                "event_level_top5",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["split"], r["seed"])))

    summary_rows = [summarize(rows, "scenario"), summarize(rows, "node_holdout")]
    summary_path = REPORTS / "chapter1_fullgraph_node_holdout_multiseed.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "n_seeds",
                "mrr_mean",
                "mrr_std",
                "top1_mean",
                "top1_std",
                "top3_mean",
                "top3_std",
                "top5_mean",
                "top5_std",
                "active_period_recall_mean",
                "detection_latency_mean_mean",
                "event_level_top1_mean",
                "event_level_top3_mean",
                "event_level_top5_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
