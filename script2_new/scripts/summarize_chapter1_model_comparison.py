from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"e:\11.16")
REPORTS = ROOT / "script2_new" / "outputs" / "reports"


ITEMS = [
    (
        "hydraulic_inverse_deepattn",
        42,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
    ),
    (
        "hydraulic_inverse_deepattn",
        7,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed7.json",
    ),
    (
        "hydraulic_inverse_deepattn",
        123,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed123.json",
    ),
    (
        "hydraulic_inverse",
        42,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
    ),
    (
        "hydraulic_inverse",
        7,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_kd0p0_akd0p0_seed7.json",
    ),
    (
        "hydraulic_inverse",
        123,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_kd0p0_akd0p0_seed123.json",
    ),
    (
        "lstm_graphsage_edge",
        42,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_lstm_graphsage_edge_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
    ),
    (
        "lstm_graphsage_edge",
        7,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_lstm_graphsage_edge_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_kd0p0_akd0p0_seed7.json",
    ),
    (
        "lstm_graphsage_edge",
        123,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_lstm_graphsage_edge_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_kd0p0_akd0p0_seed123.json",
    ),
    (
        "gru_gcn",
        42,
        REPORTS
        / "last_run_metrics_process_diagnosis_privileged_student_gru_gcn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
    ),
]


def main() -> None:
    rows = []
    for model, seed, path in ITEMS:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "mrr": data["mrr"],
                "top1": data["topk_recall_1"],
                "top3": data["topk_recall_3"],
                "top5": data["topk_recall_5"],
            }
        )

    per_seed_path = REPORTS / "chapter1_fullgraph_model_comparison_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "mrr", "top1", "top3", "top5"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["seed"])))

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    summary_rows = []
    for model, vals in grouped.items():
        summary_rows.append(
            {
                "model": model,
                "n_seeds": len(vals),
                "mrr_mean": sum(v["mrr"] for v in vals) / len(vals),
                "mrr_std": st.pstdev([v["mrr"] for v in vals]) if len(vals) > 1 else 0.0,
                "top1_mean": sum(v["top1"] for v in vals) / len(vals),
                "top1_std": st.pstdev([v["top1"] for v in vals]) if len(vals) > 1 else 0.0,
                "top3_mean": sum(v["top3"] for v in vals) / len(vals),
                "top3_std": st.pstdev([v["top3"] for v in vals]) if len(vals) > 1 else 0.0,
                "top5_mean": sum(v["top5"] for v in vals) / len(vals),
                "top5_std": st.pstdev([v["top5"] for v in vals]) if len(vals) > 1 else 0.0,
            }
        )

    summary_rows.sort(key=lambda r: r["top1_mean"], reverse=True)
    summary_path = REPORTS / "chapter1_fullgraph_model_comparison_multiseed.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "n_seeds",
                "mrr_mean",
                "mrr_std",
                "top1_mean",
                "top1_std",
                "top3_mean",
                "top3_std",
                "top5_mean",
                "top5_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
