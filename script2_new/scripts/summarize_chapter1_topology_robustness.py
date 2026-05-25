#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"e:\11.16")
REPORT_DIR = ROOT / "script2_new" / "outputs" / "reports"


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    files = {
        "baseline": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
        "edge_drop05": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_tpos1_trend0_kd0p0_akd0p0_topoedrop05_seed42.json",
        "edge_drop10": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_tpos1_trend0_kd0p0_akd0p0_topoedrop10_seed42.json",
        "edge_drop15": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_tpos1_trend0_kd0p0_akd0p0_topoedrop15_seed42.json",
    }
    drop_ratio = {"baseline": 0, "edge_drop05": 5, "edge_drop10": 10, "edge_drop15": 15}

    rows = []
    for tag, path in files.items():
        metrics = _load_metrics(path)
        rows.append(
            {
                "setting": tag,
                "drop_pct": drop_ratio[tag],
                "mrr": metrics["mrr"],
                "top1": metrics["topk_recall_1"],
                "top3": metrics["topk_recall_3"],
                "top5": metrics["topk_recall_5"],
                "active_period_recall": metrics.get("active_period_recall"),
                "detection_latency_mean": metrics.get("detection_latency_mean"),
                "event_level_top1": metrics.get("event_level_top1"),
                "event_level_top3": metrics.get("event_level_top3"),
                "event_level_top5": metrics.get("event_level_top5"),
                "metrics_file": str(path),
            }
        )

    df = pd.DataFrame(rows).sort_values("drop_pct").reset_index(drop=True)
    out = REPORT_DIR / "chapter1_topology_robustness_seed42.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
