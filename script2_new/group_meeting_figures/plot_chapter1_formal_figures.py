#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"e:\11.16")
REPORT_DIR = ROOT / "script2_new" / "outputs" / "reports"
FIG_DIR = REPORT_DIR / "figures"


def _set_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_model_comparison() -> Path:
    df = pd.read_csv(REPORT_DIR / "chapter1_fullgraph_model_comparison_multiseed.csv")
    keep = df[df["model"].isin(["hydraulic_inverse_deepattn", "lstm_graphsage_edge", "hydraulic_inverse"])].copy()
    keep["label"] = keep["model"].map(
        {
            "hydraulic_inverse_deepattn": "HydraulicInverse-DeepAttn",
            "lstm_graphsage_edge": "LSTM-GraphSAGE-Edge",
            "hydraulic_inverse": "HydraulicInverse",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(keep["label"], keep["top1_mean"], yerr=keep["top1_std"], color=["#2c7fb8", "#7fcdbb", "#fdae61"])
    axes[0].set_title("第一章主线模型对比：Top-1")
    axes[0].set_ylabel("Top-1")
    axes[0].set_ylim(0, 1.02)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(keep["label"], keep["mrr_mean"], yerr=keep["mrr_std"], color=["#2c7fb8", "#7fcdbb", "#fdae61"])
    axes[1].set_title("第一章主线模型对比：MRR")
    axes[1].set_ylabel("MRR")
    axes[1].set_ylim(0, 1.02)
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out = FIG_DIR / "chapter1_model_comparison_multiseed.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_generalization() -> Path:
    df = pd.read_csv(REPORT_DIR / "chapter1_fullgraph_node_holdout_multiseed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(df["split"], df["top1_mean"], yerr=df["top1_std"], color=["#1b9e77", "#d95f02"])
    axes[0].set_title("场景泛化 vs 节点泛化：Top-1")
    axes[0].set_ylabel("Top-1")
    axes[0].set_ylim(0, 1.02)

    axes[1].bar(df["split"], df["mrr_mean"], yerr=df["mrr_std"], color=["#1b9e77", "#d95f02"])
    axes[1].set_title("场景泛化 vs 节点泛化：MRR")
    axes[1].set_ylabel("MRR")
    axes[1].set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIG_DIR / "chapter1_generalization_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_time_ablation() -> Path:
    files = {
        "baseline(tpos on)": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_kd0p0_akd0p0_seed42.json",
        "no time pos": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_tpos0_trend0_kd0p0_akd0p0_seed42.json",
        "trend on": REPORT_DIR / "last_run_metrics_process_diagnosis_privileged_student_hydraulic_inverse_deepattn_monitor_nodes_degree_N25_time_gated_full_v2e_dense_ie_truefull_scenario_tpos1_trend1_kd0p0_akd0p0_seed42.json",
    }
    rows = []
    for label, path in files.items():
        d = _load_json(path)
        rows.append(
            {
                "setting": label,
                "top1": d["topk_recall_1"],
                "mrr": d["mrr"],
                "active_period_recall": d.get("active_period_recall", 0.0),
                "event_level_top1": d.get("event_level_top1", 0.0),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(df))
    axes[0].bar(x, df["top1"], width=0.38, label="Top-1", color="#3182bd")
    axes[0].bar([i + 0.38 for i in x], df["mrr"], width=0.38, label="MRR", color="#9ecae1")
    axes[0].set_xticks([i + 0.19 for i in x], df["setting"], rotation=15)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("时间维度首轮消融：定位指标")
    axes[0].legend()

    axes[1].bar(x, df["active_period_recall"], width=0.38, label="Active Recall", color="#31a354")
    axes[1].bar([i + 0.38 for i in x], df["event_level_top1"], width=0.38, label="Event Top-1", color="#a1d99b")
    axes[1].set_xticks([i + 0.19 for i in x], df["setting"], rotation=15)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("时间维度首轮消融：过程级指标")
    axes[1].legend()

    fig.tight_layout()
    out = FIG_DIR / "chapter1_time_ablation_seed42.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_topology_robustness() -> Path:
    df = pd.read_csv(REPORT_DIR / "chapter1_topology_robustness_seed42.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(df["drop_pct"], df["top1"], marker="o", linewidth=2.0, color="#ef6548")
    axes[0].set_title("拓扑先验扰动强度 vs Top-1")
    axes[0].set_xlabel("删边比例 (%)")
    axes[0].set_ylabel("Top-1")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["drop_pct"], df["mrr"], marker="o", linewidth=2.0, color="#2b8cbe")
    axes[1].set_title("拓扑先验扰动强度 vs MRR")
    axes[1].set_xlabel("删边比例 (%)")
    axes[1].set_ylabel("MRR")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out = FIG_DIR / "chapter1_topology_robustness_seed42.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _set_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_model_comparison(),
        plot_generalization(),
        plot_time_ablation(),
        plot_topology_robustness(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
