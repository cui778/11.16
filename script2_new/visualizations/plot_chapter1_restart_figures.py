#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports"
CKPT_DIR = ROOT / "outputs" / "model_checkpoints"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_loss_curves() -> None:
    runs = [
        ("hydraulic_inverse_deepattn", CKPT_DIR / "training_history_ch1_fullgraph_degree_ie420_s42_fix1.csv"),
        ("lstm_graphsage_edge", CKPT_DIR / "training_history_ch1_fullgraph_degree_ie420_lgse_s42.csv"),
        ("hydraulic_inverse", CKPT_DIR / "training_history_ch1_fullgraph_degree_ie420_hinv_s42.csv"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (title, path) in zip(axes, runs):
        df = pd.read_csv(path)
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#444444", linewidth=1.8)
        ax.plot(df["epoch"], df["val_mrr"], label="Val MRR", color="#d95f02", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Chapter 1 Mainline Training Dynamics", fontsize=13)
    fig.savefig(FIG_DIR / "chapter1_restart_loss_curves_seed42.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison() -> None:
    df = pd.read_csv(REPORT_DIR / "chapter1_restart_model_comparison_seed42.csv")
    metrics = ["top1", "top3", "top5", "mrr"]
    labels = ["Top-1", "Top-3", "Top-5", "MRR"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.18
    x = range(len(df))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax.bar([v + (i - 1.5) * width for v in x], df[metric], width=width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Chapter 1 Model Comparison under the Restarted Formal Protocol (seed=42)")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "chapter1_restart_model_comparison_seed42.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_split_comparison() -> None:
    scenario = pd.read_csv(REPORT_DIR / "chapter1_restart_fullgraph_degree_ie420_multiseed.csv")
    nodehold = pd.read_csv(REPORT_DIR / "chapter1_restart_nodehold_multiseed.csv")
    scenario_mean = scenario.mean(numeric_only=True)
    nodehold_mean = nodehold[nodehold["seed"] == "mean"].iloc[0]
    metrics = ["top1", "top3", "top5", "mrr"]
    labels = ["Top-1", "Top-3", "Top-5", "MRR"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(metrics))
    width = 0.34
    ax.bar([v - width / 2 for v in x], [float(scenario_mean[m]) for m in metrics], width=width, label="Scenario split", color="#4c78a8")
    ax.bar([v + width / 2 for v in x], [float(nodehold_mean[m]) for m in metrics], width=width, label="Node holdout", color="#e45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Scenario Generalization vs Node Generalization")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "chapter1_restart_split_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_time_dimension() -> None:
    df = pd.read_csv(REPORT_DIR / "chapter1_restart_time_dimension_seed42.csv")
    order = ["time_gated_baseline", "time_gated_no_time_pos", "time_gated_with_trend", "always_on"]
    df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)
    df = df.sort_values("setting")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    metrics_loc = ["top1", "top3", "top5", "mrr"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    width = 0.18
    x = range(len(df))
    for i, (metric, color) in enumerate(zip(metrics_loc, colors)):
        axes[0].bar([v + (i - 1.5) * width for v in x], df[metric], width=width, label=metric.upper(), color=color)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df["setting"], rotation=12)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Localization Metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(df["setting"], df["active_period_recall"], marker="o", linewidth=2.0, color="#54a24b", label="Active period recall")
    axes[1].plot(df["setting"], df["event_level_top1"], marker="o", linewidth=2.0, color="#e45756", label="Event-level Top-1")
    axes[1].plot(df["setting"], df["event_level_top3"], marker="o", linewidth=2.0, color="#4c78a8", label="Event-level Top-3")
    axes[1].plot(df["setting"], df["event_level_top5"], marker="o", linewidth=2.0, color="#b279a2", label="Event-level Top-5")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Process-Level Metrics")
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Chapter 1 Time-Dimension Ablation under the Restarted Formal Protocol", fontsize=13)
    fig.savefig(FIG_DIR / "chapter1_restart_time_dimension_seed42.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_nodehold_analysis() -> None:
    seed_hop = pd.read_csv(REPORT_DIR / "chapter1_restart_nodehold_seed_hop_summary.csv")
    true_profiles = pd.read_csv(REPORT_DIR / "chapter1_restart_nodehold_true_node_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    x = range(len(seed_hop))
    width = 0.34
    axes[0].bar([v - width / 2 for v in x], seed_hop["train_mean_monitor_hop"], width=width, label="Train nodes", color="#4c78a8")
    axes[0].bar([v + width / 2 for v in x], seed_hop["holdout_mean_monitor_hop"], width=width, label="Holdout nodes", color="#e45756")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f"seed{int(s)}" for s in seed_hop["seed"]])
    axes[0].set_ylabel("Mean nearest-monitor hop")
    axes[0].set_title("Holdout nodes are structurally farther from monitors")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for seed, g in true_profiles.groupby("seed"):
        axes[1].scatter(g["true_nearest_monitor_hop"], g["top1"], s=35, alpha=0.75, label=f"seed{seed}")
    axes[1].set_xlabel("Nearest-monitor hop")
    axes[1].set_ylabel("Per-node Top-1")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Node-holdout Top-1 drops as observability worsens")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(FIG_DIR / "chapter1_restart_nodehold_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_loss_curves()
    save_model_comparison()
    save_split_comparison()
    save_time_dimension()
    save_nodehold_analysis()
    print(f"saved figure pack -> {FIG_DIR}")


if __name__ == "__main__":
    main()
