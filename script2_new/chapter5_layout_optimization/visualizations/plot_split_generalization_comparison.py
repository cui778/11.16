from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"E:\11.16\script2_new\chapter5_layout_optimization")
DEFAULT_INPUT = ROOT / "outputs" / "ch5_split_generalization_comparison_N25_seed42.csv"
DEFAULT_OUTPUT = ROOT / "figures" / "ch5_split_generalization_comparison_N25_seed42.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scenario vs node_holdout layout comparison.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    layout_order = ["degree", "candidate_observability", "identifiability_driven"]
    split_order = ["scenario", "node_holdout"]
    metric_specs = [
        ("mrr", "MRR"),
        ("top1", "Top-1"),
        ("event_level_top1", "Event Top-1"),
    ]
    colors = {"scenario": "#2E5E4E", "node_holdout": "#C46A2A"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#F7F3EB")

    for ax, (metric_col, title) in zip(axes, metric_specs):
        pivot = (
            df.pivot(index="layout", columns="split", values=metric_col)
            .reindex(layout_order)
            .reindex(columns=split_order)
        )
        x = range(len(layout_order))
        width = 0.36

        for offset_idx, split in enumerate(split_order):
            offset = -width / 2 if offset_idx == 0 else width / 2
            ax.bar(
                [i + offset for i in x],
                pivot[split].tolist(),
                width=width,
                color=colors[split],
                label=split if metric_col == "mrr" else None,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(["degree", "cand_obs", "id_driven"], rotation=12)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_facecolor("#FFFDF8")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Chapter 5 Layout Comparison: Scenario Split vs Node-Holdout", fontsize=15, fontweight="bold")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
