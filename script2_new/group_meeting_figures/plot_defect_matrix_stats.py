from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import INPUT_DIR, save_fig, setup_style


def main() -> None:
    setup_style()
    df = pd.read_csv(INPUT_DIR / "defect_matrix_diverse.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    type_counts = df["defect_type"].value_counts().reindex(["I", "E", "P"]).fillna(0)
    axes[0, 0].bar(type_counts.index, type_counts.values, color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[0, 0].set_title("??????")
    axes[0, 0].set_ylabel("???")

    intensity_counts = df["intensity_pct"].value_counts().sort_index()
    axes[0, 1].bar(intensity_counts.index.astype(str), intensity_counts.values, color="#e15759")
    axes[0, 1].set_title("??????")
    axes[0, 1].set_ylabel("???")

    start_counts = df["start_hour"].value_counts().sort_index()
    axes[0, 2].plot(start_counts.index, start_counts.values, marker="o", color="#4e79a7")
    axes[0, 2].set_title("????????")
    axes[0, 2].set_xlabel("start_hour")
    axes[0, 2].set_ylabel("???")

    duration_counts = df["duration_h"].value_counts().sort_index()
    axes[1, 0].bar(duration_counts.index.astype(str), duration_counts.values, color="#76b7b2")
    axes[1, 0].set_title("????????")
    axes[1, 0].set_ylabel("???")

    top_nodes = df["node_id"].value_counts().head(10)
    axes[1, 1].barh(top_nodes.index[::-1], top_nodes.values[::-1], color="#af7aa1")
    axes[1, 1].set_title("???????? Top 10")
    axes[1, 1].set_xlabel("????")

    axes[1, 2].hist(df["baseline_flow"], bins=15, color="#9c755f", edgecolor="white")
    axes[1, 2].set_title("baseline_flow ??")
    axes[1, 2].set_xlabel("baseline_flow")
    axes[1, 2].set_ylabel("??")

    fig.suptitle("????????", fontsize=14)
    out = save_fig(fig, "10_defect_matrix_stats.png")
    plt.close(fig)
    print(f"[OK] ???: {out}")


if __name__ == "__main__":
    main()
