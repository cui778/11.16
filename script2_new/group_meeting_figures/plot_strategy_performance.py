from __future__ import annotations

import matplotlib.pyplot as plt

from common import load_metrics_json, save_fig, setup_style


def main() -> None:
    setup_style()
    data = load_metrics_json("CHAPTER1_RESULTS.json")
    ordered = sorted(data.items(), key=lambda kv: kv[1].get("mean_mrr", 0.0), reverse=True)

    strategies = [k for k, _ in ordered]
    mean_mrr = [v["mean_mrr"] for _, v in ordered]
    std_mrr = [v["std_mrr"] for _, v in ordered]
    mean_top1 = [v["mean_top1"] for _, v in ordered]
    std_top1 = [v["std_top1"] for _, v in ordered]

    fig1, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].bar(strategies, mean_mrr, yerr=std_mrr, capsize=4, color="#4e79a7")
    axes[0].set_title("??????? MRR ??")
    axes[0].set_ylabel("MRR")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(strategies, mean_top1, yerr=std_top1, capsize=4, color="#f28e2b")
    axes[1].set_title("??????? Top-1 ??")
    axes[1].set_ylabel("Top-1")
    axes[1].tick_params(axis="x", rotation=25)
    out1 = save_fig(fig1, "03_strategy_mrr_top1.png")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(9, 5.2))
    colors = {
        "downstream": "#59a14f",
        "betweenness": "#4e79a7",
        "degree": "#f28e2b",
        "observability": "#af7aa1",
        "random": "#9c755f",
        "full": "#bab0ab",
    }
    for strategy, values in ordered:
        ax.scatter([strategy] * len(values.get("mrrs", [])), values.get("mrrs", []), s=70, color=colors.get(strategy, "#4e79a7"), alpha=0.9)
    ax.set_title("????????? MRR ??")
    ax.set_ylabel("MRR")
    ax.tick_params(axis="x", rotation=25)
    out2 = save_fig(fig2, "04_strategy_seed_scatter.png")
    plt.close(fig2)

    print(f"[OK] ???: {out1}")
    print(f"[OK] ???: {out2}")


if __name__ == "__main__":
    main()
