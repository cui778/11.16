from __future__ import annotations

import matplotlib.pyplot as plt

from common import INPUT_DIR, load_json_list, save_fig, setup_style


def main() -> None:
    setup_style()
    node_list = load_json_list(INPUT_DIR / "node_list.json", ())
    candidates = set(load_json_list(INPUT_DIR / "candidate_nodes_new.json", ("candidate_nodes", "nodes")))
    monitor_files = {
        "degree": INPUT_DIR / "monitor_nodes_degree_N25.json",
        "betweenness": INPUT_DIR / "monitor_nodes_betweenness_N25.json",
        "downstream": INPUT_DIR / "monitor_nodes_downstream_N25.json",
        "observability": INPUT_DIR / "monitor_nodes_observability_N25.json",
        "random": INPUT_DIR / "monitor_nodes_random_N25.json",
        "full": INPUT_DIR / "monitor_nodes_full_N128.json",
    }

    degree_nodes = load_json_list(monitor_files["degree"], ("monitor_nodes", "selected_nodes", "nodes"))
    fig1, ax1 = plt.subplots(figsize=(7.5, 5))
    labels = ["???? V", "???? C", "?????? S"]
    values = [len(node_list), len(candidates), len(degree_nodes)]
    bars = ax1.bar(labels, values, color=["#9aa0a6", "#f28e2b", "#4e79a7"])
    ax1.set_title("V / C / S ????")
    ax1.set_ylabel("????")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 2, str(val), ha="center", va="bottom")
    out1 = save_fig(fig1, "01_vcs_counts.png")
    plt.close(fig1)

    strategies, overlaps, only_s, totals = [], [], [], []
    for strategy, path in monitor_files.items():
        s_nodes = set(load_json_list(path, ("monitor_nodes", "selected_nodes", "nodes")))
        strategies.append(strategy)
        overlaps.append(len(candidates & s_nodes))
        only_s.append(len(s_nodes - candidates))
        totals.append(len(s_nodes))

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    x = range(len(strategies))
    ax2.bar(x, overlaps, color="#d62728", label="C ? S")
    ax2.bar(x, only_s, bottom=overlaps, color="#1f77b4", label="S - C")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(strategies)
    ax2.set_ylabel("????")
    ax2.set_title("?????????????????")
    ax2.legend()
    for idx, total in enumerate(totals):
        ax2.text(idx, total + 1.5, str(total), ha="center", va="bottom", fontsize=9)
    out2 = save_fig(fig2, "02_monitor_overlap_with_candidates.png")
    plt.close(fig2)

    print(f"[OK] ???: {out1}")
    print(f"[OK] ???: {out2}")


if __name__ == "__main__":
    main()
