from __future__ import annotations

import matplotlib.pyplot as plt

from common import REPORTS_DIR, load_json, load_metrics_json, save_fig, setup_style


def main() -> None:
    setup_style()
    b1_seed_files = [
        "last_run_metrics_B1_100scen_seed42.json",
        "last_run_metrics_B1_100scen_seed7.json",
        "last_run_metrics_B1_100scen_seed123.json",
    ]
    b1_mrr = [load_metrics_json(name).get("mrr", 0.0) for name in b1_seed_files]
    b1_top1 = [load_metrics_json(name).get("top1", 0.0) for name in b1_seed_files]
    obs = load_json(REPORTS_DIR / "CHAPTER1_RESULTS.json")["observability"]

    fig1, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    axes[0].bar(["100??", "300??"], [sum(b1_mrr) / len(b1_mrr), sum(obs["mrrs"]) / len(obs["mrrs"])], color=["#e15759", "#59a14f"])
    axes[0].set_title("B1?????? MRR ???")
    axes[0].set_ylabel("MRR")
    axes[1].bar(["100??", "300??"], [sum(b1_top1) / len(b1_top1), sum(obs["top1s"]) / len(obs["top1s"])], color=["#e15759", "#59a14f"])
    axes[1].set_title("B1?????? Top-1 ???")
    axes[1].set_ylabel("Top-1")
    out1 = save_fig(fig1, "07_b1_scenarios.png")
    plt.close(fig1)

    b2 = load_json(REPORTS_DIR / "CHAPTER1_B2_RESULTS.json")
    fig2, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    labels = ["???50??", "????25??"]
    mrrs = [b2["B2_full_coverage"]["mean_mrr"], b2["B2_partial_kdef"]["mean_mrr"]]
    top1s = [b2["B2_full_coverage"]["mean_top1"], b2["B2_partial_kdef"]["mean_top1"]]
    axes[0].bar(labels, mrrs, color=["#4e79a7", "#f28e2b"])
    axes[0].set_title("B2?????? MRR ???")
    axes[0].set_ylabel("MRR")
    axes[1].bar(labels, top1s, color=["#4e79a7", "#f28e2b"])
    axes[1].set_title("B2?????? Top-1 ???")
    axes[1].set_ylabel("Top-1")
    out2 = save_fig(fig2, "08_b2_coverage.png")
    plt.close(fig2)

    b3 = load_json(REPORTS_DIR / "CHAPTER1_B_RESULTS.json")
    fig3, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    labels = ["raw", "residual"]
    mrrs = [b3["B3_raw"]["mean_mrr"], b3["B3_residual"]["mean_mrr"]]
    top1s = [b3["B3_raw"]["mean_top1"], b3["B3_residual"]["mean_top1"]]
    axes[0].bar(labels, mrrs, color=["#76b7b2", "#af7aa1"])
    axes[0].set_title("B3?????? MRR ???")
    axes[0].set_ylabel("MRR")
    axes[1].bar(labels, top1s, color=["#76b7b2", "#af7aa1"])
    axes[1].set_title("B3?????? Top-1 ???")
    axes[1].set_ylabel("Top-1")
    out3 = save_fig(fig3, "09_b3_features.png")
    plt.close(fig3)

    print(f"[OK] ???: {out1}")
    print(f"[OK] ???: {out2}")
    print(f"[OK] ???: {out3}")


if __name__ == "__main__":
    main()
