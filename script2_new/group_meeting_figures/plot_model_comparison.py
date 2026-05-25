from __future__ import annotations

import matplotlib.pyplot as plt

from common import REPORTS_DIR, load_metrics_json, save_fig, setup_style


def build_metrics(names: list[tuple[str, str]]):
    labels, mrrs, top1s = [], [], []
    for label, filename in names:
        data = load_metrics_json(filename)
        labels.append(label)
        mrrs.append(data.get("mrr", 0.0))
        top1s.append(data.get("top1", 0.0))
    return labels, mrrs, top1s


def plot_dual(labels, mrrs, top1s, title, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    axes[0].bar(labels, mrrs, color=colors[: len(labels)])
    axes[0].set_title(f"{title} - MRR")
    axes[0].set_ylabel("MRR")
    axes[1].bar(labels, top1s, color=colors[: len(labels)])
    axes[1].set_title(f"{title} - Top-1")
    axes[1].set_ylabel("Top-1")
    out = save_fig(fig, out_name)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def existing(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(label, filename) for label, filename in items if (REPORTS_DIR / filename).exists()]


def main() -> None:
    setup_style()
    print("[WARN] 该脚本仍属于历史比较图入口，只用于辅助留痕，不作为 clean 主线正式结论图。")
    scenario = existing(
        [
            ("hydraulic_inverse", "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario.json"),
            ("lstm_graphsage_edge", "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v2_rebuild_scenario.json"),
        ]
    )
    holdout = existing(
        [
            ("lstm_graphsage_edge", "last_run_metrics_process_diagnosis_lstm_graphsage_edge_time_gated_node_sensor_v1_node_holdout_seed123.json"),
            ("hydraulic_inverse", "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v1_node_holdout_dynattn_seed42.json"),
        ]
    )
    if not scenario and not holdout:
        raise FileNotFoundError("未找到可用 metrics 文件；请优先查阅 clean 主线总结文档。")
    if scenario:
        labels, mrrs, top1s = build_metrics(scenario)
        plot_dual(labels, mrrs, top1s, "Current Clean Scenario Split", "05_model_comparison_scenario.png")
    if holdout:
        labels, mrrs, top1s = build_metrics(holdout)
        plot_dual(labels, mrrs, top1s, "Legacy Holdout Reference", "06_model_comparison_holdout.png")


if __name__ == "__main__":
    main()
