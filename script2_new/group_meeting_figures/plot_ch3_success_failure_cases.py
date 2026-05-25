from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from common import CURRENT_CLEAN_PARQUET, CURRENT_DEFECT_CSV, INPUT_DIR, OUTPUT_DIR, REPORTS_DIR, first_existing_path, save_fig, setup_style


BY_SCENARIO = first_existing_path(
    REPORTS_DIR.parent / "model_checkpoints" / "by_scenario_test_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual12_scenario.csv",
    REPORTS_DIR.parent / "model_checkpoints" / "by_scenario_test_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_scenario.csv",
)
DEFECT_CSV = CURRENT_DEFECT_CSV
PARQUET = CURRENT_CLEAN_PARQUET
PARSED_INP = INPUT_DIR / "parsed_inp_data.json"

BG = "#ffffff"
TEXT = "#202632"
MUTED = "#6c7380"
LINE = "#d9dde5"
BASE = "#4f79a7"
DEFECT = "#d46a6a"
RESID = "#86b97e"


def build_graph(parsed_inp_path: Path) -> tuple[nx.DiGraph, dict]:
    data = json.loads(parsed_inp_path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    for _, conduit in data.get("conduits", {}).items():
        u = str(conduit.get("from_node", ""))
        v = str(conduit.get("to_node", ""))
        if u and v:
            graph.add_edge(u, v)
    return graph, data


def node_role(node_id: str, parsed: dict) -> str:
    if node_id in parsed.get("storage", {}):
        return "storage"
    if node_id in parsed.get("junctions", {}):
        return "junction"
    if node_id in parsed.get("outfalls", {}):
        return "outfall"
    if node_id in parsed.get("boundary_inlets", {}):
        return "boundary_inlet"
    return "other"


def topo_summary(node_id: str, graph: nx.DiGraph, parsed: dict) -> dict[str, str]:
    outfalls = set(map(str, parsed.get("outfalls", {}).keys()))
    dists = []
    for outfall in outfalls:
        try:
            dists.append(nx.shortest_path_length(graph, node_id, outfall))
        except Exception:
            continue
    return {
        "role": node_role(node_id, parsed),
        "in_degree": str(graph.in_degree(node_id)) if node_id in graph else "0",
        "out_degree": str(graph.out_degree(node_id)) if node_id in graph else "0",
        "dist_to_outfall": str(min(dists)) if dists else "NA",
    }


def select_case_series(df_ts: pd.DataFrame, scenario_id: int) -> tuple[str | None, pd.DataFrame | None]:
    cur = df_ts[df_ts["scenario_id"] == scenario_id].copy()
    if cur.empty:
        return None, None
    score = cur.groupby("node_id")["depth_residual_rel"].apply(lambda s: s.abs().max())
    node_id = str(score.idxmax())
    cur_node = cur[cur["node_id"] == node_id][["datetime", "depth", "depth_residual_rel"]].copy()
    base = df_ts[(df_ts["scenario_id"] == 0) & (df_ts["node_id"] == node_id)][["datetime", "depth"]].copy()
    merged = base.merge(cur_node, on="datetime", how="inner", suffixes=("_base", "_defect"))
    return node_id, merged


def case_diagnosis(case: pd.Series, meta: pd.Series, topo: dict[str, str], response_node: str, merged: pd.DataFrame) -> str:
    peak_res = float(merged["depth_residual_rel"].abs().max())
    active_steps = int((merged["depth_residual_rel"].abs() > 0.01).sum())
    hints: list[str] = []

    if float(case["mrr"]) >= 0.95:
        if active_steps >= 8:
            hints.append("异常响应持续时间较长，时序区分度高")
        if response_node != str(meta["node_id"]):
            hints.append("即使真实节点未被直接观测，传播路径仍形成了清晰响应")
        hints.append("该场景在当前拓扑下更容易形成稳定的逆向定位线索")
    else:
        if int(meta["duration_h"]) <= 6:
            hints.append("持续时间较短，可用于定位的窗口数量较少")
        if str(meta["defect_type"]) == "E":
            hints.append("E 类缺陷在当前模型下整体更难，容易与邻近节点混淆")
        if peak_res > 0.05:
            hints.append("失败并非单纯因为信号弱，更可能与拓扑位置和路径混淆有关")
        else:
            hints.append("响应峰值本身也偏弱，放大了定位难度")
        if topo["role"] == "junction":
            hints.append("junction 节点更容易出现局部结构相似，增加反演歧义")

    return "；".join(hints[:3])


def plot_one_case(ax, df_ts: pd.DataFrame, case: pd.Series, df_def: pd.DataFrame, graph: nx.DiGraph, parsed: dict, label: str) -> None:
    scenario_id = int(case["defect_id"])
    meta = df_def[df_def["defect_id"] == scenario_id].iloc[0]
    defect_node = str(meta["node_id"])
    topo = topo_summary(defect_node, graph, parsed)
    response_node, merged = select_case_series(df_ts, scenario_id)

    if merged is None or merged.empty or response_node is None:
        ax.text(0.5, 0.5, f"场景 {scenario_id} 无可用时序", ha="center", va="center", color=TEXT)
        ax.axis("off")
        return

    ax.set_facecolor(BG)
    ax.plot(merged["datetime"], merged["depth_base"], label="baseline", color=BASE, linewidth=1.9)
    ax.plot(merged["datetime"], merged["depth_defect"], label="defect", color=DEFECT, linewidth=1.9)
    ax2 = ax.twinx()
    ax2.plot(merged["datetime"], merged["depth_residual_rel"], label="depth_residual_rel", color=RESID, linewidth=1.8, alpha=0.95)

    ax.grid(axis="y", color=LINE, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{label}：场景 {scenario_id}  (MRR={float(case['mrr']):.3f}, Top-1={float(case['top1']):.2f})", fontsize=12.5, fontweight="bold", color=TEXT, pad=10)
    ax.set_ylabel("depth", color=TEXT)
    ax2.set_ylabel("depth_residual_rel", color=TEXT)
    ax.tick_params(axis="x", rotation=22)

    info = "\n".join(
        [
            f"真实缺陷: {defect_node} ({meta['defect_type']})",
            f"强度/时长: {int(meta['intensity_pct'])}% / {int(meta['duration_h'])}h",
            f"开始时刻: {int(meta['start_hour'])}h",
            f"拓扑角色: {topo['role']} | dist2outfall={topo['dist_to_outfall']}",
            f"最强响应节点: {response_node}",
            "可能原因: " + case_diagnosis(case, meta, topo, response_node, merged),
        ]
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fafbfc", "edgecolor": LINE},
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8, frameon=False)


def main() -> None:
    setup_style()
    df_score = pd.read_csv(BY_SCENARIO)
    df_def = pd.read_csv(DEFECT_CSV)
    df_ts = pd.read_parquet(PARQUET)
    df_ts["datetime"] = pd.to_datetime(df_ts["datetime"])
    graph, parsed = build_graph(PARSED_INP)

    best_case = df_score.nlargest(1, "mrr").iloc[0]
    worst_case = df_score.nsmallest(1, "mrr").iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2))
    fig.patch.set_facecolor(BG)

    plot_one_case(axes[0], df_ts, best_case, df_def, graph, parsed, "成功案例")
    plot_one_case(axes[1], df_ts, worst_case, df_def, graph, parsed, "失败案例")

    fig.suptitle("第3章成功/失败案例对比（hydraulic_inverse, node_holdout）", fontsize=16, fontweight="bold", color=TEXT, y=1.01)
    fig.tight_layout()
    out = save_fig(fig, "15_ch3_success_failure_cases.png")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
