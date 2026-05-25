"""
可视化：候选缺陷节点集（C）与监测节点集（S）在管网上的分布

用法：
    python script2_new/visualizations/visualize_candidates_and_monitors.py
    python script2_new/visualizations/visualize_candidates_and_monitors.py --save

输出：
    - 交互式窗口（默认）
    - 可选保存 PNG（--save）

图例：
    灰色小点   全网节点（V）
    浅灰细线   管道
    橙色圆点   候选缺陷节点（C，50个）
    蓝色菱形   监测节点（S，25个）
    红色五角星  同时属于 C 和 S 的节点（重叠）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── 路径配置（从 config 读，也可直接改这里） ──────────────────────────────────
_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
_INPUT_DIR   = _SCRIPT_ROOT / "input_1"

INP_FILE            = Path(r"E:\11.16\input_data\2_tuned_v3_merged1.inp")
CANDIDATE_JSON      = _INPUT_DIR / "candidate_nodes_new.json"
MONITOR_JSON        = _INPUT_DIR / "monitor_nodes_degree_N25.json"   # 默认（度中心性）；可被 --monitor 覆盖

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── INP 解析 ─────────────────────────────────────────────────────────────────
SECTION_RE    = re.compile(r"^\s*\[(\w+)\]\s*$", re.IGNORECASE)
COORD_RE      = re.compile(r"^(?P<node>\S+)\s+(?P<x>[-+0-9.Ee]+)\s+(?P<y>[-+0-9.Ee]+)")
JUNCTION_RE   = re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")
OUTFALL_RE    = re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")
CONDUIT_RE    = re.compile(r"^(?P<link>\S+)\s+(?P<from>\S+)\s+(?P<to>\S+)")


def parse_inp(inp_path: Path):
    coords, junctions, outfalls, conduits = {}, {}, {}, []
    section = None
    with inp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            m = SECTION_RE.match(s)
            if m:
                section = m.group(1).upper()
                continue
            if section == "COORDINATES":
                m = COORD_RE.match(s)
                if m:
                    coords[m.group("node")] = (float(m.group("x")), float(m.group("y")))
            elif section == "JUNCTIONS":
                m = JUNCTION_RE.match(s)
                if m:
                    junctions[m.group("node")] = float(m.group("elev"))
            elif section == "OUTFALLS":
                m = OUTFALL_RE.match(s)
                if m:
                    outfalls[m.group("node")] = float(m.group("elev"))
            elif section == "CONDUITS":
                m = CONDUIT_RE.match(s)
                if m:
                    conduits.append((m.group("from"), m.group("to")))

    nodes = []
    for nid, (x, y) in coords.items():
        ntype = "junction" if nid in junctions else "outfall" if nid in outfalls else "other"
        nodes.append({"node_id": nid, "x": x, "y": y, "type": ntype})
    df_nodes = pd.DataFrame(nodes)

    links = []
    for fn, tn in conduits:
        if fn in coords and tn in coords:
            links.append({
                "fx": coords[fn][0], "fy": coords[fn][1],
                "tx": coords[tn][0], "ty": coords[tn][1],
            })
    df_links = pd.DataFrame(links)
    return df_nodes, df_links


def load_json_list(path: Path, *keys) -> list[str]:
    """从 JSON 文件按候选 key 顺序读取节点列表。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        return [str(x) for x in d]
    for k in keys:
        if k in d and isinstance(d[k], list):
            return [str(x) for x in d[k]]
    raise KeyError(f"{path} 中未找到字段 {keys}")


# ── 主绘图函数 ────────────────────────────────────────────────────────────────

def visualize(df_nodes: pd.DataFrame, df_links: pd.DataFrame,
              candidates: list[str], monitors: list[str],
              save_path: Path | None = None):

    cand_set = set(candidates)
    mon_set  = set(monitors)
    both_set = cand_set & mon_set

    df_nodes["node_id"] = df_nodes["node_id"].astype(str)

    df_all   = df_nodes
    df_cand  = df_nodes[df_nodes["node_id"].isin(cand_set - both_set)]
    df_mon   = df_nodes[df_nodes["node_id"].isin(mon_set  - both_set)]
    df_both  = df_nodes[df_nodes["node_id"].isin(both_set)]

    # ── 统计摘要 ──
    print("\n" + "=" * 60)
    print(f"  全网节点 (V)         : {len(df_nodes)}")
    print(f"  候选缺陷节点 (C)     : {len(cand_set)}  (在图中: {len(df_cand) + len(df_both)})")
    print(f"  监测节点 (S)         : {len(mon_set)}  (在图中: {len(df_mon) + len(df_both)})")
    print(f"  C ∩ S (重叠)         : {len(both_set)}")
    print(f"  C - S (仅候选)       : {len(cand_set - both_set)}")
    print(f"  S - C (仅监测)       : {len(mon_set - both_set)}")
    print("=" * 60 + "\n")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=100)
    fig.suptitle("候选缺陷节点集（C）与监测节点集（S）分布", fontsize=15, fontweight="bold", y=0.98)

    titles = ["候选缺陷节点（C）", "监测节点（S）"]
    highlight_sets = [df_cand, df_mon]
    highlight_colors = ["#e8700a", "#1a6faf"]   # 橙、蓝
    highlight_markers = ["o", "D"]
    highlight_sizes   = [80, 90]

    for ax, title, df_hi, color, marker, size in zip(
            axes, titles, highlight_sets, highlight_colors, highlight_markers, highlight_sizes):

        # 管道底图
        for _, lk in df_links.iterrows():
            ax.plot([lk.fx, lk.tx], [lk.fy, lk.ty],
                    color="#cccccc", linewidth=0.5, alpha=0.6, zorder=1)

        # 全网节点（淡灰）
        ax.scatter(df_all["x"], df_all["y"],
                   s=6, color="#aaaaaa", alpha=0.25, zorder=2, label=f"全网节点 (V={len(df_all)})")

        # 高亮节点
        if not df_hi.empty:
            ax.scatter(df_hi["x"], df_hi["y"],
                       s=size, marker=marker, color=color, alpha=0.9, zorder=4,
                       edgecolors="white", linewidths=0.6,
                       label=f"{title.split('（')[0]} n={len(df_hi)}")

        # 重叠节点（红星）
        if not df_both.empty:
            ax.scatter(df_both["x"], df_both["y"],
                       s=140, marker="*", color="#d62728", alpha=1.0, zorder=5,
                       edgecolors="white", linewidths=0.5,
                       label=f"C∩S 重叠 n={len(df_both)}")

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

        # 节点 ID 标注（只标高亮节点，数量少时才显示）
        label_df = pd.concat([df_hi, df_both]) if not df_both.empty else df_hi
        if len(label_df) <= 60:
            for _, row in label_df.iterrows():
                ax.annotate(row["node_id"],
                            xy=(row["x"], row["y"]),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=5.5, color="#333333", alpha=0.85,
                            zorder=6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 图像已保存: {save_path}")

    plt.show()


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="可视化候选节点集(C)与监测节点集(S)的分布")
    parser.add_argument("--inp",       default=str(INP_FILE),       help="INP 文件路径")
    parser.add_argument("--candidate", default=str(CANDIDATE_JSON), help="候选节点 JSON")
    parser.add_argument("--monitor",   default=str(MONITOR_JSON),   help="监测节点 JSON")
    parser.add_argument("--save",      action="store_true",         help="保存为 PNG")
    parser.add_argument("--save-path", default="",                  help="保存路径（默认自动生成）")
    args = parser.parse_args()

    inp_path  = Path(args.inp)
    cand_path = Path(args.candidate)
    mon_path  = Path(args.monitor)

    for p in [inp_path, cand_path, mon_path]:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    print(f"解析 INP: {inp_path}")
    df_nodes, df_links = parse_inp(inp_path)
    print(f"  节点: {len(df_nodes)}, 管道: {len(df_links)}")

    candidates = load_json_list(cand_path, "candidate_nodes", "nodes")
    monitors   = load_json_list(mon_path,  "monitor_nodes", "selected_nodes", "nodes")
    print(f"候选节点 (C): {len(candidates)}")
    print(f"监测节点 (S): {len(monitors)}")

    save_path = None
    if args.save:
        if args.save_path:
            save_path = Path(args.save_path)
        else:
            save_path = _SCRIPT_ROOT / "visualizations" / "candidates_and_monitors.png"

    visualize(df_nodes, df_links, candidates, monitors, save_path=save_path)


if __name__ == "__main__":
    main()
