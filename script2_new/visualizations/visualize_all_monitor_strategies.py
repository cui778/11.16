"""
监测节点布置方案对比可视化

功能：
  1. 可选：先运行 build_monitors 生成所有策略（degree / betweenness / downstream）
  2. 加载 input_1 下所有 monitor_nodes_*.json
  3. 在同一张图上并排展示各策略的监测点分布，便于对比

用法：
  # 仅可视化已有文件
  python visualizations/visualize_all_monitor_strategies.py

  # 先生成所有策略，再可视化
  python visualizations/visualize_all_monitor_strategies.py --generate

  # 包含 observability（需先完成 extract_timeseries + residual_features）
  python visualizations/visualize_all_monitor_strategies.py --generate --include-observability

  # 保存为 PNG
  python visualizations/visualize_all_monitor_strategies.py --save
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ── 路径 ────────────────────────────────────────────────────────────────────
_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
_INPUT_DIR   = _SCRIPT_ROOT / "input_1"
_PREP_DIR    = _SCRIPT_ROOT / "prep"

INP_FILE       = Path(r"E:\11.16\input_data\2_tuned_v3_merged1.inp")
CANDIDATE_JSON = _INPUT_DIR / "candidate_nodes_new.json"

# 策略名称与颜色
STRATEGY_COLORS = {
    "degree":       "#e8700a",   # 橙
    "betweenness":  "#1a6faf",   # 蓝
    "downstream":   "#2ca02c",   # 绿
    "observability":"#9467bd",   # 紫
}
STRATEGY_LABELS = {
    "degree":       "度中心性 (degree)",
    "betweenness":  "介数中心性 (betweenness)",
    "downstream":   "下游覆盖 (downstream)",
    "observability":"可观测性 (observability)",
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── INP 解析 ─────────────────────────────────────────────────────────────────
SECTION_RE = re.compile(r"^\s*\[(\w+)\]\s*$", re.IGNORECASE)
COORD_RE   = re.compile(r"^(?P<node>\S+)\s+(?P<x>[-+0-9.Ee]+)\s+(?P<y>[-+0-9.Ee]+)")
JUNCTION_RE= re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")
OUTFALL_RE = re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")
CONDUIT_RE = re.compile(r"^(?P<link>\S+)\s+(?P<from>\S+)\s+(?P<to>\S+)")


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
            links.append({"fx": coords[fn][0], "fy": coords[fn][1], "tx": coords[tn][0], "ty": coords[tn][1]})
    df_links = pd.DataFrame(links)
    return df_nodes, df_links


def load_monitor_files(input_dir: Path) -> dict[str, list[str]]:
    """扫描 input_dir 下所有 monitor_nodes_*.json，返回 {strategy: [node_ids]}"""
    result = {}
    for f in sorted(input_dir.glob("monitor_nodes_*.json")):
        # 解析文件名: monitor_nodes_degree_N25.json -> strategy=degree
        stem = f.stem  # monitor_nodes_degree_N25
        parts = stem.replace("monitor_nodes_", "").split("_N")
        strategy = parts[0] if parts else "unknown"
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        nodes = d.get("monitor_nodes", d.get("selected_nodes", []))
        result[strategy] = [str(n) for n in nodes]
        print(f"  加载: {f.name} -> {strategy}, n={len(nodes)}")
    return result


def load_candidates(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [str(n) for n in d.get("candidate_nodes", [])]


def run_build_monitors(strategies: list[str], n: int = 25):
    """调用 build_monitors.py 生成指定策略的监测节点文件"""
    py = sys.executable
    prep = _PREP_DIR / "build_monitors.py"
    if not prep.exists():
        print(f"  [WARN] 未找到 {prep}，跳过生成")
        return
    for s in strategies:
        cmd = [py, str(prep), "--strategy", s, "--n", str(n)]
        print(f"  运行: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(_SCRIPT_ROOT), check=False)


def visualize_all(
    df_nodes: pd.DataFrame,
    df_links: pd.DataFrame,
    candidates: list[str],
    strategies_data: dict[str, list[str]],
    save_path: Path | None = None,
):
    """并排展示各策略的监测点分布"""
    n_strategies = len(strategies_data)
    if n_strategies == 0:
        print("  没有找到任何 monitor_nodes_*.json 文件")
        return

    # 布局：尽量接近正方形
    ncol = min(3, n_strategies)
    nrow = (n_strategies + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8 * ncol, 7 * nrow), dpi=100)
    if n_strategies == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_strategies > 1 else [axes]

    cand_set = set(candidates)
    df_nodes = df_nodes.copy()
    df_nodes["node_id"] = df_nodes["node_id"].astype(str)

    for idx, (strategy, monitor_nodes) in enumerate(strategies_data.items()):
        ax = axes[idx]
        mon_set = set(monitor_nodes)
        both_set = cand_set & mon_set
        df_cand_only = df_nodes[df_nodes["node_id"].isin(cand_set - both_set)]
        df_mon_only  = df_nodes[df_nodes["node_id"].isin(mon_set - both_set)]
        df_both      = df_nodes[df_nodes["node_id"].isin(both_set)]

        color = STRATEGY_COLORS.get(strategy, "#666666")
        label = STRATEGY_LABELS.get(strategy, strategy)

        # 管道
        for _, lk in df_links.iterrows():
            ax.plot([lk.fx, lk.tx], [lk.fy, lk.ty], color="#cccccc", linewidth=0.5, alpha=0.6, zorder=1)

        # 全网节点
        ax.scatter(df_nodes["x"], df_nodes["y"], s=5, color="#aaaaaa", alpha=0.2, zorder=2)

        # 候选节点（非重叠）
        if not df_cand_only.empty:
            ax.scatter(df_cand_only["x"], df_cand_only["y"], s=45, marker="o",
                       color="#cccccc", alpha=0.6, zorder=3, edgecolors="#999999", linewidths=0.4,
                       label=f"候选 C (n={len(cand_set)})")

        # 监测节点（该策略）
        if not df_mon_only.empty:
            ax.scatter(df_mon_only["x"], df_mon_only["y"], s=100, marker="D",
                       color=color, alpha=0.9, zorder=4, edgecolors="white", linewidths=0.6,
                       label=f"监测 S (n={len(mon_set)})")

        # 重叠
        if not df_both.empty:
            ax.scatter(df_both["x"], df_both["y"], s=180, marker="*",
                       color="#d62728", alpha=1.0, zorder=5, edgecolors="white", linewidths=0.5,
                       label=f"C∩S (n={len(both_set)})")

        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # 隐藏多余子图
    for j in range(n_strategies, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("监测节点布置方案对比（候选 C vs 各策略 S）", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 已保存: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="监测节点布置方案对比可视化")
    parser.add_argument("--generate", action="store_true",
                        help="先运行 build_monitors 生成 degree/betweenness/downstream")
    parser.add_argument("--include-observability", action="store_true",
                        help="生成时包含 observability（需先完成 extract_timeseries + residual_features）")
    parser.add_argument("--n", type=int, default=25, help="监测节点数量")
    parser.add_argument("--inp", default=str(INP_FILE))
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-path", default="")
    args = parser.parse_args()

    if args.generate:
        strategies = ["degree", "betweenness", "downstream"]
        if args.include_observability:
            strategies.append("observability")
        print("生成监测节点方案...")
        run_build_monitors(strategies, n=args.n)
        print()

    print("加载监测节点文件...")
    strategies_data = load_monitor_files(_INPUT_DIR)
    if not strategies_data:
        print("未找到 monitor_nodes_*.json。请先运行:")
        print("  python prep/build_monitors.py --strategy degree --n 25")
        print("  python prep/build_monitors.py --strategy betweenness --n 25")
        print("  python prep/build_monitors.py --strategy downstream --n 25")
        return

    inp_path = Path(args.inp)
    if not inp_path.exists():
        raise FileNotFoundError(f"INP 不存在: {inp_path}")

    print("解析 INP...")
    df_nodes, df_links = parse_inp(inp_path)
    candidates = load_candidates(CANDIDATE_JSON) if CANDIDATE_JSON.exists() else []

    save_path = None
    if args.save:
        save_path = Path(args.save_path) if args.save_path else _SCRIPT_ROOT / "visualizations" / "all_monitor_strategies.png"

    visualize_all(df_nodes, df_links, candidates, strategies_data, save_path=save_path)


if __name__ == "__main__":
    main()
