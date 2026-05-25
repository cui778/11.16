from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"E:\11.16")
SCRIPT_ROOT = ROOT / "script2_new"
CH5_ROOT = SCRIPT_ROOT / "chapter5_layout_optimization"
INPUT_DIR = SCRIPT_ROOT / "input_1"
LAYOUT_DIR = CH5_ROOT / "outputs" / "layouts"
FIGURE_DIR = CH5_ROOT / "figures"
INP_FILE = ROOT / "input_data" / "2_tuned_v3_merged1.inp"
CANDIDATE_JSON = INPUT_DIR / "candidate_nodes_new.json"

SECTION_RE = re.compile(r"^\s*\[(\w+)\]\s*$", re.IGNORECASE)
COORD_RE = re.compile(r"^(?P<node>\S+)\s+(?P<x>[-+0-9.Ee]+)\s+(?P<y>[-+0-9.Ee]+)")
CONDUIT_RE = re.compile(r"^(?P<link>\S+)\s+(?P<from>\S+)\s+(?P<to>\S+)")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_inp(inp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords: dict[str, tuple[float, float]] = {}
    conduits: list[tuple[str, str]] = []
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
            elif section == "CONDUITS":
                m = CONDUIT_RE.match(s)
                if m:
                    conduits.append((m.group("from"), m.group("to")))

    nodes = pd.DataFrame(
        [{"node_id": nid, "x": xy[0], "y": xy[1]} for nid, xy in coords.items()]
    )
    links = pd.DataFrame(
        [
            {"fx": coords[a][0], "fy": coords[a][1], "tx": coords[b][0], "ty": coords[b][1]}
            for a, b in conduits
            if a in coords and b in coords
        ]
    )
    return nodes, links


def load_json_list(path: Path, keys: tuple[str, ...]) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    for key in keys:
        if key in data and isinstance(data[key], list):
            return [str(x) for x in data[key]]
    raise KeyError(f"{path} missing any of {keys}")


def layout_path(strategy: str, n: int) -> Path:
    return LAYOUT_DIR / strategy / f"monitor_nodes_{strategy}_N{n}.json"


def draw_panel(ax, df_nodes, df_links, candidates: set[str], monitors: set[str], title: str, monitor_color: str) -> None:
    both = candidates & monitors
    cand_only = candidates - both
    mon_only = monitors - both

    for _, row in df_links.iterrows():
        ax.plot([row.fx, row.tx], [row.fy, row.ty], color="#d7d7d7", linewidth=0.45, alpha=0.75, zorder=1)

    ax.scatter(df_nodes["x"], df_nodes["y"], s=5, color="#b0b0b0", alpha=0.18, zorder=2)

    cand_df = df_nodes[df_nodes["node_id"].isin(cand_only)]
    mon_df = df_nodes[df_nodes["node_id"].isin(mon_only)]
    both_df = df_nodes[df_nodes["node_id"].isin(both)]

    if not cand_df.empty:
        ax.scatter(
            cand_df["x"], cand_df["y"],
            s=48, marker="o", color="#e67e22", alpha=0.82, zorder=3,
            label=f"候选点 only ({len(cand_df)})",
        )
    if not mon_df.empty:
        ax.scatter(
            mon_df["x"], mon_df["y"],
            s=70, marker="D", color=monitor_color, alpha=0.92, zorder=4,
            edgecolors="white", linewidths=0.5,
            label=f"监测点 only ({len(mon_df)})",
        )
    if not both_df.empty:
        ax.scatter(
            both_df["x"], both_df["y"],
            s=135, marker="*", color="#c62828", alpha=1.0, zorder=5,
            edgecolors="white", linewidths=0.6,
            label=f"重叠 ({len(both_df)})",
        )

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chapter-5 layout comparison.")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument(
        "--strategies",
        default="degree,candidate_observability,identifiability_driven",
        help="Comma separated strategy names.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output PNG path. Defaults to chapter5 figure directory.",
    )
    args = parser.parse_args()

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    nodes_df, links_df = parse_inp(INP_FILE)
    candidates = set(load_json_list(CANDIDATE_JSON, ("candidate_nodes", "nodes")))

    monitor_sets: list[set[str]] = []
    for strategy in strategies:
        monitors = set(load_json_list(layout_path(strategy, args.n), ("monitor_nodes", "selected_nodes", "nodes")))
        monitor_sets.append(monitors)

    colors = ["#1f77b4", "#2ca02c", "#7b3fb2", "#8c564b", "#17a2b8"]
    fig, axes = plt.subplots(1, len(strategies), figsize=(7.4 * len(strategies), 8.6), dpi=150)
    if len(strategies) == 1:
        axes = [axes]

    fig.suptitle(f"第5章监测布局对比 (N={args.n})", fontsize=16, fontweight="bold")

    for ax, strategy, monitor_set, color in zip(axes, strategies, monitor_sets, colors):
        title = f"{strategy} (N={args.n})"
        draw_panel(ax, nodes_df, links_df, candidates, monitor_set, title, color)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if args.output:
        out_path = Path(args.output)
    else:
        safe_name = "_vs_".join(strategies)
        out_path = FIGURE_DIR / f"ch5_layout_compare_{safe_name}_N{args.n}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(str(out_path))


if __name__ == "__main__":
    main()

