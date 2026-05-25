from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"E:\11.16")
SCRIPT_ROOT = ROOT / "script2_new"
INPUT_DIR = SCRIPT_ROOT / "input_1"
REPORT_DIR = SCRIPT_ROOT / "outputs" / "reports"
INP_FILE = ROOT / "input_data" / "2_tuned_v3_merged1.inp"

CANDIDATE_JSON = INPUT_DIR / "candidate_nodes_new.json"
DEGREE_JSON = INPUT_DIR / "monitor_nodes_degree_N25.json"
OBSAWARE_JSON = INPUT_DIR / "monitor_nodes_observability_aware_N25.json"
OUT_PNG = REPORT_DIR / "candidate_monitor_layout_degree_vs_obsaware.png"

SECTION_RE = re.compile(r"^\s*\[(\w+)\]\s*$", re.IGNORECASE)
COORD_RE = re.compile(r"^(?P<node>\S+)\s+(?P<x>[-+0-9.Ee]+)\s+(?P<y>[-+0-9.Ee]+)")
CONDUIT_RE = re.compile(r"^(?P<link>\S+)\s+(?P<from>\S+)\s+(?P<to>\S+)")


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


def draw_panel(ax, df_nodes, df_links, candidates: set[str], monitors: set[str], title: str, monitor_color: str) -> None:
    both = candidates & monitors
    cand_only = candidates - both
    mon_only = monitors - both

    for _, row in df_links.iterrows():
        ax.plot([row.fx, row.tx], [row.fy, row.ty], color="#d9d9d9", linewidth=0.5, alpha=0.7, zorder=1)

    ax.scatter(df_nodes["x"], df_nodes["y"], s=5, color="#b0b0b0", alpha=0.20, zorder=2)

    cand_df = df_nodes[df_nodes["node_id"].isin(cand_only)]
    mon_df = df_nodes[df_nodes["node_id"].isin(mon_only)]
    both_df = df_nodes[df_nodes["node_id"].isin(both)]

    if not cand_df.empty:
        ax.scatter(cand_df["x"], cand_df["y"], s=55, marker="o", color="#e67e22", alpha=0.85, zorder=3, label=f"Candidates only ({len(cand_df)})")
    if not mon_df.empty:
        ax.scatter(mon_df["x"], mon_df["y"], s=70, marker="D", color=monitor_color, alpha=0.90, zorder=4, edgecolors="white", linewidths=0.5, label=f"Monitors only ({len(mon_df)})")
    if not both_df.empty:
        ax.scatter(both_df["x"], both_df["y"], s=140, marker="*", color="#d62728", alpha=1.0, zorder=5, edgecolors="white", linewidths=0.6, label=f"Overlap ({len(both_df)})")

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_nodes, df_links = parse_inp(INP_FILE)
    candidates = set(load_json_list(CANDIDATE_JSON, ("candidate_nodes", "nodes")))
    degree_monitors = set(load_json_list(DEGREE_JSON, ("monitor_nodes", "selected_nodes", "nodes")))
    obsaware_monitors = set(load_json_list(OBSAWARE_JSON, ("monitor_nodes", "selected_nodes", "nodes")))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=140)
    fig.suptitle("Candidate Defect Nodes vs N25 Monitor Layouts", fontsize=15, fontweight="bold")

    draw_panel(
        axes[0],
        df_nodes,
        df_links,
        candidates,
        degree_monitors,
        "Degree N25",
        "#1f77b4",
    )
    draw_panel(
        axes[1],
        df_nodes,
        df_links,
        candidates,
        obsaware_monitors,
        "Observability-aware N25",
        "#2ca02c",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, bbox_inches="tight")
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()
