# -*- coding: utf-8 -*-
"""
生成候选缺陷节点集 C（输出空间）。

来源：原 script2/visualizations/缺陷矩阵/build_candidate_nodes_from_inp.py
职责：仅生成 C，不生成监测节点集 S；不依赖 defect matrix 反向定义候选集，C 先于缺陷矩阵定义。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# ---------------------- INP parsing (aligned with your visualization script) ----------------------
SECTION_RE = re.compile(r"^\s*\[(\w+)\]\s*$", re.IGNORECASE)
COORDINATE_RE = re.compile(r"^(?P<node>\S+)\s+(?P<x>[-+0-9.Ee]+)\s+(?P<y>[-+0-9.Ee]+)")
JUNCTION_RE = re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")
OUTFALL_RE  = re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")
STORAGE_RE  = re.compile(r"^(?P<node>\S+)\s+(?P<elev>[-+0-9.Ee]+)")

# NOTE: your visualization parser only captured first 3 tokens; here we parse full line (length included)
# SWMM CONDUITS format typically: Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow
# We'll safely parse tokens and use token[3] as length if present.
XSECTION_RE = re.compile(r"^(?P<link>\S+)\s+(?P<shape>\S+)\s+(?P<g1>[-+0-9.Ee]+)\s*(?P<g2>[-+0-9.Ee]+)?")

@dataclass
class InpNetwork:
    nodes_df: pd.DataFrame   # node_id, x, y, elevation, type
    links_df: pd.DataFrame   # link_id, from_node, to_node, length_m, diameter_m, shape


def parse_inp(inp_path: Path) -> InpNetwork:
    if not inp_path.exists():
        raise FileNotFoundError(f"INP not found: {inp_path}")

    coordinates: Dict[str, Tuple[float, float]] = {}
    junctions: Dict[str, float] = {}
    outfalls: Dict[str, float] = {}
    storages: Dict[str, float] = {}
    conduits_rows: List[Dict] = []
    xsections: Dict[str, Dict] = {}

    current_section = None
    with inp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";"):
                continue

            m = SECTION_RE.match(s)
            if m:
                current_section = m.group(1).upper()
                continue

            if current_section == "COORDINATES":
                m = COORDINATE_RE.match(s)
                if m:
                    node = m.group("node")
                    coordinates[node] = (float(m.group("x")), float(m.group("y")))

            elif current_section == "JUNCTIONS":
                m = JUNCTION_RE.match(s)
                if m:
                    junctions[m.group("node")] = float(m.group("elev"))

            elif current_section == "OUTFALLS":
                m = OUTFALL_RE.match(s)
                if m:
                    outfalls[m.group("node")] = float(m.group("elev"))

            elif current_section == "STORAGE":
                m = STORAGE_RE.match(s)
                if m:
                    storages[m.group("node")] = float(m.group("elev"))

            elif current_section == "CONDUITS":
                # token-based parse for robustness
                toks = s.split()
                if len(toks) >= 3:
                    link_id, fn, tn = toks[0], toks[1], toks[2]
                    length_m = float(toks[3]) if len(toks) >= 4 and _is_float(toks[3]) else math.nan
                    conduits_rows.append({
                        "link_id": link_id,
                        "from_node": fn,
                        "to_node": tn,
                        "length_m_inp": length_m,
                    })

            elif current_section == "XSECTIONS":
                m = XSECTION_RE.match(s)
                if m:
                    link_id = m.group("link")
                    shape = m.group("shape").upper()
                    g1 = float(m.group("g1")) if _is_float(m.group("g1")) else math.nan
                    g2 = float(m.group("g2")) if (m.group("g2") and _is_float(m.group("g2"))) else math.nan

                    # same convention as your visualization:
                    # circular: diameter = Geom1
                    # rect: use height (Geom2 if present else Geom1) as a characteristic size
                    if shape == "CIRCULAR":
                        diameter = g1
                    elif shape in {"RECT_CLOSED", "RECT_OPEN"}:
                        height = g2 if not math.isnan(g2) else g1
                        diameter = height
                    else:
                        diameter = g1

                    xsections[link_id] = {
                        "shape": shape,
                        "diameter_m": diameter,
                    }

    # nodes df (prefer nodes that appear in COORDINATES; type/elev derived from junction/outfall/storage)
    all_elevs = {**junctions, **outfalls, **storages}
    nodes_data = []
    for node_id, (x, y) in coordinates.items():
        ntype = "junction" if node_id in junctions else ("outfall" if node_id in outfalls else ("storage" if node_id in storages else "unknown"))
        nodes_data.append({
            "node_id": str(node_id),
            "x": float(x),
            "y": float(y),
            "elevation": float(all_elevs.get(node_id, math.nan)) if node_id in all_elevs else math.nan,
            "type": ntype,
        })
    nodes_df = pd.DataFrame(nodes_data)

    links_df = pd.DataFrame(conduits_rows)
    if links_df.empty:
        raise ValueError("No CONDUITS parsed from INP. Check INP sections / formatting.")

    # attach xsection diameter
    links_df["shape"] = links_df["link_id"].map(lambda lid: xsections.get(lid, {}).get("shape", "UNKNOWN"))
    links_df["diameter_m"] = links_df["link_id"].map(lambda lid: xsections.get(lid, {}).get("diameter_m", math.nan))

    # compute length from coordinates if INP length missing
    node_xy = nodes_df.set_index("node_id")[["x", "y"]].to_dict("index")

    def _euclid(u: str, v: str) -> float:
        pu, pv = node_xy.get(u), node_xy.get(v)
        if pu is None or pv is None:
            return math.nan
        dx = float(pu["x"]) - float(pv["x"])
        dy = float(pu["y"]) - float(pv["y"])
        return math.sqrt(dx * dx + dy * dy)

    links_df["length_m_xy"] = links_df.apply(lambda r: _euclid(str(r["from_node"]), str(r["to_node"])), axis=1)
    links_df["length_m"] = links_df["length_m_inp"]
    links_df.loc[links_df["length_m"].isna(), "length_m"] = links_df.loc[links_df["length_m"].isna(), "length_m_xy"]

    return InpNetwork(nodes_df=nodes_df, links_df=links_df)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


# ---------------------- graph building + scoring ----------------------
def load_node_list(node_list_path: str) -> List[str]:
    with open(node_list_path, "r", encoding="utf-8") as f:
        node_info = json.load(f)
    if isinstance(node_info, dict):
        nodes = node_info.get("node_list", None)
        if nodes is None:
            nodes = list(node_info.keys())
        return [str(n) for n in nodes]
    return [str(n) for n in node_info]


def load_defect_nodes(defect_csv_path: str, node_col: str = "node_id") -> List[str]:
    df = pd.read_csv(defect_csv_path)
    if node_col not in df.columns:
        raise ValueError(f"defect_csv missing '{node_col}', columns={list(df.columns)}")
    return sorted(set(df[node_col].dropna().astype(str)))


def normalize(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if np.isclose(vmax, vmin):
        return {k: 0.0 for k in d}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in d.items()}


def build_graph_from_adj(node_list: List[str], adj_path: str, directed: bool) -> nx.Graph:
    adj = np.load(adj_path)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj not square: {adj.shape}")
    if adj.shape[0] != len(node_list):
        raise ValueError(f"adj size {adj.shape[0]} != node_list {len(node_list)} (must match)")

    G = nx.from_numpy_array(adj, create_using=nx.DiGraph if directed else nx.Graph)
    mapping = {i: node_list[i] for i in range(len(node_list))}
    return nx.relabel_nodes(G, mapping)


def build_graph_from_inp(links_df: pd.DataFrame, directed: bool) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for _, r in links_df.iterrows():
        u, v = str(r["from_node"]), str(r["to_node"])
        G.add_edge(u, v)
    return G


def attach_edge_attributes(G: nx.Graph, links_df: pd.DataFrame, eps: float = 1e-6):
    """
    Map pipe attributes onto graph edges (undirected key by sorted(u,v) for robustness).
    If multiple links between same pair exist, we aggregate (sum length, take max diameter).
    """
    edge_attr: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, r in links_df.iterrows():
        u, v = str(r["from_node"]), str(r["to_node"])
        key = (u, v) if isinstance(G, nx.DiGraph) else tuple(sorted([u, v]))
        length = float(r["length_m"]) if pd.notna(r["length_m"]) else math.nan
        diam = float(r["diameter_m"]) if pd.notna(r["diameter_m"]) else math.nan

        if key not in edge_attr:
            edge_attr[key] = {"length_m": 0.0, "diameter_m": 0.0, "n_links": 0}
        if not math.isnan(length):
            edge_attr[key]["length_m"] += length
        if not math.isnan(diam):
            edge_attr[key]["diameter_m"] = max(edge_attr[key]["diameter_m"], diam)
        edge_attr[key]["n_links"] += 1

    # push into graph
    for u, v in G.edges():
        key = (u, v) if isinstance(G, nx.DiGraph) else tuple(sorted([u, v]))
        a = edge_attr.get(key, None)
        if a is None:
            # fallback defaults
            length = 1.0
            diam = 0.5
        else:
            length = a["length_m"] if a["length_m"] > 0 else 1.0
            diam = a["diameter_m"] if a["diameter_m"] > 0 else 0.5

        # weight for shortest-path-based metrics (smaller = “easier to traverse”)
        # Using length/diameter makes large-diameter pipes “shorter” in weighted paths.
        w = length / max(diam, eps)

        G[u][v]["length_m"] = float(length)
        G[u][v]["diameter_m"] = float(diam)
        G[u][v]["weight"] = float(w)


def compute_node_pipe_scores(G: nx.Graph) -> Dict[str, float]:
    """
    Pipe-attribute-based node score: sum over incident edges of (length * diameter^2)
    (roughly proportional to volume/capacity proxy).
    """
    s = {}
    for n in G.nodes():
        total = 0.0
        for _, _, data in G.edges(n, data=True):
            length = float(data.get("length_m", 1.0))
            diam = float(data.get("diameter_m", 0.5))
            total += length * (diam ** 2)
        s[str(n)] = total
    return s


def combined_score(G: nx.Graph,
                   w_degree=0.25, w_betw=0.30, w_close=0.20, w_pr=0.10, w_pipe=0.15) -> Tuple[Dict[str, float], pd.DataFrame]:
    deg = normalize(nx.degree_centrality(G))
    betw = normalize(nx.betweenness_centrality(G, weight="weight", normalized=True))
    close = normalize(nx.closeness_centrality(G, distance="weight"))
    pr = normalize(nx.pagerank(G, weight="weight"))
    pipe = normalize(compute_node_pipe_scores(G))

    score = {}
    for n in G.nodes():
        n = str(n)
        score[n] = (
            w_degree * deg.get(n, 0.0)
            + w_betw * betw.get(n, 0.0)
            + w_close * close.get(n, 0.0)
            + w_pr * pr.get(n, 0.0)
            + w_pipe * pipe.get(n, 0.0)
        )

    detail = pd.DataFrame({
        "node_id": list(G.nodes()),
        "score": [score[str(n)] for n in G.nodes()],
        "degree_c": [deg.get(str(n), 0.0) for n in G.nodes()],
        "betweenness_c": [betw.get(str(n), 0.0) for n in G.nodes()],
        "closeness_c": [close.get(str(n), 0.0) for n in G.nodes()],
        "pagerank": [pr.get(str(n), 0.0) for n in G.nodes()],
        "pipe_cap": [pipe.get(str(n), 0.0) for n in G.nodes()],
    })
    return score, detail

def diversified_fill_xy(
    candidates: list[str],
    base_score: dict[str, float],
    coords: dict[str, tuple[float, float]],
    selected: list[str],
    k: int,
    lam: float = 0.60,
    min_dist_xy: float | None = None,
):
    """
    基于“中心度/综合评分 + 空间分散度(到已选集合的最小欧氏距离)”贪心补齐到 k 个点。
    - lam 越大：越偏向 base_score（集中在关键区域）
    - lam 越小：越偏向分散（更均匀覆盖）
    - min_dist_xy：可选硬约束（例如 50/100），单位与 INP 坐标一致
    """

    selected = [str(x) for x in selected]
    selected_set = set(selected)
    candidates = [str(x) for x in candidates]

    # 归一化 score 到 [0,1]
    vals = np.array([base_score.get(n, 0.0) for n in candidates], dtype=float)
    if len(vals) == 0:
        return selected
    vmin, vmax = float(vals.min()), float(vals.max())

    def score_norm(n: str) -> float:
        if np.isclose(vmax, vmin):
            return 0.0
        return (float(base_score.get(n, 0.0)) - vmin) / (vmax - vmin)

    def dist_xy(u: str, v: str) -> float:
        pu, pv = coords.get(u), coords.get(v)
        if pu is None or pv is None:
            return 0.0
        dx = float(pu[0]) - float(pv[0])
        dy = float(pu[1]) - float(pv[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def diversity(n: str) -> float:
        if not selected:
            return 0.0
        dmin = float("inf")
        for s in selected:
            d = dist_xy(n, s)
            if d < dmin:
                dmin = d
        if dmin == float("inf"):
            return 0.0
        return dmin

    while len(selected) < k:
        best_n = None
        best_val = -1e18

        for n in candidates:
            if n in selected_set:
                continue

            dmin = diversity(n)

            # 可选硬约束：保证不太挤
            if min_dist_xy is not None and selected:
                if dmin < float(min_dist_xy):
                    continue

            val = lam * score_norm(n) + (1.0 - lam) * dmin

            if val > best_val:
                best_val = val
                best_n = n

        if best_n is None:
            # 若硬约束太严格选不出来，则退化为纯 score
            remaining = [n for n in candidates if n not in selected_set]
            if not remaining:
                break
            best_n = max(remaining, key=lambda x: base_score.get(x, 0.0))

        selected.append(best_n)
        selected_set.add(best_n)

    return selected



# ---------------------- selection ----------------------
def main():
    import sys
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=False, default="", help="INP file path (default from config)")
    ap.add_argument("--k", type=int, default=30, help="Candidate set size")

    # recommended: align with training graph
    ap.add_argument("--adj", default="", help="Optional: adj_matrix.npy (recommended)")
    ap.add_argument("--node_list", default="", help="Optional: node_list.json (required if --adj is given)")
    ap.add_argument("--directed", action="store_true", help="Treat graph as directed")

    ap.add_argument("--include", default="", help="Extra nodes to force include (comma-separated)")
    ap.add_argument("--exclude", default="", help="Nodes to exclude (comma-separated)")

    ap.add_argument("--out_json", required=True, help="Output candidate json")
    ap.add_argument("--out_report", default="", help="Output report CSV (optional)")

    # score weights
    ap.add_argument("--w_degree", type=float, default=0.25)
    ap.add_argument("--w_betw", type=float, default=0.30)
    ap.add_argument("--w_close", type=float, default=0.20)
    ap.add_argument("--w_pr", type=float, default=0.10)
    ap.add_argument("--w_pipe", type=float, default=0.15)

    args = ap.parse_args()
    inp_path = Path(args.inp or cfg.inp_file)
    net = parse_inp(inp_path)

    # ====== (新增) 从 INP 坐标构造 coords: node_id -> (x, y) ======
    coords = {}
    if not net.nodes_df.empty:
        for _, r in net.nodes_df.iterrows():
            nid = str(r["node_id"])
            if pd.notna(r.get("x")) and pd.notna(r.get("y")):
                coords[nid] = (float(r["x"]), float(r["y"]))

    # ====== 排除入流口和出流口 ======
    # 先获取用户指定的排除节点
    extra_exclude_user = set(x.strip() for x in args.exclude.split(",") if x.strip())
    
    # 获取所有出流口节点（从 nodes_df 中筛选 type='outfall' 的节点）
    outfall_nodes = set()
    if not net.nodes_df.empty and 'type' in net.nodes_df.columns:
        outfall_mask = net.nodes_df['type'].str.upper() == 'OUTFALL'
        outfall_nodes = set(net.nodes_df.loc[outfall_mask, 'node_id'].astype(str).tolist())
    
    # 从解析的INP数据中获取边界入流点
    boundary_inlet_nodes = set()
    try:
        parsed_data_path = Path(cfg.parsed_inp_data_file)
        if parsed_data_path.exists():
            with open(parsed_data_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
            boundary_inlet_nodes = set(parsed_data.get('boundary_inlets', []))
            print(f"[INFO] 从解析数据加载边界入流点: {len(boundary_inlet_nodes)}个")
        else:
            print("[WARNING] 未找到 parsed_inp_data.json，将用 prep.inp_parser 解析INP")
            from prep.inp_parser import UnifiedINPParser
            parser = UnifiedINPParser(str(inp_path))
            network = parser.parse()
            boundary_inlet_nodes = set(network.boundary_inlets)
    except Exception as e:
        print(f"[WARNING] 加载边界入流点失败: {e}")
        boundary_inlet_nodes = set()
    
    # 调试：打印节点类型分布
    if not net.nodes_df.empty and 'type' in net.nodes_df.columns:
        type_counts = net.nodes_df['type'].value_counts()
        print(f"\n[INFO] 节点类型分布:")
        for node_type, count in type_counts.items():
            print(f"  {node_type}: {count}个")
        print(f"  总计: {len(net.nodes_df)}个节点\n")
    
    # 基于命名规则识别入流口（作为备用方案）
    inflow_patterns = ['INLET', 'INFLOW', 'IN', 'INLET_', 'INFLOW_', 'DRAIN', 'DISCHARGE']
    inflow_nodes = set()
    if not net.nodes_df.empty:
        for node_id in net.nodes_df['node_id']:
            node_str = str(node_id).upper()
            if any(pattern in node_str for pattern in inflow_patterns):
                inflow_nodes.add(str(node_id))
    
    # 调试：打印识别到的入流口
    if boundary_inlet_nodes:
        print(f"[INFO] 识别的边界入流点 ({len(boundary_inlet_nodes)}个):")
        print(f"  {sorted(list(boundary_inlet_nodes))}\n")
    
    if inflow_nodes:
        print(f"[INFO] 基于命名规则识别的入流口 ({len(inflow_nodes)}个):")
        print(f"  {sorted(list(inflow_nodes))}\n")
    
    # 合并需要排除的节点（优先使用边界入流点，备用命名规则）
    all_inlet_nodes = boundary_inlet_nodes | inflow_nodes
    excluded_nodes = outfall_nodes | all_inlet_nodes | extra_exclude_user
    
    print(f"\n[INFO] 排除节点统计:")
    print(f"  出流口节点: {len(outfall_nodes)}个")
    print(f"  边界入流点: {len(boundary_inlet_nodes)}个")
    print(f"  命名入流口: {len(inflow_nodes)}个")
    print(f"  用户排除: {len(extra_exclude_user)}个")
    print(f"  总排除节点: {len(excluded_nodes)}个")
    
    extra_include = [x.strip() for x in args.include.split(",") if x.strip()]
    extra_exclude = excluded_nodes

    # Build graph (prefer adj+node_list to match training)
    if args.adj:
        if not args.node_list:
            raise ValueError("If --adj is provided, --node_list must also be provided.")
        node_list = load_node_list(args.node_list)
        G = build_graph_from_adj(node_list=node_list, adj_path=args.adj, directed=args.directed)
    else:
        G = build_graph_from_inp(net.links_df, directed=args.directed)

    # Attach pipe attributes onto G
    attach_edge_attributes(G, net.links_df)

    # Node universe (for final candidate validity)
    graph_nodes = set(str(n) for n in G.nodes())

    # ====== 不再强制包含缺陷节点 ======
    must = []
    for n in extra_include:
        n = str(n)
        if n in graph_nodes and n not in must:
            must.append(n)

    # Compute combined scores
    score, detail = combined_score(
        G,
        w_degree=args.w_degree,
        w_betw=args.w_betw,
        w_close=args.w_close,
        w_pr=args.w_pr,
        w_pipe=args.w_pipe,
    )

    # ====== 必须：基于 baseline 流量过滤无效节点 ======
    # 候选节点与原始流量相关：无/低流量节点、末端低流节点不会作为缺陷发生点；出口已在上方 extra_exclude 中排除。
    # 因此 baseline_flow_stats 必须先算，再跑 build_candidates。
    baseline_stats_path = Path(cfg.baseline_flow_stats_file)
    if not baseline_stats_path.exists():
        raise FileNotFoundError(
            f"未找到 baseline 流量统计: {baseline_stats_path}\n"
            "候选节点需基于流量过滤（剔除近零流、末端低流）；出口已排除。请先运行:\n"
            "  python prep/defect_matrix.py --baseline-only"
        )

    node_mean_flow = {}
    try:
        with open(baseline_stats_path, "r", encoding="utf-8") as f:
            baseline_stats = json.load(f)
        raw_nodes = baseline_stats.get("nodes", {})
        for nid, stats in raw_nodes.items():
            try:
                node_mean_flow[str(nid)] = float(stats.get("mean", 0.0))
            except Exception:
                continue
    except Exception as e:
        raise RuntimeError(f"读取 baseline_flow_stats 失败: {baseline_stats_path}") from e

    def _is_valid_candidate(nid: str) -> bool:
        """基于 baseline 流量和度数过滤：剔除近零流节点、末端低流节点（与原始流量相关，不发生缺陷的节点不进入候选集）。"""
        nid = str(nid)
        mean_flow = node_mean_flow.get(nid, 0.0)
        degree_val = G.degree(nid) if nid in G else 0

        if mean_flow < 1e-3:
            return False
        if degree_val == 1 and mean_flow < 1e-2:
            return False
        return True

    # ====== (替换) 用“score + 空间分散度”补齐到 K ======
    selected = list(must)

    pool = [str(n) for n in G.nodes()
            if str(n) not in set(selected)
            and str(n) not in extra_exclude
            and _is_valid_candidate(str(n))]

    print(f"\n[INFO] 候选节点池分析:")
    print(f"  图中总节点数: {len(G.nodes())}")
    print(f"  强制包含节点: {len(selected)}")
    print(f"  排除节点数: {len(extra_exclude)}")
    print(f"  有效候选池: {len(pool)}")
    
    # 检查是否有边界入流点漏网
    leaked_boundary = [n for n in pool if n in boundary_inlet_nodes]
    if leaked_boundary:
        print(f"  [WARNING] 边界入流点漏网: {len(leaked_boundary)}个")
        print(f"    {leaked_boundary}")
    else:
        print(f"  [OK] 边界入流点已完全排除")

    selected = diversified_fill_xy(
        candidates=pool,
        base_score=score,
        coords=coords,
        selected=selected,
        k=args.k,
        lam=0.60,  # 0.55~0.70 之间试；越小越分散
        min_dist_xy=None  # 可选：比如 50 或 100（看你坐标尺度）
    )

    # Final sanity
    selected = [n for i, n in enumerate(selected) if n not in selected[:i]]
    if len(selected) != args.k:
        raise ValueError(f"Internal error: selected size {len(selected)} != k {args.k}")

    # Build metadata / output
    out = {
        "candidate_nodes": selected,  # <-- required by your dataset loader
        "k": args.k,
        "strategy": "exclude_inlets_outfalls_then_fill_by_combined_centrality_and_pipe_attributes",
        "weights": {
            "degree": args.w_degree,
            "betweenness": args.w_betw,
            "closeness": args.w_close,
            "pagerank": args.w_pr,
            "pipe_cap": args.w_pipe,
        },
        "excluded_inlets_outfalls": sorted(list(outfall_nodes | inflow_nodes)),
        "extra_include": extra_include,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "inp": os.path.basename(str(inp_path)),
            "adj": os.path.basename(args.adj) if args.adj else None,
            "node_list": os.path.basename(args.node_list) if args.node_list else None,
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[OK] Candidate nodes generated")
    print(f"  out_json: {args.out_json}")
    print(f"  k={args.k} | excluded_outfalls={len(outfall_nodes)} | excluded_boundary_inlets={len(boundary_inlet_nodes)} | excluded_name_inlets={len(inflow_nodes)} | final_selected={len(selected)}")
    print("  first 10:", selected[:10])
    print("=" * 80)

    # Optional report (with node coords/type if available)
    if args.out_report:
        nodes_meta = net.nodes_df.set_index("node_id").to_dict("index") if not net.nodes_df.empty else {}
        rows = []
        excluded_set = excluded_nodes

        for n in selected:
            m = nodes_meta.get(n, {})
            rows.append({
                "node_id": n,
                "is_excluded": int(n in excluded_set),
                "is_boundary_inlet": int(n in boundary_inlet_nodes),
                "is_name_inlet": int(n in inflow_nodes),
                "is_outfall": int(n in outfall_nodes),
                "score": float(score.get(n, 0.0)),
                "degree": int(G.degree(n)) if n in G else 0,
                "x": m.get("x", np.nan),
                "y": m.get("y", np.nan),
                "elevation": m.get("elevation", np.nan),
                "type": m.get("type", "unknown"),
            })

        rep = pd.DataFrame(rows).sort_values(["is_excluded", "score"], ascending=[True, False])
        os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
        rep.to_csv(args.out_report, index=False, encoding="utf-8-sig")
        print(f"[OK] Report saved: {args.out_report}")


if __name__ == "__main__":
    import sys
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    _cfg = Config()

    if len(sys.argv) == 1:
        print("=" * 60)
        print("使用默认参数运行候选节点生成...")
        print("=" * 60)
        default_args = [
            "--inp", _cfg.inp_file,
            "--k", "50",
            "--out_json", _cfg.candidate_nodes_file,
            "--out_report", str(Path(_cfg.input_dir_new) / "candidate_nodes_50_report.csv")
        ]
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] + default_args
        try:
            main()
        finally:
            sys.argv = original_argv
    else:
        main()
