# -*- coding: utf-8 -*-
"""
生成监测节点集 S（传感器布置方案）。

职责：
- 只负责生成 S，不负责生成候选缺陷节点集 C
- 支持多种策略，每种策略独立生成 monitor_nodes_{strategy}_N{n}.json
- 便于 chapter1 对比实验

支持的策略：
  1. degree        - 度中心性最高的 N 个节点
  2. betweenness   - 介数中心性最高的 N 个节点
  3. downstream    - 下游覆盖度最大的 N 个节点（上游节点数最多）
  4. observability - 基于缺陷影响范围的可观测性覆盖最大化（贪心）
  5. full          - 全节点 V（用于 pretest 性能上限，N 通常为图节点总数）
  6. random        - 随机选 N 个节点（用于 A5 基线，可 --seed 复现）

输出：input_1/monitor_nodes_{strategy}_N{n}.json（路径由 config 管理）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# 工具函数
# ============================================================

def load_node_list(node_list_path: str) -> List[str]:
    with open(node_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        nodes = data.get("node_list", None)
        if nodes is None:
            nodes = list(data.keys())
        return [str(n) for n in nodes]
    return [str(n) for n in data]


def load_candidate_nodes(candidate_path: str) -> List[str]:
    with open(candidate_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(n) for n in data.get("candidate_nodes", [])]


def build_directed_graph(node_list: List[str], adj_path: str) -> nx.DiGraph:
    """
    从 adj_matrix.npy 构建有向图。
    约定：A[u, v] = 1 表示 u -> v（水流方向）。
    """
    adj = np.load(adj_path)
    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    n = len(node_list)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(node_list[i], node_list[j], weight=float(adj[i, j]))
    return G


def save_monitor_nodes(
    selected: List[str],
    strategy: str,
    n: int,
    out_path: str,
    extra_meta: dict = None,
):
    out = {
        "monitor_nodes": selected,
        "strategy": strategy,
        "n": n,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra_meta:
        out.update(extra_meta)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] 监测节点集已保存: {out_path}")
    print(f"     策略={strategy}, N={n}")
    print(f"     前10个节点: {selected[:10]}")


# ============================================================
# 策略1：度中心性
# ============================================================

def strategy_degree(G: nx.DiGraph, n: int, exclude: Set[str] = None) -> List[str]:
    """选无向度数（入度+出度）最高的 N 个节点。"""
    exclude = exclude or set()
    degree_dict = {
        node: G.in_degree(node) + G.out_degree(node)
        for node in G.nodes() if node not in exclude
    }
    sorted_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)
    selected = sorted_nodes[:n]
    print(f"[度中心性] 选出 {len(selected)} 个节点，"
          f"最高度数={degree_dict[selected[0]]}")
    return selected


# ============================================================
# 策略2：介数中心性
# ============================================================

def strategy_betweenness(G: nx.DiGraph, n: int, exclude: Set[str] = None) -> List[str]:
    """选介数中心性最高的 N 个节点。"""
    exclude = exclude or set()
    print("[介数中心性] 开始计算...")
    bc = nx.betweenness_centrality(G, normalized=True)
    bc_filtered = {k: v for k, v in bc.items() if k not in exclude}
    sorted_nodes = sorted(bc_filtered, key=lambda x: bc_filtered[x], reverse=True)
    selected = sorted_nodes[:n]
    print(f"[介数中心性] 选出 {len(selected)} 个节点，"
          f"最高介数={bc_filtered[selected[0]]:.4f}")
    return selected


# ============================================================
# 策略3：下游覆盖度
# ============================================================

def strategy_downstream(G: nx.DiGraph, n: int, exclude: Set[str] = None) -> List[str]:
    """选上游可达节点数最多的 N 个节点（逆水流方向能到达的节点最多）。"""
    exclude = exclude or set()
    G_reverse = G.reverse(copy=True)
    upstream_count = {}
    for node in G.nodes():
        if node in exclude:
            continue
        reachable = nx.descendants(G_reverse, node)
        upstream_count[node] = len(reachable)

    sorted_nodes = sorted(upstream_count, key=lambda x: upstream_count[x], reverse=True)
    selected = sorted_nodes[:n]
    print(f"[下游覆盖度] 选出 {len(selected)} 个节点，"
          f"最多上游节点数={upstream_count[selected[0]]}")
    return selected


# ============================================================
# 策略4：可观测性覆盖（贪心）
# ============================================================

def compute_influence_sets(
    timeseries_path: str,
    candidate_nodes: List[str],
    threshold_rel: float = 0.01,
    defect_matrix_path: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """
    从时序数据（带残差）计算每个候选节点的缺陷影响节点集合。

    influence_set[candidate] = 在该候选节点发生缺陷时，
    residual 超过 threshold_rel 阈值的节点集合。

    若 parquet 中无 defect_node/target_node_id 列，则通过 defect_matrix_path
    按 scenario_id 与缺陷矩阵合并得到缺陷节点（当前 pipeline 的 parquet 只有 scenario_id）。
    """
    print(f"[可观测性] 读取时序数据: {timeseries_path}")
    df = pd.read_parquet(timeseries_path)

    defect_col = None
    for col in ["defect_node", "target_node_id"]:
        if col in df.columns:
            defect_col = col
            break

    if defect_col is None and defect_matrix_path and os.path.isfile(defect_matrix_path):
        # 当前 parquet 只有 scenario_id；用缺陷矩阵按场景得到 defect_node
        if "scenario_id" not in df.columns:
            raise ValueError("时序数据缺少 scenario_id，无法与缺陷矩阵合并。")
        defect_df = pd.read_csv(defect_matrix_path)
        if "defect_id" not in defect_df.columns or "node_id" not in defect_df.columns:
            raise ValueError("缺陷矩阵需包含 defect_id 与 node_id 列。")
        scenario_to_defect = defect_df.set_index("defect_id")["node_id"].astype(str).to_dict()
        df["defect_node"] = df["scenario_id"].map(lambda s: scenario_to_defect.get(int(s), ""))
        defect_col = "defect_node"
        print(f"[可观测性] 已通过缺陷矩阵按 scenario_id 合并得到 defect_node 列")
    elif defect_col is None:
        raise ValueError(
            "时序数据中缺少 'defect_node' 或 'target_node_id' 列，且未提供 defect_matrix_path。"
            "请提供 --defect_matrix 参数（缺陷矩阵 CSV，含 defect_id 与 node_id）。"
        )

    residual_col = None
    for col in ["depth_residual_rel", "depth_relative_residual", "depth_residual"]:
        if col in df.columns:
            residual_col = col
            break
    if residual_col is None:
        raise ValueError(
            "时序数据中未找到残差列（depth_residual 或 depth_residual_rel 等）。"
            "请先运行 residual_features.py。"
        )

    print(f"[可观测性] 缺陷列={defect_col}, 残差列={residual_col}, 阈值={threshold_rel}")

    influence_sets: Dict[str, Set[str]] = {c: set() for c in candidate_nodes}

    for candidate in candidate_nodes:
        sub = df[df[defect_col].astype(str) == str(candidate)]
        if sub.empty:
            continue

        if "residual_rel" in residual_col or "relative" in residual_col:
            influenced = sub[sub[residual_col].abs() > threshold_rel]["node_id"].unique()
        else:
            max_res = sub[residual_col].abs().max()
            if max_res < 1e-8:
                continue
            influenced = sub[
                sub[residual_col].abs() > max_res * threshold_rel
            ]["node_id"].unique()

        influence_sets[candidate] = set(str(n) for n in influenced)

    covered = sum(1 for s in influence_sets.values() if len(s) > 0)
    print(f"[可观测性] 有影响集的候选节点: {covered}/{len(candidate_nodes)}")
    return influence_sets


def strategy_observability(
    G: nx.DiGraph,
    n: int,
    candidate_nodes: List[str],
    influence_sets: Dict[str, Set[str]],
    exclude: Set[str] = None,
) -> List[str]:
    """贪心最大化候选节点覆盖度：每轮选一个节点使新增覆盖的候选缺陷节点数最多。"""
    exclude = exclude or set()
    all_nodes = [node for node in G.nodes() if node not in exclude]

    node_coverage: Dict[str, Set[str]] = {v: set() for v in all_nodes}
    for candidate, inf_set in influence_sets.items():
        for v in inf_set:
            if v in node_coverage:
                node_coverage[v].add(candidate)

    selected = []
    covered_candidates: Set[str] = set()

    for step in range(n):
        best_node = None
        best_gain = -1
        for v in all_nodes:
            if v in selected:
                continue
            gain = len(node_coverage[v] - covered_candidates)
            if gain > best_gain:
                best_gain = gain
                best_node = v
        if best_node is None:
            break
        selected.append(best_node)
        covered_candidates |= node_coverage[best_node]
        print(f"  步骤{step+1:2d}: 选 {best_node:<15s} "
              f"新增覆盖 {best_gain:2d} 个候选节点，"
              f"累计 {len(covered_candidates)}/{len(candidate_nodes)}")

    final_coverage = len(covered_candidates) / max(len(candidate_nodes), 1) * 100
    print(f"[可观测性] 最终覆盖率: {final_coverage:.1f}% "
          f"({len(covered_candidates)}/{len(candidate_nodes)})")
    return selected


# ============================================================
# 策略5：全节点（pretest）
# ============================================================

def strategy_full(G: nx.DiGraph, n: int, exclude: Set[str] = None) -> List[str]:
    """全图节点作为监测集（用于 A0 pretest 性能上限）。n 通常取图节点总数。"""
    exclude = exclude or set()
    all_nodes = [node for node in G.nodes() if node not in exclude]
    selected = all_nodes[:min(n, len(all_nodes))] if n < len(all_nodes) else all_nodes
    print(f"[全节点] 监测节点数: {len(selected)}")
    return selected


# ============================================================
# 策略6：随机
# ============================================================

def strategy_random(G: nx.DiGraph, n: int, exclude: Set[str] = None, seed: int = 42) -> List[str]:
    """随机选 N 个节点（用于 A5 基线）。"""
    exclude = exclude or set()
    pool = [node for node in G.nodes() if node not in exclude]
    rng = np.random.default_rng(seed)
    if len(pool) <= n:
        selected = pool
    else:
        selected = list(rng.choice(pool, size=n, replace=False))
    print(f"[随机] 种子={seed}, 选出 {len(selected)} 个节点")
    return selected


# ============================================================
# 主函数
# ============================================================

def main():
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()

    parser = argparse.ArgumentParser(description="生成监测节点集 S")
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["degree", "betweenness", "downstream", "observability", "full", "random"],
        help="监测节点选择策略"
    )
    parser.add_argument("--n", type=int, default=25, help="监测节点数量（默认25）；full 时建议取图节点总数如 128")
    parser.add_argument("--seed", type=int, default=42, help="random 策略的随机种子")
    parser.add_argument("--adj", type=str, default="", help="邻接矩阵路径（默认从 config）")
    parser.add_argument("--node_list", type=str, default="", help="节点列表路径（默认从 config）")
    parser.add_argument("--candidate_nodes", type=str, default="", help="候选节点 JSON（默认从 config）")
    parser.add_argument("--timeseries_path", type=str, default="",
                        help="带残差时序 parquet（仅 observability 需要，默认从 config）")
    parser.add_argument("--defect_matrix", type=str, default="",
                        help="缺陷矩阵 CSV（仅 observability 且 parquet 无 defect_node 时需要）")
    parser.add_argument("--obs_threshold", type=float, default=0.01,
                        help="可观测性策略的残差阈值（默认0.01）")
    parser.add_argument("--out_dir", type=str, default="", help="输出目录（默认从 config）")
    args = parser.parse_args()

    adj = args.adj or cfg.adjacency_matrix_file
    node_list_path = args.node_list or cfg.node_list_file
    candidate_path = args.candidate_nodes or cfg.candidate_nodes_file
    timeseries_path = args.timeseries_path or cfg.node_timeseries_file
    defect_matrix_path = args.defect_matrix or cfg.defect_matrix_file
    out_dir = args.out_dir or cfg.input_dir_new

    print("=" * 60)
    print(f"策略: {args.strategy} | N={args.n}")
    print("=" * 60)
    node_list = load_node_list(node_list_path)
    G = build_directed_graph(node_list, adj)
    print(f"图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    candidate_nodes = load_candidate_nodes(candidate_path)
    print(f"候选缺陷节点集 C: {len(candidate_nodes)} 个节点")

    exclude = set(n for n in G.nodes() if G.in_degree(n) + G.out_degree(n) == 0)
    if exclude:
        print(f"排除零度节点: {len(exclude)} 个")

    if args.strategy == "degree":
        selected = strategy_degree(G, args.n, exclude)
    elif args.strategy == "betweenness":
        selected = strategy_betweenness(G, args.n, exclude)
    elif args.strategy == "downstream":
        selected = strategy_downstream(G, args.n, exclude)
    elif args.strategy == "full":
        selected = strategy_full(G, args.n, exclude)
    elif args.strategy == "random":
        selected = strategy_random(G, args.n, exclude, seed=args.seed)
    elif args.strategy == "observability":
        influence_sets = compute_influence_sets(
            timeseries_path=timeseries_path,
            candidate_nodes=candidate_nodes,
            threshold_rel=args.obs_threshold,
            defect_matrix_path=defect_matrix_path if os.path.isfile(defect_matrix_path) else None,
        )
        selected = strategy_observability(
            G=G, n=args.n, candidate_nodes=candidate_nodes,
            influence_sets=influence_sets, exclude=exclude,
        )

    out_path = os.path.join(out_dir, f"monitor_nodes_{args.strategy}_N{args.n}.json")
    save_monitor_nodes(
        selected=selected, strategy=args.strategy, n=args.n, out_path=out_path,
        extra_meta={
            "candidate_nodes_file": os.path.basename(candidate_path),
            "adj_file": os.path.basename(adj),
            "node_list_file": os.path.basename(node_list_path),
        }
    )

    overlap = set(selected) & set(candidate_nodes)
    print(f"\n[统计] S ∩ C 重叠节点数: {len(overlap)}/{args.n}")
    if overlap:
        print(f"       重叠节点示例: {sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")
    print("=" * 60)
    print("[完成] 监测节点集生成结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
