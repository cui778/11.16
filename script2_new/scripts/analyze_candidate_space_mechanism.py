#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze Chapter 4 V/S/C/D candidate-space mechanism.

This script is intentionally analysis-only. It does not train models and does
not redefine the formal Chapter 4 task. C128 and C50+Neg are documented as
boundary probes rather than promoted to the main result table.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "chapter4_diagnosis_model" / "outputs"
DEFAULT_DOC = (
    SCRIPT_ROOT
    / "chapter4_diagnosis_model"
    / "docs"
    / "CH4_CANDIDATE_SPACE_MECHANISM_20260505.md"
)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_nodes(obj, keys: Iterable[str]) -> list[str]:
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
        if "node_list" in obj and isinstance(obj["node_list"], list):
            return [str(x) for x in obj["node_list"]]
    raise ValueError("Cannot extract node list from JSON object")


def _undirected_min_distances(adj: np.ndarray, source_idx: int) -> list[int | None]:
    graph = (adj != 0) | (adj.T != 0)
    n = graph.shape[0]
    dist: list[int | None] = [None] * n
    dist[source_idx] = 0
    q: deque[int] = deque([source_idx])
    while q:
        cur = q.popleft()
        neighbors = np.flatnonzero(graph[cur])
        for nb in neighbors:
            if dist[int(nb)] is None:
                dist[int(nb)] = int(dist[cur]) + 1
                q.append(int(nb))
    return dist


def _tier(min_hop: int | None, in_monitor: bool) -> str:
    if in_monitor:
        return "direct"
    if min_hop is None:
        return "unreachable"
    if min_hop <= 2:
        return "near"
    return "far"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Chapter 4 candidate-space mechanism")
    parser.add_argument("--node-list", default=str(SCRIPT_ROOT / "input_1" / "node_list.json"))
    parser.add_argument("--candidate-nodes", default=str(SCRIPT_ROOT / "input_1" / "candidate_nodes_new.json"))
    parser.add_argument("--monitor-nodes", default=str(SCRIPT_ROOT / "input_1" / "monitor_nodes_degree_N25.json"))
    parser.add_argument(
        "--defect-matrix",
        default=str(SCRIPT_ROOT / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"),
    )
    parser.add_argument("--adj-matrix", default=str(SCRIPT_ROOT / "input_1" / "adj_matrix.npy"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--doc-output", default=str(DEFAULT_DOC))
    args = parser.parse_args()

    node_list = _extract_nodes(_load_json(Path(args.node_list)), ["node_list", "nodes"])
    candidate_nodes = _extract_nodes(_load_json(Path(args.candidate_nodes)), ["candidate_nodes", "nodes", "candidates"])
    monitor_nodes = _extract_nodes(_load_json(Path(args.monitor_nodes)), ["monitor_nodes", "nodes", "observed_nodes"])
    defect_df = pd.read_csv(args.defect_matrix)
    adj = np.load(args.adj_matrix)

    node_set = set(node_list)
    candidate_set = set(candidate_nodes)
    monitor_set = set(monitor_nodes)
    active_set = set(defect_df["node_id"].astype(str))
    i_set = set(defect_df.loc[defect_df["defect_type"] == "I", "node_id"].astype(str))
    e_set = set(defect_df.loc[defect_df["defect_type"] == "E", "node_id"].astype(str))

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    monitor_indices = [node_to_idx[n] for n in monitor_nodes if n in node_to_idx]
    candidate_rows = []

    monitor_dist_cache = {
        monitor_idx: _undirected_min_distances(adj, monitor_idx)
        for monitor_idx in monitor_indices
    }

    for node_id in candidate_nodes:
        idx = node_to_idx.get(node_id)
        if idx is None:
            min_hop = None
            nearest_monitors = []
        else:
            distances = [
                (monitor_nodes[pos], dist_list[idx])
                for pos, (monitor_idx, dist_list) in enumerate(monitor_dist_cache.items())
                if dist_list[idx] is not None
            ]
            if distances:
                min_hop = min(int(d) for _, d in distances)
                nearest_monitors = [m for m, d in distances if int(d) == min_hop]
            else:
                min_hop = None
                nearest_monitors = []

        in_monitor = node_id in monitor_set
        has_i = node_id in i_set
        has_e = node_id in e_set
        candidate_rows.append(
            {
                "node_id": node_id,
                "in_monitor_S": int(in_monitor),
                "in_active_D": int(node_id in active_set),
                "has_I_scenario": int(has_i),
                "has_E_scenario": int(has_e),
                "min_monitor_hop": "" if min_hop is None else int(min_hop),
                "observability_tier": _tier(min_hop, in_monitor),
                "nearest_monitors": ";".join(nearest_monitors[:5]),
            }
        )

    relationships = [
        {"set_name": "V_full_graph", "count": len(node_set), "description": "128-node topology input space"},
        {"set_name": "S_degree_N25_monitors", "count": len(monitor_set), "description": "dynamic observation space"},
        {"set_name": "C_candidate_defect_nodes", "count": len(candidate_set), "description": "formal localization output space"},
        {"set_name": "D_active_defect_nodes", "count": len(active_set), "description": "active defect nodes in formal IE420 matrix"},
        {"set_name": "S_intersect_C", "count": len(monitor_set & candidate_set), "description": "monitor-candidate overlap"},
        {"set_name": "C_intersect_D", "count": len(candidate_set & active_set), "description": "candidate nodes activated by matrix"},
        {"set_name": "I_active_nodes", "count": len(i_set), "description": "I-type active nodes"},
        {"set_name": "E_active_nodes", "count": len(e_set), "description": "E-type legal active nodes"},
        {"set_name": "E_missing_candidate_nodes", "count": len(candidate_set - e_set), "description": "candidate nodes without legal E scenario"},
    ]

    tiers_df = pd.DataFrame(candidate_rows)
    rel_df = pd.DataFrame(relationships)
    boundary_df = pd.DataFrame(
        [
            {
                "probe": "C50_formal",
                "role": "mainline",
                "output_space": "50 candidate defect nodes",
                "status": "formal Chapter 4 task",
            },
            {
                "probe": "C128_full_graph",
                "role": "boundary_exploration",
                "output_space": "all 128 graph nodes",
                "status": "definition only; not promoted to main table",
            },
            {
                "probe": "C50_plus_hard_negatives",
                "role": "boundary_exploration",
                "output_space": "50 candidates plus non-candidate distractors",
                "status": "definition only; not promoted to main table",
            },
        ]
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_path = out_dir / "candidate_space_relationships.csv"
    tier_path = out_dir / "candidate_observability_tiers.csv"
    boundary_path = out_dir / "candidate_space_boundary_probe_definitions.csv"
    rel_df.to_csv(rel_path, index=False, encoding="utf-8-sig")
    tiers_df.to_csv(tier_path, index=False, encoding="utf-8-sig")
    boundary_df.to_csv(boundary_path, index=False, encoding="utf-8-sig")

    tier_counts = tiers_df["observability_tier"].value_counts().to_dict()
    doc_path = Path(args.doc_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "\n".join(
            [
                "# 第4章候选节点机制分析",
                "",
                "日期：2026-05-05",
                "",
                "## 正式口径",
                "",
                "- V：128 个全网节点，是图结构输入空间。",
                "- S：25 个 degree 监测节点，是动态观测空间。",
                "- C：50 个候选缺陷节点，是正式定位输出空间。",
                "- D：正式 IE420 矩阵中的实际激活缺陷节点。",
                "",
                "第4章主线是全图拓扑约束下的候选缺陷节点排序定位，不是 128 节点全网自由定位。",
                "",
                "## 核验结果",
                "",
                f"- |V| = {len(node_set)}",
                f"- |S| = {len(monitor_set)}",
                f"- |C| = {len(candidate_set)}",
                f"- |D| = {len(active_set)}",
                f"- |S ∩ C| = {len(monitor_set & candidate_set)}",
                f"- |C ∩ D| = {len(candidate_set & active_set)}",
                f"- I 覆盖节点数 = {len(i_set)}",
                f"- E 覆盖节点数 = {len(e_set)}",
                f"- E 类未覆盖候选节点数 = {len(candidate_set - e_set)}",
                f"- 可观测性分层 = {tier_counts}",
                "",
                "## 边界探索",
                "",
                "`C128` 与 `C50+Neg` 仅作为后续边界探索定义，不进入第4章正式主表。第5章继续固定 V 与 C，只优化 S。",
                "",
                "## 输出文件",
                "",
                f"- `{rel_path}`",
                f"- `{tier_path}`",
                f"- `{boundary_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] relationships -> {rel_path}")
    print(f"[OK] tiers -> {tier_path}")
    print(f"[OK] boundary definitions -> {boundary_path}")
    print(f"[OK] doc -> {doc_path}")


if __name__ == "__main__":
    main()
