#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预计算图路径特征矩阵，供 HydraulicInverseAttention 模型使用。

输入：adj_matrix.npy + node_list.json + parsed_inp_data.json
输出：input_1/graph_path_features.npz
  - shortest_dist [N, N]: 最短路径跳数（无连接填 0 或大常数）
  - pipe_length_dist [N, N]: 最短路径上管道长度之和（无连接填 0）
  - flow_direction [N, N]: i→j 顺流 +1 / 逆流 -1 / 无连接 0
  - elevation_diff [N, N]: 高程差 elevation[j]-elevation[i]（缺失填 0）
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    nx = None


def load_node_list(node_list_path):
    with open(node_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "node_list" in data:
        return [str(n) for n in data["node_list"]]
    return [str(n) for n in data]


def build_graph_features(adj_matrix_path, node_list_path, parsed_inp_path, output_npz_path):
    adj = np.load(adj_matrix_path)
    node_list = load_node_list(node_list_path)
    N = len(node_list)
    assert adj.shape == (N, N), f"adj {adj.shape} vs node_list len {N}"

    node_to_idx = {str(n): i for i, n in enumerate(node_list)}

    # 高程：仅 junctions 有，其余填 0
    elevation = np.zeros(N, dtype=np.float32)
    if os.path.exists(parsed_inp_path):
        with open(parsed_inp_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        junctions = parsed.get("junctions", {})
        for i, nid in enumerate(node_list):
            if str(nid) in junctions:
                elevation[i] = float(junctions[str(nid)].get("elevation", 0))

    # 有向图：边 (u,v) 表示 u→v（水从 u 流向 v），与 adj 一致
    G = nx.DiGraph()
    edge_lengths = {}  # (u_idx, v_idx) -> length
    conduits = {}
    if os.path.exists(parsed_inp_path):
        with open(parsed_inp_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        conduits = parsed.get("conduits", {})

    for eid, info in conduits.items():
        u_id = str(info.get("from_node", ""))
        v_id = str(info.get("to_node", ""))
        length = float(info.get("length", 0.0))
        if u_id in node_to_idx and v_id in node_to_idx:
            u, v = node_to_idx[u_id], node_to_idx[v_id]
            G.add_edge(u, v, weight=1.0)
            edge_lengths[(u, v)] = length

    # 若 parsed 里没有 conduit，用邻接矩阵建图（无长度信息，长度填 0）
    if G.number_of_edges() == 0:
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    G.add_edge(i, j, weight=1.0)
                    edge_lengths[(i, j)] = 0.0

    INF_HOP = 999
    INF_LEN = 1e6

    shortest_dist = np.zeros((N, N), dtype=np.float32)
    pipe_length_dist = np.zeros((N, N), dtype=np.float32)
    flow_direction = np.zeros((N, N), dtype=np.float32)
    elevation_diff = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            elevation_diff[i, j] = elevation[j] - elevation[i]
            if i == j:
                shortest_dist[i, j] = 0.0
                pipe_length_dist[i, j] = 0.0
                flow_direction[i, j] = 0.0
                continue
            try:
                path = nx.shortest_path(G, source=i, target=j, weight="weight")
                hops = len(path) - 1
                total_len = 0.0
                for k in range(len(path) - 1):
                    total_len += edge_lengths.get((path[k], path[k + 1]), 0.0)
                shortest_dist[i, j] = float(hops)
                pipe_length_dist[i, j] = total_len
                flow_direction[i, j] = 1.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    path = nx.shortest_path(G, source=j, target=i, weight="weight")
                    flow_direction[i, j] = -1.0
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                shortest_dist[i, j] = INF_HOP
                pipe_length_dist[i, j] = INF_LEN

    # 归一化/裁剪，便于模型使用：大常数改为固定上界
    max_hop = 15
    max_len = 5000.0
    shortest_dist = np.clip(shortest_dist, 0, max_hop).astype(np.float32)
    pipe_length_dist = np.clip(pipe_length_dist, 0, max_len).astype(np.float32)

    os.makedirs(os.path.dirname(output_npz_path) or ".", exist_ok=True)
    np.savez(
        output_npz_path,
        shortest_dist=shortest_dist,
        pipe_length_dist=pipe_length_dist,
        flow_direction=flow_direction,
        elevation_diff=elevation_diff,
    )
    print(f"已写入: {output_npz_path}")
    print(f"  shortest_dist: {shortest_dist.shape}, [0, {shortest_dist.max():.1f}]")
    print(f"  pipe_length_dist: {pipe_length_dist.shape}, [0, {pipe_length_dist.max():.1f}]")
    print(f"  flow_direction: 顺流 {(flow_direction > 0).sum()}, 逆流 {(flow_direction < 0).sum()}, 无连接 {(flow_direction == 0).sum() - N}")
    return output_npz_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import Config

    cfg = Config()
    build_graph_features(
        adj_matrix_path=cfg.adjacency_matrix_file,
        node_list_path=cfg.node_list_file,
        parsed_inp_path=cfg.parsed_inp_data_file,
        output_npz_path=os.path.join(cfg.input_dir_new, "graph_path_features.npz"),
    )
