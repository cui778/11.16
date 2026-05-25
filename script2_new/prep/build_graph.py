# -*- coding: utf-8 -*-
"""
构建全网图，得到节点集 V 及邻接等图结构。

来源：原 script2/visualizations/缺陷矩阵/6_build_graph_parquet.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


def parse_inp_and_build_graph(inp_file_path, output_dir):
    """
    解析 INP，构建图结构，并保存为标准格式 (npy + json)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inlet_nodes = []
    outlet_nodes = []

    print(f"正在读取: {inp_file_path}")
    with open(inp_file_path, 'r') as f:
        lines = f.readlines()
        section = None
        for line in lines:
            line = line.strip()
            if line.startswith("[CONDUITS]"):
                section = "conduits"
                continue
            elif line.startswith("["):
                section = None

            if section == "conduits" and line and not line.startswith(";;"):
                parts = line.split()
                if len(parts) >= 3:
                    inlet_nodes.append(parts[1])
                    outlet_nodes.append(parts[2])

    all_nodes = sorted(list(set(inlet_nodes + outlet_nodes)))
    print(f"找到 {len(all_nodes)} 个唯一节点。")

    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

    row_indices = []
    col_indices = []
    for i in range(len(inlet_nodes)):
        if inlet_nodes[i] in node_to_index and outlet_nodes[i] in node_to_index:
            u = node_to_index[inlet_nodes[i]]
            v = node_to_index[outlet_nodes[i]]
            row_indices.append(u)
            col_indices.append(v)

    adj_matrix = np.zeros((len(all_nodes), len(all_nodes)), dtype=np.float32)
    adj_matrix[row_indices, col_indices] = 1.0

    json_path = os.path.join(output_dir, "node_list.json")
    with open(json_path, 'w') as f:
        json.dump(all_nodes, f)

    npy_path = os.path.join(output_dir, "adj_matrix.npy")
    np.save(npy_path, adj_matrix)

    print("=" * 40)
    print("✅ 图结构构建完成！")
    print(f"1. 节点列表已保存: {json_path}")
    print(f"2. 邻接矩阵已保存: {npy_path} (Shape: {adj_matrix.shape})")
    print("=" * 40)

    return all_nodes, adj_matrix


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()
    INP_FILE = cfg.inp_file
    OUTPUT_DIR = cfg.input_dir_new

    parse_inp_and_build_graph(INP_FILE, OUTPUT_DIR)
