#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的INP文件解析器

来源：原 script2/visualizations/缺陷矩阵/1_inp_parser.py
用途：替代所有脚本中重复的INP解析逻辑，确保所有脚本使用相同的解析结果。
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InpNetwork:
    """INP网络数据结构"""
    nodes_df: pd.DataFrame  # node_id, x, y, elevation, type
    links_df: pd.DataFrame  # link_id, from_node, to_node, length_m, diameter_m, shape
    boundary_inlets: List[str]  # 边界入流节点


class UnifiedINPParser:
    """统一的INP文件解析器"""

    def __init__(self, inp_path: str):
        self.inp_path = Path(inp_path)
        self.data = {
            'junctions': {},
            'outfalls': {},
            'storage': {},
            'conduits': {},
            'xsections': {},
            'coordinates': {},
            'boundary_inlets': []
        }

    def parse(self) -> InpNetwork:
        """解析INP文件，返回统一格式的网络数据"""
        logger.info(f"开始解析INP文件: {self.inp_path}")

        with open(self.inp_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        section = None
        section_patterns = {
            'JUNCTIONS': self._parse_junctions,
            'OUTFALLS': self._parse_outfalls,
            'STORAGE': self._parse_storage,
            'CONDUITS': self._parse_conduits,
            'XSECTIONS': self._parse_xsections,
            'COORDINATES': self._parse_coordinates
        }

        for line in lines:
            line = line.strip()

            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1].upper()
                continue

            if not line or line.startswith(';;'):
                continue

            if section in section_patterns:
                section_patterns[section](line)

        return self._build_network()

    def _parse_junctions(self, line: str):
        parts = line.split()
        if len(parts) >= 2:
            node_id = parts[0]
            elevation = float(parts[1])
            self.data['junctions'][node_id] = {
                'elevation': elevation,
                'type': 'JUNCTION'
            }

    def _parse_outfalls(self, line: str):
        parts = line.split()
        if len(parts) >= 2:
            node_id = parts[0]
            elevation = float(parts[1])
            self.data['outfalls'][node_id] = {
                'elevation': elevation,
                'type': 'OUTFALL'
            }

    def _parse_storage(self, line: str):
        parts = line.split()
        if len(parts) >= 2:
            node_id = parts[0]
            elevation = float(parts[1])
            self.data['storage'][node_id] = {
                'elevation': elevation,
                'type': 'STORAGE'
            }

    def _parse_conduits(self, line: str):
        parts = line.split()
        if len(parts) >= 3:
            link_id = parts[0]
            from_node = parts[1]
            to_node = parts[2]
            length = float(parts[3]) if len(parts) > 3 else 0.0

            self.data['conduits'][link_id] = {
                'from_node': from_node,
                'to_node': to_node,
                'length': length
            }

    def _parse_xsections(self, line: str):
        parts = line.split()
        if len(parts) >= 3:
            link_id = parts[0]
            shape = parts[1]
            geom1 = float(parts[2])
            geom2 = float(parts[3]) if len(parts) > 3 else 0.0

            self.data['xsections'][link_id] = {
                'shape': shape,
                'geom1': geom1,
                'geom2': geom2
            }

    def _parse_coordinates(self, line: str):
        parts = line.split()
        if len(parts) >= 3:
            node_id = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            self.data['coordinates'][node_id] = {'x': x, 'y': y}

    def _build_network(self) -> InpNetwork:
        """构建统一的网络数据结构"""
        all_nodes = {}

        for node_id, info in self.data['junctions'].items():
            all_nodes[node_id] = {
                'elevation': info['elevation'],
                'type': info['type']
            }

        for node_id, info in self.data['outfalls'].items():
            all_nodes[node_id] = {
                'elevation': info['elevation'],
                'type': info['type']
            }

        for node_id, info in self.data['storage'].items():
            all_nodes[node_id] = {
                'elevation': info['elevation'],
                'type': info['type']
            }

        nodes_data = []
        for node_id, info in all_nodes.items():
            coord = self.data['coordinates'].get(node_id, {'x': 0, 'y': 0})
            nodes_data.append({
                'node_id': node_id,
                'x': coord['x'],
                'y': coord['y'],
                'elevation': info['elevation'],
                'type': info['type']
            })

        nodes_df = pd.DataFrame(nodes_data)

        links_data = []
        for link_id, conduit in self.data['conduits'].items():
            xsection = self.data['xsections'].get(link_id, {'shape': 'CIRCULAR', 'geom1': 0.0, 'geom2': 0.0})
            diameter = xsection['geom1'] if xsection['shape'] == 'CIRCULAR' else xsection['geom1']

            links_data.append({
                'link_id': link_id,
                'from_node': conduit['from_node'],
                'to_node': conduit['to_node'],
                'length_m': conduit['length'],
                'diameter_m': diameter,
                'shape': xsection['shape']
            })

        links_df = pd.DataFrame(links_data)
        boundary_inlets = self._identify_boundary_inlets(links_df, nodes_df)

        logger.info(f"解析完成: {len(nodes_df)}个节点, {len(links_df)}个管道, {len(boundary_inlets)}个边界入流")

        return InpNetwork(
            nodes_df=nodes_df,
            links_df=links_df,
            boundary_inlets=boundary_inlets
        )

    def _identify_boundary_inlets(self, links_df: pd.DataFrame, nodes_df: pd.DataFrame) -> List[str]:
        from_counts = links_df['from_node'].value_counts()
        to_counts = links_df['to_node'].value_counts()
        all_degrees = (from_counts + to_counts).fillna(0)
        junction_nodes = set(nodes_df[nodes_df['type'] == 'JUNCTION']['node_id'])
        boundary_nodes = []
        for node, degree in all_degrees.items():
            if node in junction_nodes and degree == 1:
                boundary_nodes.append(node)
        return sorted(boundary_nodes)

    def save_to_json(self, output_path: str):
        """保存解析结果到JSON文件（兼容现有脚本）"""
        legacy_data = {
            'junctions': self.data['junctions'],
            'outfalls': self.data['outfalls'],
            'storage': self.data['storage'],
            'conduits': self.data['conduits'],
            'xsections': self.data['xsections'],
            'coordinates': self.data['coordinates'],
            'boundary_inlets': self._identify_boundary_inlets(
                pd.DataFrame([{
                    'link_id': k, 'from_node': v['from_node'], 'to_node': v['to_node']
                } for k, v in self.data['conduits'].items()]),
                pd.DataFrame([{
                    'node_id': k, 'elevation': v['elevation'], 'type': v['type']
                } for k, v in {**self.data['junctions'], **self.data['outfalls'], **self.data['storage']}.items()])
            )
        }

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(legacy_data, f, indent=2, ensure_ascii=False)

        logger.info(f"解析结果已保存到: {output_path}")


def main():
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()
    inp_path = cfg.inp_file
    output_path = cfg.parsed_inp_data_file

    parser = UnifiedINPParser(inp_path)
    network = parser.parse()
    parser.save_to_json(output_path)

    print(f"\n解析统计:")
    print(f"  节点数: {len(network.nodes_df)}")
    print(f"  管道数: {len(network.links_df)}")
    print(f"  边界入流: {len(network.boundary_inlets)}")
    print(f"  节点类型分布: {network.nodes_df['type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
