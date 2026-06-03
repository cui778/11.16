# -*- coding: utf-8 -*-
"""
Validate V / C / S / D set consistency for the current experiment configuration.

Checks:
- C subset V
- S subset V
- D subset C
- Optional overlap statistics between S and C
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def _load_node_list(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        nodes = data.get("node_list", [])
    else:
        nodes = data
    return {str(x) for x in nodes}


def _load_json_node_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        raw = data.get("candidate_nodes") or data.get("monitor_nodes") or data.get("selected_nodes") or data.get("nodes") or []
    else:
        raw = data
    return [str(x) for x in raw]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import Config

    cfg = Config()
    parser = argparse.ArgumentParser(description="Validate node set consistency for V/C/S/D.")
    parser.add_argument("--node-list-file", default=cfg.node_list_file)
    parser.add_argument("--candidate-file", default=cfg.candidate_nodes_file)
    parser.add_argument("--monitor-file", default=cfg.monitor_nodes_file)
    parser.add_argument("--defect-matrix-file", default=cfg.defect_matrix_file)
    args = parser.parse_args()

    node_list_file = Path(args.node_list_file)
    candidate_file = Path(args.candidate_file)
    monitor_file = Path(args.monitor_file)
    defect_matrix_file = Path(args.defect_matrix_file)

    v_nodes = _load_node_list(node_list_file)
    c_nodes = set(_load_json_node_list(candidate_file))
    s_nodes = set(_load_json_node_list(monitor_file))

    defect_df = pd.read_csv(defect_matrix_file)
    d_nodes = {
        str(node_id)
        for node_id in defect_df.get("node_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
        if str(node_id).strip() and str(node_id).strip().upper() != "N/A"
    }

    missing_c = sorted(c_nodes - v_nodes)
    missing_s = sorted(s_nodes - v_nodes)
    missing_d = sorted(d_nodes - c_nodes)

    report = {
        "n_v": len(v_nodes),
        "n_c": len(c_nodes),
        "n_s": len(s_nodes),
        "n_d": len(d_nodes),
        "c_subset_v": len(missing_c) == 0,
        "s_subset_v": len(missing_s) == 0,
        "d_subset_c": len(missing_d) == 0,
        "s_intersect_c": len(s_nodes & c_nodes),
        "s_only": len(s_nodes - c_nodes),
        "c_only": len(c_nodes - s_nodes),
        "missing_c_examples": missing_c[:10],
        "missing_s_examples": missing_s[:10],
        "missing_d_examples": missing_d[:10],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not (report["c_subset_v"] and report["s_subset_v"] and report["d_subset_c"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
