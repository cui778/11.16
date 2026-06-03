#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the conservative Chapter-1 restart IE matrix.

Definition:
- candidate set C is fixed to 50 nodes
- I covers all 50 candidate nodes
- E covers only nodes that are legal under the current pipe-based E semantics
  (a candidate node must appear as the to_node of at least one valid link)
- each active node/type gets the same number of scenes

This script does not redefine E semantics. It only restarts the formal matrix
under the currently executable definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_int_list(text: str) -> List[int]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError(f"Empty integer list: {text!r}")
    return values


def _all_templates(
    intensity_choices: Sequence[int],
    start_hour_choices: Sequence[int],
    duration_choices: Sequence[int],
) -> List[dict]:
    out = []
    for intensity in intensity_choices:
        for start_hour in start_hour_choices:
            for duration_h in duration_choices:
                out.append(
                    {
                        "intensity_pct": int(intensity),
                        "start_hour": int(start_hour),
                        "duration_h": int(duration_h),
                    }
                )
    return out


def _pick_templates(
    templates: Sequence[dict],
    scenes_per_node: int,
    seed: int,
) -> List[dict]:
    if scenes_per_node > len(templates):
        raise ValueError(
            f"scenes_per_node={scenes_per_node} exceeds available templates={len(templates)}"
        )
    arr = list(templates)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    return arr[:scenes_per_node]


def _load_e_legal_nodes(
    inp_root: Path,
    candidate_nodes: Sequence[str],
) -> Dict[str, str]:
    """
    Return mapping: e_node_id -> representative_link_id
    under the current E semantics.
    """
    import importlib.util

    dm_path = inp_root / "prep" / "defect_matrix.py"
    spec = importlib.util.spec_from_file_location("defect_matrix_runtime", dm_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    generator = module.RealisticDefectMatrixGenerator(
        inp_root.parent / "input_data" / "2_tuned_v3_merged1.inp",
        json_path=inp_root / "input_1" / "parsed_inp_data.json",
        candidate_nodes_file=inp_root / "input_1" / "candidate_nodes_new.json",
    )
    ok = generator.calculate_baseline(sample_interval=100)
    if not ok:
        raise RuntimeError("Failed to calculate baseline for E feasibility audit.")

    candidates = generator._select_defect_candidates(None, None)
    link_ids = list(candidates["E"])
    e_node_to_link: Dict[str, str] = {}
    candidate_set = set(str(n) for n in candidate_nodes)
    for lid in link_ids:
        props = generator.link_properties.get(lid, {})
        to_node = str(props.get("to_node", ""))
        if to_node and to_node in candidate_set and to_node not in e_node_to_link:
            e_node_to_link[to_node] = str(lid)
    return e_node_to_link


def build_matrix(
    repo_root: Path,
    output_csv: Path,
    scenes_per_node: int,
    i_intensities: Sequence[int],
    e_intensities: Sequence[int],
    seed: int,
) -> dict:
    input_dir = repo_root / "input_1"
    baseline_stats = _load_json(input_dir / "baseline_flow_stats.json")
    candidate_nodes = _load_json(input_dir / "candidate_nodes_new.json")["candidate_nodes"]
    monitor_nodes = _load_json(input_dir / "monitor_nodes_degree_N25.json")["monitor_nodes"]

    e_node_to_link = _load_e_legal_nodes(repo_root, candidate_nodes)

    i_nodes = [str(n) for n in candidate_nodes]
    e_nodes = sorted(e_node_to_link.keys())

    i_templates = _all_templates(
        intensity_choices=i_intensities,
        start_hour_choices=[2, 8, 14, 20],
        duration_choices=[6, 12, 18, 24],
    )
    e_templates = _all_templates(
        intensity_choices=e_intensities,
        start_hour_choices=[0, 6, 12, 18],
        duration_choices=[6, 12, 18, 24],
    )

    rows = []
    defect_id = 1

    for node_pos, node_id in enumerate(i_nodes):
        baseline_flow = float(baseline_stats["nodes"].get(node_id, {}).get("mean", 1.0))
        selected = _pick_templates(i_templates, scenes_per_node, seed + node_pos)
        for tpl in selected:
            intensity = int(tpl["intensity_pct"])
            flow = round(float(baseline_flow * 0.01 * intensity), 4)
            rows.append(
                {
                    "defect_id": defect_id,
                    "defect_type": "I",
                    "node_id": node_id,
                    "link_id": "",
                    "intensity_pct": intensity,
                    "start_hour": int(tpl["start_hour"]),
                    "duration_h": int(tpl["duration_h"]),
                    "flow": flow,
                    "BODf": 0.0,
                    "NH4": 0.0,
                    "DO": 0.0,
                    "baseline_flow": round(float(baseline_flow), 4),
                }
            )
            defect_id += 1

    for node_pos, node_id in enumerate(e_nodes):
        link_id = e_node_to_link[node_id]
        baseline_flow = float(baseline_stats["links"].get(link_id, {}).get("mean", 1.0))
        selected = _pick_templates(e_templates, scenes_per_node, seed + 1000 + node_pos)
        for tpl in selected:
            intensity = int(tpl["intensity_pct"])
            flow = round(float(-baseline_flow * 0.01 * intensity), 4)
            rows.append(
                {
                    "defect_id": defect_id,
                    "defect_type": "E",
                    "node_id": node_id,
                    "link_id": link_id,
                    "intensity_pct": intensity,
                    "start_hour": int(tpl["start_hour"]),
                    "duration_h": int(tpl["duration_h"]),
                    "flow": flow,
                    "BODf": 0.0,
                    "NH4": 0.0,
                    "DO": 0.0,
                    "baseline_flow": round(float(baseline_flow), 4),
                }
            )
            defect_id += 1

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    candidate_set = set(i_nodes)
    monitor_set = set(str(n) for n in monitor_nodes)
    active_set = set(df["node_id"].astype(str))
    summary = {
        "output_csv": str(output_csv),
        "n_rows": int(len(df)),
        "candidate_count": int(len(candidate_set)),
        "monitor_count": int(len(monitor_set)),
        "i_rows": int((df["defect_type"] == "I").sum()),
        "e_rows": int((df["defect_type"] == "E").sum()),
        "i_unique_nodes": int(df.loc[df["defect_type"] == "I", "node_id"].nunique()),
        "e_unique_nodes": int(df.loc[df["defect_type"] == "E", "node_id"].nunique()),
        "active_unique_nodes": int(len(active_set)),
        "candidate_active_overlap": int(len(candidate_set & active_set)),
        "monitor_active_overlap": int(len(monitor_set & active_set)),
        "missing_e_candidate_nodes": sorted(candidate_set - set(e_nodes)),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build formal conservative IE matrix for Chapter-1 restart.")
    parser.add_argument(
        "--output-csv",
        default=str(repo_root / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative.csv"),
    )
    parser.add_argument("--scenes-per-node", type=int, default=4)
    parser.add_argument("--i-intensities", default="40,50,60")
    parser.add_argument("--e-intensities", default="40,50,60")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_matrix(
        repo_root=repo_root,
        output_csv=Path(args.output_csv),
        scenes_per_node=int(args.scenes_per_node),
        i_intensities=_parse_int_list(args.i_intensities),
        e_intensities=_parse_int_list(args.e_intensities),
        seed=int(args.seed),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
