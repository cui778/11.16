#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Chapter-5 dataset-view folders.

Each layout gets its own lightweight dataset-view directory, but all of them
share the same frozen full-graph source dataset. The only changing factor is
the monitor layout / observed mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter-5 dataset-view folders.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument(
        "--strategies",
        default="random,degree,betweenness,downstream,candidate_observability,identifiability_driven",
    )
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[2]
    ch5_root = script_root / "chapter5_layout_optimization"
    outputs_root = ch5_root / "outputs"
    layouts_root = outputs_root / "layouts"
    datasets_root = outputs_root / "datasets"

    shared_dataset_dir = script_root / "training_data_new" / "time_gated_full_ie_v4_formal_conservative420_seed42"
    shared_manifest = shared_dataset_dir / "dataset_manifest.json"
    shared_parquet = shared_dataset_dir / "node_timeseries_with_residuals.parquet"
    defect_matrix = script_root / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"
    candidate_nodes = script_root / "input_1" / "candidate_nodes_new.json"

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    for strategy in strategies:
        for n in budgets:
            layout_file = layouts_root / strategy / f"monitor_nodes_{strategy}_N{n}.json"
            out_dir = datasets_root / f"{strategy}_N{n}"
            out_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "dataset_view_name": f"{strategy}_N{n}",
                "shared_source_dataset_dir": str(shared_dataset_dir),
                "shared_source_manifest": str(shared_manifest),
                "shared_source_parquet": str(shared_parquet),
                "defect_matrix_file": str(defect_matrix),
                "candidate_nodes_file": str(candidate_nodes),
                "monitor_nodes_file": str(layout_file),
                "data_semantics": {
                    "scenarios_fixed": True,
                    "full_graph_data_fixed": True,
                    "defect_matrix_fixed": True,
                    "layout_changes_observed_mask_only": True,
                },
                "note": "This is a lightweight Chapter-5 dataset view. The full-graph source data is shared; only monitor layout and observed mask change.",
            }

            (out_dir / "dataset_view_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(str(datasets_root))


if __name__ == "__main__":
    main()

