#!/usr/bin/env python3
"""Build targeted Chapter 5 supplementary layouts and training manifest.

Experiments:
1. Embedding-source ablation at N=25, diagnosis seed 42:
   - topology-diversity
   - coordinate-diversity
   The formal Embedding-Guided result is reused.
2. Low-budget confirmation at N=5, diagnosis seeds 7 and 123:
   - Degree
   - Two-stage v1
   - Embedding-Guided
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("E:/11.16")
SCRIPT2 = ROOT / "script2_new"
CH5 = SCRIPT2 / "chapter5_layout_optimization"
INPUT = SCRIPT2 / "input_1"
LAYOUT_ROOT = CH5 / "outputs/layouts"
OUT_DIR = CH5 / "outputs/targeted_supplements"
REPORTS = SCRIPT2 / "outputs/reports"
PYTHON = Path("D:/conda3/envs/swmm_gpu/python.exe")
TRAIN = SCRIPT2 / "scripts/train_privileged_teacher_student.py"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-8)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def max_min_select(features: np.ndarray, node_list: list[str], budget: int) -> list[str]:
    center = features.mean(axis=0, keepdims=True)
    center_dist = np.linalg.norm(features - center, axis=1)
    first = int(np.argmax(center_dist))
    selected = [first]
    remaining = set(range(len(node_list)))
    remaining.remove(first)
    min_dist = np.linalg.norm(features - features[first : first + 1], axis=1)
    while len(selected) < budget:
        best = max(remaining, key=lambda idx: (float(min_dist[idx]), -idx))
        selected.append(best)
        remaining.remove(best)
        distance = np.linalg.norm(features - features[best : best + 1], axis=1)
        min_dist = np.minimum(min_dist, distance)
    return [node_list[index] for index in selected]


def graph_degrees(parsed: dict, node_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {node: i for i, node in enumerate(node_list)}
    in_degree = np.zeros(len(node_list), dtype=np.float64)
    out_degree = np.zeros(len(node_list), dtype=np.float64)
    for conduit in parsed.get("conduits", {}).values():
        start = conduit.get("from_node")
        end = conduit.get("to_node")
        if start in index:
            out_degree[index[start]] += 1
        if end in index:
            in_degree[index[end]] += 1
    return in_degree, out_degree


def finite_row_stats(matrix: np.ndarray, invalid_at: float = 14.9) -> tuple[np.ndarray, np.ndarray]:
    means = []
    minima = []
    for row in np.asarray(matrix, dtype=np.float64):
        finite = row[np.isfinite(row) & (row < invalid_at)]
        means.append(float(finite.mean()) if finite.size else invalid_at)
        positive = finite[finite > 0]
        minima.append(float(positive.min()) if positive.size else 0.0)
    return np.asarray(means), np.asarray(minima)


def build_feature_views() -> tuple[list[str], dict[str, np.ndarray], dict[str, list[str]]]:
    node_list = [str(node) for node in read_json(INPUT / "node_list.json")]
    parsed = read_json(INPUT / "parsed_inp_data.json")
    graph = np.load(INPUT / "graph_path_features.npz")
    in_degree, out_degree = graph_degrees(parsed, node_list)

    elevations = np.asarray(
        [
            float(
                parsed.get("junctions", {})
                .get(node, {})
                .get("elevation", 0.0)
            )
            for node in node_list
        ],
        dtype=np.float64,
    )
    mean_hop, min_hop = finite_row_stats(graph["shortest_dist"], invalid_at=14.9)
    pipe = np.asarray(graph["pipe_length_dist"], dtype=np.float64)
    finite_pipe = np.where(np.isfinite(pipe) & (pipe < 1e8), pipe, np.nan)
    mean_pipe = np.nanmean(finite_pipe, axis=1)
    mean_pipe = np.nan_to_num(mean_pipe, nan=float(np.nanmean(mean_pipe)))

    topology_raw = np.column_stack(
        [in_degree, out_degree, in_degree + out_degree, elevations, mean_hop, min_hop, mean_pipe]
    )

    coordinates = parsed.get("coordinates", {})
    coordinate_raw = np.asarray(
        [
            [
                float(coordinates.get(node, {}).get("x", 0.0)),
                float(coordinates.get(node, {}).get("y", 0.0)),
            ]
            for node in node_list
        ],
        dtype=np.float64,
    )

    views = {
        "topology_diversity": l2_normalize(zscore(topology_raw)),
        "coordinate_diversity": l2_normalize(zscore(coordinate_raw)),
    }
    names = {
        "topology_diversity": [
            "in_degree",
            "out_degree",
            "total_degree",
            "elevation",
            "mean_reachable_hop",
            "nearest_positive_hop",
            "mean_pipe_length_distance",
        ],
        "coordinate_diversity": ["x", "y"],
    }
    return node_list, views, names


def training_command(layout: Path, tag: str, seed: int) -> str:
    parts = [
        str(PYTHON),
        str(TRAIN),
        "--student-monitors",
        str(layout),
        "--output-tag",
        tag,
        "--seed",
        str(seed),
        "--teacher-subdir",
        "ie420_plus_normal20_v1",
        "--split-mode",
        "scenario",
        "--feature-set",
        "raw_plus_residual",
        "--lambda-loc",
        "0.5",
        "--student-model-type",
        "hydraulic_inverse_deepattn",
        "--lambda-kd",
        "0",
        "--lambda-active-kd",
        "0",
    ]
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def add_row(
    rows: list[dict[str, object]],
    experiment: str,
    method_key: str,
    method: str,
    budget: int,
    seed: int,
    layout: Path,
    tag: str,
) -> None:
    metrics = REPORTS / f"last_run_metrics_{tag}.json"
    rows.append(
        {
            "experiment": experiment,
            "method_key": method_key,
            "method": method,
            "budget": budget,
            "diagnosis_seed": seed,
            "layout_file": str(layout),
            "output_tag": tag,
            "metrics_file": str(metrics),
            "status": "done" if metrics.exists() else "pending",
            "command": training_command(layout, tag, seed),
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ablation_layout_dir = LAYOUT_ROOT / "embedding_source_ablation"
    ablation_layout_dir.mkdir(parents=True, exist_ok=True)

    node_list, views, feature_names = build_feature_views()
    for method_key, features in views.items():
        selected = max_min_select(features, node_list, budget=25)
        path = ablation_layout_dir / f"monitor_nodes_{method_key}_N25.json"
        payload = {
            "monitor_nodes": selected,
            "strategy": method_key,
            "n": 25,
            "selection_rule": "max_min_diversity_on_l2_normalized_features",
            "feature_names": feature_names[method_key],
            "formal_protocol": {
                "teacher_subdir": "ie420_plus_normal20_v1",
                "diagnosis_model": "hydraulic_inverse_deepattn",
                "split_mode": "scenario",
                "feature_set": "raw_plus_residual",
                "lambda_loc": 0.5,
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(OUT_DIR / f"{method_key}_node_features.npy", features)

    rows: list[dict[str, object]] = []
    for method_key, label in (
        ("topology_diversity", "Topology-Diversity"),
        ("coordinate_diversity", "Coordinate-Diversity"),
    ):
        layout = ablation_layout_dir / f"monitor_nodes_{method_key}_N25.json"
        tag = f"ch5_embedding_source_{method_key}_N25_normal20_rawres_loc0p5_s42"
        add_row(rows, "embedding_source_ablation", method_key, label, 25, 42, layout, tag)

    n5_methods = (
        (
            "degree",
            "Degree",
            LAYOUT_ROOT / "degree/monitor_nodes_degree_N5.json",
            "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N5_s{seed}",
        ),
        (
            "two_stage_v1",
            "Two-stage v1",
            LAYOUT_ROOT
            / "two_stage_balanced_layout_v1/monitor_nodes_two_stage_balanced_layout_v1_N5.json",
            "ch5_fixed_two_stage_balanced_layout_v1_N5_normal20_rawres_loc0p5_s{seed}",
        ),
        (
            "embedding_guided",
            "Embedding-Guided",
            LAYOUT_ROOT / "embedding_guided_clean_fixed/monitor_nodes_embedding_guided_clean_N5.json",
            "ch5_fixed_embedding_guided_clean_new_N5_normal20_rawres_loc0p5_s{seed}",
        ),
    )
    for method_key, label, layout, tag_template in n5_methods:
        if not layout.exists():
            raise FileNotFoundError(layout)
        for seed in (7, 123):
            add_row(
                rows,
                "low_budget_confirmation",
                method_key,
                label,
                5,
                seed,
                layout,
                tag_template.format(seed=seed),
            )

    manifest = OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "experiments": {
            "embedding_source_ablation": {
                "new_runs": 2,
                "formal_embedding_guided_result_reused": True,
                "diagnosis_seed": 42,
            },
            "low_budget_confirmation": {
                "new_runs": 6,
                "methods": ["Degree", "Two-stage v1", "Embedding-Guided"],
                "diagnosis_seeds": [7, 123],
            },
        },
        "total_new_runs": len(rows),
        "manifest": str(manifest),
    }
    (OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_PROTOCOL.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] wrote {manifest}")
    print(f"rows={len(rows)} pending={sum(row['status'] == 'pending' for row in rows)}")


if __name__ == "__main__":
    main()
