#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a new IE-only curriculum matrix with configurable profiles.

Profiles:
- lowpct: realistic low-intensity curriculum (5% / 10% / 15%)
- balanced: moderate curriculum (15% / 25% / 35%)
- mainrange: keep the original mainline intensity scale (40% / 50% / 60%)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


LOWPCT_I_TEMPLATES = [
    {"difficulty": "easy", "intensity_pct": 15, "start_hour": 6, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 15, "start_hour": 18, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 15, "start_hour": 10, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 6, "duration_h": 24},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 14, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 10, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 15, "start_hour": 22, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 5, "start_hour": 18, "duration_h": 24},
    {"difficulty": "hard", "intensity_pct": 5, "start_hour": 10, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 5, "start_hour": 14, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 10, "start_hour": 18, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 5, "start_hour": 22, "duration_h": 6},
]

LOWPCT_E_TEMPLATES = [
    {"difficulty": "easy", "intensity_pct": 15, "start_hour": 4, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 15, "start_hour": 16, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 15, "start_hour": 8, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 0, "duration_h": 24},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 4, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 12, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 10, "start_hour": 8, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 15, "start_hour": 20, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 5, "start_hour": 0, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 5, "start_hour": 12, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 5, "start_hour": 20, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 10, "start_hour": 16, "duration_h": 6},
]

BALANCED_I_TEMPLATES = [
    {"difficulty": "easy", "intensity_pct": 35, "start_hour": 6, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 35, "start_hour": 18, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 35, "start_hour": 10, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 25, "start_hour": 6, "duration_h": 24},
    {"difficulty": "medium", "intensity_pct": 25, "start_hour": 14, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 25, "start_hour": 10, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 20, "start_hour": 18, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 20, "start_hour": 22, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 15, "start_hour": 10, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 15, "start_hour": 14, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 20, "start_hour": 18, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 15, "start_hour": 22, "duration_h": 6},
]

BALANCED_E_TEMPLATES = [
    {"difficulty": "easy", "intensity_pct": 35, "start_hour": 4, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 35, "start_hour": 16, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 35, "start_hour": 8, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 25, "start_hour": 0, "duration_h": 24},
    {"difficulty": "medium", "intensity_pct": 25, "start_hour": 4, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 25, "start_hour": 12, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 20, "start_hour": 8, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 20, "start_hour": 20, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 15, "start_hour": 0, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 15, "start_hour": 12, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 15, "start_hour": 20, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 20, "start_hour": 16, "duration_h": 6},
]

MAINRANGE_I_TEMPLATES = [
    {"difficulty": "easy", "intensity_pct": 60, "start_hour": 6, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 60, "start_hour": 18, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 60, "start_hour": 10, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 6, "duration_h": 24},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 14, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 10, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 60, "start_hour": 22, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 40, "start_hour": 18, "duration_h": 24},
    {"difficulty": "hard", "intensity_pct": 40, "start_hour": 10, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 40, "start_hour": 14, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 50, "start_hour": 18, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 40, "start_hour": 22, "duration_h": 6},
]

MAINRANGE_E_TEMPLATES = [
    {"difficulty": "easy", "intensity_pct": 60, "start_hour": 4, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 60, "start_hour": 16, "duration_h": 24},
    {"difficulty": "easy", "intensity_pct": 60, "start_hour": 8, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 0, "duration_h": 24},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 4, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 12, "duration_h": 18},
    {"difficulty": "medium", "intensity_pct": 50, "start_hour": 8, "duration_h": 12},
    {"difficulty": "medium", "intensity_pct": 60, "start_hour": 20, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 40, "start_hour": 0, "duration_h": 12},
    {"difficulty": "hard", "intensity_pct": 40, "start_hour": 12, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 40, "start_hour": 20, "duration_h": 6},
    {"difficulty": "hard", "intensity_pct": 50, "start_hour": 16, "duration_h": 6},
]


PROFILE_MAP: Dict[str, Dict[str, List[dict]]] = {
    "lowpct": {"I": LOWPCT_I_TEMPLATES, "E": LOWPCT_E_TEMPLATES},
    "balanced": {"I": BALANCED_I_TEMPLATES, "E": BALANCED_E_TEMPLATES},
    "mainrange": {"I": MAINRANGE_I_TEMPLATES, "E": MAINRANGE_E_TEMPLATES},
}


def _sample_quality(rng: np.random.Generator) -> tuple[float, float, float]:
    return (
        round(float(rng.uniform(120.0, 260.0)), 2),
        round(float(rng.uniform(10.0, 18.0)), 2),
        round(float(rng.uniform(2.0, 7.0)), 2),
    )


def _build_rows_for_node(
    node_df: pd.DataFrame,
    templates: List[dict],
    rng: np.random.Generator,
) -> List[dict]:
    node_id = str(node_df["node_id"].iloc[0])
    defect_type = str(node_df["defect_type"].iloc[0])
    baseline_flow = float(node_df["baseline_flow"].iloc[0])
    flow_sign = -1.0 if defect_type == "E" else 1.0
    flow_factor = 0.01

    rows: List[dict] = []
    for tpl in templates:
        bodf, nh4, do = _sample_quality(rng)
        flow = flow_sign * baseline_flow * flow_factor * float(tpl["intensity_pct"])
        rows.append(
            {
                "defect_type": defect_type,
                "node_id": node_id,
                "link_id": "",
                "intensity_pct": int(tpl["intensity_pct"]),
                "start_hour": int(tpl["start_hour"]),
                "duration_h": int(tpl["duration_h"]),
                "flow": round(float(flow), 4),
                "BODf": bodf,
                "NH4": nh4,
                "DO": do,
                "baseline_flow": round(float(baseline_flow), 4),
                "curriculum_difficulty": str(tpl["difficulty"]),
            }
        )
    return rows


def build_curriculum_matrix(input_csv: Path, output_csv: Path, profile: str, seed: int) -> pd.DataFrame:
    if profile not in PROFILE_MAP:
        raise ValueError(f"Unsupported profile: {profile}")

    df = pd.read_csv(input_csv)
    if set(df["defect_type"].unique()) - {"I", "E"}:
        raise ValueError("Curriculum v3 builder expects an IE-only source matrix.")

    rows: List[dict] = []
    next_defect_id = 1
    templates_map = PROFILE_MAP[profile]

    for defect_type in ("I", "E"):
        type_df = df[df["defect_type"] == defect_type].copy()
        node_ids = sorted(type_df["node_id"].unique().tolist())
        for node_pos, node_id in enumerate(node_ids):
            node_df = type_df[type_df["node_id"] == node_id].copy()
            node_rng = np.random.default_rng(seed + (31 * node_pos) + (101 if defect_type == "E" else 0))
            node_rows = _build_rows_for_node(node_df=node_df, templates=templates_map[defect_type], rng=node_rng)
            for row in node_rows:
                row["defect_id"] = next_defect_id
                next_defect_id += 1
            rows.extend(node_rows)

    out_df = pd.DataFrame(rows)
    cols = [
        "defect_id",
        "defect_type",
        "node_id",
        "link_id",
        "intensity_pct",
        "start_hour",
        "duration_h",
        "flow",
        "BODf",
        "NH4",
        "DO",
        "baseline_flow",
        "curriculum_difficulty",
    ]
    out_df = out_df[cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out_df


def summarize(df: pd.DataFrame) -> dict:
    difficulty_counts = {
        f"{dtype}:{difficulty}": int(count)
        for (dtype, difficulty), count in df.groupby(["defect_type", "curriculum_difficulty"]).size().to_dict().items()
    }
    return {
        "n_rows": int(len(df)),
        "type_counts": df["defect_type"].value_counts().to_dict(),
        "type_unique_nodes": df.groupby("defect_type")["node_id"].nunique().to_dict(),
        "scenes_per_node": (
            df.groupby(["defect_type", "node_id"]).size().groupby(level=0).describe().round(3).to_dict()
        ),
        "intensity_stats": df.groupby("defect_type")["intensity_pct"].describe().round(3).to_dict(),
        "difficulty_counts": difficulty_counts,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    input_dir = repo_root / "input_1"

    parser = argparse.ArgumentParser(description="Build curriculum-style IE defect matrix v3.")
    parser.add_argument(
        "--input-csv",
        default=str(input_dir / "defect_matrix_diverse_ie_v2e_dense.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(input_dir / "defect_matrix_diverse_ie_v3a_curriculum_lowpct.csv"),
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_MAP.keys()), default="lowpct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    out_df = build_curriculum_matrix(input_csv=input_csv, output_csv=output_csv, profile=args.profile, seed=args.seed)

    manifest = {
        "seed": int(args.seed),
        "profile": args.profile,
        "source_matrix": str(input_csv),
        "output_matrix": str(output_csv),
        "summary": summarize(out_df),
        "templates": PROFILE_MAP[args.profile],
    }
    manifest_path = output_csv.with_name(f"{output_csv.stem}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
