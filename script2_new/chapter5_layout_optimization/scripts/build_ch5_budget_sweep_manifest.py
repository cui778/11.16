#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the formal Chapter-5 monitoring-budget experiment manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(r"E:\11.16\script2_new")
CH5_ROOT = ROOT / "chapter5_layout_optimization"
LAYOUT_ROOT = CH5_ROOT / "outputs" / "layouts"
OUT_DIR = CH5_ROOT / "outputs" / "budget_sweep_formal"
REPORTS = ROOT / "outputs" / "reports"

PYTHON = Path(r"D:\conda3\envs\swmm_gpu\python.exe")
TRAIN_SCRIPT = ROOT / "scripts" / "train_privileged_teacher_student.py"

METHODS = {
    "degree": {
        "label": "Degree",
        "layout": LAYOUT_ROOT / "degree" / "monitor_nodes_degree_N{budget}.json",
        "tag": "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N{budget}_s{seed}",
    },
    "betweenness": {
        "label": "Betweenness",
        "layout": LAYOUT_ROOT / "betweenness" / "monitor_nodes_betweenness_N{budget}.json",
        "tag": "48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N{budget}_s{seed}",
    },
    "candidate_observability": {
        "label": "Cand-Obs",
        "layout": (
            LAYOUT_ROOT
            / "candidate_observability"
            / "monitor_nodes_candidate_observability_N{budget}.json"
        ),
        "tag": "ch5_fixed_candidate_observability_N{budget}_normal20_rawres_loc0p5_s{seed}",
    },
    "two_stage_v1": {
        "label": "Two-stage v1",
        "layout": (
            LAYOUT_ROOT
            / "two_stage_balanced_layout_v1"
            / "monitor_nodes_two_stage_balanced_layout_v1_N{budget}.json"
        ),
        "tag": "ch5_fixed_two_stage_balanced_layout_v1_N{budget}_normal20_rawres_loc0p5_s{seed}",
    },
    "embedding_guided": {
        "label": "Embedding-Guided",
        "layout": (
            LAYOUT_ROOT
            / "embedding_guided_clean_fixed"
            / "monitor_nodes_embedding_guided_clean_N{budget}.json"
        ),
        "tag": "ch5_fixed_embedding_guided_clean_new_N{budget}_normal20_rawres_loc0p5_s{seed}",
    },
}


def build_command(layout_file: Path, output_tag: str, seed: int) -> str:
    parts = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--student-monitors",
        str(layout_file),
        "--output-tag",
        output_tag,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal Ch5 budget-sweep manifest.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--methods",
        default="degree,betweenness,candidate_observability,two_stage_v1,embedding_guided",
    )
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    method_keys = [x.strip() for x in args.methods.split(",") if x.strip()]

    unknown = set(method_keys) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown method keys: {sorted(unknown)}")

    rows = []
    for method_key in method_keys:
        spec = METHODS[method_key]
        for budget in budgets:
            layout_file = Path(str(spec["layout"]).format(budget=budget))
            if not layout_file.exists():
                raise FileNotFoundError(
                    f"Missing {spec['label']} N={budget} layout: {layout_file}"
                )
            for seed in seeds:
                output_tag = str(spec["tag"]).format(budget=budget, seed=seed)
                metrics_file = REPORTS / f"last_run_metrics_{output_tag}.json"
                rows.append(
                    {
                        "method_key": method_key,
                        "method": spec["label"],
                        "budget": budget,
                        "diagnosis_seed": seed,
                        "teacher_subdir": "ie420_plus_normal20_v1",
                        "split_mode": "scenario",
                        "feature_set": "raw_plus_residual",
                        "lambda_loc": 0.5,
                        "student_model_type": "hydraulic_inverse_deepattn",
                        "layout_file": str(layout_file),
                        "output_tag": output_tag,
                        "metrics_file": str(metrics_file),
                        "status": "done" if metrics_file.exists() else "pending",
                        "command": build_command(layout_file, output_tag, seed),
                    }
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "s" + "-".join(str(seed) for seed in seeds)
    manifest_path = OUT_DIR / f"CH5_budget_sweep_manifest_{suffix}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    done = sum(row["status"] == "done" for row in rows)
    print(f"[OK] wrote {manifest_path}")
    print(f"rows={len(rows)} done={done} pending={len(rows) - done}")


if __name__ == "__main__":
    main()
