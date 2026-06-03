#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Chapter-5 core budget-performance retraining manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(r"E:\11.16\script2_new")
CH5 = ROOT / "chapter5_layout_optimization"
OUTPUTS = CH5 / "outputs"
PLANS = CH5 / "plans"

METHODS = [
    {
        "method_key": "degree",
        "method_short": "Degree",
        "method": "degree",
        "layout_template": OUTPUTS / "layouts" / "degree" / "monitor_nodes_degree_N{budget}.json",
    },
    {
        "method_key": "candidate_observability",
        "method_short": "Cand-Obs",
        "method": "candidate_observability",
        "layout_template": OUTPUTS
        / "layouts"
        / "candidate_observability"
        / "monitor_nodes_candidate_observability_N{budget}.json",
    },
    {
        "method_key": "two_stage_balanced_layout_v1",
        "method_short": "Two-stage v1",
        "method": "two_stage_balanced_layout_v1",
        "layout_template": OUTPUTS
        / "layouts"
        / "two_stage_balanced_layout_v1"
        / "monitor_nodes_two_stage_balanced_layout_v1_N{budget}.json",
    },
    {
        "method_key": "v0_2_clean_scenario",
        "method_short": "v0_2 clean",
        "method": "learnable_layout_network_v0_2_clean_scenario",
        "layout_template": OUTPUTS
        / "layout_stability"
        / "layout_files"
        / "v0_2_clean_scenario"
        / "N{budget}"
        / "layout_seed_42"
        / "monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N{budget}_layoutseed42.json",
    },
    {
        "method_key": "v2_2_clean_generalization",
        "method_short": "v2_2 clean",
        "method": "learnable_layout_network_v2_2_clean_generalization",
        "layout_template": OUTPUTS
        / "layout_stability"
        / "layout_files"
        / "v2_2_clean_generalization"
        / "N{budget}"
        / "layout_seed_42"
        / "monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N{budget}_layoutseed42.json",
    },
]


def command_for(layout_file: Path, output_tag: str, seed: int) -> str:
    return (
        "conda run -n swmm_gpu python "
        f"\"{ROOT / 'scripts' / 'train_privileged_teacher_student.py'}\" "
        f"--student-monitors \"{layout_file}\" "
        "--teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 "
        "--student-model-type hydraulic_inverse_deepattn "
        "--num-epochs 25 "
        "--lambda-kd 0 "
        "--lambda-active-kd 0 "
        "--split-mode scenario "
        f"--seed {seed} "
        f"--output-tag {output_tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build core Chapter-5 budget retraining manifest.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument("--diagnosis-seeds", default="7,42,123")
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    diagnosis_seeds = [int(x.strip()) for x in args.diagnosis_seeds.split(",") if x.strip()]

    rows = []
    for budget in budgets:
        for spec in METHODS:
            layout_file = Path(str(spec["layout_template"]).format(budget=budget))
            if not layout_file.exists():
                raise FileNotFoundError(f"Missing layout for {spec['method_key']} N={budget}: {layout_file}")
            for seed in diagnosis_seeds:
                output_tag = f"ch5_core_{spec['method_key']}_N{budget}_scenario_s{seed}"
                rows.append(
                    {
                        "method_key": spec["method_key"],
                        "method_short": spec["method_short"],
                        "method": spec["method"],
                        "budget": budget,
                        "diagnosis_seed": seed,
                        "split_mode": "scenario",
                        "layout_file": str(layout_file),
                        "output_tag": output_tag,
                        "metrics_file": str(ROOT / "outputs" / "reports" / f"last_run_metrics_{output_tag}.json"),
                        "command": command_for(layout_file, output_tag, seed),
                    }
                )

    if len(rows) != len(METHODS) * len(budgets) * len(diagnosis_seeds):
        raise AssertionError(f"Unexpected manifest row count: {len(rows)}")
    if len({row["output_tag"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate output_tag in manifest")

    PLANS.mkdir(parents=True, exist_ok=True)
    csv_path = PLANS / "CH5_CORE_BUDGET_PERFORMANCE_MANIFEST.csv"
    md_path = PLANS / "CH5_CORE_BUDGET_PERFORMANCE_MANIFEST.md"
    fieldnames = [
        "method_key",
        "method_short",
        "method",
        "budget",
        "diagnosis_seed",
        "split_mode",
        "layout_file",
        "output_tag",
        "metrics_file",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# CH5 core budget performance manifest",
        "",
        "- Methods: Degree / Cand-Obs / Two-stage v1 / v0_2 clean / v2_2 clean",
        "- Budgets: " + ", ".join(str(x) for x in budgets),
        "- Diagnosis seeds: " + ", ".join(str(x) for x in diagnosis_seeds),
        "- Split mode: scenario",
        "- Row count: " + str(len(rows)),
        "",
    ]
    for row in rows:
        md_lines.append(f"## {row['output_tag']}")
        md_lines.append("")
        md_lines.append(f"- layout: `{row['layout_file']}`")
        md_lines.append(f"- metrics: `{row['metrics_file']}`")
        md_lines.append("")
        md_lines.append("```powershell")
        md_lines.append(row["command"])
        md_lines.append("```")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(csv_path)
    print(md_path)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
