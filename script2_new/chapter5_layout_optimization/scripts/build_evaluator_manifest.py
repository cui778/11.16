#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a formal run manifest for Chapter-5 layout experiments.

The defect matrix is fixed. Only monitor layout / observed nodes change.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter-5 evaluator manifest.")
    parser.add_argument("--budgets", default="5,10,15,20,25")
    parser.add_argument(
        "--strategies",
        default="random,degree,betweenness,downstream,candidate_observability,identifiability_driven",
    )
    parser.add_argument("--seeds", default="42,7,123")
    parser.add_argument("--split-modes", default="scenario")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[2]
    chapter5_root = script_root / "chapter5_layout_optimization"
    outputs_dir = chapter5_root / "outputs"
    manifest_csv = chapter5_root / "plans" / "CH5_FORMAL_RUN_MANIFEST.csv"
    manifest_md = chapter5_root / "plans" / "CH5_FORMAL_RUN_MANIFEST.md"

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    split_modes = [x.strip() for x in args.split_modes.split(",") if x.strip()]

    rows = []
    md_lines = [
        "# 第5章正式实验运行清单",
        "",
        "默认冻结条件：",
        "",
        "- 缺陷矩阵固定",
        "- 候选节点固定",
        "- full-graph sparse-observation 固定",
        "- evaluator 固定为 `hydraulic_inverse_deepattn` 同协议训练入口",
        "",
    ]

    for split_mode in split_modes:
        md_lines.append(f"## split_mode = `{split_mode}`")
        md_lines.append("")
        for n in budgets:
            for strategy in strategies:
                layout_file = outputs_dir / "layouts" / strategy / f"monitor_nodes_{strategy}_N{n}.json"
                for seed in seeds:
                    tag = f"ch5_{strategy}_N{n}_{split_mode}_s{seed}"
                    command = (
                        "conda run -n swmm_gpu python "
                        f"\"{script_root / 'scripts' / 'train_privileged_teacher_student.py'}\" "
                        f"--student-monitors \"{layout_file}\" "
                        "--teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 "
                        "--student-model-type hydraulic_inverse_deepattn "
                        "--num-epochs 25 "
                        "--lambda-kd 0 "
                        "--lambda-active-kd 0 "
                        f"--split-mode {split_mode} "
                        f"--seed {seed} "
                        f"--output-tag {tag}"
                    )
                    rows.append(
                        {
                            "strategy": strategy,
                            "n": n,
                            "seed": seed,
                            "split_mode": split_mode,
                            "layout_file": str(layout_file),
                            "output_tag": tag,
                            "command": command,
                        }
                    )
                    md_lines.append(f"- `{tag}`")
                    md_lines.append(f"  - `{command}`")
        md_lines.append("")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "n", "seed", "split_mode", "layout_file", "output_tag", "command"],
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(str(manifest_csv))
    print(str(manifest_md))


if __name__ == "__main__":
    main()

