#!/usr/bin/env python3
"""Collect targeted Chapter 5 supplement results into auditable tables."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("E:/11.16")
SCRIPT2 = ROOT / "script2_new"
OUT_DIR = SCRIPT2 / "chapter5_layout_optimization/outputs/targeted_supplements"
MANIFEST = OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_MANIFEST.csv"
REPORTS = SCRIPT2 / "outputs/reports"

METRICS = (
    "mrr",
    "top1",
    "top3",
    "top5",
    "event_top1",
    "event_top3",
    "normal_window_fpr",
    "scene_f1",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric(data: dict, name: str) -> float:
    aliases = {
        "top1": ("topk_recall_1", "top1"),
        "top3": ("topk_recall_3", "top3"),
        "top5": ("topk_recall_5", "top5"),
        "event_top1": ("event_level_top1",),
        "event_top3": ("event_level_top3",),
    }
    for key in aliases.get(name, (name,)):
        if data.get(key) is not None:
            return float(data[key])
    return float("nan")


def append_existing_controls(rows: list[dict[str, str]]) -> None:
    controls = (
        {
            "experiment": "embedding_source_ablation",
            "method_key": "embedding_guided",
            "method": "Embedding-Guided",
            "budget": "25",
            "diagnosis_seed": "42",
            "metrics_file": str(
                REPORTS
                / "last_run_metrics_ch5_fixed_embedding_guided_clean_new_N25_normal20_rawres_loc0p5_s42.json"
            ),
        },
        {
            "experiment": "low_budget_confirmation",
            "method_key": "degree",
            "method": "Degree",
            "budget": "5",
            "diagnosis_seed": "42",
            "metrics_file": str(
                REPORTS
                / "last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N5_s42.json"
            ),
        },
        {
            "experiment": "low_budget_confirmation",
            "method_key": "two_stage_v1",
            "method": "Two-stage v1",
            "budget": "5",
            "diagnosis_seed": "42",
            "metrics_file": str(
                REPORTS
                / "last_run_metrics_ch5_fixed_two_stage_balanced_layout_v1_N5_normal20_rawres_loc0p5_s42.json"
            ),
        },
        {
            "experiment": "low_budget_confirmation",
            "method_key": "embedding_guided",
            "method": "Embedding-Guided",
            "budget": "5",
            "diagnosis_seed": "42",
            "metrics_file": str(
                REPORTS
                / "last_run_metrics_ch5_fixed_embedding_guided_clean_new_N5_normal20_rawres_loc0p5_s42.json"
            ),
        },
    )
    for row in controls:
        rows.append({**row, "source": "existing_formal_result"})


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    manifest_rows = read_rows(MANIFEST)
    rows: list[dict] = []
    for item in manifest_rows:
        rows.append({**item, "source": "new_targeted_run"})
    append_existing_controls(rows)

    result_rows: list[dict] = []
    status_rows: list[dict] = []
    for row in rows:
        path = Path(row["metrics_file"])
        status = {
            "experiment": row["experiment"],
            "method": row["method"],
            "budget": row["budget"],
            "diagnosis_seed": row["diagnosis_seed"],
            "metrics_file": str(path),
            "source": row["source"],
            "status": "missing",
        }
        if not path.exists():
            status_rows.append(status)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        status["status"] = "ok"
        status_rows.append(status)
        result = {
            "experiment": row["experiment"],
            "method_key": row["method_key"],
            "method": row["method"],
            "budget": int(row["budget"]),
            "diagnosis_seed": int(row["diagnosis_seed"]),
            "source": row["source"],
            "metrics_file": str(path),
        }
        result.update({name: metric(data, name) for name in METRICS})
        result_rows.append(result)

    fields = [
        "experiment",
        "method_key",
        "method",
        "budget",
        "diagnosis_seed",
        "source",
        "metrics_file",
        *METRICS,
    ]
    write_csv(OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_BY_RUN.csv", result_rows, fields)
    write_csv(
        OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_STATUS.csv",
        status_rows,
        [
            "experiment",
            "method",
            "budget",
            "diagnosis_seed",
            "source",
            "metrics_file",
            "status",
        ],
    )

    summary_rows: list[dict] = []
    groups = sorted(
        {
            (row["experiment"], row["method_key"], row["method"], row["budget"])
            for row in result_rows
        }
    )
    for experiment, method_key, method_name, budget in groups:
        subset = [
            row
            for row in result_rows
            if (
                row["experiment"],
                row["method_key"],
                row["method"],
                row["budget"],
            )
            == (experiment, method_key, method_name, budget)
        ]
        summary = {
            "experiment": experiment,
            "method_key": method_key,
            "method": method_name,
            "budget": budget,
            "n_runs": len(subset),
        }
        for name in METRICS:
            values = [float(row[name]) for row in subset]
            summary[f"{name}_mean"] = statistics.mean(values)
            summary[f"{name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary)

    summary_fields = [
        "experiment",
        "method_key",
        "method",
        "budget",
        "n_runs",
    ]
    for name in METRICS:
        summary_fields.extend((f"{name}_mean", f"{name}_std"))
    write_csv(
        OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_SUMMARY.csv",
        summary_rows,
        summary_fields,
    )
    ok = sum(row["status"] == "ok" for row in status_rows)
    print(f"[OK] collected {ok}/{len(status_rows)} expected result files")
    return 0 if ok == len(status_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
