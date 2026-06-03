#!/usr/bin/env python3
"""Run and summarize the formal Chapter 4 model comparison experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("E:/11.16")
SCRIPT2 = ROOT / "script2_new"
PYTHON = Path("D:/conda3/envs/swmm_gpu/python.exe")
TRAIN_SCRIPT = SCRIPT2 / "scripts/train_privileged_teacher_student.py"
REPORTS_DIR = SCRIPT2 / "outputs/reports"
LAYOUT_FILE = (
    SCRIPT2
    / "chapter5_layout_optimization/outputs/layouts/degree/monitor_nodes_degree_N25.json"
)
OUTPUT_DIR = SCRIPT2 / "chapter4_diagnosis_model/outputs/formal_model_comparison"
THESIS_SOURCE_DIR = ROOT / "thesis_writing_repo/figures/ch4/source_data"
THESIS_SUMMARY_FILE = (
    THESIS_SOURCE_DIR / "CH4-F08_formal_model_comparison_multiseed_summary.csv"
)
LEGACY_SOURCE_FILE = THESIS_SOURCE_DIR / "CH4-F08_model_comparison_multiseed_summary.csv"
LEGACY_TARGET_FILE = (
    THESIS_SOURCE_DIR
    / "legacy_old_protocol/CH4-F08_model_comparison_multiseed_summary.csv"
)

BASELINE_MODELS = [
    "gru_only",
    "gru_gcn",
    "lstm_graphsage_edge",
    "hydraulic_inverse",
]
MAIN_MODEL = "hydraulic_inverse_deepattn"
FORMAL_MODELS = BASELINE_MODELS + [MAIN_MODEL]
SEEDS = [7, 42, 123]

TEACHER_SUBDIR = "ie420_plus_normal20_v1"
FEATURE_SET = "raw_plus_residual"
LAMBDA_LOC = 0.5
LAMBDA_KD = 0.0
LAMBDA_ACTIVE_KD = 0.0
SPLIT_MODE = "scenario"
SCENE_SCORE_MODE = "topk_mean"
SCENE_TOPK = 5
FORMAL_EPOCHS = 25

BY_SEED_FIELDS = [
    "model",
    "seed",
    "tag",
    "metrics_file",
    "mrr",
    "top1",
    "top3",
    "top5",
    "active_f1",
    "normal_window_fpr",
    "scene_f1",
    "event_top1",
    "event_top3",
    "event_top5",
]


def baseline_tag(model: str, seed: int, smoke: bool = False) -> str:
    prefix = "ch4_modelcmp_smoke1ep" if smoke else "ch4_modelcmp"
    return f"{prefix}_normal20_rawres_loc0p5_degree_N25_{model}_s{seed}"


def main_model_tag(seed: int) -> str:
    return f"48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s{seed}"


def audit_main_model_tag(seed: int) -> str:
    return (
        "ch4_modelcmp_audit_currentcode_normal20_rawres_loc0p5_degree_N25_"
        f"{MAIN_MODEL}_s{seed}"
    )


def metrics_path(tag: str) -> Path:
    return REPORTS_DIR / f"last_run_metrics_{tag}.json"


def expected_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for model in BASELINE_MODELS:
        for seed in SEEDS:
            tag = baseline_tag(model, seed)
            runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "tag": tag,
                    "metrics_file": metrics_path(tag),
                    "source": "new_formal_baseline",
                }
            )
    for seed in SEEDS:
        tag = main_model_tag(seed)
        runs.append(
            {
                "model": MAIN_MODEL,
                "seed": seed,
                "tag": tag,
                "metrics_file": metrics_path(tag),
                "source": "existing_formal_main_model",
            }
        )
    return runs


def ensure_inputs() -> None:
    required = [PYTHON, TRAIN_SCRIPT, LAYOUT_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def training_command(model: str, seed: int, epochs: int, tag: str) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--seed",
        str(seed),
        "--teacher-subdir",
        TEACHER_SUBDIR,
        "--student-monitors",
        str(LAYOUT_FILE),
        "--student-model-type",
        model,
        "--feature-set",
        FEATURE_SET,
        "--lambda-loc",
        str(LAMBDA_LOC),
        "--lambda-kd",
        str(LAMBDA_KD),
        "--lambda-active-kd",
        str(LAMBDA_ACTIVE_KD),
        "--split-mode",
        SPLIT_MODE,
        "--scene-score-mode",
        SCENE_SCORE_MODE,
        "--scene-topk",
        str(SCENE_TOPK),
        "--num-epochs",
        str(epochs),
        "--output-tag",
        tag,
    ]


def run_jobs(jobs: list[tuple[str, int, int, str]], cooldown_seconds: int) -> None:
    ensure_inputs()
    total = len(jobs)
    for index, (model, seed, epochs, tag) in enumerate(jobs, start=1):
        report = metrics_path(tag)
        if report.exists():
            print(f"[{index}/{total}] SKIP existing metrics: {report}", flush=True)
            continue

        print(
            f"[{index}/{total}] START model={model} seed={seed} epochs={epochs} tag={tag}",
            flush=True,
        )
        started = time.time()
        child_env = os.environ.copy()
        child_env["SWMM_PARQUET_SINGLE_THREAD"] = "1"
        result = subprocess.run(
            training_command(model, seed, epochs, tag),
            cwd=ROOT,
            env=child_env,
        )
        elapsed = time.time() - started
        gc.collect()

        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed: model={model}, seed={seed}, returncode={result.returncode}"
            )
        if not report.exists():
            raise FileNotFoundError(f"Training completed but metrics file is missing: {report}")

        print(
            f"[{index}/{total}] DONE model={model} seed={seed} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if index < total and cooldown_seconds > 0:
            print(f"Cooldown: {cooldown_seconds}s", flush=True)
            time.sleep(cooldown_seconds)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close_enough(actual: Any, expected: float) -> bool:
    try:
        return abs(float(actual) - expected) < 1e-9
    except (TypeError, ValueError):
        return False


def validate_metrics(data: dict[str, Any], run: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = {
        "teacher_subdir": TEACHER_SUBDIR,
        "student_subdir": TEACHER_SUBDIR,
        "feature_set": FEATURE_SET,
        "student_model_type": run["model"],
    }
    for key, expected in checks.items():
        if data.get(key) != expected:
            errors.append(f"{key}={data.get(key)!r}, expected {expected!r}")
    numeric_checks = {
        "lambda_loc": LAMBDA_LOC,
        "lambda_kd": LAMBDA_KD,
        "lambda_active_kd": LAMBDA_ACTIVE_KD,
    }
    for key, expected in numeric_checks.items():
        if not close_enough(data.get(key), expected):
            errors.append(f"{key}={data.get(key)!r}, expected {expected!r}")
    monitor_file = data.get("student_monitor_nodes_file")
    if monitor_file and Path(monitor_file).resolve() != LAYOUT_FILE.resolve():
        errors.append(f"student_monitor_nodes_file={monitor_file!r}")
    return errors


def metric_value(data: dict[str, Any], key: str, fallback: str | None = None) -> float:
    value = data.get(key)
    if value is None and fallback:
        value = data.get(fallback)
    return float(value) if value is not None else float("nan")


def by_seed_row(run: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": run["model"],
        "seed": run["seed"],
        "tag": run["tag"],
        "metrics_file": str(run["metrics_file"]),
        "mrr": metric_value(data, "mrr"),
        "top1": metric_value(data, "topk_recall_1"),
        "top3": metric_value(data, "topk_recall_3"),
        "top5": metric_value(data, "topk_recall_5"),
        "active_f1": metric_value(data, "active_f1"),
        "normal_window_fpr": metric_value(data, "normal_window_fpr"),
        "scene_f1": metric_value(data, "scene_f1"),
        "event_top1": metric_value(data, "event_level_top1"),
        "event_top3": metric_value(data, "event_level_top3"),
        "event_top5": metric_value(data, "event_level_top5"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "mrr",
        "top1",
        "top3",
        "top5",
        "active_f1",
        "normal_window_fpr",
        "scene_f1",
        "event_top1",
        "event_top3",
        "event_top5",
    ]
    for model in FORMAL_MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        record: dict[str, Any] = {"model": model, "n_seeds": len(model_rows)}
        for name in metric_names:
            values = [float(row[name]) for row in model_rows]
            record[f"{name}_mean"] = statistics.mean(values)
            record[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(record)
    return summary_rows


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sync_thesis_source(summary_file: Path) -> None:
    THESIS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary_file, THESIS_SUMMARY_FILE)
    if LEGACY_SOURCE_FILE.exists() and not LEGACY_TARGET_FILE.exists():
        if not is_within(LEGACY_SOURCE_FILE, THESIS_SOURCE_DIR):
            raise RuntimeError(f"Unsafe legacy source path: {LEGACY_SOURCE_FILE}")
        if not is_within(LEGACY_TARGET_FILE, THESIS_SOURCE_DIR):
            raise RuntimeError(f"Unsafe legacy target path: {LEGACY_TARGET_FILE}")
        LEGACY_TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(LEGACY_SOURCE_FILE), str(LEGACY_TARGET_FILE))


def summarize() -> int:
    ensure_inputs()
    run_status: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    missing: list[str] = []

    for run in expected_runs():
        report = run["metrics_file"]
        status_row = {
            "model": run["model"],
            "seed": run["seed"],
            "source": run["source"],
            "tag": run["tag"],
            "metrics_file": str(report),
            "status": "missing",
            "audit_errors": "",
        }
        if not report.exists():
            missing.append(str(report))
            run_status.append(status_row)
            continue

        data = read_json(report)
        errors = validate_metrics(data, run)
        if errors:
            status_row["status"] = "invalid"
            status_row["audit_errors"] = " | ".join(errors)
            invalid.append(f"{report}: {' | '.join(errors)}")
        else:
            status_row["status"] = "ok"
            rows.append(by_seed_row(run, data))
        run_status.append(status_row)

    write_csv(
        OUTPUT_DIR / "CH4_FORMAL_MODEL_COMPARISON_RUN_STATUS.csv",
        run_status,
        ["model", "seed", "source", "tag", "metrics_file", "status", "audit_errors"],
    )
    write_csv(OUTPUT_DIR / "CH4_FORMAL_MODEL_COMPARISON_BY_SEED.csv", rows, BY_SEED_FIELDS)

    summary_rows = aggregate_rows(rows)
    summary_fields = ["model", "n_seeds"]
    for metric in [
        "mrr",
        "top1",
        "top3",
        "top5",
        "active_f1",
        "normal_window_fpr",
        "scene_f1",
        "event_top1",
        "event_top3",
        "event_top5",
    ]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    summary_file = OUTPUT_DIR / "CH4_FORMAL_MODEL_COMPARISON_SUMMARY.csv"
    write_csv(summary_file, summary_rows, summary_fields)

    complete = not missing and not invalid and len(rows) == len(FORMAL_MODELS) * len(SEEDS)
    audit_lines = [
        "# Chapter 4 Formal Model Comparison Audit",
        "",
        "## Frozen Protocol",
        "",
        f"- dataset: `{TEACHER_SUBDIR}`",
        f"- layout: `{LAYOUT_FILE}`",
        f"- feature_set: `{FEATURE_SET}`",
        f"- lambda_loc: `{LAMBDA_LOC}`",
        f"- lambda_kd: `{LAMBDA_KD}`",
        f"- lambda_active_kd: `{LAMBDA_ACTIVE_KD}`",
        f"- split_mode: `{SPLIT_MODE}`",
        f"- epochs: `{FORMAL_EPOCHS}`",
        f"- seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        "",
        "## Audit Result",
        "",
        f"- complete: `{complete}`",
        f"- valid metrics files: `{len(rows)}/{len(FORMAL_MODELS) * len(SEEDS)}`",
        f"- missing files: `{len(missing)}`",
        f"- invalid files: `{len(invalid)}`",
        "",
    ]
    if missing:
        audit_lines.extend(["## Missing Files", ""])
        audit_lines.extend(f"- `{item}`" for item in missing)
        audit_lines.append("")
    if invalid:
        audit_lines.extend(["## Invalid Files", ""])
        audit_lines.extend(f"- `{item}`" for item in invalid)
        audit_lines.append("")
    audit_lines.extend(
        [
            "## Notes",
            "",
            "- Smoke metrics are excluded from the formal summary.",
            "- Existing deep-attention checkpoints are reused and are not overwritten.",
            "- The historical old-protocol comparison is archived only after a complete audit.",
            "",
        ]
    )
    (OUTPUT_DIR / "CH4_FORMAL_MODEL_COMPARISON_AUDIT.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )

    if complete:
        sync_thesis_source(summary_file)
        print(f"Formal summary synced: {THESIS_SUMMARY_FILE}", flush=True)
        return 0

    print("Formal summary is incomplete. See the audit report.", flush=True)
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "seed42", "remaining", "audit_main42", "summarize"],
        required=True,
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=60,
        help="Seconds to wait between serial training runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        jobs = [
            (model, 42, 1, baseline_tag(model, 42, smoke=True))
            for model in BASELINE_MODELS
        ]
        run_jobs(jobs, args.cooldown_seconds)
        return 0
    if args.mode == "seed42":
        jobs = [(model, 42, FORMAL_EPOCHS, baseline_tag(model, 42)) for model in BASELINE_MODELS]
        run_jobs(jobs, args.cooldown_seconds)
        return 0
    if args.mode == "remaining":
        jobs = [
            (model, seed, FORMAL_EPOCHS, baseline_tag(model, seed))
            for seed in [7, 123]
            for model in BASELINE_MODELS
        ]
        run_jobs(jobs, args.cooldown_seconds)
        return 0
    if args.mode == "audit_main42":
        run_jobs(
            [(MAIN_MODEL, 42, FORMAL_EPOCHS, audit_main_model_tag(42))],
            args.cooldown_seconds,
        )
        return 0
    return summarize()


if __name__ == "__main__":
    sys.exit(main())
