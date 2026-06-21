#!/usr/bin/env python3
"""Run Chapter 4 mechanism ablations under the frozen formal protocol.

The seed-42 screening covers three questions:
1. DeepAttn depth: L1/L2/L3/L4.
2. Path prior: full/distance-only/content-only at L3.
3. Temporal encoder control: Hydraulic-Inverse-LSTM.

The content-only variant removes path-feature values from attention logits but
retains the reachability mask. It therefore tests path-value contribution
without changing the set of physically connected node pairs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
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
OUTPUT_DIR = SCRIPT2 / "chapter4_diagnosis_model/outputs/formal_method_ablation"
LOG_DIR = OUTPUT_DIR / "logs"

TEACHER_SUBDIR = "ie420_plus_normal20_v1"
FEATURE_SET = "raw_plus_residual"
LAMBDA_LOC = 0.5
FORMAL_EPOCHS = 25
SEEDS = (7, 42, 123)


@dataclass(frozen=True)
class Variant:
    experiment: str
    variant: str
    model: str
    layers: int
    path_mode: str


VARIANTS = (
    Variant("depth", "deepattn_L1", "hydraulic_inverse_deepattn", 1, "full"),
    Variant("depth", "deepattn_L2", "hydraulic_inverse_deepattn", 2, "full"),
    Variant("depth", "deepattn_L3", "hydraulic_inverse_deepattn", 3, "full"),
    Variant("depth", "deepattn_L4", "hydraulic_inverse_deepattn", 4, "full"),
    Variant("path_prior", "deepattn_L3_distance_only", "hydraulic_inverse_deepattn", 3, "distance_only"),
    Variant("path_prior", "deepattn_L3_content_only", "hydraulic_inverse_deepattn", 3, "none"),
    Variant("temporal_encoder", "hydraulic_inverse_LSTM", "hydraulic_inverse_lstm", 1, "full"),
)

CONFIRM_VARIANTS = tuple(
    variant
    for variant in VARIANTS
    if variant.variant != "deepattn_L3_distance_only"
)

METRICS = (
    "mrr",
    "top1",
    "top3",
    "active_f1",
    "normal_window_fpr",
    "scene_f1",
    "event_top1",
    "event_top3",
)


def tag_for(variant: Variant, seed: int, smoke: bool = False) -> str:
    prefix = "ch4_ablation_smoke1ep" if smoke else "ch4_ablation"
    return f"{prefix}_{variant.variant}_normal20_rawres_loc0p5_degree_N25_s{seed}"


def metrics_path(tag: str) -> Path:
    return REPORTS_DIR / f"last_run_metrics_{tag}.json"


def ensure_inputs() -> None:
    missing = [
        str(path)
        for path in (PYTHON, TRAIN_SCRIPT, LAYOUT_FILE)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def command_for(variant: Variant, seed: int, epochs: int, tag: str) -> list[str]:
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
        variant.model,
        "--num-spatial-layers",
        str(variant.layers),
        "--path-prior-mode",
        variant.path_mode,
        "--feature-set",
        FEATURE_SET,
        "--lambda-loc",
        str(LAMBDA_LOC),
        "--lambda-kd",
        "0",
        "--lambda-active-kd",
        "0",
        "--split-mode",
        "scenario",
        "--scene-score-mode",
        "topk_mean",
        "--scene-topk",
        "5",
        "--num-epochs",
        str(epochs),
        "--output-tag",
        tag,
    ]


def run_variants(variants: tuple[Variant, ...], seeds: tuple[int, ...], epochs: int, cooldown: int, smoke: bool) -> None:
    ensure_inputs()
    jobs = [(variant, seed) for variant in variants for seed in seeds]
    for index, (variant, seed) in enumerate(jobs, start=1):
        tag = tag_for(variant, seed, smoke=smoke)
        report = metrics_path(tag)
        if report.exists():
            print(f"[{index}/{len(jobs)}] SKIP {report}", flush=True)
            continue

        stdout_file = LOG_DIR / f"{tag}.stdout.log"
        stderr_file = LOG_DIR / f"{tag}.stderr.log"
        print(
            f"[{index}/{len(jobs)}] START {variant.variant} seed={seed}",
            flush=True,
        )
        started = time.time()
        env = os.environ.copy()
        env["SWMM_PARQUET_SINGLE_THREAD"] = "1"
        with stdout_file.open("w", encoding="utf-8") as stdout, stderr_file.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                command_for(variant, seed, epochs, tag),
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
        gc.collect()
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed: {variant.variant} seed={seed}; "
                f"returncode={result.returncode}; see {stderr_file}"
            )
        if not report.exists():
            raise FileNotFoundError(f"Metrics missing after successful run: {report}")
        print(
            f"[{index}/{len(jobs)}] DONE {variant.variant} "
            f"seed={seed} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        if index < len(jobs) and cooldown > 0:
            time.sleep(cooldown)


def metric_value(data: dict[str, Any], key: str) -> float:
    aliases = {
        "top1": ("topk_recall_1", "top1"),
        "top3": ("topk_recall_3", "top3"),
        "event_top1": ("event_level_top1",),
        "event_top3": ("event_level_top3",),
    }
    for candidate in aliases.get(key, (key,)):
        if data.get(candidate) is not None:
            return float(data[candidate])
    return float("nan")


def validate(data: dict[str, Any], variant: Variant, seed: int) -> list[str]:
    expected = {
        "teacher_subdir": TEACHER_SUBDIR,
        "student_subdir": TEACHER_SUBDIR,
        "student_model_type": variant.model,
        "feature_set": FEATURE_SET,
        "num_spatial_layers": variant.layers,
        "resolved_num_spatial_layers": variant.layers,
        "path_prior_mode": variant.path_mode,
    }
    errors = [
        f"{key}={data.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if data.get(key) != value
    ]
    for key, value in {
        "lambda_loc": LAMBDA_LOC,
        "lambda_kd": 0.0,
        "lambda_active_kd": 0.0,
    }.items():
        try:
            valid = abs(float(data.get(key)) - value) < 1e-9
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{key}={data.get(key)!r}, expected {value!r}")
    if int(data.get("seed", seed)) != seed:
        errors.append(f"seed={data.get('seed')!r}, expected {seed}")
    return errors


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize() -> int:
    ensure_inputs()
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        expected_seeds = (
            (42,) if variant.variant == "deepattn_L3_distance_only" else SEEDS
        )
        for seed in expected_seeds:
            tag = tag_for(variant, seed)
            report = metrics_path(tag)
            status: dict[str, Any] = {
                "experiment": variant.experiment,
                "variant": variant.variant,
                "seed": seed,
                "metrics_file": str(report),
                "status": "missing",
                "audit_errors": "",
            }
            if not report.exists():
                status_rows.append(status)
                continue
            data = json.loads(report.read_text(encoding="utf-8"))
            errors = validate(data, variant, seed)
            if errors:
                status["status"] = "invalid"
                status["audit_errors"] = " | ".join(errors)
                status_rows.append(status)
                continue
            status["status"] = "ok"
            status_rows.append(status)
            row: dict[str, Any] = {
                "experiment": variant.experiment,
                "variant": variant.variant,
                "model": variant.model,
                "layers": variant.layers,
                "path_prior_mode": variant.path_mode,
                "seed": seed,
                "metrics_file": str(report),
            }
            row.update({metric: metric_value(data, metric) for metric in METRICS})
            rows.append(row)

    status_fields = [
        "experiment",
        "variant",
        "seed",
        "metrics_file",
        "status",
        "audit_errors",
    ]
    by_seed_fields = [
        "experiment",
        "variant",
        "model",
        "layers",
        "path_prior_mode",
        "seed",
        "metrics_file",
        *METRICS,
    ]
    write_csv(OUTPUT_DIR / "CH4_METHOD_ABLATION_RUN_STATUS.csv", status_rows, status_fields)
    write_csv(OUTPUT_DIR / "CH4_METHOD_ABLATION_BY_SEED.csv", rows, by_seed_fields)

    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant.variant]
        if not subset:
            continue
        summary: dict[str, Any] = {
            "experiment": variant.experiment,
            "variant": variant.variant,
            "model": variant.model,
            "layers": variant.layers,
            "path_prior_mode": variant.path_mode,
            "n_seeds": len(subset),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary)

    summary_fields = [
        "experiment",
        "variant",
        "model",
        "layers",
        "path_prior_mode",
        "n_seeds",
    ]
    for metric in METRICS:
        summary_fields.extend((f"{metric}_mean", f"{metric}_std"))
    write_csv(
        OUTPUT_DIR / "CH4_METHOD_ABLATION_SUMMARY.csv",
        summary_rows,
        summary_fields,
    )

    ok_count = sum(row["status"] == "ok" for row in status_rows)
    invalid_count = sum(row["status"] == "invalid" for row in status_rows)
    audit = [
        "# Chapter 4 Method Ablation Audit",
        "",
        "## Frozen protocol",
        "",
        f"- dataset: `{TEACHER_SUBDIR}`",
        f"- layout: `{LAYOUT_FILE}`",
        f"- features: `{FEATURE_SET}`",
        f"- lambda_loc: `{LAMBDA_LOC}`",
        "- split: `scenario`",
        "- KD: disabled",
        f"- epochs: `{FORMAL_EPOCHS}`",
        "",
        "## Mechanism controls",
        "",
        "- depth: DeepAttn L1/L2/L3/L4 with full path prior",
        "- path prior: full/distance_only/content_only at DeepAttn L3",
        "- temporal encoder: Hydraulic-Inverse-LSTM with full path prior",
        "- content_only retains the reachability mask but removes all path-feature values from attention logits",
        "",
        "## Status",
        "",
        f"- complete: `{ok_count == len(status_rows) and invalid_count == 0}`",
        f"- valid: `{ok_count}/{len(status_rows)}`",
        f"- invalid: `{invalid_count}`",
        f"- missing: `{len(status_rows) - ok_count - invalid_count}`",
        "",
        "Depth, content-only path ablation and the LSTM encoder control use",
        "seeds 7/42/123. Distance-only is retained as a seed-42 mechanism probe.",
    ]
    (OUTPUT_DIR / "CH4_METHOD_ABLATION_AUDIT.md").write_text(
        "\n".join(audit), encoding="utf-8"
    )
    return 1 if invalid_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "seed42", "confirm", "all-seeds", "summarize"],
        required=True,
    )
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        run_variants(VARIANTS, (42,), 1, args.cooldown_seconds, smoke=True)
        return 0
    if args.mode == "seed42":
        run_variants(VARIANTS, (42,), FORMAL_EPOCHS, args.cooldown_seconds, smoke=False)
        return summarize()
    if args.mode == "confirm":
        run_variants(
            CONFIRM_VARIANTS,
            (7, 123),
            FORMAL_EPOCHS,
            args.cooldown_seconds,
            smoke=False,
        )
        return summarize()
    if args.mode == "all-seeds":
        run_variants(VARIANTS, SEEDS, FORMAL_EPOCHS, args.cooldown_seconds, smoke=False)
        return summarize()
    return summarize()


if __name__ == "__main__":
    sys.exit(main())
