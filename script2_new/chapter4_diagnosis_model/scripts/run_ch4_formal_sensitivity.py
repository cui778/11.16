#!/usr/bin/env python3
"""Run and summarize the formal Chapter 4 sensitivity experiments."""

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
TIMELINE_SCRIPT = SCRIPT2 / "scripts/evaluate_scene_timeline_diagnosis.py"
REPORTS_DIR = SCRIPT2 / "outputs/reports"
CHECKPOINTS_DIR = SCRIPT2 / "outputs/model_checkpoints"
LAYOUT_FILE = (
    SCRIPT2
    / "chapter5_layout_optimization/outputs/layouts/degree/monitor_nodes_degree_N25.json"
)
WINDOW_DIR = SCRIPT2 / "chapter4_diagnosis_model/outputs/formal_window_length"
IE_DIR = SCRIPT2 / "chapter4_diagnosis_model/outputs/formal_ie_group"
THESIS_SOURCE_DIR = ROOT / "thesis_writing_repo/figures/ch4/source_data"
THESIS_WINDOW = THESIS_SOURCE_DIR / "CH4-F09b_formal_time_window_length_eval_summary.csv"
THESIS_IE = THESIS_SOURCE_DIR / "CH4-F10a_formal_ie_type_group_multiseed_summary.csv"
LEGACY_DIR = THESIS_SOURCE_DIR / "legacy_old_protocol"

TEACHER_SUBDIR = "ie420_plus_normal20_v1"
MODEL = "hydraulic_inverse_deepattn"
FEATURE_SET = "raw_plus_residual"
LAMBDA_LOC = 0.5
WINDOW_STRIDE = 6
SEED = 42
FORMAL_EPOCHS = 25
SCENE_SCORE_MODE = "topk_mean"
SCENE_TOPK = 5
WINDOWS = [(2, 12), (3, 18), (4, 24), (6, 36)]
IE_SEEDS = [7, 42, 123]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main_tag(seed: int) -> str:
    return f"48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s{seed}"


def window_tag(hours: int, smoke: bool = False) -> str:
    prefix = "ch4_window_smoke1ep" if smoke else "ch4_window"
    return f"{prefix}_normal20_rawres_loc0p5_degree_N25_{hours}h_s42"


def checkpoint(tag: str) -> Path:
    return CHECKPOINTS_DIR / f"best_model_{tag}.pth"


def metrics(tag: str) -> Path:
    return REPORTS_DIR / f"last_run_metrics_{tag}.json"


def ensure_inputs() -> None:
    missing = [
        str(path)
        for path in [PYTHON, TRAIN_SCRIPT, TIMELINE_SCRIPT, LAYOUT_FILE]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    IE_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def train_command(hours: int, length: int, epochs: int, tag: str) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--seed",
        str(SEED),
        "--teacher-subdir",
        TEACHER_SUBDIR,
        "--student-monitors",
        str(LAYOUT_FILE),
        "--student-model-type",
        MODEL,
        "--feature-set",
        FEATURE_SET,
        "--lambda-loc",
        str(LAMBDA_LOC),
        "--lambda-kd",
        "0.0",
        "--lambda-active-kd",
        "0.0",
        "--split-mode",
        "scenario",
        "--scene-score-mode",
        SCENE_SCORE_MODE,
        "--scene-topk",
        str(SCENE_TOPK),
        "--sequence-length",
        str(length),
        "--window-stride",
        str(WINDOW_STRIDE),
        "--num-epochs",
        str(epochs),
        "--output-tag",
        tag,
    ]


def run_train(
    hours: int,
    length: int,
    epochs: int,
    smoke: bool = False,
    max_attempts: int = 3,
    retry_cooldown_seconds: int = 120,
) -> None:
    tag = window_tag(hours, smoke=smoke)
    report = metrics(tag)
    if report.exists():
        print(f"SKIP existing metrics: {report}", flush=True)
        return
    child_env = os.environ.copy()
    child_env["SWMM_PARQUET_SINGLE_THREAD"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    result = None
    for attempt in range(1, max_attempts + 1):
        print(
            f"START train window={hours}h length={length} epochs={epochs} "
            f"attempt={attempt}/{max_attempts}",
            flush=True,
        )
        result = subprocess.run(train_command(hours, length, epochs, tag), cwd=ROOT, env=child_env)
        gc.collect()
        if result.returncode == 0:
            break
        if attempt < max_attempts:
            print(
                f"RETRY train window={hours}h returncode={result.returncode} "
                f"cooldown={retry_cooldown_seconds}s",
                flush=True,
            )
            time.sleep(retry_cooldown_seconds)
    if result is None or result.returncode != 0:
        code = None if result is None else result.returncode
        raise RuntimeError(f"Training failed: window={hours}h returncode={code}")
    if not report.exists() or not checkpoint(tag).exists():
        raise FileNotFoundError(f"Missing training artifacts for {tag}")
    print(f"DONE train window={hours}h", flush=True)


def formal_checkpoint(hours: int) -> Path:
    return checkpoint(main_tag(SEED) if hours == 6 else window_tag(hours))


def formal_training_metrics(hours: int) -> Path:
    return metrics(main_tag(SEED) if hours == 6 else window_tag(hours))


def timeline_prefix(hours: int, seed: int, group: str) -> str:
    return f"ch4_formal_{group}_{hours}h_s{seed}"


def run_timeline(hours: int, length: int, seed: int, group: str, ckpt: Path) -> None:
    out_dir = WINDOW_DIR if group == "window" else IE_DIR
    prefix = timeline_prefix(hours, seed, group)
    out_metrics = out_dir / f"{prefix}_metrics.json"
    out_events = out_dir / f"{prefix}_event_predictions.csv"
    if out_metrics.exists():
        if group != "ie" or (
            out_events.exists()
            and "true_active_true_node_rank" in out_events.read_text(encoding="utf-8-sig").splitlines()[0]
        ):
            print(f"SKIP existing timeline metrics: {out_metrics}", flush=True)
            return
        print(f"REFRESH timeline artifacts with true-active event ranks: {prefix}", flush=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    cmd = [
        str(PYTHON),
        str(TIMELINE_SCRIPT),
        "--subdir",
        TEACHER_SUBDIR,
        "--checkpoint",
        str(ckpt),
        "--monitor-nodes",
        str(LAYOUT_FILE),
        "--model-type",
        MODEL,
        "--feature-set",
        FEATURE_SET,
        "--sequence-length",
        str(length),
        "--window-stride",
        str(WINDOW_STRIDE),
        "--seed",
        str(seed),
        "--split-mode",
        "scenario",
        "--threshold",
        "0.5",
        "--min-consecutive",
        "2",
        "--output-dir",
        str(out_dir),
        "--output-prefix",
        prefix,
    ]
    child_env = os.environ.copy()
    child_env["SWMM_PARQUET_SINGLE_THREAD"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    print(f"START timeline group={group} window={hours}h seed={seed}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=child_env)
    gc.collect()
    if result.returncode != 0:
        raise RuntimeError(
            f"Timeline evaluation failed: group={group} window={hours}h seed={seed}"
        )
    if not out_metrics.exists():
        raise FileNotFoundError(f"Missing timeline metrics: {out_metrics}")
    print(f"DONE timeline group={group} window={hours}h seed={seed}", flush=True)


def archive_legacy(source: Path) -> None:
    if not source.exists():
        return
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    target = LEGACY_DIR / source.name
    if not target.exists():
        shutil.copy2(source, target)


def summarize_window() -> None:
    rows = []
    for hours, length in WINDOWS:
        train_data = read_json(formal_training_metrics(hours))
        timeline = read_json(
            WINDOW_DIR / f"{timeline_prefix(hours, SEED, 'window')}_metrics.json"
        )
        rows.append(
            {
                "window_hours": hours,
                "sequence_length": length,
                "window_stride": WINDOW_STRIDE,
                "stride_hours": timeline["stride_hours"],
                "active_f1": train_data["active_f1"],
                "normal_window_fpr": train_data["normal_window_fpr"],
                "window_mrr": train_data["mrr"],
                "window_top1": train_data["topk_recall_1"],
                "window_top3": train_data["topk_recall_3"],
                "window_top5": train_data["topk_recall_5"],
                "scene_recall": timeline["scene_defect_recall"],
                "onset_error_hours_mean": timeline["onset_error_hours_mean"],
                "onset_accuracy_within_1h": timeline["onset_accuracy_within_1_stride"],
                "onset_accuracy_within_2h": timeline["onset_accuracy_within_2_strides"],
                "onset_accuracy_within_3h": timeline["onset_accuracy_within_3_strides"],
                "active_interval_iou_mean": timeline["active_interval_iou_mean"],
                "duration_error_hours_mean": timeline["duration_error_hours_mean"],
                "scene_node_mrr": timeline["scene_node_mrr"],
                "scene_node_top1": timeline["scene_node_top1"],
                "scene_node_top3": timeline["scene_node_top3"],
                "scene_node_top5": timeline["scene_node_top5"],
                "training_metrics_file": str(formal_training_metrics(hours)),
                "timeline_metrics_file": str(
                    WINDOW_DIR / f"{timeline_prefix(hours, SEED, 'window')}_metrics.json"
                ),
            }
        )
    fields = list(rows[0].keys())
    write_csv(WINDOW_DIR / "CH4_FORMAL_WINDOW_LENGTH_BY_RUN.csv", rows, fields)
    write_csv(WINDOW_DIR / "CH4_FORMAL_WINDOW_LENGTH_SUMMARY.csv", rows, fields)
    archive_legacy(THESIS_SOURCE_DIR / "CH4-F09b_time_window_length_eval_summary.csv")
    write_csv(THESIS_WINDOW, rows, fields)
    audit = [
        "# Chapter 4 Formal Window-Length Sensitivity Audit",
        "",
        "- dataset: `ie420_plus_normal20_v1`",
        "- model: `hydraulic_inverse_deepattn`",
        "- layout: `degree_N25`",
        "- feature set: `raw_plus_residual`",
        "- lambda_loc: `0.5`",
        "- split: `scenario`",
        "- seed: `42`",
        "- stride: `6` samples = `1 h`",
        "- varied field only: `sequence_length`",
        "",
        "The 6 h row reuses the existing formal Chapter 4 main-model checkpoint.",
        "Window-level detection and localization metrics are the formal sensitivity evidence.",
        "Predicted-segment timeline fields are exploratory audit fields only: the simple first-consecutive-active-segment postprocessor is sensitive to early alarms and is not used as headline evidence for temporal boundary recovery.",
    ]
    (WINDOW_DIR / "CH4_FORMAL_WINDOW_LENGTH_AUDIT.md").write_text(
        "\n".join(audit) + "\n", encoding="utf-8"
    )


def event_group_rows(seed: int) -> list[dict[str, Any]]:
    event_path = IE_DIR / f"{timeline_prefix(6, seed, 'ie')}_event_predictions.csv"
    import pandas as pd

    df = pd.read_csv(event_path)
    df = df[df["defect_type"].isin(["I", "E"])].copy()
    df["rank"] = pd.to_numeric(df["true_active_true_node_rank"], errors="coerce")
    df["integrated_rank"] = pd.to_numeric(df["pred_true_node_rank"], errors="coerce")
    rows = []
    for defect_type in ["I", "E"]:
        group = df[df["defect_type"] == defect_type]
        ranks = group["rank"]
        integrated_ranks = group["integrated_rank"]
        rows.append(
            {
                "seed": seed,
                "defect_type": defect_type,
                "event_n": int(len(group)),
                "event_mrr": float((1.0 / ranks.dropna()).sum() / max(len(group), 1)),
                "event_top1": float((ranks <= 1).sum() / max(len(group), 1)),
                "event_top3": float((ranks <= 3).sum() / max(len(group), 1)),
                "event_top5": float((ranks <= 5).sum() / max(len(group), 1)),
                "integrated_scene_mrr": float((1.0 / integrated_ranks.dropna()).sum() / max(len(group), 1)),
                "integrated_scene_top1": float((integrated_ranks <= 1).sum() / max(len(group), 1)),
                "integrated_scene_top3": float((integrated_ranks <= 3).sum() / max(len(group), 1)),
                "integrated_scene_top5": float((integrated_ranks <= 5).sum() / max(len(group), 1)),
            }
        )
    return rows


def summarize_ie() -> None:
    event_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in IE_SEEDS:
        for row in event_group_rows(seed):
            event_lookup[(seed, row["defect_type"])] = row

    rows = []
    for seed in IE_SEEDS:
        data = read_json(metrics(main_tag(seed)))
        for defect_type in ["I", "E"]:
            window = data["by_defect_type"][defect_type]
            event = event_lookup[(seed, defect_type)]
            rows.append(
                {
                    "seed": seed,
                    "defect_type": defect_type,
                    "window_n": window["n"],
                    "window_mrr": window["mrr"],
                    "window_top1": window["top1"],
                    "window_top3": window["top3"],
                    "window_top5": window["top5"],
                    **{key: value for key, value in event.items() if key not in {"seed", "defect_type"}},
                    "metrics_file": str(metrics(main_tag(seed))),
                }
            )
    by_seed_fields = list(rows[0].keys())
    write_csv(IE_DIR / "CH4_FORMAL_IE_GROUP_BY_SEED.csv", rows, by_seed_fields)

    summary = []
    metric_names = [
        "window_mrr",
        "window_top1",
        "window_top3",
        "window_top5",
        "event_mrr",
        "event_top1",
        "event_top3",
        "event_top5",
        "integrated_scene_mrr",
        "integrated_scene_top1",
        "integrated_scene_top3",
        "integrated_scene_top5",
    ]
    for defect_type in ["I", "E"]:
        group = [row for row in rows if row["defect_type"] == defect_type]
        record: dict[str, Any] = {
            "defect_type": defect_type,
            "n_seeds": len(group),
            "window_n_mean": mean([float(row["window_n"]) for row in group]),
            "event_n_mean": mean([float(row["event_n"]) for row in group]),
        }
        for name in metric_names:
            values = [float(row[name]) for row in group]
            record[f"{name}_mean"] = mean(values)
            record[f"{name}_std"] = stdev(values)
        summary.append(record)
    summary_fields = list(summary[0].keys())
    write_csv(IE_DIR / "CH4_FORMAL_IE_GROUP_SUMMARY.csv", summary, summary_fields)
    archive_legacy(THESIS_SOURCE_DIR / "CH4-F10a_ie_type_group_summary.csv")
    write_csv(THESIS_IE, summary, summary_fields)
    audit = [
        "# Chapter 4 Formal I/E Grouped Localization Audit",
        "",
        "- dataset: `ie420_plus_normal20_v1`",
        "- model: `hydraulic_inverse_deepattn`",
        "- layout: `degree_N25`",
        "- feature set: `raw_plus_residual`",
        "- lambda_loc: `0.5`",
        "- split: `scenario`",
        "- seeds: `7, 42, 123`",
        "",
        "I/E grouping is a post-hoc stratification of the same localization task.",
        "It is not a defect-type classification task and Type Accuracy is not reported.",
        "Formal event-level localization metrics aggregate node scores over true active windows.",
        "Integrated-scene metrics use predicted active segments and remain diagnostic audit fields because they additionally depend on the temporal postprocessor.",
    ]
    (IE_DIR / "CH4_FORMAL_IE_GROUP_AUDIT.md").write_text(
        "\n".join(audit) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "smoke",
            "train_window",
            "eval_window",
            "eval_ie",
            "summarize_window",
            "summarize_ie",
            "summarize",
            "all",
        ],
    )
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    args = parser.parse_args()
    ensure_inputs()

    if args.mode == "smoke":
        run_train(2, 12, 1, smoke=True)
        return
    if args.mode in {"train_window", "all"}:
        for index, (hours, length) in enumerate(WINDOWS[:3]):
            run_train(hours, length, FORMAL_EPOCHS)
            if index < len(WINDOWS[:3]) - 1 and args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)
    if args.mode in {"eval_window", "all"}:
        for hours, length in WINDOWS:
            run_timeline(hours, length, SEED, "window", formal_checkpoint(hours))
    if args.mode in {"eval_ie", "all"}:
        for seed in IE_SEEDS:
            run_timeline(6, 36, seed, "ie", checkpoint(main_tag(seed)))
    if args.mode in {"summarize_window", "summarize", "all"}:
        summarize_window()
    if args.mode in {"summarize_ie", "summarize", "all"}:
        summarize_ie()


if __name__ == "__main__":
    main()
