#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_ROOT / "outputs" / "reports"

DEFAULT_MODELS = [
    "time_mean_linear",
    "gru_only",
    "gru_gcn",
    "hydraulic_inverse",
    "lstm_graphsage_edge",
    "tcn_graphsage_edge",
]

DEFAULT_SPLITS = ["scenario", "node_holdout"]


def resolve_training_data_dir(subdir: str, explicit_dir: str | None) -> str:
    if explicit_dir:
        return explicit_dir

    local_dir = SCRIPT_ROOT / "training_data"
    if (local_dir / subdir).exists():
        return str(local_dir)

    legacy_dir = SCRIPT_ROOT.parent / "script2_new" / "training_data_new"
    if (legacy_dir / subdir).exists():
        return str(legacy_dir)

    return str(local_dir)


def run_one(model_type: str, split_mode: str, args) -> dict | None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_type}_{split_mode}_{args.feature_mode}"
    train_cmd = [
        sys.executable,
        "train/train.py",
        "--model-type",
        model_type,
        "--split-mode",
        split_mode,
        "--subdir",
        args.subdir,
        "--training-data-dir",
        resolve_training_data_dir(args.subdir, args.training_data_dir),
        "--feature-mode",
        args.feature_mode,
        "--lambda-type",
        str(args.lambda_type),
        "--run-name",
        run_name,
    ]

    if split_mode == "node_holdout":
        train_cmd.extend(["--n-holdout-nodes", str(args.n_holdout_nodes)])
    if args.candidate_file:
        train_cmd.extend(["--candidate-file", args.candidate_file])
    if args.monitor_file:
        train_cmd.extend(["--monitor-file", args.monitor_file])
    if args.defect_csv:
        train_cmd.extend(["--defect-csv", args.defect_csv])

    result = subprocess.run(train_cmd, cwd=SCRIPT_ROOT)
    if result.returncode != 0:
        return None

    metrics_path = REPORTS_DIR / f"last_run_metrics_{run_name}.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", type=str, default="time_gated_downstream")
    parser.add_argument("--training-data-dir", type=str, default=None)
    parser.add_argument("--feature-mode", type=str, default="raw_residual", choices=["raw_residual", "raw", "residual"])
    parser.add_argument("--lambda-type", type=float, default=0.2)
    parser.add_argument("--n-holdout-nodes", type=int, default=10)
    parser.add_argument("--candidate-file", type=str, default=None)
    parser.add_argument("--monitor-file", type=str, default=None)
    parser.add_argument("--defect-csv", type=str, default=None)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    args = parser.parse_args()

    summary = {}
    for split_mode in args.splits:
        summary[split_mode] = {}
        for model_type in args.models:
            metrics = run_one(model_type, split_mode, args)
            if metrics is None:
                summary[split_mode][model_type] = {"status": "failed"}
                continue
            summary[split_mode][model_type] = {
                "status": "ok",
                "mrr": metrics.get("mrr", 0.0),
                "top1": metrics.get("top1", 0.0),
                "top3": metrics.get("top3", 0.0),
                "top5": metrics.get("top5", 0.0),
                "by_type_top1": metrics.get("by_type_top1", {}),
            }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "CHAPTER1_MODEL_COMPARISON.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
