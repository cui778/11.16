from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\11.16\script2_new\chapter5_layout_optimization")
STRUCT_DIR = ROOT / "outputs" / "structural_innovation"
SURROGATE_DIR = STRUCT_DIR / "surrogate"
DATASET_PATH = STRUCT_DIR / "layout_quality_dataset_wide.csv"

TARGET_COLUMNS = [
    "scenario_mrr",
    "scenario_top1",
    "node_holdout_mrr",
    "node_holdout_top1",
]

DROP_FEATURE_COLUMNS = {
    "method",
    "budget",
    "layout_file",
    "representative_pool_file",
    "metrics_file",
    "weight_preset",
}


def ridge_fit_predict_loo(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    n_samples = x.shape[0]
    preds = np.zeros(n_samples, dtype=float)
    coef_sum = np.zeros(x.shape[1] + 1, dtype=float)

    for holdout in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[holdout] = False
        x_train = x[mask]
        y_train = y[mask]

        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0

        x_train_std = (x_train - mean) / std
        x_test_std = (x[holdout : holdout + 1] - mean) / std

        design = np.concatenate([np.ones((x_train_std.shape[0], 1)), x_train_std], axis=1)
        lhs = design.T @ design + alpha * np.eye(design.shape[1])
        lhs[0, 0] -= alpha
        rhs = design.T @ y_train
        coef = np.linalg.solve(lhs, rhs)

        test_design = np.concatenate([np.ones((1, 1)), x_test_std], axis=1)
        preds[holdout] = float((test_design @ coef)[0])
        coef_sum += coef

    return preds, coef_sum / n_samples


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train layout surrogate model.")
    parser.add_argument("--dataset", default="", help="Path to quality dataset CSV (overrides default)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"缺少布局质量数据集：{dataset_path}")

    SURROGATE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    available_targets = [column for column in TARGET_COLUMNS if column in df.columns and df[column].notna().sum() >= 4]
    if not available_targets:
        raise RuntimeError("可用目标列不足，无法训练 surrogate。")

    feature_columns = [
        column
        for column in df.columns
        if column not in DROP_FEATURE_COLUMNS
        and column not in TARGET_COLUMNS
        and not column.startswith("scenario_")
        and not column.startswith("node_holdout_")
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    prediction_rows = []
    coefficient_rows = []
    report = {
        "dataset_path": str(DATASET_PATH),
        "sample_count": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "targets": {},
    }

    alpha = 1.0
    for target in available_targets:
        target_df = df[df[target].notna()].reset_index(drop=True)
        x = target_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y = target_df[target].to_numpy(dtype=float)

        preds, coef = ridge_fit_predict_loo(x, y, alpha=alpha)
        mae = float(np.mean(np.abs(y - preds)))
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        r2 = float(r2_score(y, preds))

        report["targets"][target] = {
            "sample_count": int(len(target_df)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "alpha": alpha,
        }

        for _, row, pred, truth in zip(target_df.index, target_df.itertuples(index=False), preds, y):
            prediction_rows.append(
                {
                    "target": target,
                    "method": getattr(row, "method"),
                    "budget": getattr(row, "budget"),
                    "truth": truth,
                    "loo_prediction": float(pred),
                    "abs_error": float(abs(truth - pred)),
                }
            )

        coefficient_rows.append({"target": target, "feature": "__intercept__", "coefficient": float(coef[0])})
        for feature, value in zip(feature_columns, coef[1:]):
            coefficient_rows.append({"target": target, "feature": feature, "coefficient": float(value)})

    predictions_df = pd.DataFrame(prediction_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    predictions_path = SURROGATE_DIR / "surrogate_predictions.csv"
    coefficients_path = SURROGATE_DIR / "surrogate_coefficients.csv"
    report_path = SURROGATE_DIR / "surrogate_training_report.json"

    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    coefficients_df.to_csv(coefficients_path, index=False, encoding="utf-8-sig")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] surrogate predictions -> {predictions_path}")
    print(f"[OK] surrogate coefficients -> {coefficients_path}")
    print(f"[OK] surrogate report -> {report_path}")
    print(f"[INFO] trained targets = {list(report['targets'].keys())}")


if __name__ == "__main__":
    main()
