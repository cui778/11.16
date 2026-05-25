# -*- coding: utf-8 -*-
"""
第一章 B2：缺陷矩阵覆盖对比（固定监测策略 A4=observability）

B2 目标（来自 research_plan_full.md）：
  - 对比「C 全覆盖」 vs 「随机（部分节点未出现）」对定位性能的影响

实现方式（一次只改一个变量=覆盖程度）：
  - 固定：监测点=observability_N25、训练超参、候选集 C、特征（残差版）、时间门控
  - 仅改变：缺陷矩阵中实际缺陷节点覆盖范围

输出：
  - 缺陷矩阵：
      input_1/defect_matrix_b2_full_coverage.csv
      input_1/defect_matrix_b2_partial_k25.csv
  - 时序数据：
      training_data_new/time_gated_observability_b2_full/
      training_data_new/time_gated_observability_b2_partial/
  - 结果汇总：
      outputs/reports/CHAPTER1_B2_RESULTS.json

用法（用你指定解释器跑）：
  D:\conda3\envs\swmm_gpu\python.exe E:\11.16\script2_new\run_chapter1_B2.py
  D:\conda3\envs\swmm_gpu\python.exe E:\11.16\script2_new\run_chapter1_B2.py --force
"""

from __future__ import annotations

# Warning: this script belongs to the legacy Chapter 1 B2 pipeline and is kept
# for historical replay only. It is not aligned with the clean monitor-only
# main line by default.

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
import os
os.chdir(SCRIPT_ROOT)

SEEDS = [42, 7, 123]


def run(cmd: list[str], timeout: int | None = 3600) -> bool:
    try:
        r = subprocess.run(cmd, cwd=SCRIPT_ROOT, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[WARN] 超时: {' '.join(cmd)}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def load_candidate_nodes() -> list[str]:
    cand_path = Path("input_1/candidate_nodes_new.json")
    with open(cand_path, encoding="utf-8") as f:
        data = json.load(f)
    return [str(n) for n in data.get("candidate_nodes", [])]


def coverage_ok(defect_csv: Path, candidates: set[str]) -> tuple[bool, int]:
    import pandas as pd
    df = pd.read_csv(defect_csv)
    present = set(str(x) for x in df["node_id"].dropna().astype(str).tolist())
    missing = candidates - present
    return (len(missing) == 0, len(missing))


def ensure_observability_monitors(full_parquet: Path, monitor_n: int | None = None) -> bool:
    from config import Config
    cfg = Config()
    if monitor_n is not None:
        cfg.monitor_node_count = int(monitor_n)
    out = Path(cfg.get_monitor_nodes_file("observability"))
    if out.exists():
        return True
    return run([
        sys.executable, "prep/build_monitors.py",
        "--strategy", "observability", "--n", str(cfg.monitor_node_count),
        "--timeseries_path", str(full_parquet),
    ], timeout=600)


def suffixed_subdir(name: str, monitor_n: int) -> str:
    return f"{name}_N{monitor_n}"


def suffixed_report(name: str, monitor_n: int) -> Path:
    return Path("outputs/reports") / f"{name}_N{monitor_n}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重跑（重新训练并覆盖汇总JSON）")
    ap.add_argument("--num-scenarios", type=int, default=300, help="B2 的场景数（默认 300）")
    ap.add_argument("--partial-kdef", type=int, default=25, help="partial 覆盖：仅从 C 中抽 k_def 个节点（默认 25）")
    ap.add_argument("--monitor-n", type=int, default=None, help="监测节点数，默认读取 config.monitor_node_count")
    args = ap.parse_args()

    from config import Config
    cfg = Config()
    if args.monitor_n is not None:
        cfg.monitor_node_count = int(args.monitor_n)
    monitor_n = cfg.monitor_node_count
    obs_monitor_file = cfg.get_monitor_nodes_file("observability")

    out_report = suffixed_report("CHAPTER1_B2_RESULTS", monitor_n)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    if out_report.exists() and not args.force:
        with open(out_report, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    # observability 依赖 full 节点残差数据
    full_parquet = Path(f"training_data_new/{suffixed_subdir('time_gated_full', monitor_n)}/node_timeseries_with_residuals.parquet")
    if not full_parquet.exists():
        raise FileNotFoundError("缺少 time_gated_full 的全节点残差 parquet，请先完成 A 路线的 time_gated_full。")

    if not ensure_observability_monitors(full_parquet, monitor_n):
        raise RuntimeError("生成 observability 监测节点失败。")

    candidates = set(load_candidate_nodes())

    # ---------- 1) 生成两份缺陷矩阵 ----------
    full_csv = Path(f"input_1/defect_matrix_b2_full_coverage_N{monitor_n}.csv")
    partial_csv = Path(f"input_1/defect_matrix_b2_partial_k25_N{monitor_n}.csv")

    # full coverage：反复生成直到覆盖所有候选节点（最多 50 次尝试）
    if args.force or not full_csv.exists():
        for attempt in range(1, 51):
            seed = 1000 + attempt
            ok = run([
                sys.executable, "prep/defect_matrix.py",
                "--num-scenarios", str(args.num_scenarios),
                "--output-name", full_csv.name,
                "--time-mode", "diverse",
                "--seed", str(seed),
            ], timeout=1200)
            if not ok:
                continue
            ok2, n_missing = coverage_ok(full_csv, candidates)
            print(f"[B2-full] attempt={attempt}, seed={seed}, missing={n_missing}")
            if ok2:
                break
        ok2, n_missing = coverage_ok(full_csv, candidates)
        if not ok2:
            print("[WARN] 未能做到 100% 覆盖（可能因为 SWMM 失败场景导致缺口），仍继续跑。")

    # partial：通过 k_def 抽子集制造“部分节点未出现”
    if args.force or not partial_csv.exists():
        run([
            sys.executable, "prep/defect_matrix.py",
            "--num-scenarios", str(args.num_scenarios),
            "--output-name", partial_csv.name,
            "--time-mode", "diverse",
            "--k-def", str(args.partial_kdef),
            "--seed", "4242",
        ], timeout=1200)

    # ---------- 2) 提取 + 残差 ----------
    def extract_and_residual(defect_csv: Path, out_subdir: str) -> Path:
        out_dir = Path(f"training_data_new/{out_subdir}")
        parquet = out_dir / "node_timeseries_with_residuals.parquet"
        if parquet.exists() and not args.force:
            print(f"[SKIP] 已存在 {out_subdir}")
            return parquet
        out_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "prep/extract_timeseries.py",
            "--time-gated",
            "--defect-csv", str(defect_csv),
            "--output-dir", str(out_dir),
            "--monitor-nodes-file", obs_monitor_file,
        ], timeout=600000)
        run([
            sys.executable, "prep/residual_features.py",
            "--input-dir", str(out_dir),
            "--output-dir", str(out_dir),
        ], timeout=7200)
        return parquet

    sub_full = suffixed_subdir("time_gated_observability_b2_full", monitor_n)
    sub_partial = suffixed_subdir("time_gated_observability_b2_partial", monitor_n)
    extract_and_residual(full_csv, sub_full)
    extract_and_residual(partial_csv, sub_partial)

    # ---------- 3) 训练（两组各三种子） ----------
    def train_group(key: str, subdir: str, defect_csv: Path):
        if key not in results or args.force:
            results[key] = {"seeds": {}, "mrrs": [], "top1s": []}
        for seed in SEEDS:
            if (not args.force) and str(seed) in results[key].get("seeds", {}):
                print(f"[SKIP] {key} seed={seed} 已有结果")
                continue
            run_name = f"{key}_N{monitor_n}_seed{seed}"
            ok = run([
                sys.executable, "train/train.py",
                "--seed", str(seed),
                "--subdir", subdir,
                "--defect-csv", str(defect_csv),
                "--run-name", run_name,
            ], timeout=7200)
            if not ok:
                continue
            metrics_path = Path(f"outputs/reports/last_run_metrics_{run_name}.json")
            if not metrics_path.exists():
                metrics_path = Path("outputs/reports/last_run_metrics.json")
            with open(metrics_path, encoding="utf-8") as f:
                last = json.load(f)
            mrr = float(last.get("mrr", 0.0))
            top1 = float(last.get("top1", 0.0))
            results[key]["seeds"][str(seed)] = {"mrr": mrr, "top1": top1}
            results[key]["mrrs"].append(mrr)
            results[key]["top1s"].append(top1)
            with open(out_report, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [SAVE] {key} seed={seed} -> MRR={mrr:.4f}, Top1={top1:.4f}")

        if results[key]["mrrs"]:
            import numpy as np
            results[key]["mean_mrr"] = float(np.mean(results[key]["mrrs"]))
            results[key]["std_mrr"] = float(np.std(results[key]["mrrs"]))
            results[key]["mean_top1"] = float(np.mean(results[key]["top1s"]))
            results[key]["std_top1"] = float(np.std(results[key]["top1s"]))
            with open(out_report, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    train_group("B2_full_coverage", sub_full, full_csv)
    train_group("B2_partial_kdef", sub_partial, partial_csv)

    print(f"\n[OK] B2 完成，结果: {out_report}")


if __name__ == "__main__":
    main()

