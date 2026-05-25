# -*- coding: utf-8 -*-
"""
第一章 B 组实验：数据层面对比（验证设计选择的合理性）

用法（在 script2_new 目录下）:
  D:\conda3\envs\swmm_gpu\python.exe run_chapter1_B.py              # 跑 B1 + B3
  D:\conda3\envs\swmm_gpu\python.exe run_chapter1_B.py --only-b1    # 只跑 B1
  D:\conda3\envs\swmm_gpu\python.exe run_chapter1_B.py --only-b3    # 只跑 B3
  D:\conda3\envs\swmm_gpu\python.exe run_chapter1_B.py --force      # 强制重跑

B1：场景数 100 vs 300
  - 固定 observability 策略
  - 生成 defect_matrix_100.csv → 提取 → 残差 → 用该矩阵训练 3 种子
  - 300 场景对照从 A 组 observability 读取

B3：残差特征 vs 原始观测值
  - 固定 observability 策略、300 场景
  - raw：用 node_timeseries.parquet + 仅原始特征列（train --raw-features）
  - residual：从 A 组 observability 读取（已有）

输出：outputs/reports/CHAPTER1_B_RESULTS.json
"""

from __future__ import annotations

# Warning: this script belongs to the legacy Chapter 1 B-series pipeline and
# is kept for historical replay only. It is not aligned with the clean
# monitor-only main line by default.

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
import os
os.chdir(SCRIPT_ROOT)
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

SEEDS = [42, 7, 123]


def run(cmd: list, timeout: int | None = 3600) -> bool:
    try:
        r = subprocess.run(cmd, cwd=SCRIPT_ROOT, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[WARN] 超时: {' '.join(str(c) for c in cmd)}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def train_group(results: dict, report_path: Path, key: str, subdir: str,
                extra_args: list[str], seeds: list[int], force: bool, monitor_n: int):
    """训练一组实验（3 种子），实时落盘。"""
    if key not in results or force:
        results[key] = {"seeds": {}, "mrrs": [], "top1s": []}
    for seed in seeds:
        if (not force) and str(seed) in results[key].get("seeds", {}):
            print(f"[SKIP] {key} seed={seed} 已有结果")
            continue
        run_name = f"{key}_N{monitor_n}_seed{seed}"
        cmd = [sys.executable, "train/train.py",
               "--seed", str(seed), "--subdir", subdir,
               "--run-name", run_name] + extra_args
        ok = run(cmd, timeout=7200)
        if not ok:
            continue
        metrics_path = Path(f"outputs/reports/last_run_metrics_{run_name}.json")
        if not metrics_path.exists():
            metrics_path = Path("outputs/reports/last_run_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                last = json.load(f)
            mrr = float(last.get("mrr", 0.0))
            top1 = float(last.get("top1", 0.0))
        else:
            mrr, top1 = 0.0, 0.0
        results[key]["seeds"][str(seed)] = {"mrr": mrr, "top1": top1}
        results[key]["mrrs"].append(mrr)
        results[key]["top1s"].append(top1)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  [SAVE] {key} seed={seed} -> MRR={mrr:.4f}, Top1={top1:.4f}")

    if results[key]["mrrs"]:
        import numpy as np
        results[key]["mean_mrr"] = float(np.mean(results[key]["mrrs"]))
        results[key]["std_mrr"] = float(np.std(results[key]["mrrs"]))
        results[key]["mean_top1"] = float(np.mean(results[key]["top1s"]))
        results[key]["std_top1"] = float(np.std(results[key]["top1s"]))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] {key} 完成: mean_MRR={results[key].get('mean_mrr', 'N/A')}")


def ensure_observability_monitors(full_parquet: Path, monitor_n: int) -> bool:
    from config import Config
    cfg = Config()
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
    ap.add_argument("--only-b1", action="store_true", help="只跑 B1")
    ap.add_argument("--only-b3", action="store_true", help="只跑 B3")
    ap.add_argument("--force", action="store_true", help="强制重跑，忽略已有结果")
    ap.add_argument("--monitor-n", type=int, default=None, help="监测节点数，默认读取 config.monitor_node_count")
    args = ap.parse_args()

    from config import Config
    cfg = Config()
    if args.monitor_n is not None:
        cfg.monitor_node_count = int(args.monitor_n)
    monitor_n = cfg.monitor_node_count
    obs_monitor_file = cfg.get_monitor_nodes_file("observability")
    obs_full_parquet = Path(f"training_data_new/{suffixed_subdir('time_gated_full', monitor_n)}/node_timeseries_with_residuals.parquet")

    do_b1 = (not args.only_b3)
    do_b3 = (not args.only_b1)

    results = {}
    report_path = suffixed_report("CHAPTER1_B_RESULTS", monitor_n)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and not args.force:
        with open(report_path, encoding="utf-8") as f:
            results = json.load(f)

    # ========== B1：场景数 100 vs 300 ==========
    if do_b1:
        if obs_full_parquet.exists() and not ensure_observability_monitors(obs_full_parquet, monitor_n):
            raise RuntimeError("生成 observability 监测节点失败。")
        print("\n" + "=" * 60)
        print("B1：场景数 100 vs 300（固定 observability）")
        print("=" * 60)

        defect_100 = Path("input_1/defect_matrix_100.csv")
        subdir_100 = suffixed_subdir("time_gated_observability_100scen", monitor_n)
        parquet_100 = Path(f"training_data_new/{subdir_100}/node_timeseries_with_residuals.parquet")

        if not defect_100.exists():
            print("[B1] 生成 100 场景缺陷矩阵...")
            run([sys.executable, "prep/defect_matrix.py", "--num-scenarios", "100",
                 "--output-name", "defect_matrix_100.csv", "--time-mode", "diverse"], timeout=600)

        if not parquet_100.exists():
            Path(f"training_data_new/{subdir_100}").mkdir(parents=True, exist_ok=True)
            print(f"[B1] 提取 100 场景时序（observability {monitor_n} 节点）...")
            run([sys.executable, "prep/extract_timeseries.py", "--time-gated",
                 "--defect-csv", str(defect_100),
                 "--output-dir", f"training_data_new/{subdir_100}",
                 "--monitor-nodes-file", obs_monitor_file], timeout=600000)
            print("[B1] 残差特征...")
            run([sys.executable, "prep/residual_features.py",
                 "--input-dir", f"training_data_new/{subdir_100}",
                 "--output-dir", f"training_data_new/{subdir_100}"], timeout=3600)
        else:
            print(f"[B1] 已存在 {subdir_100}，跳过提取")

        train_group(results, report_path, "B1_100scen", subdir_100,
                    ["--defect-csv", str(defect_100.resolve())],
                    SEEDS, args.force, monitor_n)

        # 300 场景对照从 A 组读取
        a_path = suffixed_report("CHAPTER1_RESULTS", monitor_n)
        if a_path.exists():
            with open(a_path, encoding="utf-8") as f:
                a_results = json.load(f)
            obs = a_results.get("observability", {})
            if obs and obs.get("seeds"):
                results["B1_300scen"] = obs
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"[OK] B1 300 场景（A 组 observability）: mean_MRR={obs.get('mean_mrr', 'N/A')}")
            else:
                print("[WARN] A 组 observability 结果为空，B1_300scen 未写入")
        else:
            print("[WARN] CHAPTER1_RESULTS.json 不存在，B1_300scen 未写入")

    # ========== B3：残差特征 vs 原始观测值 ==========
    if do_b3:
        print("\n" + "=" * 60)
        print("B3：残差特征 vs 原始观测值（固定 observability 300 场景）")
        print("=" * 60)

        subdir_obs = suffixed_subdir("time_gated_observability", monitor_n)
        raw_parquet = Path(f"training_data_new/{subdir_obs}/node_timeseries.parquet")
        if not raw_parquet.exists():
            print(f"[WARN] 未找到原始时序 {raw_parquet}，B3 raw 无法运行。")
            print("       请确认 time_gated_observability 目录下有 node_timeseries.parquet")
        else:
            train_group(results, report_path, "B3_raw", subdir_obs,
                        ["--raw-features"],
                        SEEDS, args.force, monitor_n)

        # B3 residual 对照从 A 组 observability 读取
        a_path = suffixed_report("CHAPTER1_RESULTS", monitor_n)
        if a_path.exists():
            with open(a_path, encoding="utf-8") as f:
                a_results = json.load(f)
            obs = a_results.get("observability", {})
            if obs and obs.get("seeds"):
                results["B3_residual"] = obs
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"[OK] B3 residual（A 组 observability）: mean_MRR={obs.get('mean_mrr', 'N/A')}")

    # ========== 汇总输出 ==========
    print("\n" + "=" * 60)
    print("B 组结果汇总")
    print("=" * 60)
    all_keys = ["B1_100scen", "B1_300scen", "B3_raw", "B3_residual"]
    print("| 实验 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |")
    print("|------|---------|---------|-----------|----------|")
    for k in all_keys:
        r = results.get(k, {})
        if not r:
            continue
        mm = r.get("mean_mrr")
        sm = r.get("std_mrr")
        mt = r.get("mean_top1")
        st = r.get("std_top1")
        mm_s = f"{mm:.4f}" if mm is not None else "-"
        sm_s = f"{sm:.4f}" if sm is not None else "-"
        mt_s = f"{mt:.4f}" if mt is not None else "-"
        st_s = f"{st:.4f}" if st is not None else "-"
        print(f"| {k} | {mm_s} | {sm_s} | {mt_s} | {st_s} |")
    print(f"\n[OK] 结果已写: {report_path}")


if __name__ == "__main__":
    main()
