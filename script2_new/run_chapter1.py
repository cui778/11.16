# -*- coding: utf-8 -*-
"""
第一章实验一键脚本（在已安装 pyswmm 等依赖的环境下运行）。

用法（在 script2_new 目录下）:
  python run_chapter1.py

或分步（跳过已完成的步骤）:
  python run_chapter1.py --skip-extract   # 若已有 time_gated 等 parquet
  python run_chapter1.py --only-train     # 只跑训练（需已有各策略的 parquet）

执行顺序：
  1. 生成监测集 full / random（若尚未生成）
  2. 为 degree/betweenness/downstream/random/full 各策略：提取时序 -> 残差
  3. Step 0 & A1：degree 三种子 (42, 7, 123)
  4. A2/A3/A5/A0：betweenness / downstream / random / full 各三种子
  5. A4：observability（先 full-nodes 提取+残差 -> build observability -> 再提取 25 节点 -> 残差 -> 三种子）
  6. 汇总写入 outputs/reports/CHAPTER1_RESULTS.json 并追加 STEP_RESULTS_LOG.md
"""

# Warning: this script is retained for replaying the old Chapter 1 strategy
# experiments. It is not the clean main-line experiment entry after the
# monitor-only leakage fix.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
os.chdir(SCRIPT_ROOT)
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

SEEDS = [42, 7, 123]

def run(cmd: list[str], timeout: int | None = 3600) -> bool:
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(cmd, cwd=SCRIPT_ROOT, timeout=timeout, env=env)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[WARN] 超时: {' '.join(cmd)}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def build_strategy_config(cfg, monitor_n: int):
    return [
        ("degree", cfg.get_monitor_nodes_file("degree"), suffixed_subdir("time_gated", monitor_n)),
        ("betweenness", cfg.get_monitor_nodes_file("betweenness"), suffixed_subdir("time_gated_betweenness", monitor_n)),
        ("downstream", cfg.get_monitor_nodes_file("downstream"), suffixed_subdir("time_gated_downstream", monitor_n)),
        ("random", cfg.get_monitor_nodes_file("random"), suffixed_subdir("time_gated_random", monitor_n)),
        ("full", cfg.get_monitor_nodes_file("full"), suffixed_subdir("time_gated_full", monitor_n)),
    ]


def suffixed_subdir(name: str, monitor_n: int) -> str:
    return f"{name}_N{monitor_n}"


def suffixed_report(name: str, monitor_n: int) -> Path:
    return Path("outputs/reports") / f"{name}_N{monitor_n}.json"


def suffixed_log(name: str, monitor_n: int) -> Path:
    return Path("outputs/reports") / f"{name}_N{monitor_n}.md"


def parse_csv_args(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true", help="跳过提取与残差，仅用已有数据训练")
    ap.add_argument("--only-train", action="store_true", help="仅训练（需已有各 subdir 的 parquet）")
    ap.add_argument("--force", action="store_true", help="强制重新训练，忽略已有 CHAPTER1_RESULTS.json")
    ap.add_argument("--max-strategies", type=int, default=999, help="最多跑前 N 个策略（调试用）")
    ap.add_argument("--monitor-n", type=int, default=None, help="监测节点数，默认读取 config.monitor_node_count")
    ap.add_argument("--strategies", type=str, default="", help="只跑指定策略，逗号分隔，如 degree,observability")
    ap.add_argument("--seeds", type=str, default="", help="只跑指定随机种子，逗号分隔，如 42 或 42,7,123")
    args = ap.parse_args()
    print("[WARN] run_chapter1.py 属于旧实验编排入口，当前 clean 主线请优先参考 clean 总结文档与独立训练命令。")

    from config import Config
    cfg = Config()
    if args.monitor_n is not None:
        cfg.monitor_node_count = int(args.monitor_n)
    monitor_n = cfg.monitor_node_count
    selected_strategies = set(parse_csv_args(args.strategies))
    seeds = [int(x) for x in parse_csv_args(args.seeds)] if args.seeds else list(SEEDS)

    strategy_config = build_strategy_config(cfg, monitor_n)
    obs_monitor_file = cfg.get_monitor_nodes_file("observability")
    if selected_strategies:
        strategy_config = [item for item in strategy_config if item[0] in selected_strategies]

    results = {}  # strategy -> { seed -> metrics, mean_mrr, std_mrr, ... }

    # ---------- 1. 监测集 ----------
    if not args.only_train:
        for strategy, monitor_file, _subdir in strategy_config:
            if Path(monitor_file).exists():
                continue
            n_value = cfg.full_monitor_node_count if strategy == "full" else cfg.monitor_node_count
            cmd = [sys.executable, "prep/build_monitors.py", "--strategy", strategy, "--n", str(n_value)]
            if strategy == "random":
                cmd.extend(["--seed", "42"])
            run(cmd)

    # ---------- 2. 提取 + 残差 ----------
    if not args.skip_extract and not args.only_train:
        for item in strategy_config[: args.max_strategies]:
            name, monitor_file, subdir = item[0], item[1], item[2]
            out_dir = f"training_data_new/{subdir}"
            parquet = Path(out_dir) / "node_timeseries_with_residuals.parquet"
            if parquet.exists():
                print(f"[SKIP] 已存在 {subdir}")
                continue
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            ok = run([sys.executable, "prep/extract_timeseries.py", "--time-gated",
                     "--output-dir", out_dir, "--monitor-nodes-file", monitor_file], timeout=600000)
            if not ok:
                print(f"[FAIL] extract {subdir}")
                continue
            ok = run([sys.executable, "prep/residual_features.py", "--input-dir", out_dir, "--output-dir", out_dir], timeout=3600)
            if not ok:
                print(f"[FAIL] residual {subdir}")
                continue
            print(f"[OK] {subdir} 提取+残差完成")

        # A4 observability：复用 time_gated_full 全节点数据 -> build observability -> N 节点提取+残差
        obs_full_parquet = Path(f"training_data_new/{suffixed_subdir('time_gated_full', monitor_n)}/node_timeseries_with_residuals.parquet")
        obs_dir = f"training_data_new/{suffixed_subdir('time_gated_observability', monitor_n)}"
        if not (Path(obs_dir) / "node_timeseries_with_residuals.parquet").exists():
            if not obs_full_parquet.exists():
                print("[FAIL] observability 需要 time_gated_full 的全节点数据，但不存在，跳过 A4")
            else:
                if not Path(obs_monitor_file).exists():
                    run([sys.executable, "prep/build_monitors.py", "--strategy", "observability", "--n", str(cfg.monitor_node_count),
                        "--timeseries_path", str(obs_full_parquet)], timeout=300)
                Path(obs_dir).mkdir(parents=True, exist_ok=True)
                run([sys.executable, "prep/extract_timeseries.py", "--time-gated", "--output-dir", obs_dir,
                    "--monitor-nodes-file", obs_monitor_file], timeout=600000)
                run([sys.executable, "prep/residual_features.py", "--input-dir", obs_dir, "--output-dir", obs_dir], timeout=3600)
        obs_subdir_name = suffixed_subdir("time_gated_observability", monitor_n)
        need_observability = (not selected_strategies) or ("observability" in selected_strategies)
        if need_observability and ("observability", obs_monitor_file, obs_subdir_name) not in strategy_config:
            strategy_config.append(("observability", obs_monitor_file, obs_subdir_name))

    # 训练列表：所有已有 parquet 的策略（含 observability 若存在）
    to_train = []
    for item in strategy_config[: args.max_strategies]:
        name, _, subdir = item[0], item[1], item[2]
        if (Path(f"training_data_new/{subdir}") / "node_timeseries_with_residuals.parquet").exists():
            to_train.append(item)
    obs_subdir = suffixed_subdir("time_gated_observability", monitor_n)
    if (Path(f"training_data_new/{obs_subdir}") / "node_timeseries_with_residuals.parquet").exists():
        if not any(x[0] == "observability" for x in to_train):
            to_train.append(("observability", obs_monitor_file, obs_subdir))

    # ---------- 3. 训练（各策略三种子）----------
    # 增量结果文件：每跑完一次就落盘，防止中途崩溃丢失
    incremental_path = suffixed_report("CHAPTER1_RESULTS", monitor_n)
    incremental_path.parent.mkdir(parents=True, exist_ok=True)
    if incremental_path.exists() and not args.force:
        with open(incremental_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    for item in to_train:
        name, _monitor_file, subdir = item[0], item[1], item[2]
        if name not in results:
            results[name] = {"seeds": {}, "mrrs": [], "top1s": []}
        for seed in seeds:
            if not args.force and str(seed) in results[name].get("seeds", {}):
                print(f"[SKIP] {name} seed={seed} 已有结果，跳过训练")
                continue
            run_name = f"{name}_N{monitor_n}_seed{seed}"
            ok = run([sys.executable, "train/train.py", "--seed", str(seed), "--subdir", subdir,
                      "--run-name", run_name], timeout=7200)
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
            results[name]["seeds"][str(seed)] = {"mrr": mrr, "top1": top1}
            results[name]["mrrs"].append(mrr)
            results[name]["top1s"].append(top1)

            # 每次训练完立刻落盘
            with open(incremental_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [SAVE] {name} seed={seed} -> MRR={mrr:.4f}, Top1={top1:.4f}")

        if results[name]["mrrs"]:
            import numpy as np
            results[name]["mean_mrr"] = float(np.mean(results[name]["mrrs"]))
            results[name]["std_mrr"] = float(np.std(results[name]["mrrs"]))
            results[name]["mean_top1"] = float(np.mean(results[name]["top1s"]))
            results[name]["std_top1"] = float(np.std(results[name]["top1s"]))
            with open(incremental_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] {name} 三种子完成: mean_mrr={results[name].get('mean_mrr', 'N/A')}")

    # ---------- 4. 写结果 ----------
    out_report = suffixed_report("CHAPTER1_RESULTS", monitor_n)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 结果已写: {out_report}")

    # 追加 STEP_RESULTS_LOG
    log_path = suffixed_log("STEP_RESULTS_LOG", monitor_n)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n\n## 第一章批量实验（run_chapter1.py）\n\n")
        f.write("| 策略 | mean_MRR | std_MRR | mean_Top1 | std_Top1 |\n")
        f.write("|------|--------|--------|----------|----------|\n")
        for name in list(results.keys()):
            r = results.get(name, {})
            mm = r.get("mean_mrr")
            sm = r.get("std_mrr")
            mt = r.get("mean_top1")
            st = r.get("std_top1")
            mm_s = f"{mm:.4f}" if mm is not None else "-"
            sm_s = f"{sm:.4f}" if sm is not None else "-"
            mt_s = f"{mt:.4f}" if mt is not None else "-"
            st_s = f"{st:.4f}" if st is not None else "-"
            f.write(f"| {name} | {mm_s} | {sm_s} | {mt_s} | {st_s} |\n")
    print(f"[OK] 已追加: {log_path}")


if __name__ == "__main__":
    main()
