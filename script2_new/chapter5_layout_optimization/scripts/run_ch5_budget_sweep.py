#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run formal Chapter-5 budget experiments sequentially from a manifest."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def memory_snapshot() -> dict[str, float]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "ram_total_gb": round(vm.total / 1024**3, 3),
            "ram_available_gb": round(vm.available / 1024**3, 3),
            "ram_percent": float(vm.percent),
            "swap_total_gb": round(swap.total / 1024**3, 3),
            "swap_used_gb": round(swap.used / 1024**3, 3),
        }
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal Ch5 budget sweep.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--methods", default="", help="Optional comma-separated method keys.")
    parser.add_argument("--budgets", default="", help="Optional comma-separated budgets.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed job. Default is to stop immediately.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = load_rows(manifest_path)
    log_dir = manifest_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_file = manifest_path.parent / "CH5_budget_sweep_runtime_status.jsonl"
    method_filter = {x.strip() for x in args.methods.split(",") if x.strip()}
    budget_filter = {int(x.strip()) for x in args.budgets.split(",") if x.strip()}

    selected = []
    for row in rows:
        if method_filter and row["method_key"] not in method_filter:
            continue
        if budget_filter and int(row["budget"]) not in budget_filter:
            continue
        if Path(row["metrics_file"]).exists():
            continue
        selected.append(row)

    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        print("[OK] no unfinished jobs selected")
        return

    print(f"[INFO] unfinished jobs selected: {len(selected)}")
    for index, row in enumerate(selected, start=1):
        command = row["command"]
        print(
            f"\n[{index}/{len(selected)}] {row['method']} "
            f"N={row['budget']} seed={row['diagnosis_seed']}"
        )
        print(command)
        if args.dry_run:
            continue

        started_at = datetime.now()
        before = memory_snapshot()
        log_path = log_dir / f"{row['output_tag']}.log"
        print(f"[LOG] {log_path}")
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n=== START {started_at.isoformat()} ===\n")
            log.write(f"command={command}\n")
            log.write(f"memory_before={json.dumps(before, ensure_ascii=False)}\n")
            log.flush()
            result = subprocess.run(
                shlex.split(command, posix=False),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            finished_at = datetime.now()
            after = memory_snapshot()
            log.write(f"returncode={result.returncode}\n")
            log.write(f"memory_after={json.dumps(after, ensure_ascii=False)}\n")
            log.write(f"=== END {finished_at.isoformat()} ===\n")

        runtime_record = {
            "output_tag": row["output_tag"],
            "method": row["method"],
            "budget": int(row["budget"]),
            "diagnosis_seed": int(row["diagnosis_seed"]),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "returncode": int(result.returncode),
            "metrics_exists": Path(row["metrics_file"]).exists(),
            "log_file": str(log_path),
            "memory_before": before,
            "memory_after": after,
        }
        with status_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(runtime_record, ensure_ascii=False) + "\n")

        if result.returncode != 0:
            print(f"[FAILED] {row['output_tag']} exit={result.returncode}; see {log_path}")
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        elif not Path(row["metrics_file"]).exists():
            print(f"[FAILED] process returned 0 but metrics file is missing: {row['metrics_file']}")
            if not args.continue_on_error:
                raise SystemExit(2)
        else:
            print(f"[DONE] {row['output_tag']}")
        gc.collect()
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
