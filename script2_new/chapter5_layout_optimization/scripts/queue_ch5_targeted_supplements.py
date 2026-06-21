#!/usr/bin/env python3
"""Wait for a process to finish, then run targeted Chapter 5 supplements."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("E:/11.16")
PYTHON = Path("D:/conda3/envs/swmm_gpu/python.exe")
RUNNER = ROOT / "script2_new/chapter5_layout_optimization/scripts/run_ch5_budget_sweep.py"
CH4_RUNNER = (
    ROOT
    / "script2_new/chapter4_diagnosis_model/scripts/run_ch4_method_ablation.py"
)
COLLECTOR = (
    ROOT
    / "script2_new/chapter5_layout_optimization/scripts/collect_ch5_targeted_supplements.py"
)
OUT_DIR = ROOT / "script2_new/chapter5_layout_optimization/outputs/targeted_supplements"


def process_exists(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    parser.add_argument("--resume-ch4-confirm", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queue_log = OUT_DIR / "CH5_TARGETED_SUPPLEMENTS_QUEUE.log"
    with queue_log.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"{datetime.now().isoformat()} waiting for PID {args.wait_pid}\n"
        )
        while process_exists(args.wait_pid):
            time.sleep(args.poll_seconds)
        log.write(f"{datetime.now().isoformat()} wait PID finished\n")
        if args.resume_ch4_confirm:
            log.write(f"{datetime.now().isoformat()} resuming Ch4 confirm\n")
            ch4_result = subprocess.run(
                [
                    str(PYTHON),
                    str(CH4_RUNNER),
                    "--mode",
                    "confirm",
                    "--cooldown-seconds",
                    str(args.cooldown_seconds),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.write(
                f"{datetime.now().isoformat()} Ch4 resume returncode={ch4_result.returncode}\n"
            )
            if ch4_result.returncode != 0:
                return int(ch4_result.returncode)
        log.write(f"{datetime.now().isoformat()} starting Ch5\n")
        command = [
            str(PYTHON),
            str(RUNNER),
            "--manifest",
            str(Path(args.manifest)),
            "--sleep",
            str(args.cooldown_seconds),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(
            f"{datetime.now().isoformat()} Ch5 runner returncode={result.returncode}\n"
        )
        if result.returncode != 0:
            return int(result.returncode)
        collect_result = subprocess.run(
            [str(PYTHON), str(COLLECTOR)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(
            f"{datetime.now().isoformat()} collector returncode={collect_result.returncode}\n"
        )
        return int(collect_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
