#!/usr/bin/env python3
"""Launch the Chapter 4 ablation runner as a detached Windows process."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("E:/11.16")
PYTHON = Path("D:/conda3/envs/swmm_gpu/python.exe")
RUNNER = (
    ROOT
    / "script2_new/chapter4_diagnosis_model/scripts/run_ch4_method_ablation.py"
)
OUTPUT_DIR = (
    ROOT / "script2_new/chapter4_diagnosis_model/outputs/formal_method_ablation"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "seed42", "confirm", "all-seeds", "summarize"],
        default="seed42",
    )
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    args = parser.parse_args()

    log_dir = OUTPUT_DIR / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = log_dir / f"ch4_ablation_{args.mode}_{stamp}.stdout.log"
    stderr_path = log_dir / f"ch4_ablation_{args.mode}_{stamp}.stderr.log"
    pid_path = log_dir / f"ch4_ablation_{args.mode}_{stamp}.pid"

    command = [
        str(PYTHON),
        str(RUNNER),
        "--mode",
        args.mode,
        "--cooldown-seconds",
        str(args.cooldown_seconds),
    ]
    env = os.environ.copy()
    env["SWMM_PARQUET_SINGLE_THREAD"] = "1"
    # CREATE_NEW_PROCESS_GROUP keeps the runner independent after this short
    # launcher exits. DETACHED_PROCESS is deliberately avoided because nested
    # CUDA training children can fail to initialize under some Windows job
    # hosts when that flag is inherited.
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
    pid_path.write_text(str(process.pid), encoding="ascii")

    print("Started detached Chapter 4 method ablation.")
    print(f"Mode: {args.mode}")
    print(f"PID: {process.pid}")
    print(f"stdout: {stdout_path}")
    print(f"stderr: {stderr_path}")
    print(f"per-job logs: {OUTPUT_DIR / 'logs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
