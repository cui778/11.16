#!/usr/bin/env python3
"""Launch the targeted Chapter 5 queue as an independent Windows process."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path("E:/11.16")
PYTHON = Path("D:/conda3/envs/swmm_gpu/python.exe")
QUEUE = ROOT / "script2_new/chapter5_layout_optimization/scripts/queue_ch5_targeted_supplements.py"
OUT_DIR = ROOT / "script2_new/chapter5_layout_optimization/outputs/targeted_supplements"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resume-ch4-confirm", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = OUT_DIR / "CH5_TARGETED_QUEUE_LAUNCHER.stdout.log"
    stderr_path = OUT_DIR / "CH5_TARGETED_QUEUE_LAUNCHER.stderr.log"
    pid_path = OUT_DIR / "CH5_TARGETED_QUEUE_LAUNCHER.pid"
    command = [
        str(PYTHON),
        str(QUEUE),
        "--wait-pid",
        str(args.wait_pid),
        "--manifest",
        args.manifest,
    ]
    if args.resume_ch4_confirm:
        command.append("--resume-ch4-confirm")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=flags,
        )
    pid_path.write_text(str(process.pid), encoding="ascii")
    print(f"Queued Chapter 5 targeted supplements. PID: {process.pid}")
    print(f"Waiting for Chapter 4 PID: {args.wait_pid}")
    print(f"Queue log: {OUT_DIR / 'CH5_TARGETED_SUPPLEMENTS_QUEUE.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
