#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=SCRIPT_ROOT).returncode == 0


def main():
    python_exe = sys.executable

    steps = [
        [python_exe, "prep/build_segments.py"],
        [python_exe, "prep/filter_defect_matrix.py"],
        [python_exe, "prep/residual_features.py"],
        [python_exe, "prep/link_residual_features.py"],
        [python_exe, "prep/edge_feature_engineering.py"],
    ]

    for step in steps:
        ok = run(step)
        if not ok:
            print("[FAIL]", " ".join(step))
            break


if __name__ == "__main__":
    main()
