from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    scripts = [
        "plot_vcs_overview.py",
        "plot_strategy_performance.py",
        "plot_model_comparison.py",
        "plot_b_experiments.py",
        "plot_defect_matrix_stats.py",
    ]
    for script in scripts:
        path = root / script
        print(f"[RUN] {script}")
        result = subprocess.run([sys.executable, str(path)], cwd=str(root.parent))
        if result.returncode != 0:
            print(f"[WARN] {script} ????????={result.returncode}")
        else:
            print(f"[OK] {script} ??")


if __name__ == "__main__":
    main()
