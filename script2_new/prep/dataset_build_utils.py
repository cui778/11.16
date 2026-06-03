#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utility functions for dataset construction scripts.

Functions extracted from exploration scripts to decouple formal scripts
from legacy dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\11.16\script2_new")


def _load_manifest(path: Path) -> dict:
    """Load a dataset_manifest.json file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_ie_defect_csv(ie_dir: Path) -> Path:
    """Infer the defect CSV path from an IE dataset manifest.

    Falls back to the formal conservative420 defect matrix if not found.
    """
    manifest = _load_manifest(ie_dir / "dataset_manifest.json")
    csv_path = str(manifest.get("defect_csv_path", "")).strip()
    if csv_path:
        return Path(csv_path)
    return ROOT / "input_1" / "defect_matrix_diverse_ie_v4_formal_conservative420_seed42.csv"
