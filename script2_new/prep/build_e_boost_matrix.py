#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an IE-only matrix variant that strengthens E-type scenes without
changing total scene count.

This is meant as a focused probe after the v2e_dense_ie line:
- keep node coverage and scene count fixed
- make E scenes stronger / longer
- preserve reproducibility via an explicit derived matrix file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--intensity-add", type=float, default=10.0)
    parser.add_argument("--intensity-cap", type=float, default=70.0)
    parser.add_argument("--duration-add", type=float, default=6.0)
    parser.add_argument("--duration-cap", type=float, default=24.0)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df_out = df.copy()

    mask_e = df_out["defect_type"].astype(str) == "E"
    if int(mask_e.sum()) == 0:
        raise RuntimeError("No E-type rows found in input matrix.")

    df_out.loc[mask_e, "intensity_pct"] = np.minimum(
        df_out.loc[mask_e, "intensity_pct"].astype(float) + float(args.intensity_add),
        float(args.intensity_cap),
    )
    df_out.loc[mask_e, "duration_h"] = np.minimum(
        df_out.loc[mask_e, "duration_h"].astype(float) + float(args.duration_add),
        float(args.duration_cap),
    )
    df_out.loc[mask_e, "flow"] = -(
        df_out.loc[mask_e, "intensity_pct"].astype(float)
        * df_out.loc[mask_e, "baseline_flow"].astype(float)
        * 0.01
    ).round(4)

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    manifest = {
        "source_csv": str(in_path.resolve()),
        "output_csv": str(out_path.resolve()),
        "rule": "E-boost",
        "intensity_add": float(args.intensity_add),
        "intensity_cap": float(args.intensity_cap),
        "duration_add": float(args.duration_add),
        "duration_cap": float(args.duration_cap),
        "scene_count": int(len(df_out)),
        "type_counts": df_out.groupby("defect_type").size().to_dict(),
        "type_unique_nodes": df_out.groupby("defect_type")["node_id"].nunique().to_dict(),
    }
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
