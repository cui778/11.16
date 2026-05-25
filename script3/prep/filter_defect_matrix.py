#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd


def filter_defect_matrix(input_csv, output_csv, keep_types=("I", "E"), merge_p_to_i=False):
    df = pd.read_csv(input_csv)
    keep_types = tuple(str(x).upper() for x in keep_types)
    dtype = df["defect_type"].astype(str).str.upper()
    if merge_p_to_i:
        df.loc[dtype.eq("P"), "defect_type"] = "I"
        dtype = df["defect_type"].astype(str).str.upper()
    filtered = df[dtype.isin(keep_types)].copy()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return filtered


def main():
    import argparse
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import Config

    cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=cfg.defect_matrix_file)
    parser.add_argument("--output-csv", default=str(Path(cfg.input_dir_new) / "defect_matrix_ie.csv"))
    parser.add_argument("--merge-p-to-i", action="store_true")
    args = parser.parse_args()

    filtered = filter_defect_matrix(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        keep_types=("I", "E"),
        merge_p_to_i=args.merge_p_to_i,
    )
    print(f"[OK] wrote {args.output_csv} ({len(filtered)} rows)")


if __name__ == "__main__":
    main()
