#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def main():
    import sys
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import Config
    from prep.residual_features import apply_baseline_feature_engineering_generic

    cfg = Config()
    input_file = cfg.link_timeseries_file
    output_file = cfg.link_timeseries_residual_file

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Missing link timeseries file: {input_file}")

    df, _ = apply_baseline_feature_engineering_generic(
        timeseries_file=input_file,
        id_col="link_id",
        baseline_scenario_id=0,
    )
    if df is None:
        raise RuntimeError("Link baseline fit failed")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False, compression="gzip")
    print(f"[OK] wrote {output_file}")


if __name__ == "__main__":
    main()
