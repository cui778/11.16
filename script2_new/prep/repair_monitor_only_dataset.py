import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def load_monitor_nodes(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(node) for node in data}
    if isinstance(data, dict) and isinstance(data.get("monitor_nodes"), list):
        return {str(node) for node in data["monitor_nodes"]}
    raise ValueError(f"Unsupported monitor file format: {path}")


def filter_parquet(path: Path, monitor_nodes: set[str]) -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(path)
    if "node_id" not in df.columns:
        raise ValueError(f"`node_id` column missing in {path}")
    before = df["node_id"].nunique()
    df = df[df["node_id"].astype(str).isin(monitor_nodes)].copy()
    after = df["node_id"].nunique()
    return df, before - after


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair leaked monitor-subset datasets into strict monitor-only datasets."
    )
    parser.add_argument("--input-dir", required=True, help="Source dataset directory")
    parser.add_argument("--output-dir", required=True, help="Destination dataset directory")
    parser.add_argument("--monitor-file", required=True, help="JSON list of fixed monitor nodes")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    monitor_file = Path(args.monitor_file)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_nodes = load_monitor_nodes(monitor_file)

    node_src = input_dir / "node_timeseries.parquet"
    residual_src = input_dir / "node_timeseries_with_residuals.parquet"
    summary_src = input_dir / "scenario_summary.csv"

    node_df, node_removed = filter_parquet(node_src, monitor_nodes)
    residual_df, residual_removed = filter_parquet(residual_src, monitor_nodes)

    node_df.to_parquet(output_dir / node_src.name, index=False)
    residual_df.to_parquet(output_dir / residual_src.name, index=False)
    copy_if_exists(summary_src, output_dir / summary_src.name)

    manifest = {
        "source_dir": str(input_dir),
        "monitor_file": str(monitor_file),
        "monitor_node_count": len(monitor_nodes),
        "node_timeseries_unique_nodes": int(node_df["node_id"].nunique()),
        "node_timeseries_removed_nodes": int(node_removed),
        "residual_timeseries_unique_nodes": int(residual_df["node_id"].nunique()),
        "residual_timeseries_removed_nodes": int(residual_removed),
        "strict_monitor_only": True,
    }
    with (output_dir / "repair_manifest_monitor_only.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
