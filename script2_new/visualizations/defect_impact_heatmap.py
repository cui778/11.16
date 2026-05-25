from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from config import Config


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_inp_coordinates(inp_path: str | Path) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    inp_path = Path(inp_path)
    if not inp_path.exists():
        return coords

    in_section = False
    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.upper().startswith("[COORDINATES]"):
                in_section = True
                continue
            if in_section:
                if line.startswith("["):
                    break
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    coords[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    continue
    return coords


def resolve_timeseries_path(cfg: Config, provided_path: str = "", subdir: str = "") -> Path:
    candidates: list[Path] = []
    if provided_path:
        candidates.append(Path(provided_path))
    if subdir:
        candidates.append(Path(cfg.training_data_dir) / subdir / "node_timeseries_with_residuals.parquet")
        candidates.append(Path(cfg.training_data_dir) / subdir / "node_timeseries.parquet")

    candidates.append(Path(cfg.node_timeseries_file))
    candidates.extend(
        [
            Path(cfg.training_data_dir) / "time_gated_node_sensor_v2_rebuild" / "node_timeseries_with_residuals.parquet",
            Path(cfg.training_data_dir) / "time_gated_node_sensor_v2_rebuild" / "node_timeseries.parquet",
            Path(cfg.training_data_dir) / "time_gated" / "node_timeseries_with_residuals.parquet",
            Path(cfg.training_data_dir) / "time_gated" / "node_timeseries.parquet",
            Path(cfg.training_data_dir) / "time_gated_downstream" / "node_timeseries_with_residuals.parquet",
            Path(cfg.training_data_dir) / "time_gated_observability" / "node_timeseries_with_residuals.parquet",
        ]
    )

    seen: set[str] = set()
    ordered: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)

    for path in ordered:
        if path.exists():
            return path

    tried = "\n".join(f"  - {path}" for path in ordered)
    raise FileNotFoundError(f"未找到可用时序数据文件，已尝试：\n{tried}")


def compute_impact_for_scenario(
    df_ts: pd.DataFrame,
    df_base: pd.DataFrame,
    scenario_id: int,
) -> tuple[dict[str, float], dict[str, float]] | None:
    df_def = df_ts[df_ts["scenario_id"] == scenario_id].copy()
    if df_def.empty:
        return None

    df_b = df_base[["datetime", "node_id", "depth", "total_inflow"]].rename(
        columns={"depth": "depth_base", "total_inflow": "inflow_base"}
    )
    df_d = df_def[["datetime", "node_id", "depth", "total_inflow"]].rename(
        columns={"depth": "depth_def", "total_inflow": "inflow_def"}
    )
    df_merged = pd.merge(df_b, df_d, on=["datetime", "node_id"], how="inner")
    if df_merged.empty:
        return None

    df_merged["abs_diff_depth"] = (df_merged["depth_def"] - df_merged["depth_base"]).abs()
    denom = np.maximum(np.abs(df_merged["depth_base"]), 0.01)
    df_merged["rel_diff_depth"] = df_merged["abs_diff_depth"] / denom * 100.0

    impact_abs = df_merged.groupby("node_id")["abs_diff_depth"].max().to_dict()
    impact_rel = df_merged.groupby("node_id")["rel_diff_depth"].max().to_dict()
    return impact_abs, impact_rel


def plot_heatmap_single(
    ax,
    xs: list[float],
    ys: list[float],
    cs: np.ndarray,
    defect_node: str,
    defect_type: str,
    intensity_pct: float,
    scenario_id: int,
    node_coords: dict[str, tuple[float, float]],
) -> None:
    vmax = np.percentile(cs[cs > 0], 95) if np.any(cs > 0) else 0.01
    ax.scatter(xs, ys, c=cs, cmap="Reds", vmin=0, vmax=vmax, s=35, alpha=0.85, edgecolors="none")
    if defect_node in node_coords:
        xd, yd = node_coords[defect_node]
        ax.scatter([xd], [yd], c="blue", marker="*", s=220, edgecolors="black", linewidths=0.8)
    ax.set_title(f"场景 {scenario_id}: {defect_type} @ {intensity_pct}%\n{defect_node}", fontsize=9, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制缺陷影响范围热力图。")
    parser.add_argument("--defect-csv", default="", help="缺陷矩阵 CSV 路径。")
    parser.add_argument("--timeseries", default="", help="时序 parquet 路径。")
    parser.add_argument("--subdir", default="", help="training_data_new 下的子目录，如 time_gated。")
    parser.add_argument("--inp", default="", help="SWMM inp 文件路径。")
    parser.add_argument("--output-dir", default="", help="输出目录。")
    parser.add_argument("--max-scenarios", type=int, default=None, help="最多绘制多少个场景。")
    parser.add_argument("--grid-size", type=int, default=4, help="每页网格大小。")
    return parser.parse_args()


def main() -> int:
    cfg = Config()
    args = parse_args()

    defect_csv = Path(args.defect_csv or cfg.defect_matrix_file)
    inp_path = Path(args.inp or cfg.inp_file)
    output_dir = Path(args.output_dir or (_SCRIPT_ROOT / "visualizations" / "defect_impact_figures"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not defect_csv.exists():
        print(f"未找到缺陷矩阵: {defect_csv}")
        print("请先运行 prep/defect_matrix.py 生成缺陷矩阵。")
        return 1

    try:
        ts_path = resolve_timeseries_path(cfg, args.timeseries, args.subdir)
    except FileNotFoundError as exc:
        print(str(exc))
        print("请先运行 prep/extract_timeseries.py 和 prep/residual_features.py，或显式传入 --timeseries / --subdir。")
        return 1

    print("=" * 60)
    print("缺陷影响范围热力图")
    print("=" * 60)
    print(f"缺陷矩阵: {defect_csv}")
    print(f"时序数据: {ts_path}")
    print(f"输出目录: {output_dir}")
    print()

    df_def = pd.read_csv(defect_csv).dropna(subset=["node_id"])
    if args.max_scenarios:
        df_def = df_def.head(args.max_scenarios)
        print(f"仅处理前 {args.max_scenarios} 个场景")
    print(f"缺陷场景数: {len(df_def)}")

    df_ts = pd.read_parquet(ts_path)
    if not pd.api.types.is_datetime64_any_dtype(df_ts["datetime"]):
        df_ts["datetime"] = pd.to_datetime(df_ts["datetime"])
    df_base = df_ts[df_ts["scenario_id"] == 0].copy()
    print(
        f"时序记录: {len(df_ts):,}, 场景数: {df_ts['scenario_id'].nunique()}, "
        f"节点数: {df_ts['node_id'].nunique()}"
    )

    node_coords = load_inp_coordinates(inp_path)
    print(f"节点坐标数: {len(node_coords)}")
    print()

    grid = args.grid_size
    n_plots = len(df_def)
    n_batches = (n_plots + grid * grid - 1) // (grid * grid)

    for batch_idx in range(n_batches):
        start = batch_idx * grid * grid
        end = min(start + grid * grid, n_plots)
        batch_df = df_def.iloc[start:end]
        n_curr = len(batch_df)

        ncol = min(grid, n_curr)
        nrow = (n_curr + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), dpi=100)
        if n_curr == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, (_, row) in enumerate(batch_df.iterrows()):
            scenario_id = int(row["defect_id"])
            defect_node = str(row["node_id"])
            defect_type = str(row["defect_type"])
            intensity_pct = float(row.get("intensity_pct", 0))

            result = compute_impact_for_scenario(df_ts, df_base, scenario_id)
            if not result:
                axes[i].set_visible(False)
                continue

            impact_abs, _impact_rel = result
            xs: list[float] = []
            ys: list[float] = []
            cs: list[float] = []
            for node_id, value in impact_abs.items():
                if node_id in node_coords:
                    x, y = node_coords[node_id]
                    xs.append(x)
                    ys.append(y)
                    cs.append(value)
            if not xs:
                axes[i].set_visible(False)
                continue

            plot_heatmap_single(
                axes[i],
                xs,
                ys,
                np.array(cs),
                defect_node,
                defect_type,
                intensity_pct,
                scenario_id,
                node_coords,
            )

        for j in range(n_curr, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"缺陷影响范围热力图（场景 {start + 1}-{end}）", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = output_dir / f"impact_batch_{batch_idx + 1:02d}_scenarios_{start + 1}-{end}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] 已保存: {out_path}")

    print()
    print("完成。可用这些图检查：")
    print("  - 缺陷影响是否主要沿下游传播")
    print("  - 某些候选节点作为缺陷源时影响范围是否异常小")
    print("  - 当前候选/监测设置是否覆盖了主要影响区域")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
