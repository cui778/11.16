import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = {"datetime", "scenario_id", "defect_type", "node_id", "time_step"}


def load_inputs(input_dir: Path, defect_matrix_file: Path, node_list_file: Path, adj_file: Path):
    data_file = input_dir / "node_timeseries_with_residuals.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing file: {data_file}")

    df = pd.read_parquet(data_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    defect_df = pd.read_csv(defect_matrix_file)
    node_list = json.loads(node_list_file.read_text(encoding="utf-8"))
    adj = np.load(adj_file)
    return df, defect_df, node_list, adj


def get_feature_groups(df: pd.DataFrame):
    raw_features = []
    residual_features = []
    residual_rel_features = []

    for col in df.columns:
        if col in META_COLS:
            continue
        if col.endswith("_residual_rel"):
            residual_rel_features.append(col)
        elif col.endswith("_residual"):
            residual_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            raw_features.append(col)

    return raw_features, residual_features, residual_rel_features


def build_active_mask(df: pd.DataFrame, defect_df: pd.DataFrame):
    active = pd.Series(False, index=df.index)
    defect_map = defect_df.set_index("defect_id")

    for scen_id in sorted(df["scenario_id"].unique()):
        if scen_id == 0:
            continue
        if scen_id not in defect_map.index:
            continue
        row = defect_map.loc[scen_id]
        start_dt = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=float(row["start_hour"]))
        end_dt = start_dt + pd.Timedelta(hours=float(row["duration_h"]))
        scen_mask = df["scenario_id"] == scen_id
        active.loc[scen_mask] = (df.loc[scen_mask, "datetime"] >= start_dt) & (
            df.loc[scen_mask, "datetime"] < end_dt
        )

    return active


def temporal_delta_correlation(df: pd.DataFrame, features, active_mask=None):
    work = df.copy()
    work = work.sort_values(["scenario_id", "node_id", "datetime"])

    delta_cols = []
    for feature in features:
        delta_col = f"{feature}__delta"
        work[delta_col] = work.groupby(["scenario_id", "node_id"])[feature].diff()
        delta_cols.append(delta_col)

    valid = work[delta_cols].notna().all(axis=1)
    if active_mask is not None:
        valid &= active_mask.reindex(work.index, fill_value=False)

    delta_df = work.loc[valid, delta_cols].rename(columns={f"{c}__delta": c for c in features})
    return delta_df.corr(), len(delta_df)


def compute_hop_distances(adj: np.ndarray):
    graph = (adj != 0).astype(np.int32)
    graph = ((graph + graph.T) > 0).astype(np.int32)
    n = graph.shape[0]
    inf = 10**9
    dist = np.full((n, n), inf, dtype=np.int32)
    for i in range(n):
        dist[i, i] = 0
        frontier = [i]
        while frontier:
            current = frontier.pop(0)
            next_nodes = np.where(graph[current] != 0)[0]
            for nxt in next_nodes:
                if dist[i, nxt] == inf:
                    dist[i, nxt] = dist[i, current] + 1
                    frontier.append(nxt)
    return dist


def spatial_pairwise_summary(df: pd.DataFrame, features, active_mask, node_list, hop_dist):
    defect_df = df[(df["scenario_id"] != 0) & active_mask].copy()
    defect_df["sample_key"] = (
        defect_df["scenario_id"].astype(str)
        + "__"
        + defect_df["time_step"].astype(str)
    )

    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    observed_nodes = [n for n in sorted(defect_df["node_id"].unique()) if n in node_to_idx]

    records = []
    for feature in features:
        pivot = defect_df.pivot_table(index="sample_key", columns="node_id", values=feature, aggfunc="mean")
        pivot = pivot[[n for n in observed_nodes if n in pivot.columns]]
        if pivot.shape[1] < 2:
            continue
        corr = pivot.corr()

        pair_rows = []
        for i, src in enumerate(pivot.columns):
            for j in range(i + 1, len(pivot.columns)):
                dst = pivot.columns[j]
                value = corr.loc[src, dst]
                if pd.isna(value):
                    continue
                hop = int(hop_dist[node_to_idx[src], node_to_idx[dst]])
                if hop >= 10**9:
                    continue
                pair_rows.append((hop, float(value), abs(float(value))))

        if not pair_rows:
            continue

        pair_df = pd.DataFrame(pair_rows, columns=["hop", "corr", "abs_corr"])
        hop_summary = (
            pair_df.groupby("hop")
            .agg(mean_corr=("corr", "mean"), mean_abs_corr=("abs_corr", "mean"), pairs=("hop", "size"))
            .reset_index()
        )
        hop_summary["feature"] = feature

        locality = pair_df[["hop", "abs_corr"]].corr().iloc[0, 1]
        records.append(
            {
                "feature": feature,
                "pair_count": int(len(pair_df)),
                "locality_corr_hop_vs_abscorr": float(locality) if not pd.isna(locality) else np.nan,
                "hop1_abs_corr": float(hop_summary.loc[hop_summary["hop"] == 1, "mean_abs_corr"].mean())
                if (hop_summary["hop"] == 1).any()
                else np.nan,
                "hop2_abs_corr": float(hop_summary.loc[hop_summary["hop"] == 2, "mean_abs_corr"].mean())
                if (hop_summary["hop"] == 2).any()
                else np.nan,
                "hop3plus_abs_corr": float(hop_summary.loc[hop_summary["hop"] >= 3, "mean_abs_corr"].mean())
                if (hop_summary["hop"] >= 3).any()
                else np.nan,
            }
        )
        yield feature, hop_summary

    summary_df = pd.DataFrame(records).sort_values(
        ["locality_corr_hop_vs_abscorr", "hop1_abs_corr"], ascending=[True, False]
    )
    yield "__summary__", summary_df


def activation_summary(df: pd.DataFrame, features, active_mask):
    defect = df[df["scenario_id"] != 0].copy()
    active_df = defect.loc[active_mask.reindex(defect.index, fill_value=False)]
    inactive_df = defect.loc[~active_mask.reindex(defect.index, fill_value=False)]

    rows = []
    for feature in features:
        active_mean = float(active_df[feature].abs().mean())
        inactive_mean = float(inactive_df[feature].abs().mean())
        ratio = active_mean / (inactive_mean + 1e-9)
        rows.append(
            {
                "feature": feature,
                "active_abs_mean": active_mean,
                "inactive_abs_mean": inactive_mean,
                "active_inactive_ratio": ratio,
            }
        )
    return pd.DataFrame(rows).sort_values("active_inactive_ratio", ascending=False)


def top_corr_pairs(corr_df: pd.DataFrame, top_k: int = 10):
    rows = []
    cols = list(corr_df.columns)
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            rows.append({"feature_1": c1, "feature_2": c2, "corr": float(corr_df.loc[c1, c2])})
    result = pd.DataFrame(rows)
    result["abs_corr"] = result["corr"].abs()
    return result.sort_values("abs_corr", ascending=False).head(top_k)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze temporal and spatial feature correlations.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--defect-matrix-file", default="script2_new/input_1/defect_matrix_diverse.csv")
    parser.add_argument("--node-list-file", default="script2_new/input_1/node_list.json")
    parser.add_argument("--adj-file", default="script2_new/input_1/adj_matrix.npy")
    parser.add_argument("--output-dir", default="script2_new/outputs/reports")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, defect_df, node_list, adj = load_inputs(
        input_dir=input_dir,
        defect_matrix_file=Path(args.defect_matrix_file),
        node_list_file=Path(args.node_list_file),
        adj_file=Path(args.adj_file),
    )

    raw_features, residual_features, residual_rel_features = get_feature_groups(df)
    active_mask = build_active_mask(df, defect_df)
    hop_dist = compute_hop_distances(adj)

    raw_delta_corr, raw_n = temporal_delta_correlation(df[df["scenario_id"] != 0], raw_features, active_mask[df["scenario_id"] != 0])
    residual_delta_corr, residual_n = temporal_delta_correlation(
        df[df["scenario_id"] != 0], residual_features, active_mask[df["scenario_id"] != 0]
    )
    residual_rel_delta_corr, residual_rel_n = temporal_delta_correlation(
        df[df["scenario_id"] != 0], residual_rel_features, active_mask[df["scenario_id"] != 0]
    )

    raw_activation = activation_summary(df, raw_features, active_mask)
    residual_activation = activation_summary(df, residual_features, active_mask)
    residual_rel_activation = activation_summary(df, residual_rel_features, active_mask)

    raw_delta_corr.to_csv(output_dir / f"{args.tag}_raw_delta_corr.csv", encoding="utf-8-sig")
    residual_delta_corr.to_csv(output_dir / f"{args.tag}_residual_delta_corr.csv", encoding="utf-8-sig")
    residual_rel_delta_corr.to_csv(output_dir / f"{args.tag}_residual_rel_delta_corr.csv", encoding="utf-8-sig")
    raw_activation.to_csv(output_dir / f"{args.tag}_raw_activation_summary.csv", index=False, encoding="utf-8-sig")
    residual_activation.to_csv(output_dir / f"{args.tag}_residual_activation_summary.csv", index=False, encoding="utf-8-sig")
    residual_rel_activation.to_csv(output_dir / f"{args.tag}_residual_rel_activation_summary.csv", index=False, encoding="utf-8-sig")

    spatial_residual_summary = None
    spatial_residual_rel_summary = None
    for feature, hop_summary in spatial_pairwise_summary(df, residual_features, active_mask, node_list, hop_dist):
        if feature == "__summary__":
            spatial_residual_summary = hop_summary
        else:
            hop_summary.to_csv(output_dir / f"{args.tag}_spatial_{feature}.csv", index=False, encoding="utf-8-sig")

    for feature, hop_summary in spatial_pairwise_summary(df, residual_rel_features, active_mask, node_list, hop_dist):
        if feature == "__summary__":
            spatial_residual_rel_summary = hop_summary
        else:
            hop_summary.to_csv(output_dir / f"{args.tag}_spatial_{feature}.csv", index=False, encoding="utf-8-sig")

    if spatial_residual_summary is not None:
        spatial_residual_summary.to_csv(
            output_dir / f"{args.tag}_spatial_residual_summary.csv", index=False, encoding="utf-8-sig"
        )
    if spatial_residual_rel_summary is not None:
        spatial_residual_rel_summary.to_csv(
            output_dir / f"{args.tag}_spatial_residual_rel_summary.csv", index=False, encoding="utf-8-sig"
        )

    lines = []
    lines.append(f"# Temporal and Spatial Feature Analysis: {args.tag}")
    lines.append("")
    lines.append(f"- input_dir: `{input_dir}`")
    lines.append(f"- rows: {len(df):,}")
    lines.append(f"- scenarios: {df['scenario_id'].nunique()}")
    lines.append(f"- observed_nodes: {df['node_id'].nunique()}")
    lines.append(f"- active_rows: {int(active_mask.sum()):,}")
    lines.append("")
    lines.append("## Temporal Delta Correlation")
    lines.append("")
    lines.append(f"- raw_active_delta_samples: {raw_n:,}")
    lines.append(f"- residual_active_delta_samples: {residual_n:,}")
    lines.append(f"- residual_rel_active_delta_samples: {residual_rel_n:,}")
    lines.append("")
    lines.append("### Top Raw Delta Correlation Pairs")
    lines.append("")
    lines.append(dataframe_to_markdown(top_corr_pairs(raw_delta_corr)))
    lines.append("")
    lines.append("### Top Residual Delta Correlation Pairs")
    lines.append("")
    lines.append(dataframe_to_markdown(top_corr_pairs(residual_delta_corr)))
    lines.append("")
    lines.append("### Top Residual-Rel Delta Correlation Pairs")
    lines.append("")
    lines.append(dataframe_to_markdown(top_corr_pairs(residual_rel_delta_corr)))
    lines.append("")
    lines.append("## Active vs Inactive Signal Strength")
    lines.append("")
    lines.append("### Raw Features")
    lines.append("")
    lines.append(dataframe_to_markdown(raw_activation.head(10)))
    lines.append("")
    lines.append("### Residual Features")
    lines.append("")
    lines.append(dataframe_to_markdown(residual_activation.head(10)))
    lines.append("")
    lines.append("### Residual-Rel Features")
    lines.append("")
    lines.append(dataframe_to_markdown(residual_rel_activation.head(10)))
    lines.append("")
    lines.append("## Spatial Propagation Summary")
    lines.append("")
    if spatial_residual_summary is not None:
        lines.append("### Residual Features")
        lines.append("")
        lines.append(dataframe_to_markdown(spatial_residual_summary.head(12)))
        lines.append("")
    if spatial_residual_rel_summary is not None:
        lines.append("### Residual-Rel Features")
        lines.append("")
        lines.append(dataframe_to_markdown(spatial_residual_rel_summary.head(12)))
        lines.append("")

    (output_dir / f"{args.tag}_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
