#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene-level timeline diagnosis from the existing window model.

This is the minimal-change A1 implementation:
- no model architecture change
- no retraining
- restore p_active(t) from window predictions
- estimate onset/interval with predicted active windows
- aggregate node scores only from predicted active windows
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from dataset.processor import create_dataloaders
from models.anomaly_detection_model import create_model


RESIDUAL8 = [
    "depth_residual",
    "total_outflow_residual",
    "pollut_NH4_residual",
    "pollut_TSSs_residual",
    "depth_residual_rel",
    "total_outflow_residual_rel",
    "pollut_NH4_residual_rel",
    "pollut_TSSs_residual_rel",
]


HYDRAULIC_MODEL_TYPES = {
    "hydraulic_inverse",
    "hydraulic_inverse_ctx",
    "hydraulic_inverse_ctxfix",
    "hydraulic_inverse_typeaware",
    "hydraulic_inverse_lstm",
    "hydraulic_inverse_static",
    "hydraulic_inverse_staticbias",
    "hydraulic_inverse_deepattn_static",
    "hydraulic_inverse_deepattn_staticbias",
    "hydraulic_inverse_deepattn_recon",
    "hydraulic_inverse_deepattn",
}


def _mask_unobserved_features(x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    mask = observed_mask.bool().unsqueeze(1).unsqueeze(-1)
    x_masked = x.clone()
    x_masked[..., :-1] = x_masked[..., :-1].masked_fill(~mask, 0.0)
    return x_masked


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.training_data_subdir = args.subdir
    cfg.node_timeseries_file = os.path.normpath(
        os.path.join(cfg.training_data_dir, args.subdir, "node_timeseries_with_residuals.parquet")
    )
    cfg.monitor_nodes_file = os.path.normpath(args.monitor_nodes)
    cfg.selected_features = list(RESIDUAL8)
    cfg.random_seed = int(args.seed)
    cfg.dataset_split_mode = str(args.split_mode)
    cfg.dataset_n_holdout_nodes = int(args.n_holdout_nodes)
    cfg.model_type = str(args.model_type)
    cfg.dataset_label_mode = "time_gated"
    cfg.use_time_pos_encoding = True
    cfg.use_trend_feature = False
    cfg.ensure_output_dirs()
    return cfg


def _create_model(cfg: Config, dataset, device: torch.device):
    model_kw: dict[str, Any] = {
        "model_type": cfg.model_type,
        "input_dim": int(dataset.input_feature_dim),
        "time_hidden_dim": cfg.time_hidden_dim,
        "spatial_hidden_dim": cfg.spatial_hidden_dim,
        "num_time_layers": cfg.num_time_layers,
        "num_spatial_layers": cfg.num_spatial_layers,
        "num_nodes": int(len(dataset.node_list)),
        "dropout": cfg.dropout,
    }
    if cfg.model_type in HYDRAULIC_MODEL_TYPES:
        npz = np.load(cfg.graph_path_features_file)
        model_kw["graph_features_dict"] = {k: npz[k] for k in npz.files}
        model_kw["use_flow_direction"] = getattr(cfg, "use_flow_direction", True)
        model_kw["use_propagation_delay"] = getattr(cfg, "use_propagation_delay", False)
        model_kw["propagation_delay_velocity_mps"] = getattr(cfg, "propagation_delay_velocity_mps", 0.5)
        model_kw["attention_max_hops"] = getattr(cfg, "hydraulic_attention_max_hops", 0)
        model_kw["observed_source_only"] = getattr(cfg, "hydraulic_observed_source_only", False)
    return create_model(**model_kw).to(device)


def _to_list(value, length: int) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * length


def _find_first_segment(flags: list[bool], min_consecutive: int) -> tuple[int, int] | None:
    start = None
    run = 0
    for idx, flag in enumerate(flags):
        if flag:
            if start is None:
                start = idx
            run += 1
        else:
            if start is not None and run >= min_consecutive:
                return start, idx - 1
            start = None
            run = 0
    if start is not None and run >= min_consecutive:
        return start, len(flags) - 1
    return None


def _interval_iou(a_start, a_end, b_start, b_end) -> float:
    latest_start = max(pd.Timestamp(a_start), pd.Timestamp(b_start))
    earliest_end = min(pd.Timestamp(a_end), pd.Timestamp(b_end))
    overlap = max(0.0, (earliest_end - latest_start).total_seconds())
    union_start = min(pd.Timestamp(a_start), pd.Timestamp(b_start))
    union_end = max(pd.Timestamp(a_end), pd.Timestamp(b_end))
    union = max(1.0, (union_end - union_start).total_seconds())
    return float(overlap / union)


def _rank_from_scores(scores: np.ndarray, true_idx: int) -> int | None:
    if true_idx < 0:
        return None
    order = np.argsort(-scores)
    hits = np.where(order == int(true_idx))[0]
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scene-level timeline diagnosis from window predictions")
    parser.add_argument("--subdir", default="time_gated_full_ie_v4_formal_conservative420_seed42")
    parser.add_argument("--checkpoint", default=str(SCRIPT_ROOT / "outputs" / "model_checkpoints" / "best_model_ch1_fullgraph_degree_ie420_s42_fix1.pth"))
    parser.add_argument("--monitor-nodes", default=str(SCRIPT_ROOT / "input_1" / "monitor_nodes_degree_N25.json"))
    parser.add_argument("--model-type", default="hydraulic_inverse_deepattn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", default="scenario", choices=["scenario", "node_holdout"])
    parser.add_argument("--n-holdout-nodes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--output-prefix", default="scene_timeline_ch1_s42")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _build_config(args)
    _, _, test_loader, dataset = create_dataloaders(
        node_timeseries_file=cfg.node_timeseries_file,
        adjacency_matrix_file=cfg.adjacency_matrix_file,
        node_list_file=cfg.node_list_file,
        defect_matrix_file=cfg.defect_matrix_file,
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        window_stride=cfg.window_stride,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        normalize=True,
        random_seed=cfg.random_seed,
        feature_names=cfg.selected_features,
        candidate_nodes_file=cfg.candidate_nodes_file,
        soft_label_sigma=cfg.soft_label_sigma,
        use_full_graph=cfg.train_with_full_graph,
        label_mode=cfg.dataset_label_mode,
        use_time_pos_encoding=cfg.use_time_pos_encoding,
        use_observed_mask_feature=cfg.use_observed_mask_feature,
        observed_nodes_file=cfg.monitor_nodes_file,
        use_trend_feature=cfg.use_trend_feature,
        active_overlap_threshold=cfg.dataset_active_overlap_threshold,
        transition_overlap_threshold=cfg.dataset_transition_overlap_threshold,
        split_mode=cfg.dataset_split_mode,
        n_holdout_nodes=cfg.dataset_n_holdout_nodes,
    )

    model = _create_model(cfg, dataset, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    candidate_mask_np = dataset.candidate_mask_np if dataset.candidate_mask_np is not None else np.ones(len(dataset.node_list), dtype=np.float32)
    candidate_indices = np.flatnonzero(candidate_mask_np > 0.5)
    node_list = list(dataset.node_list)

    window_rows: list[dict[str, Any]] = []
    grouped_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            observed_mask = batch["observed_mask"].to(device).bool()
            x_masked = _mask_unobserved_features(x, observed_mask)
            adj = batch["adj_matrix"]
            if isinstance(adj, (list, tuple)):
                adj = adj[0].to(device)
            elif adj.dim() == 3:
                adj = adj[0].to(device)
            else:
                adj = adj.to(device)

            outputs = model(x_masked, adj)
            logits_node = outputs[0]
            logits_active = outputs[1]
            p_active = torch.softmax(logits_active, dim=-1)[:, 1].detach().cpu().numpy()
            scores = logits_node[:, :, 1]
            cand_mask = batch["candidate_mask"].to(device).bool()
            if cand_mask.dim() == 1:
                cand_mask = cand_mask.unsqueeze(0).expand(scores.shape[0], -1)
            scores_masked = scores.masked_fill(~cand_mask, -1e9)
            probs = torch.softmax(scores_masked, dim=-1).detach().cpu().numpy()

            bsz = int(x.shape[0])
            scenario_ids = _to_list(batch["scenario_id"], bsz)
            starts = _to_list(batch["window_start_time"], bsz)
            ends = _to_list(batch["window_end_time"], bsz)
            true_active = _to_list(batch["active_label"], bsz)
            overlap = _to_list(batch["overlap_ratio"], bsz)
            target_idx = _to_list(batch["target_node_idx"], bsz)
            defect_node_id = _to_list(batch["defect_node_id"], bsz)

            for i in range(bsz):
                sc = str(scenario_ids[i])
                row = {
                    "scenario_id": sc,
                    "window_start_time": str(starts[i]),
                    "window_end_time": str(ends[i]),
                    "p_active": float(p_active[i]),
                    "pred_active": int(float(p_active[i]) > float(args.threshold)),
                    "true_active": int(true_active[i]),
                    "overlap_ratio": float(overlap[i]),
                    "target_node_idx": int(target_idx[i]),
                    "defect_node_id": str(defect_node_id[i]),
                    "top_candidate_node": node_list[int(np.argmax(probs[i]))],
                }
                window_rows.append(row)
                grouped_scores[sc].append(
                    {
                        **row,
                        "candidate_probs": probs[i].astype(np.float64),
                    }
                )

    defect_info = dataset.defect_info
    event_rows = []
    for sc, rows in grouped_scores.items():
        rows = sorted(rows, key=lambda r: pd.Timestamp(r["window_start_time"]))
        flags = [bool(float(r["p_active"]) > float(args.threshold)) for r in rows]
        segment = _find_first_segment(flags, int(args.min_consecutive))
        pred_has_defect = int(segment is not None)

        scenario_id = int(float(sc))
        info = defect_info.get(scenario_id, {})
        defect_type = str(info.get("defect_type", "BASELINE"))
        true_has_defect = int(defect_type != "BASELINE")
        base_time = pd.Timestamp(rows[0]["window_start_time"]).normalize()
        if true_has_defect:
            true_start = base_time + pd.Timedelta(hours=float(info.get("start_hour", 0.0)))
            true_end = true_start + pd.Timedelta(hours=float(info.get("duration_h", 0.0)))
        else:
            true_start = pd.NaT
            true_end = pd.NaT

        pred_start = pd.NaT
        pred_end = pd.NaT
        aggregate_scores = np.zeros(len(node_list), dtype=np.float64)
        aggregate_weight = 0.0
        if segment is not None:
            seg_start, seg_end = segment
            pred_start = pd.Timestamp(rows[seg_start]["window_start_time"])
            pred_end = pd.Timestamp(rows[seg_end]["window_end_time"])
            for r in rows[seg_start : seg_end + 1]:
                weight = max(float(r["p_active"]), 1e-6)
                aggregate_scores += weight * r["candidate_probs"]
                aggregate_weight += weight

        if aggregate_weight > 0:
            aggregate_scores /= aggregate_weight
            pred_rank = _rank_from_scores(aggregate_scores, int(rows[0]["target_node_idx"]))
            top_node = node_list[int(np.argmax(aggregate_scores))]
        else:
            pred_rank = None
            top_node = ""

        if true_has_defect and segment is not None:
            onset_error_h = abs((pd.Timestamp(pred_start) - true_start).total_seconds()) / 3600.0
            signed_onset_error_h = (pd.Timestamp(pred_start) - true_start).total_seconds() / 3600.0
            interval_iou = _interval_iou(pred_start, pred_end, true_start, true_end)
            duration_error_h = abs((pd.Timestamp(pred_end) - pd.Timestamp(pred_start)).total_seconds() - (true_end - true_start).total_seconds()) / 3600.0
            false_alarm_before_start = int(pd.Timestamp(pred_start) < true_start)
        else:
            onset_error_h = math.nan
            signed_onset_error_h = math.nan
            interval_iou = math.nan
            duration_error_h = math.nan
            false_alarm_before_start = 0

        event_rows.append(
            {
                "scenario_id": sc,
                "defect_type": defect_type,
                "true_has_defect": true_has_defect,
                "pred_has_defect": pred_has_defect,
                "true_start": "" if pd.isna(true_start) else str(true_start),
                "true_end": "" if pd.isna(true_end) else str(true_end),
                "pred_start": "" if pd.isna(pred_start) else str(pred_start),
                "pred_end": "" if pd.isna(pred_end) else str(pred_end),
                "onset_error_hours": onset_error_h,
                "signed_onset_error_hours": signed_onset_error_h,
                "active_interval_iou": interval_iou,
                "duration_error_hours": duration_error_h,
                "false_alarm_before_start": false_alarm_before_start,
                "true_node_id": str(info.get("node_id", "")),
                "pred_top_node_id": top_node,
                "pred_true_node_rank": "" if pred_rank is None else int(pred_rank),
                "n_windows": len(rows),
                "n_pred_active_windows": int(sum(flags)),
                "n_true_active_windows": int(sum(int(r["true_active"]) for r in rows)),
            }
        )

    event_df = pd.DataFrame(event_rows)
    defect_events = event_df[event_df["true_has_defect"] == 1].copy()
    detected_defects = defect_events[defect_events["pred_has_defect"] == 1].copy()
    ranks = pd.to_numeric(defect_events["pred_true_node_rank"], errors="coerce")
    stride_hours = float(cfg.window_stride * 10.0 / 60.0)

    metrics = {
        "threshold": float(args.threshold),
        "min_consecutive": int(args.min_consecutive),
        "n_events": int(len(event_df)),
        "n_defect_events": int(len(defect_events)),
        "scene_defect_recall": float(len(detected_defects) / max(len(defect_events), 1)),
        "missed_detection_rate": float(1.0 - len(detected_defects) / max(len(defect_events), 1)),
        "false_positive_baseline_events": int(((event_df["true_has_defect"] == 0) & (event_df["pred_has_defect"] == 1)).sum()),
        "onset_error_hours_mean": float(pd.to_numeric(detected_defects["onset_error_hours"], errors="coerce").mean()) if len(detected_defects) else math.nan,
        "onset_accuracy_within_1_window": float((pd.to_numeric(detected_defects["onset_error_hours"], errors="coerce") <= stride_hours).mean()) if len(detected_defects) else math.nan,
        "onset_accuracy_within_2_windows": float((pd.to_numeric(detected_defects["onset_error_hours"], errors="coerce") <= 2 * stride_hours).mean()) if len(detected_defects) else math.nan,
        "active_interval_iou_mean": float(pd.to_numeric(detected_defects["active_interval_iou"], errors="coerce").mean()) if len(detected_defects) else math.nan,
        "duration_error_hours_mean": float(pd.to_numeric(detected_defects["duration_error_hours"], errors="coerce").mean()) if len(detected_defects) else math.nan,
        "false_alarm_before_start_rate": float(pd.to_numeric(detected_defects["false_alarm_before_start"], errors="coerce").mean()) if len(detected_defects) else math.nan,
        "scene_node_mrr": float((1.0 / ranks.dropna()).sum() / max(len(defect_events), 1)),
        "scene_node_top1": float((ranks <= 1).sum() / max(len(defect_events), 1)),
        "scene_node_top3": float((ranks <= 3).sum() / max(len(defect_events), 1)),
        "scene_node_top5": float((ranks <= 5).sum() / max(len(defect_events), 1)),
    }
    if len(detected_defects):
        onset_err = pd.to_numeric(detected_defects["onset_error_hours"], errors="coerce")
        rank_det = pd.to_numeric(detected_defects["pred_true_node_rank"], errors="coerce")
        metrics["onset_within_1_window_and_node_top3"] = float(((onset_err <= stride_hours) & (rank_det <= 3)).sum() / max(len(defect_events), 1))
        metrics["onset_within_2_windows_and_node_top5"] = float(((onset_err <= 2 * stride_hours) & (rank_det <= 5)).sum() / max(len(defect_events), 1))
    else:
        metrics["onset_within_1_window_and_node_top3"] = math.nan
        metrics["onset_within_2_windows_and_node_top5"] = math.nan

    out_dir = Path(cfg.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_path = out_dir / f"{args.output_prefix}_window_predictions.csv"
    event_path = out_dir / f"{args.output_prefix}_event_predictions.csv"
    metrics_path = out_dir / f"{args.output_prefix}_metrics.json"
    pd.DataFrame(window_rows).to_csv(window_path, index=False, encoding="utf-8-sig")
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] window predictions -> {window_path}")
    print(f"[OK] event predictions -> {event_path}")
    print(f"[OK] metrics -> {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
