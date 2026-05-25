#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from config import Config
from dataset.processor import create_dataloaders
from models.anomaly_detection_model import create_model


ROOT = SCRIPT_ROOT
REPORT_DIR = ROOT / "outputs" / "reports"
CKPT_DIR = ROOT / "outputs" / "model_checkpoints"

RUNS = [
    {
        "seed": 42,
        "checkpoint": CKPT_DIR / "best_model_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual8_scenario_refresh42.pth",
        "metrics": REPORT_DIR / "last_run_metrics_process_diagnosis_hydraulic_inverse_time_gated_node_sensor_v2_rebuild_residual8_scenario_refresh42.json",
    },
]

FEATURES = [
    "depth_residual",
    "total_outflow_residual",
    "pollut_NH4_residual",
    "pollut_TSSs_residual",
    "depth_residual_rel",
    "total_outflow_residual_rel",
    "pollut_NH4_residual_rel",
    "pollut_TSSs_residual_rel",
]


def build_adj_info(node_list, adj_matrix):
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    neigh = [set(np.where((adj_matrix[i] > 0) | (adj_matrix[:, i] > 0))[0].tolist()) for i in range(n)]
    return node_to_idx, neigh


def shortest_hop(neigh, src, dst):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        u, d = q.popleft()
        for v in neigh[u]:
            if v == dst:
                return d + 1
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return -1


def create_test_loader(cfg):
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
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
        signal_threshold=cfg.dataset_signal_threshold,
        label_mode=cfg.dataset_label_mode,
        overlap_threshold=cfg.dataset_overlap_threshold,
        use_full_graph=cfg.train_with_full_graph,
        split_mode=cfg.dataset_split_mode,
        n_holdout_nodes=cfg.dataset_n_holdout_nodes,
        always_on_force_target=cfg.dataset_always_on_force_target,
        use_time_pos_encoding=cfg.use_time_pos_encoding,
        use_observed_mask_feature=cfg.use_observed_mask_feature,
        use_trend_feature=cfg.use_trend_feature,
        active_overlap_threshold=cfg.dataset_active_overlap_threshold,
        transition_overlap_threshold=cfg.dataset_transition_overlap_threshold,
        active_signal_threshold=None,
        transition_signal_threshold=None,
    )
    return test_loader, dataset


def load_model(cfg, dataset, device):
    npz = np.load(cfg.graph_path_features_file)
    model = create_model(
        model_type="hydraulic_inverse",
        input_dim=int(getattr(dataset, "input_feature_dim", 0) or dataset.samples[0]["features"].shape[-1]),
        time_hidden_dim=cfg.time_hidden_dim,
        spatial_hidden_dim=cfg.spatial_hidden_dim,
        num_time_layers=cfg.num_time_layers,
        num_spatial_layers=cfg.num_spatial_layers,
        num_nodes=len(dataset.node_list),
        dropout=cfg.dropout,
        graph_features_dict={k: npz[k] for k in npz.files},
        use_flow_direction=cfg.use_flow_direction,
        use_propagation_delay=cfg.use_propagation_delay,
    ).to(device)
    return model


def analyze_run(run):
    cfg = Config()
    cfg.training_data_subdir = "time_gated_node_sensor_v2_rebuild"
    cfg.dataset_split_mode = "scenario"
    cfg.random_seed = run["seed"]
    cfg.selected_features = list(FEATURES)
    cfg.model_type = "hydraulic_inverse"

    test_loader, dataset = create_test_loader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, dataset, device)
    checkpoint = torch.load(run["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    node_to_idx, neigh = build_adj_info(dataset.node_list, dataset.adj_matrix)
    monitor_set = set(getattr(dataset, "monitor_node_list", []))

    false_pred_counter = Counter()
    false_pair_counter = Counter()
    hop_counter = Counter()
    by_type_counter = defaultdict(Counter)
    scenario_counter = Counter()
    top1_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            adj = batch["adj_matrix"]
            if isinstance(adj, (list, tuple)):
                adj = adj[0]
            if adj.dim() == 3:
                adj = adj[0]
            adj = adj.to(device)

            logits = model(x, adj)
            logits_node = logits[0] if isinstance(logits, tuple) else logits
            logits_active = logits[1] if isinstance(logits, tuple) else None
            scores = logits_node[:, :, 1]

            candidate_mask = batch["candidate_mask"].to(device).bool()
            if candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.unsqueeze(0).expand_as(scores)

            y_idx = batch["target_node_idx"].long().to(device)
            valid_mask = y_idx >= 0
            if "active_label" in batch:
                valid_mask = valid_mask & (batch["active_label"].long().to(device) > 0)
            if "loc_enabled" in batch:
                valid_mask = valid_mask & (batch["loc_enabled"].long().to(device) > 0)
            valid_mask = valid_mask & candidate_mask[torch.arange(candidate_mask.size(0), device=device), y_idx]
            if valid_mask.sum().item() == 0:
                continue

            scores = scores[valid_mask]
            y_idx = y_idx[valid_mask]
            candidate_mask = candidate_mask[valid_mask]
            scores = scores.masked_fill(~candidate_mask, -1e9)
            pred_idx = torch.argmax(scores, dim=1).cpu().tolist()
            true_idx = y_idx.cpu().tolist()

            defect_types = batch["defect_type_str"]
            scenario_ids = batch["scenario_id"]
            if isinstance(defect_types, tuple):
                defect_types = list(defect_types)
            if isinstance(scenario_ids, torch.Tensor):
                scenario_ids = scenario_ids.view(-1).cpu().tolist()
            else:
                scenario_ids = list(scenario_ids)
            valid_list = valid_mask.cpu().tolist()
            defect_types = [str(defect_types[i]) for i, ok in enumerate(valid_list) if ok]
            scenario_ids = [scenario_ids[i] for i, ok in enumerate(valid_list) if ok]

            for p, t, typ, sid in zip(pred_idx, true_idx, defect_types, scenario_ids):
                total += 1
                pred_node = dataset.node_list[p]
                true_node = dataset.node_list[t]
                scenario_counter[str(sid)] += 1
                hop = shortest_hop(neigh, p, t)
                hop_counter[hop] += 1
                if p == t:
                    top1_correct += 1
                    continue
                false_pred_counter[pred_node] += 1
                false_pair_counter[(true_node, pred_node)] += 1
                by_type_counter[typ][pred_node] += 1

    rows_false_pred = []
    for node_id, count in false_pred_counter.most_common(20):
        idx = node_to_idx[node_id]
        rows_false_pred.append({
            "pred_node_id": node_id,
            "pred_node_idx": idx,
            "false_top1_count": count,
            "is_monitor": int(node_id in monitor_set),
            "is_candidate": int(node_id in set(dataset.candidate_nodes)),
        })

    rows_false_pair = []
    for (true_node, pred_node), count in false_pair_counter.most_common(30):
        rows_false_pair.append({
            "true_node_id": true_node,
            "pred_node_id": pred_node,
            "count": count,
        })

    rows_by_type = []
    for defect_type, counter in sorted(by_type_counter.items()):
        for node_id, count in counter.most_common(10):
            rows_by_type.append({
                "defect_type": defect_type,
                "pred_node_id": node_id,
                "false_top1_count": count,
            })

    summary = {
        "seed": run["seed"],
        "total_loc_windows": total,
        "top1_correct_windows": top1_correct,
        "top1_acc": top1_correct / max(total, 1),
        "error_windows": total - top1_correct,
        "hop_distribution": dict(sorted(hop_counter.items())),
        "top_false_pred_nodes": rows_false_pred[:10],
        "top_false_pairs": rows_false_pair[:10],
    }
    return summary, rows_false_pred, rows_false_pair, rows_by_type


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_summary = []
    all_false_pred = []
    all_false_pair = []
    all_by_type = []
    for run in RUNS:
        summary, rows_false_pred, rows_false_pair, rows_by_type = analyze_run(run)
        all_summary.append(summary)
        for row in rows_false_pred:
            all_false_pred.append({"seed": run["seed"], **row})
        for row in rows_false_pair:
            all_false_pair.append({"seed": run["seed"], **row})
        for row in rows_by_type:
            all_by_type.append({"seed": run["seed"], **row})

    write_csv(REPORT_DIR / "clean_error_pattern_false_pred_nodes.csv", all_false_pred)
    write_csv(REPORT_DIR / "clean_error_pattern_false_pairs.csv", all_false_pair)
    write_csv(REPORT_DIR / "clean_error_pattern_false_pred_by_type.csv", all_by_type)
    (REPORT_DIR / "clean_error_pattern_summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
