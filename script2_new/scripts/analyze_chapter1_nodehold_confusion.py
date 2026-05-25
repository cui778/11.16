#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_privileged_teacher_student import (
    _build_config,
    _create_model,
    _get_subset_base_and_indices,
    _make_loaders,
    _mask_unobserved_features,
)


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports"
CKPT_DIR = ROOT / "outputs" / "model_checkpoints"
MONITOR_FILE = str(ROOT / "input_1" / "monitor_nodes_degree_N25.json")
SUBDIR = "time_gated_full_ie_v4_formal_conservative420_seed42"
N_HOLDOUT = 10

RUNS = [
    {"seed": 42, "tag": "ch1_fullgraph_degree_ie420_nodehold_s42_fix1"},
    {"seed": 7, "tag": "ch1_fullgraph_degree_ie420_nodehold_s7_fix1"},
    {"seed": 123, "tag": "ch1_fullgraph_degree_ie420_nodehold_s123_fix1"},
]


def build_neighbors(adj_matrix: np.ndarray) -> list[set[int]]:
    n = adj_matrix.shape[0]
    return [set(np.where((adj_matrix[i] > 0) | (adj_matrix[:, i] > 0))[0].tolist()) for i in range(n)]


def shortest_hop(neigh: list[set[int]], src: int, dst: int) -> int:
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


def nearest_monitor_hop(neigh: list[set[int]], node_idx: int, monitor_indices: list[int]) -> int:
    hops = [shortest_hop(neigh, node_idx, m) for m in monitor_indices]
    hops = [h for h in hops if h >= 0]
    return min(hops) if hops else -1


def collect_node_set(loader) -> set[int]:
    base, indices = _get_subset_base_and_indices(loader.dataset)
    node_set = set()
    for i in indices:
        defect_idx = int(base.samples[i].get("defect_node_idx", -1))
        if defect_idx >= 0:
            node_set.add(defect_idx)
    return node_set


def main() -> None:
    rows_false_pred = []
    rows_false_pair = []
    rows_true_profile = []
    rows_seed_summary = []

    for run in RUNS:
        cfg = _build_config(
            subdir=SUBDIR,
            seed=run["seed"],
            monitor_nodes_file=MONITOR_FILE,
            split_mode="node_holdout",
            n_holdout_nodes=N_HOLDOUT,
        )
        cfg.model_type = "hydraulic_inverse_deepattn"

        train_loader, val_loader, test_loader, dataset = _make_loaders(cfg, observed_nodes_file=cfg.monitor_nodes_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _create_model(cfg, dataset, device)
        ckpt_path = CKPT_DIR / f"best_model_{run['tag']}.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        node_list = [str(n) for n in dataset.node_list]
        node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
        adj = np.asarray(dataset.adj_matrix)
        neigh = build_neighbors(adj)
        monitor_nodes = [str(n) for n in sorted(getattr(dataset, "observed_node_ids", set()))]
        monitor_indices = [node_to_idx[n] for n in monitor_nodes if n in node_to_idx]

        train_nodes = collect_node_set(train_loader)
        holdout_nodes = collect_node_set(test_loader)

        false_pred_counter = Counter()
        false_pair_counter = Counter()
        true_node_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "top5": 0})

        with torch.no_grad():
            for batch in test_loader:
                x = batch["features"].to(device)
                observed_mask = batch["observed_mask"].to(device).bool()
                x_masked = _mask_unobserved_features(x, observed_mask)

                batch_adj = batch["adj_matrix"]
                if isinstance(batch_adj, (list, tuple)):
                    batch_adj = batch_adj[0].to(device)
                elif batch_adj.dim() == 3:
                    batch_adj = batch_adj[0].to(device)
                else:
                    batch_adj = batch_adj.to(device)

                outputs = model(x_masked, batch_adj)
                logits_node = outputs[0] if isinstance(outputs, tuple) else outputs
                scores = logits_node[:, :, 1]

                active_label = batch["active_label"].long().to(device)
                target_node_idx = batch["target_node_idx"].long().to(device)
                candidate_mask = batch["candidate_mask"].to(device).bool()
                if candidate_mask.dim() == 1:
                    candidate_mask = candidate_mask.unsqueeze(0).expand(scores.shape[0], -1)

                valid_mask = target_node_idx >= 0
                valid_mask = valid_mask & (active_label > 0)
                if "loc_enabled" in batch:
                    valid_mask = valid_mask & (batch["loc_enabled"].long().to(device) > 0)
                valid_mask = valid_mask & candidate_mask[torch.arange(candidate_mask.size(0), device=device), target_node_idx]
                if valid_mask.sum().item() == 0:
                    continue

                scores = scores[valid_mask]
                target_node_idx = target_node_idx[valid_mask]
                candidate_mask = candidate_mask[valid_mask]
                scores = scores.masked_fill(~candidate_mask, -1e9)
                topk_idx = torch.argsort(scores, dim=1, descending=True)[:, :5].cpu().numpy()
                true_idx = target_node_idx.cpu().numpy()

                for pred_top5, true_node_idx in zip(topk_idx, true_idx):
                    true_node_idx = int(true_node_idx)
                    pred_top5 = pred_top5.tolist()
                    pred_top1 = int(pred_top5[0])
                    true_node_id = str(node_list[true_node_idx])
                    pred_node_id = str(node_list[pred_top1])

                    stats = true_node_stats[true_node_id]
                    stats["n"] += 1
                    stats["top1"] += int(pred_top1 == true_node_idx)
                    stats["top3"] += int(true_node_idx in pred_top5[:3])
                    stats["top5"] += int(true_node_idx in pred_top5[:5])

                    if pred_top1 != true_node_idx:
                        false_pred_counter[pred_node_id] += 1
                        false_pair_counter[(true_node_id, pred_node_id)] += 1

        for pred_node_id, count in false_pred_counter.most_common(20):
            pred_idx = node_to_idx[pred_node_id]
            rows_false_pred.append(
                {
                    "seed": run["seed"],
                    "pred_node_id": pred_node_id,
                    "false_top1_count": count,
                    "pred_nearest_monitor_hop": nearest_monitor_hop(neigh, pred_idx, monitor_indices),
                    "pred_in_train_nodes": int(pred_idx in train_nodes),
                    "pred_in_holdout_nodes": int(pred_idx in holdout_nodes),
                    "pred_is_monitor": int(pred_node_id in monitor_nodes),
                }
            )

        for (true_node_id, pred_node_id), count in false_pair_counter.most_common(40):
            true_idx = node_to_idx[true_node_id]
            pred_idx = node_to_idx[pred_node_id]
            rows_false_pair.append(
                {
                    "seed": run["seed"],
                    "true_node_id": true_node_id,
                    "pred_node_id": pred_node_id,
                    "count": count,
                    "true_nearest_monitor_hop": nearest_monitor_hop(neigh, true_idx, monitor_indices),
                    "pred_nearest_monitor_hop": nearest_monitor_hop(neigh, pred_idx, monitor_indices),
                    "true_pred_hop": shortest_hop(neigh, true_idx, pred_idx),
                    "pred_in_train_nodes": int(pred_idx in train_nodes),
                    "pred_is_monitor": int(pred_node_id in monitor_nodes),
                }
            )

        train_mean_hop = np.mean([nearest_monitor_hop(neigh, idx, monitor_indices) for idx in sorted(train_nodes)]) if train_nodes else -1.0
        holdout_mean_hop = np.mean([nearest_monitor_hop(neigh, idx, monitor_indices) for idx in sorted(holdout_nodes)]) if holdout_nodes else -1.0
        rows_seed_summary.append(
            {
                "seed": run["seed"],
                "n_train_nodes": len(train_nodes),
                "n_holdout_nodes": len(holdout_nodes),
                "train_mean_monitor_hop": float(train_mean_hop),
                "holdout_mean_monitor_hop": float(holdout_mean_hop),
            }
        )

        for true_node_id in sorted(true_node_stats.keys()):
            true_idx = node_to_idx[true_node_id]
            s = true_node_stats[true_node_id]
            n = max(int(s["n"]), 1)
            rows_true_profile.append(
                {
                    "seed": run["seed"],
                    "true_node_id": true_node_id,
                    "n_windows": int(s["n"]),
                    "top1": float(s["top1"] / n),
                    "top3": float(s["top3"] / n),
                    "top5": float(s["top5"] / n),
                    "true_nearest_monitor_hop": nearest_monitor_hop(neigh, true_idx, monitor_indices),
                    "true_is_monitor": int(true_node_id in monitor_nodes),
                    "is_holdout_node": int(true_idx in holdout_nodes),
                }
            )

    pd.DataFrame(rows_false_pred).to_csv(
        REPORT_DIR / "chapter1_restart_nodehold_false_pred_nodes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rows_false_pair).to_csv(
        REPORT_DIR / "chapter1_restart_nodehold_false_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rows_true_profile).to_csv(
        REPORT_DIR / "chapter1_restart_nodehold_true_node_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rows_seed_summary).to_csv(
        REPORT_DIR / "chapter1_restart_nodehold_seed_hop_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("saved -> chapter1_restart_nodehold_false_pred_nodes.csv")
    print("saved -> chapter1_restart_nodehold_false_pairs.csv")
    print("saved -> chapter1_restart_nodehold_true_node_profiles.csv")
    print("saved -> chapter1_restart_nodehold_seed_hop_summary.csv")


if __name__ == "__main__":
    main()
