#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose stage-1 shortlist quality for a trained localization model.

This script reloads the current dataset split and base checkpoint, collects
candidate probabilities on the test split, and reports:
  - overall Top-k inclusion rates of the true defect node
  - per-type shortlist recall
  - per-observability-tier shortlist recall
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.train_topk_reranker import (  # type: ignore
    _build_model,
    _collect_split_arrays,
    _create_loaders,
    _make_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage1 shortlist quality")
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--model-type", default="hydraulic_inverse_deepattn")
    parser.add_argument("--split-mode", default="scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", default="")
    parser.add_argument("--candidate-tier-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _compute_rank_matrix(probs: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    scores = np.array(probs, copy=True)
    scores[~candidate_mask] = -1.0
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]
    return ranks


def _summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    def _agg(sub: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_samples": int(len(sub)),
                "top1_rate": float(sub["in_top1"].mean()),
                "top3_rate": float(sub["in_top3"].mean()),
                "top5_rate": float(sub["in_top5"].mean()),
                "top10_rate": float(sub["in_top10"].mean()),
                "top20_rate": float(sub["in_top20"].mean()),
                "mean_rank": float(sub["true_rank"].mean()),
                "median_rank": float(sub["true_rank"].median()),
            }
        )

    return df.groupby(group_cols, dropna=False).apply(_agg).reset_index()


def main() -> None:
    args = _parse_args()
    cfg = _make_config(args)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, dataset = _create_loaders(cfg)
    model = _build_model(cfg, dataset, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    payload = _collect_split_arrays(model, test_loader, device)
    probs = payload["probs"]
    targets = payload["targets"]
    candidate_mask = payload["candidate_mask"]
    defect_types = np.asarray(payload.get("defect_type_str", ["UNK"] * len(targets)), dtype=object)

    node_list = list(dataset.node_list)
    true_nodes = np.asarray([str(node_list[int(idx)]) for idx in targets], dtype=object)

    ranks = _compute_rank_matrix(probs, candidate_mask)
    true_ranks = ranks[np.arange(len(targets)), targets]

    tier_df = pd.read_csv(args.candidate_tier_csv)
    tier_map = dict(zip(tier_df["candidate_node"].astype(str), tier_df["observability_tier"].astype(str)))
    hop_map = dict(zip(tier_df["candidate_node"].astype(str), tier_df["nearest_monitor_hop"]))

    df = pd.DataFrame(
        {
            "true_node": true_nodes,
            "defect_type": defect_types,
            "true_rank": true_ranks.astype(int),
            "observability_tier": [tier_map.get(n, "unknown") for n in true_nodes],
            "nearest_monitor_hop": [hop_map.get(n, np.nan) for n in true_nodes],
        }
    )
    for k in (1, 3, 5, 10, 20):
        df[f"in_top{k}"] = (df["true_rank"] <= k).astype(int)

    overall = {
        "n_samples": int(len(df)),
        "top1_rate": float(df["in_top1"].mean()),
        "top3_rate": float(df["in_top3"].mean()),
        "top5_rate": float(df["in_top5"].mean()),
        "top10_rate": float(df["in_top10"].mean()),
        "top20_rate": float(df["in_top20"].mean()),
        "mean_rank": float(df["true_rank"].mean()),
        "median_rank": float(df["true_rank"].median()),
    }

    by_type = _summarize_group(df, ["defect_type"])
    by_tier = _summarize_group(df, ["observability_tier"])
    by_tier_type = _summarize_group(df, ["observability_tier", "defect_type"])

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_name(out_prefix.name + "_samples.csv"), index=False)
    by_type.to_csv(out_prefix.with_name(out_prefix.name + "_by_type.csv"), index=False)
    by_tier.to_csv(out_prefix.with_name(out_prefix.name + "_by_tier.csv"), index=False)
    by_tier_type.to_csv(out_prefix.with_name(out_prefix.name + "_by_tier_type.csv"), index=False)

    with open(out_prefix.with_name(out_prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "subdir": args.subdir,
                "model_type": args.model_type,
                "split_mode": args.split_mode,
                "seed": int(args.seed),
                "checkpoint": str(args.checkpoint),
                "candidate_tier_csv": str(args.candidate_tier_csv),
                "overall": overall,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
