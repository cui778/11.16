#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate stage-1 localization under observability-based candidate subsets.

This script reloads the base model and test split, then compares:
  - all 50 candidates
  - direct+near candidates
  - near-only candidates
for the current test windows.
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

from scripts.train_topk_reranker import _build_model, _collect_split_arrays, _create_loaders, _make_config  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--model-type", default="hydraulic_inverse_deepattn")
    parser.add_argument("--split-mode", default="scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--candidate-tier-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_tier_masks(candidate_tier_csv: str, node_list: List[str]) -> Dict[str, np.ndarray]:
    df = pd.read_csv(candidate_tier_csv)
    tier_map = dict(zip(df["candidate_node"].astype(str), df["observability_tier"].astype(str)))
    tiers = np.asarray([tier_map.get(str(node), "non_candidate") for node in node_list], dtype=object)
    masks = {
        "all50": tiers != "non_candidate",
        "direct_near": np.isin(tiers, ["direct", "near"]),
        "near_only": tiers == "near",
        "direct_only": tiers == "direct",
    }
    return masks


def _safe_mrr(ranks: np.ndarray) -> float:
    finite = ranks[np.isfinite(ranks)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(1.0 / finite))


def _evaluate_subset(
    probs: np.ndarray,
    targets: np.ndarray,
    candidate_mask: np.ndarray,
    allowed_mask: np.ndarray,
) -> Dict[str, float]:
    allowed_mask = allowed_mask.astype(bool)[None, :]
    eval_mask = candidate_mask & allowed_mask
    scores = np.where(eval_mask, probs, -1.0)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int64)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]

    target_allowed = eval_mask[np.arange(eval_mask.shape[0]), targets]
    true_rank = np.full(targets.shape[0], np.inf, dtype=np.float64)
    true_rank[target_allowed] = ranks[np.arange(ranks.shape[0]), targets][target_allowed].astype(np.float64)

    covered = target_allowed.astype(np.float32)
    return {
        "n_samples": int(len(targets)),
        "coverage": float(covered.mean()),
        "covered_samples": int(target_allowed.sum()),
        "top1_all": float(np.mean(true_rank <= 1)),
        "top3_all": float(np.mean(true_rank <= 3)),
        "top5_all": float(np.mean(true_rank <= 5)),
        "top10_all": float(np.mean(true_rank <= 10)),
        "mrr_all": _safe_mrr(true_rank),
        "top1_covered": float(np.mean((true_rank[target_allowed] <= 1))) if target_allowed.any() else 0.0,
        "top3_covered": float(np.mean((true_rank[target_allowed] <= 3))) if target_allowed.any() else 0.0,
        "top5_covered": float(np.mean((true_rank[target_allowed] <= 5))) if target_allowed.any() else 0.0,
        "top10_covered": float(np.mean((true_rank[target_allowed] <= 10))) if target_allowed.any() else 0.0,
        "mrr_covered": _safe_mrr(true_rank[target_allowed]),
        "mean_rank_covered": float(np.mean(true_rank[target_allowed])) if target_allowed.any() else 0.0,
    }


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

    tier_masks = _load_tier_masks(args.candidate_tier_csv, list(dataset.node_list))
    rows = []
    for subset_name, allowed_mask in tier_masks.items():
        row = {"subset": subset_name}
        row.update(_evaluate_subset(probs, targets, candidate_mask, allowed_mask))
        rows.append(row)

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "subdir": args.subdir,
                "model_type": args.model_type,
                "split_mode": args.split_mode,
                "seed": int(args.seed),
                "checkpoint": str(args.checkpoint),
                "candidate_tier_csv": args.candidate_tier_csv,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
