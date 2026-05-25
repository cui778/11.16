"""
模型评估工具（patched v2）

目标：
- 支持“候选集合上的单标签定位”作为主任务（CrossEntropy over nodes with candidate mask）
- 输出更符合定位任务的指标：MRR、Top-K、k-hop、距离分布
- 支持按 scenario_id 分组统计（窗口级别）
- 预测案例分析使用 softmax 概率（对候选节点归一化），避免出现“0.99 概率同时有很多个”的假象
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# 只依赖 patched 指标实现（含 khop / distance_distribution 的 index 版）
from .metrics_patched import (
    compute_khop_hit_from_indices,
    compute_distance_distribution_from_indices,
)

logger = logging.getLogger(__name__)


def _ensure_adj_2d(adj_matrix: torch.Tensor) -> torch.Tensor:
    """兼容 dataloader 返回 [B,N,N] / [N,N] / list/tuple 的情况，统一为 [N,N]."""
    if isinstance(adj_matrix, (list, tuple)):
        adj = adj_matrix[0]
    else:
        adj = adj_matrix

    if adj.dim() == 3:
        adj = adj[0]
    return adj


def evaluate_model(model,
                   dataloader,
                   criterion_node,
                   device,
                   khop_values: List[int] = [0, 1, 2],
                   topk_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    """
    完整的模型评估（兼容两种任务）：
      1) 旧版：节点级二分类（batch里只有 node_labels）
      2) 新版：候选集合上的单标签定位（batch里包含 target_node_idx + candidate_mask）

    说明：
      - 如果 batch 里存在 target_node_idx 且其值 >=0，则默认走“定位任务”分支。
      - 定位任务的 logits 期望为 [B, N, 2]，使用 class=1 的 logit 作为节点得分。
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    # --------- 新版定位的累积量 ---------
    n_samples = 0
    topk_hit = {k: 0 for k in topk_values}
    mrr_sum = 0.0
    khop_hit = {k: 0 for k in khop_values}
    distance_distribution_total: Dict[Any, int] = {}

    # --------- 按 scenario 分组（定位任务）---------
    scenario_stats: Dict[str, Dict[str, float]] = {}

    # --------- 旧版二分类（保留接口，避免你旧脚本报错）---------
    # 这里不再输出 sklearn 的一堆分类报告，避免与你当前定位任务混淆
    legacy_mode_seen = False

    with torch.no_grad():
        for batch in dataloader:
            x = batch['features'].to(device)

            adj = _ensure_adj_2d(batch['adj_matrix']).to(device)
            logits = model(x, adj)

            # 判断任务类型
            is_localization = ('target_node_idx' in batch) and ('candidate_mask' in batch)

            if is_localization:
                y_idx = batch['target_node_idx'].long().to(device)  # [B]
                # 过滤掉无效样本（例如 baseline 或标注缺失）
                valid_mask = (y_idx >= 0)

                # 对齐 batch 维度
                scores = logits[:, :, 1]  # [B, N]
                cand_mask = batch['candidate_mask'].to(device).bool()  # [B, N] or [N]
                if cand_mask.dim() == 1:
                    cand_mask = cand_mask.unsqueeze(0).expand(scores.size(0), -1)

                # 进一步要求：真实缺陷节点必须在候选集合内（否则定位任务无意义）
                in_cand = cand_mask[torch.arange(cand_mask.size(0), device=device), y_idx].bool()
                if (valid_mask & (~in_cand)).any():
                    # 只提示一次即可；多数情况下说明候选点集生成不包含缺陷点
                    logger.warning("[evaluate] 发现真实缺陷节点不在 candidate_mask 内，已跳过这些样本。")
                valid_mask = valid_mask & in_cand
                if valid_mask.sum().item() == 0:
                    continue


                # 只取有效样本
                scores = scores[valid_mask]
                y_idx = y_idx[valid_mask]
                cand_mask = cand_mask[valid_mask]

                # 候选外置极小值（避免 -inf 在 softmax/排序时产生数值坑）
                scores_masked = scores.masked_fill(~cand_mask, -1e9)

                # loss（CrossEntropy over nodes）
                loss = criterion_node(scores_masked, y_idx)
                total_loss += float(loss.item())
                total_batches += 1

                B = scores_masked.size(0)
                n_samples += B

                # 排序得到 rank / TopK / MRR
                sorted_idx = torch.argsort(scores_masked, dim=1, descending=True)  # [B, N]
                # pos[b] = 真实节点在排序中的位置（0-based）
                pos = (sorted_idx == y_idx.unsqueeze(1)).nonzero(as_tuple=False)
                # 保险：按 batch 排序
                pos = pos[pos[:, 0].argsort()]
                ranks = pos[:, 1] + 1  # 1-based
                mrr_sum += float(torch.mean(1.0 / ranks.float()).item())

                for k in topk_values:
                    topk_hit[k] += int((ranks <= k).sum().item())

                # 预测节点（候选内 argmax）
                pred_idx = sorted_idx[:, 0].detach().cpu().numpy()
                true_idx = y_idx.detach().cpu().numpy()

                adj_np = adj.detach().cpu().numpy()

                # k-hop / distance distribution：逐样本统计
                for b in range(B):
                    p = int(pred_idx[b])
                    t = int(true_idx[b])

                    for k in khop_values:
                        khop_hit[k] += int(compute_khop_hit_from_indices(p, t, adj_np, k_hops=int(k)))

                    dd = compute_distance_distribution_from_indices(p, t, adj_np)
                    for key, val in dd.items():
                        distance_distribution_total[key] = distance_distribution_total.get(key, 0) + int(val)

                # ---------- scenario 分组 ----------
                if 'scenario_id' in batch:
                    scenario_ids = batch['scenario_id']
                    # collate 后可能是 list[str] 或 np.ndarray/object
                    if isinstance(scenario_ids, (list, tuple)):
                        scenario_ids_list = list(scenario_ids)
                    elif isinstance(scenario_ids, torch.Tensor):
                        # 处理 tensor 类型的 scenario_id
                        scenario_ids_list = scenario_ids.tolist()
                    else:
                        # 单个值或张量（极少见）
                        scenario_ids_list = [str(scenario_ids)] * B

                    # 注意：我们对 valid_mask 做了筛选，所以也要同步筛选 scenario_id
                    scenario_ids_list = [scenario_ids_list[i] for i, ok in enumerate(valid_mask.cpu().tolist()) if ok]

                    for b in range(B):
                        sc = str(scenario_ids_list[b])
                        if sc not in scenario_stats:
                            scenario_stats[sc] = {
                                'n': 0,
                                'mrr_sum': 0.0,
                                **{f'hit@{k}': 0 for k in topk_values}
                            }
                        scenario_stats[sc]['n'] += 1
                        scenario_stats[sc]['mrr_sum'] += float(1.0 / float(ranks[b].item()))
                        for k in topk_values:
                            scenario_stats[sc][f'hit@{k}'] += int(ranks[b].item() <= k)

            else:
                # 旧版二分类：保留入口以防旧脚本调用，但不再作为主口径输出
                legacy_mode_seen = True
                # 如果你还需要旧版评估，可在这里扩展；目前不建议混用

    metrics: Dict[str, Any] = {}

    if n_samples > 0:
        metrics['loss'] = (total_loss / max(total_batches, 1))
        metrics['mrr'] = (mrr_sum / n_samples)

        for k in topk_values:
            metrics[f'topk_recall_{k}'] = topk_hit[k] / n_samples

        for k in khop_values:
            metrics[f'khop_recall_{k}'] = khop_hit[k] / n_samples

        metrics['distance_distribution'] = distance_distribution_total

        # scenario 汇总（窗口级别）
        scenario_breakdown = {}
        for sc, d in scenario_stats.items():
            n = max(int(d['n']), 1)
            scenario_breakdown[sc] = {
                'n_windows': int(d['n']),
                'mrr': float(d['mrr_sum'] / n),
                **{f'top{k}': float(d[f'hit@{k}'] / n) for k in topk_values},
            }
        metrics['by_scenario'] = scenario_breakdown

    else:
        # 没有定位样本（通常说明你的 batch 没有 target_node_idx 或全是 -1）
        metrics['loss'] = float('nan')
        metrics['mrr'] = 0.0
        for k in topk_values:
            metrics[f'topk_recall_{k}'] = 0.0
        for k in khop_values:
            metrics[f'khop_recall_{k}'] = 0.0
        metrics['distance_distribution'] = {}
        metrics['by_scenario'] = {}

    if legacy_mode_seen:
        metrics['legacy_mode_seen'] = True

    return metrics


def analyze_predictions(model,
                        dataloader,
                        device,
                        num_samples: int = 5):
    """
    预测案例分析（定位任务优先）：
    - 使用 softmax(scores_masked) 作为候选集合内的概率
    - 打印：scenario_id、真实缺陷节点、Top-5预测（节点idx/节点ID/概率）、真实节点排名与概率
    """
    model.eval()

    # 尝试从 dataset 里拿 node_id 映射
    node_list: Optional[List[str]] = None
    try:
        ds = dataloader.dataset
        if hasattr(ds, 'node_list') and isinstance(ds.node_list, list):
            node_list = ds.node_list
    except Exception:
        node_list = None

    logger.info("\n" + "=" * 80)
    logger.info("预测案例分析（Top-5预测 vs 真实缺陷）")
    logger.info("=" * 80)

    shown = 0
    with torch.no_grad():
        for batch in dataloader:
            if shown >= num_samples:
                break

            if ('target_node_idx' not in batch) or ('candidate_mask' not in batch):
                logger.info("  当前 dataloader batch 不包含 target_node_idx / candidate_mask，跳过案例分析。")
                return

            x = batch['features'].to(device)
            adj = _ensure_adj_2d(batch['adj_matrix']).to(device)
            logits = model(x, adj)  # [B,N,2]

            scores = logits[:, :, 1]  # [B,N]
            cand_mask = batch['candidate_mask'].to(device).bool()
            if cand_mask.dim() == 1:
                cand_mask = cand_mask.unsqueeze(0).expand(scores.size(0), -1)

            y_idx = batch['target_node_idx'].long().to(device)
            valid_mask = (y_idx >= 0)

            # softmax 只对候选集合
            scores_masked = scores.masked_fill(~cand_mask, -1e9)
            probs = torch.softmax(scores_masked, dim=1)  # [B,N]

            scenario_ids = batch.get('scenario_id', None)
            defect_node_ids = batch.get('defect_node_id', None)

            B = scores.size(0)
            for b in range(B):
                if shown >= num_samples:
                    break
                if not bool(valid_mask[b].item()):
                    continue

                true_i = int(y_idx[b].item())
                if isinstance(scenario_ids, (list, tuple)):
                    sc = str(scenario_ids[b])
                elif isinstance(scenario_ids, torch.Tensor):
                    sc = str(scenario_ids[b].item())
                else:
                    sc = str(scenario_ids) if scenario_ids is not None else "N/A"
                true_node_id = None
                if node_list is not None and 0 <= true_i < len(node_list):
                    true_node_id = node_list[true_i]

                logger.info(f"\n样本 {shown + 1}:")
                logger.info(f"  scenario_id: {sc}")
                if defect_node_ids is not None:
                    try:
                        logger.info(f"  缺陷节点ID(原始): {defect_node_ids[b]}")
                    except Exception:
                        pass
                logger.info(f"  真实缺陷节点: idx={true_i}" + (f", id={true_node_id}" if true_node_id else ""))

                # Top-5
                topk = 5
                top_probs, top_idx = torch.topk(probs[b], k=topk)
                logger.info("  Top-5 预测:")
                for rank in range(topk):
                    ni = int(top_idx[rank].item())
                    p = float(top_probs[rank].item())
                    nid = None
                    if node_list is not None and 0 <= ni < len(node_list):
                        nid = node_list[ni]
                    ok = "✅" if ni == true_i else "❌"
                    logger.info(f"    #{rank+1}: 节点 {ni}" + (f"({nid})" if nid else "") + f" (概率={p:.4f}) {ok}")

                # 真实节点排名
                sorted_idx = torch.argsort(probs[b], descending=True)
                true_rank = int((sorted_idx == true_i).nonzero(as_tuple=False)[0, 0].item()) + 1
                true_prob = float(probs[b, true_i].item())
                logger.info(f"  真实节点排名: #{true_rank} (概率={true_prob:.4f})")

                shown += 1
            if shown >= num_samples:
                break




