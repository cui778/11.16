# -*- coding: utf-8 -*-
"""
模型评估工具。来源：原 script2/utils/evaluation_patched.py
支持候选集上的单标签定位、MRR/Top-K/k-hop、按见过/未见过节点与 I/E/P 拆分等。
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# 只依赖指标实现（含 khop / distance_distribution 的 index 版）
from .metrics import (
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
                   criterion_loc,  # 应传入 KLDivLoss（与训练一致）
                   device,
                   criterion_active=None,
                   active_prob_threshold: Optional[float] = None,
                   node_logit_bias: Optional[torch.Tensor] = None,
                   khop_values: List[int] = [0, 1, 2],
                   topk_values: List[int] = [1, 3, 5],
                   train_seen_node_indices: Optional[set] = None) -> Dict[str, Any]:
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

    # --------- 定位子集：active_label==1 的累积量 ---------
    n_samples_active = 0
    topk_hit_active = {k: 0 for k in topk_values}
    mrr_sum_active = 0.0

    # --------- active 二分类（全过程检测） ---------
    active_loss_sum = 0.0
    active_correct = 0
    active_samples = 0
    active_true_positive = 0
    active_pred_positive = 0
    active_actual_positive = 0

    # --------- 按 scenario 分组（定位任务）---------
    scenario_stats: Dict[str, Dict[str, float]] = {}
    scenario_windows: Dict[str, List[Dict[str, Any]]] = {}
    scenario_event_probs: Dict[str, List[np.ndarray]] = {}
    scenario_event_targets: Dict[str, int] = {}

    # --------- Task4：按“见过/未见过节点”、按缺陷类型 I/E/P 收集 rank --------
    per_sample_ranks: List[int] = []
    per_sample_target_idx: List[int] = []
    per_sample_defect_type: List[str] = []
    per_sample_scenario_id: List[Any] = []

    # --------- 旧版二分类（保留接口，避免你旧脚本报错）---------
    # 这里不再输出 sklearn 的一堆分类报告，避免与你当前定位任务混淆
    legacy_mode_seen = False

    with torch.no_grad():
        for batch in dataloader:
            x = batch['features'].to(device)

            adj = _ensure_adj_2d(batch['adj_matrix']).to(device)
            logits = model(x, adj)
            if isinstance(logits, tuple):
                logits_node = logits[0]
                logits_active = logits[1] if len(logits) > 1 else None
            else:
                logits_node = logits
                logits_active = None

            if logits_active is not None and ('active_label' in batch or 'has_defect' in batch):
                active_target_batch = batch['active_label'].long().to(device) if 'active_label' in batch else batch['has_defect'].long().to(device)
                if criterion_active is None:
                    criterion_active = torch.nn.CrossEntropyLoss(reduction='mean')
                active_loss = criterion_active(logits_active, active_target_batch)
                active_loss_sum += float(active_loss.item()) * int(active_target_batch.numel())
                if active_prob_threshold is not None:
                    active_prob = torch.softmax(logits_active, dim=1)[:, 1]
                    active_pred = (active_prob >= float(active_prob_threshold)).long()
                else:
                    active_pred = torch.argmax(logits_active, dim=1)
                active_correct += int((active_pred == active_target_batch).sum().item())
                active_samples += int(active_target_batch.numel())
                active_true_positive += int(((active_pred == 1) & (active_target_batch == 1)).sum().item())
                active_pred_positive += int((active_pred == 1).sum().item())
                active_actual_positive += int((active_target_batch == 1).sum().item())
            else:
                active_target_batch = None
                active_pred = None

            # 判断任务类型
            is_localization = ('target_node_idx' in batch) and ('candidate_mask' in batch)

            if is_localization:
                y_idx = batch['target_node_idx'].long().to(device)  # [B]
                # 过滤掉无效样本（例如 baseline 或标注缺失）
                valid_mask = (y_idx >= 0)

                active_mask = None
                if active_target_batch is not None:
                    active_mask = (active_target_batch > 0)
                elif 'active_label' in batch:
                    active_mask = (batch['active_label'].long().to(device) > 0)
                elif 'has_defect' in batch:
                    active_mask = (batch['has_defect'].long().to(device) > 0)

                loc_enabled_mask = None
                if 'loc_enabled' in batch:
                    loc_enabled_mask = (batch['loc_enabled'].long().to(device) > 0)

                # 对齐 batch 维度
                scores = logits_node[:, :, 1]  # [B, N]
                if node_logit_bias is not None:
                    bias = node_logit_bias.to(device)
                    if bias.dim() == 1:
                        scores = scores + bias.unsqueeze(0)
                    else:
                        scores = scores + bias
                cand_mask = batch['candidate_mask'].to(device).bool()  # [B, N] or [N]
                if cand_mask.dim() == 1:
                    cand_mask = cand_mask.unsqueeze(0).expand(scores.size(0), -1)

                if active_mask is not None:
                    valid_mask = valid_mask & active_mask
                if loc_enabled_mask is not None:
                    valid_mask = valid_mask & loc_enabled_mask

                # 过程级时间轴记录：基于所有窗口的 active 预测
                if active_target_batch is not None and active_pred is not None and 'scenario_id' in batch and 'window_start_time' in batch:
                    scenario_ids_raw = batch['scenario_id']
                    if isinstance(scenario_ids_raw, torch.Tensor):
                        scenario_ids_full = scenario_ids_raw.view(-1).cpu().tolist()
                    elif isinstance(scenario_ids_raw, (list, tuple)):
                        scenario_ids_full = list(scenario_ids_raw)
                    else:
                        scenario_ids_full = [scenario_ids_raw] * int(scores.size(0))

                    window_times_raw = batch['window_start_time']
                    if isinstance(window_times_raw, (list, tuple)):
                        window_times_full = list(window_times_raw)
                    else:
                        window_times_full = [window_times_raw] * int(scores.size(0))

                    for i in range(int(scores.size(0))):
                        sc_key = str(scenario_ids_full[i])
                        scenario_windows.setdefault(sc_key, []).append({
                            'window_start_time': str(window_times_full[i]),
                            'true_active': int(active_target_batch[i].item()),
                            'pred_active': int(active_pred[i].item()),
                        })

                # 进一步要求：真实缺陷节点必须在候选集合内（否则定位任务无意义）
                in_cand = cand_mask[torch.arange(cand_mask.size(0), device=device), y_idx].bool()
                if (valid_mask & (~in_cand)).any():
                    # 只提示一次即可；多数情况下说明候选点集生成不包含缺陷点
                    logger.warning("[evaluate] 发现真实缺陷节点不在 candidate_mask 内，已跳过这些样本。")
                valid_mask = valid_mask & in_cand
                if valid_mask.sum().item() == 0:
                    continue

                # 只取有效样本
                scores_valid = scores[valid_mask]
                y_idx_valid = y_idx[valid_mask]
                cand_mask_valid = cand_mask[valid_mask]

                soft_labels_valid = None
                if 'soft_label' in batch:
                    soft_labels_valid = batch['soft_label'].to(device)[valid_mask]

                # loss（与训练保持一致：KL散度 + 软标签）
                if soft_labels_valid is not None:
                    # 候选外置极小值（避免 -inf 在 log_softmax 时产生数值坑）
                    scores_masked = scores_valid.masked_fill(~cand_mask_valid, -1e9)
                    soft_labels_masked = soft_labels_valid.masked_fill(~cand_mask_valid, 0.0)
                    # 归一化软标签（仅在候选集内）
                    soft_sum = soft_labels_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    soft_labels_norm = soft_labels_masked / soft_sum

                    log_probs = torch.log_softmax(scores_masked, dim=-1)
                    loss = criterion_loc(log_probs, soft_labels_norm)
                else:
                    # 回退：硬标签 + CE（兼容旧数据）
                    scores_masked = scores_valid.masked_fill(~cand_mask_valid, -1e9)
                    loss = criterion_loc(scores_masked, y_idx_valid)
                total_loss += float(loss.item())
                total_batches += 1

                B = scores_masked.size(0)
                n_samples += B

                # 排序得到 rank / TopK / MRR
                sorted_idx = torch.argsort(scores_masked, dim=1, descending=True)  # [B, N]
                # pos[b] = 真实节点在排序中的位置（0-based）
                pos = (sorted_idx == y_idx_valid.unsqueeze(1)).nonzero(as_tuple=False)
                # 保险：按 batch 排序
                pos = pos[pos[:, 0].argsort()]
                ranks = pos[:, 1] + 1  # 1-based
                mrr_sum += float(torch.sum(1.0 / ranks.float()).item())

                n_samples_active += B
                mrr_sum_active += float(torch.sum(1.0 / ranks.float()).item())

                for k in topk_values:
                    topk_hit[k] += int((ranks <= k).sum().item())
                    topk_hit_active[k] += int((ranks <= k).sum().item())

                # 预测节点（候选内 argmax）
                pred_idx = sorted_idx[:, 0].detach().cpu().numpy()
                true_idx = y_idx_valid.detach().cpu().numpy()
                probs_valid = torch.softmax(scores_masked, dim=1).detach().cpu().numpy()

                adj_np = adj.detach().cpu().numpy()

                # k-hop / distance distribution：逐样本统计
                for b in range(B):
                    p = int(pred_idx[b])
                    t = int(true_idx[b])

                    for k in khop_values:
                        khop_hit[k] += int(compute_khop_hit_from_indices(
                            p, t, adj_np, k_hops=int(k), undirected=True
                        ))

                    dd = compute_distance_distribution_from_indices(p, t, adj_np, undirected=True)
                    for key, val in dd.items():
                        distance_distribution_total[key] = distance_distribution_total.get(key, 0) + int(val)

                # ---------- scenario 分组 ----------
                if 'scenario_id' in batch:
                    scenario_ids_raw = batch['scenario_id']

                    # 统一转换为长度为 batch_size 的 list[int]
                    if isinstance(scenario_ids_raw, torch.Tensor):
                        scenario_ids_list = scenario_ids_raw.view(-1).cpu().tolist()
                    elif isinstance(scenario_ids_raw, (list, tuple)):
                        scenario_ids_list = [int(s) if str(s).isdigit() else str(s) for s in scenario_ids_raw]
                    else:
                        # 标量或单值，重复扩展
                        scenario_ids_list = [scenario_ids_raw] * scores.size(0)
                        scenario_ids_list = [int(s) if str(s).isdigit() else str(s) for s in scenario_ids_list]

                    # 同步 valid_mask 过滤
                    scenario_ids_list = [scenario_ids_list[i] for i, ok in enumerate(valid_mask.cpu().tolist()) if ok]

                    # defect_type_str 同长度过滤（用于 Task4 按类型统计）
                    defect_type_list = [None] * B
                    if 'defect_type_str' in batch:
                        raw = batch['defect_type_str']
                        vm = valid_mask.cpu().tolist()
                        if isinstance(raw, (list, tuple)):
                            defect_type_list = [str(raw[i]) for i, ok in enumerate(vm) if ok]
                        elif hasattr(raw, 'view'):
                            defect_type_list = [str(raw.view(-1).cpu().tolist()[i]) for i, ok in enumerate(vm) if ok]
                        else:
                            defect_type_list = [str(raw)] * B

                    for b in range(B):
                        sc = scenario_ids_list[b]
                        sc_key = str(sc)
                        if sc_key not in scenario_stats:
                            scenario_stats[sc_key] = {
                                'n': 0,
                                'mrr_sum': 0.0,
                                **{f'hit@{k}': 0 for k in topk_values}
                            }
                        scenario_stats[sc_key]['n'] += 1
                        scenario_stats[sc_key]['mrr_sum'] += float(1.0 / float(ranks[b].item()))
                        for k in topk_values:
                            scenario_stats[sc_key][f'hit@{k}'] += int(ranks[b].item() <= k)

                        # Task4：收集每样本 rank / target_idx / defect_type / scenario_id
                        per_sample_ranks.append(int(ranks[b].item()))
                        per_sample_target_idx.append(int(y_idx_valid[b].item()))
                        per_sample_defect_type.append(defect_type_list[b] if defect_type_list[b] else '')
                        per_sample_scenario_id.append(sc)
                        scenario_event_probs.setdefault(sc_key, []).append(probs_valid[b])
                        scenario_event_targets[sc_key] = int(y_idx_valid[b].item())

            else:
                # 旧版二分类：保留入口以防旧脚本调用，但不再作为主口径输出
                legacy_mode_seen = True
                # 如果你还需要旧版评估，可在这里扩展；目前不建议混用

    # --------- Task4：按“见过/未见过节点”、按缺陷类型 I/E/P 汇总指标 ---------
    seen_nodes_metrics = {}
    unseen_nodes_metrics = {}
    by_type_metrics: Dict[str, Dict[str, Any]] = {}

    if per_sample_ranks and (train_seen_node_indices is not None or any(per_sample_defect_type)):
        ranks_arr = np.array(per_sample_ranks)
        target_arr = np.array(per_sample_target_idx)
        type_arr = np.array(per_sample_defect_type)

        def _agg(r: np.ndarray) -> Dict[str, Any]:
            if len(r) == 0:
                return {'n': 0, 'mrr': 0.0, **{f'top{k}': 0.0 for k in topk_values}}
            return {
                'n': int(len(r)),
                'mrr': float(np.mean(1.0 / r)),
                **{f'top{k}': float(np.mean(r <= k)) for k in topk_values}
            }

        if train_seen_node_indices is not None:
            seen_mask = np.array([t in train_seen_node_indices for t in per_sample_target_idx])
            unseen_mask = ~seen_mask
            if seen_mask.any():
                seen_nodes_metrics = _agg(ranks_arr[seen_mask])
            else:
                seen_nodes_metrics = {'n': 0, 'mrr': 0.0, **{f'top{k}': 0.0 for k in topk_values}}
            if unseen_mask.any():
                unseen_nodes_metrics = _agg(ranks_arr[unseen_mask])
            else:
                unseen_nodes_metrics = {'n': 0, 'mrr': 0.0, **{f'top{k}': 0.0 for k in topk_values}}

        for defect_type in ('I', 'E', 'P'):
            mask = (type_arr == defect_type)
            if mask.any():
                by_type_metrics[defect_type] = _agg(ranks_arr[mask])
            else:
                by_type_metrics[defect_type] = {'n': 0, 'mrr': 0.0, **{f'top{k}': 0.0 for k in topk_values}}

    # 原有指标计算
    metrics: Dict[str, Any] = {}

    if n_samples > 0:
        metrics['loss'] = (total_loss / max(total_batches, 1))
        metrics['mrr'] = (mrr_sum / n_samples)

        for k in topk_values:
            metrics[f'topk_recall_{k}'] = topk_hit[k] / n_samples

        for k in khop_values:
            metrics[f'khop_recall_{k}'] = khop_hit[k] / n_samples

        metrics['distance_distribution'] = distance_distribution_total

        if n_samples_active > 0:
            metrics['mrr_has_defect'] = (mrr_sum_active / n_samples_active)
            for k in topk_values:
                metrics[f'topk_recall_{k}_has_defect'] = topk_hit_active[k] / n_samples_active
        else:
            metrics['mrr_has_defect'] = 0.0
            for k in topk_values:
                metrics[f'topk_recall_{k}_has_defect'] = 0.0

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

        # Task4：见过/未见过节点、按类型 I/E/P
        metrics['seen_nodes'] = seen_nodes_metrics
        metrics['unseen_nodes'] = unseen_nodes_metrics
        metrics['by_defect_type'] = by_type_metrics

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
        metrics['seen_nodes'] = {}
        metrics['unseen_nodes'] = {}
        metrics['by_defect_type'] = {}

    if active_samples > 0:
        active_acc = active_correct / max(1, active_samples)
        active_precision = active_true_positive / max(1, active_pred_positive)
        active_recall = active_true_positive / max(1, active_actual_positive)
        metrics['active_loss'] = active_loss_sum / max(1, active_samples)
        metrics['active_acc'] = active_acc
        metrics['active_precision'] = active_precision
        metrics['active_period_recall'] = active_recall
        metrics['cls_loss'] = metrics['active_loss']
        metrics['cls_acc'] = active_acc
    else:
        metrics['active_loss'] = float('nan')
        metrics['active_acc'] = 0.0
        metrics['active_precision'] = 0.0
        metrics['active_period_recall'] = 0.0
        metrics['cls_loss'] = float('nan')
        metrics['cls_acc'] = 0.0

    detection_latencies = []
    event_level_hits = {k: 0 for k in topk_values}
    n_events = 0

    for sc_key, windows in scenario_windows.items():
        if not windows:
            continue
        ordered = sorted(windows, key=lambda item: item['window_start_time'])
        active_truth = [w for w in ordered if int(w.get('true_active', 0)) == 1]
        active_detected = [w for w in ordered if int(w.get('true_active', 0)) == 1 and int(w.get('pred_active', 0)) == 1]
        if active_truth and active_detected:
            t0 = np.datetime64(active_truth[0]['window_start_time'])
            td = np.datetime64(active_detected[0]['window_start_time'])
            latency_hours = float((td - t0) / np.timedelta64(1, 'h'))
            detection_latencies.append(latency_hours)

        probs_list = scenario_event_probs.get(sc_key, [])
        target_idx = scenario_event_targets.get(sc_key, None)
        if probs_list and target_idx is not None:
            agg_prob = np.mean(np.asarray(probs_list), axis=0)
            rank = int(np.where(np.argsort(-agg_prob) == int(target_idx))[0][0]) + 1
            n_events += 1
            for k in topk_values:
                event_level_hits[k] += int(rank <= k)

    metrics['detection_latency_mean'] = float(np.mean(detection_latencies)) if detection_latencies else float('nan')
    metrics['detection_latency_count'] = int(len(detection_latencies))
    metrics['event_level_top1'] = event_level_hits.get(1, 0) / max(1, n_events)
    metrics['event_level_top3'] = event_level_hits.get(3, 0) / max(1, n_events)
    metrics['event_level_top5'] = event_level_hits.get(5, 0) / max(1, n_events)
    metrics['event_level_count'] = int(n_events)

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
            outputs = model(x, adj)  # [B,N,2] or tuple
            if isinstance(outputs, tuple):
                logits_node = outputs[0]
            else:
                logits_node = outputs

            scores = logits_node[:, :, 1]  # [B,N]
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
                sc = str(scenario_ids[b]) if isinstance(scenario_ids, (list, tuple)) else (
                    str(scenario_ids) if scenario_ids is not None else "N/A")
                true_node_id = None
                if node_list is not None and 0 <= true_i < len(node_list):
                    true_node_id = node_list[true_i]

                logger.info(f"\n样本 {shown + 1}:")
                # 打印 scenario_id 为标量，避免 tensor 列表长串
                try:
                    if isinstance(scenario_ids, torch.Tensor):
                        sc_val = scenario_ids.view(-1).cpu().tolist()
                        sc_val = sc_val[b] if b < len(sc_val) else sc
                    elif isinstance(scenario_ids, (list, tuple)):
                        sc_val = scenario_ids[b] if b < len(scenario_ids) else sc
                    else:
                        sc_val = sc
                except Exception:
                    sc_val = sc
                logger.info(f"  scenario_id: {sc_val}")
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
                    logger.info(f"    #{rank + 1}: 节点 {ni}" + (f"({nid})" if nid else "") + f" (概率={p:.4f}) {ok}")

                # 真实节点排名
                sorted_idx = torch.argsort(probs[b], descending=True)
                true_rank = int((sorted_idx == true_i).nonzero(as_tuple=False)[0, 0].item()) + 1
                true_prob = float(probs[b, true_i].item())
                logger.info(f"  真实节点排名: #{true_rank} (概率={true_prob:.4f})")

                shown += 1
            if shown >= num_samples:
                break
