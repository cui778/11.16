#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测模型训练脚本

功能：
- 加载数据集
- 训练模型
- 验证和测试
- 保存最佳模型
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

# 添加 script2_new 根目录到路径，以便导入 config、models、utils
SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from models.anomaly_detection_model import create_model
from utils.evaluation import evaluate_model, analyze_predictions
from config import Config

# 配置日志（强制 UTF-8，避免 Windows GBK 终端乱码）
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys.stderr, 'reconfigure'):
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_module(config):
    """动态导入数据处理模块"""
    spec = importlib.util.spec_from_file_location(
        "dataset_processor_fixed",
        config.dataset_processor_path
    )
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    return dataset_module.create_dataloaders


# I/E/P -> 0/1/2，仅对有缺陷样本监督类型头
DEFECT_TYPE_STR_TO_ID = {'I': 0, 'E': 1, 'P': 2}

def _get_subset_base_and_indices(ds):
    # torch.utils.data.Subset has .dataset and .indices
    base = getattr(ds, 'dataset', ds)
    indices = getattr(ds, 'indices', range(len(ds)))
    return base, indices


def compute_split_diagnostics(train_loader, val_loader, test_loader):
    """
    返回可序列化的 split 诊断信息：
    - VAL/TEST 缺陷节点是否为 TRAIN 的子集
    - 各 split 的去重缺陷节点数、场景数、窗口数、定位启用占比
    """
    def _stats(loader):
        ds = loader.dataset
        base, indices = _get_subset_base_and_indices(ds)
        scenario_set = set()
        defect_nodes = set()
        n_loc_enabled = 0
        n_windows = 0
        samples = getattr(base, 'samples', None)
        if samples is None:
            return {
                'n_scenarios': 0,
                'n_windows': 0,
                'n_defect_nodes': 0,
                'loc_enabled': 0,
                'loc_enabled_ratio': 0.0,
            }
        for i in indices:
            s = samples[i]
            scenario_set.add(int(s.get('scenario_id', -1)))
            dn = str(s.get('defect_node_id', '')).strip()
            if dn:
                defect_nodes.add(dn)
            if int(s.get('loc_enabled', 0)) > 0:
                n_loc_enabled += 1
            n_windows += 1
        loc_ratio = float(n_loc_enabled / max(n_windows, 1))
        # scenario_id 可能包含 -1（异常样本），剔除不影响子集判断但影响统计可读性
        if -1 in scenario_set:
            scenario_set.discard(-1)
        return {
            'n_scenarios': int(len(scenario_set)),
            'n_windows': int(n_windows),
            'n_defect_nodes': int(len(defect_nodes)),
            'loc_enabled': int(n_loc_enabled),
            'loc_enabled_ratio': float(loc_ratio),
            '_defect_node_set': defect_nodes,  # 内部用，后面会删掉
        }

    tr = _stats(train_loader)
    va = _stats(val_loader)
    te = _stats(test_loader)

    train_nodes = tr.pop('_defect_node_set', set())
    val_nodes = va.pop('_defect_node_set', set())
    test_nodes = te.pop('_defect_node_set', set())

    val_in_train = val_nodes <= train_nodes
    test_in_train = test_nodes <= train_nodes

    diag = {
        'train': tr,
        'val': va,
        'test': te,
        'val_defect_nodes_subset_train': bool(val_in_train),
        'test_defect_nodes_subset_train': bool(test_in_train),
    }
    if not val_in_train:
        diag['val_only_defect_nodes_sample'] = list(sorted(val_nodes - train_nodes))[:10]
    if not test_in_train:
        diag['test_only_defect_nodes_sample'] = list(sorted(test_nodes - train_nodes))[:10]
    return diag


def estimate_process_signal_thresholds(dataset, train_indices, active_quantile=0.95, transition_quantile=0.90):
    """
    基于训练集中的低重叠窗口估计 always_on 场景的残差阈值。
    若当前数据主要是 time_gated，可能拿不到有效样本，此时返回 None。
    """
    signal_scores = []
    for idx in train_indices:
        sample = dataset.samples[int(idx)]
        if str(sample.get('defect_type_str', 'BASELINE')).upper() != 'BASELINE':
            continue
        signal_scores.append(float(sample.get('signal_score', 0.0)))

    if not signal_scores:
        return None, None

    scores = np.asarray(signal_scores, dtype=np.float32)
    transition_threshold = float(np.quantile(scores, transition_quantile))
    active_threshold = float(np.quantile(scores, active_quantile))
    return active_threshold, transition_threshold


def _load_id_list(json_path, keys=("monitor_nodes", "candidate_nodes", "nodes", "candidates")):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = []
        for key in keys:
            if key in data and isinstance(data[key], list):
                raw = data[key]
                break
    else:
        raw = []
    return [str(x) for x in raw]


def _safe_minmax_scale(values):
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def build_node_static_features(config, dataset, graph_features_dict):
    node_list = [str(n) for n in dataset.node_list]
    node_to_idx = {str(n): i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)

    adj = np.asarray(dataset.adj_matrix, dtype=np.float32)
    out_degree = _safe_minmax_scale(adj.sum(axis=1))
    in_degree = _safe_minmax_scale(adj.sum(axis=0))

    elevation_map = {}
    with open(config.parsed_inp_data_file, 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    for section in ("junctions", "outfalls", "storage"):
        for node_id, info in parsed.get(section, {}).items():
            elevation_map[str(node_id)] = float(info.get("elevation", 0.0))
    elevation = _safe_minmax_scale([elevation_map.get(node_id, 0.0) for node_id in node_list])

    candidate_mask = getattr(dataset, 'candidate_mask_np', None)
    if candidate_mask is None:
        candidate_flag = np.ones(n_nodes, dtype=np.float32)
    else:
        candidate_flag = np.asarray(candidate_mask, dtype=np.float32)

    monitor_ids = _load_id_list(config.monitor_nodes_file, keys=("monitor_nodes", "nodes"))
    monitor_flag = np.zeros(n_nodes, dtype=np.float32)
    monitor_indices = []
    for node_id in monitor_ids:
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            monitor_flag[idx] = 1.0
            monitor_indices.append(idx)

    monitor_closeness = np.zeros(n_nodes, dtype=np.float32)
    if monitor_indices:
        shortest = np.asarray(graph_features_dict["shortest_dist"], dtype=np.float32)
        forward = shortest[:, monitor_indices]
        backward = shortest[monitor_indices, :].T
        min_hops = np.minimum(forward, backward).min(axis=1)
        max_hop = max(float(np.max(min_hops)), 1.0)
        monitor_closeness = 1.0 - np.clip(min_hops / max_hop, 0.0, 1.0)

    static_features = np.stack([
        elevation,
        in_degree,
        out_degree,
        monitor_closeness,
        candidate_flag,
        monitor_flag,
    ], axis=-1).astype(np.float32)
    feature_names = [
        "elevation_norm",
        "in_degree_norm",
        "out_degree_norm",
        "monitor_closeness",
        "candidate_flag",
        "monitor_flag",
    ]
    return static_features, feature_names


def train_epoch(model, train_loader, optimizer, criterion_loc, criterion_active, device, use_full_graph=False,
                lambda_type=0.0, criterion_type=None, lambda_loc=1.0, lambda_rank=0.0, rank_margin=0.1,
                lambda_balance=0.0, loc_loss_mode: str = "soft", loc_soft_weight: float = 0.5,
                lambda_recon: float = 0.0, recon_mask_prob: float = 0.3):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion_loc: 节点级定位损失函数
        criterion_active: 活跃期检测损失函数
        device: 计算设备
        use_full_graph: 是否在训练时使用全图（不限制候选集）
        lambda_type: 缺陷类型辅助头损失权重（0 表示不计算）
        criterion_type: 类型分类 CE 损失函数
        lambda_loc: 定位损失权重

    Returns:
        metrics: 训练指标字典
    """
    model.train()
    total_loss = 0.0
    total_active_loss = 0.0
    total_loc_loss = 0.0
    total_rank_loss = 0.0
    total_balance_loss = 0.0
    total_recon_loss = 0.0
    total_active_correct = 0
    total_active_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        x = batch['features'].to(device)
        active_label = batch['active_label'].long().to(device)

        # 处理邻接矩阵
        adj_matrix = batch['adj_matrix']
        if isinstance(adj_matrix, (list, tuple)):
            adj = adj_matrix[0].to(device)
        elif adj_matrix.dim() == 3:
            adj = adj_matrix[0].to(device)
        else:
            adj = adj_matrix.to(device)

        # ✅ 新增：获取软标签（空间感知）
        soft_labels = batch['soft_label'].to(device)  # [B, N]
        
        # ✅ 保留：硬标签（兼容性）
        target_node_idx = batch['target_node_idx'].long().to(device)  # [B]

        # ✅ 候选节点mask（仅在评估时使用）
        candidate_mask = batch['candidate_mask'].to(device).bool()    # [B, N] 或 [N]

        optimizer.zero_grad()

        recon_target = None
        recon_mask_nodes = None
        x_model = x
        if getattr(model, "reconstruction_target_dim", 0) > 0:
            target_dim = int(model.reconstruction_target_dim)
            recon_target = x[:, -1, :, :target_dim].detach()
            observed_mask = batch.get('observed_mask', None)
            if observed_mask is not None:
                observed_mask = observed_mask.to(device).bool()
                recon_mask_nodes = (torch.rand_like(observed_mask.float()) < float(recon_mask_prob)) & observed_mask
                if recon_mask_nodes.any():
                    x_model = x.clone()
                    mask_expand = recon_mask_nodes.unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[1], -1, target_dim)
                    x_model[:, :, :, :target_dim] = x_model[:, :, :, :target_dim].masked_fill(mask_expand, 0.0)

        type_override = None
        if getattr(model, 'type_conditioned_head', False):
            defect_type_str_batch = batch['defect_type_str']
            type_override_list = []
            for i in range(active_label.shape[0]):
                s = defect_type_str_batch[i] if isinstance(defect_type_str_batch, (list, tuple)) else defect_type_str_batch
                if isinstance(s, torch.Tensor):
                    s = s.item() if s.numel() == 1 else str(s.cpu().tolist()[0])
                else:
                    s = str(s).strip().upper()
                type_override_list.append(DEFECT_TYPE_STR_TO_ID.get(s, -1))
            type_override = torch.tensor(type_override_list, dtype=torch.long, device=device)

        # 前向传播（含缺陷类型头 logits_defect_type [B, 3]）
        if type_override is not None:
            outputs = model(x_model, adj, defect_type_override=type_override)
        else:
            outputs = model(x_model, adj)  # [B, N, 2], [B, 2], [B, 3]

        if isinstance(outputs, tuple):
            logits_node = outputs[0]
            logits_active = outputs[1] if len(outputs) > 1 else None
            logits_defect_type = outputs[2] if len(outputs) > 2 else None
            recon_pred = outputs[3] if len(outputs) > 3 else None
        else:
            logits_node = outputs
            logits_active = None
            logits_defect_type = None
            recon_pred = None

        loss_active = criterion_active(logits_active, active_label)
        active_pred = torch.argmax(logits_active, dim=1)
        total_active_correct += int((active_pred == active_label).sum().item())
        total_active_samples += int(active_label.numel())

        # ✅ 改进：使用软标签 + KL散度损失
        scores = logits_node[:, :, 1]  # [B, N]

        # 训练时可以选择是否使用全图
        if candidate_mask.dim() == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand_as(scores)
        if not use_full_graph:
            scores = scores.masked_fill(~candidate_mask, -1e9)

        # 仅对活跃期窗口计算定位损失
        loc_mask = (active_label > 0) & (soft_labels.sum(dim=-1) > 1e-8)
        if loc_mask.sum() > 0:
            scores = scores[loc_mask]
            soft_labels = soft_labels[loc_mask]
            candidate_mask = candidate_mask[loc_mask]
            target_node_idx = target_node_idx[loc_mask]

            candidate_mask_bool = candidate_mask.bool()
            scores_masked = scores.masked_fill(~candidate_mask_bool, -1e9)
            soft_labels_masked = soft_labels.masked_fill(~candidate_mask_bool, 0.0)
            soft_sum = soft_labels_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            soft_labels_norm = soft_labels_masked / soft_sum

            log_probs = torch.log_softmax(scores_masked, dim=-1)
            loss_loc_soft = criterion_loc(log_probs, soft_labels_norm)
            loss_loc_hard = F.cross_entropy(scores_masked, target_node_idx)
            loc_mode = str(loc_loss_mode or "soft").lower()
            if loc_mode == "hard":
                loss_loc = loss_loc_hard
            elif loc_mode == "hybrid":
                alpha = float(loc_soft_weight)
                loss_loc = alpha * loss_loc_soft + (1.0 - alpha) * loss_loc_hard
            else:
                loss_loc = loss_loc_soft
            loss_rank = torch.zeros((), device=device)
            loss_balance = torch.zeros((), device=device)
            if lambda_rank > 0:
                true_scores = scores_masked.gather(1, target_node_idx.unsqueeze(1)).squeeze(1)
                target_onehot = torch.zeros_like(scores_masked, dtype=torch.bool)
                target_onehot.scatter_(1, target_node_idx.unsqueeze(1), True)
                hardest_neg = scores_masked.masked_fill(target_onehot, -1e9).max(dim=-1).values
                loss_rank = F.relu(float(rank_margin) + hardest_neg - true_scores).mean()
            if lambda_balance > 0:
                probs = torch.softmax(scores_masked, dim=-1)
                mean_probs = probs.mean(dim=0)
                candidate_any = candidate_mask_bool.any(dim=0)
                if candidate_any.any():
                    mean_probs_valid = mean_probs[candidate_any]
                    target_uniform = torch.full_like(mean_probs_valid, 1.0 / float(mean_probs_valid.numel()))
                    loss_balance = F.mse_loss(mean_probs_valid, target_uniform)
        else:
            loss_loc = torch.zeros((), device=device)
            loss_rank = torch.zeros((), device=device)
            loss_balance = torch.zeros((), device=device)

        loss_recon = torch.zeros((), device=device)
        if (
            lambda_recon > 0
            and recon_pred is not None
            and recon_target is not None
            and recon_mask_nodes is not None
            and recon_mask_nodes.any()
        ):
            recon_pred_valid = recon_pred[recon_mask_nodes]
            recon_target_valid = recon_target[recon_mask_nodes]
            loss_recon = F.mse_loss(recon_pred_valid, recon_target_valid)

        loss = loss_active + float(lambda_loc) * loss_loc
        if lambda_rank > 0:
            loss = loss + float(lambda_rank) * loss_rank
        if lambda_balance > 0:
            loss = loss + float(lambda_balance) * loss_balance
        if lambda_recon > 0:
            loss = loss + float(lambda_recon) * loss_recon

        # 缺陷类型辅助头：仅对 has_defect=1 的样本计算 I/E/P 分类损失
        if lambda_type > 0 and criterion_type is not None:
            defect_type_str_batch = batch['defect_type_str']  # list of str or tensor
            B = active_label.shape[0]
            type_ids = []
            for i in range(B):
                if active_label[i].item() != 1:
                    continue
                s = defect_type_str_batch[i] if isinstance(defect_type_str_batch, (list, tuple)) else defect_type_str_batch
                if isinstance(s, torch.Tensor):
                    s = s.item() if s.numel() == 1 else str(s.cpu().tolist()[0])
                else:
                    s = str(s).strip().upper()
                if s in DEFECT_TYPE_STR_TO_ID:
                    type_ids.append((i, DEFECT_TYPE_STR_TO_ID[s]))
            if type_ids:
                indices = [t[0] for t in type_ids]
                labels = [t[1] for t in type_ids]
                logits_type = logits_defect_type[indices].to(device)
                targets_type = torch.tensor(labels, dtype=torch.long, device=device)
                loss_type = criterion_type(logits_type, targets_type)
                loss = loss + lambda_type * loss_type
            # 若 batch 中无缺陷样本，仅 loss_loc
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_active_loss += float(loss_active.item())
        total_loc_loss += float(loss_loc.item())
        total_rank_loss += float(loss_rank.item())
        total_balance_loss += float(loss_balance.item())
        total_recon_loss += float(loss_recon.item())

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return {
        'loss': total_loss / max(len(train_loader), 1),
        'active_loss': total_active_loss / max(len(train_loader), 1),
        'loc_loss': total_loc_loss / max(len(train_loader), 1),
        'rank_loss': total_rank_loss / max(len(train_loader), 1),
        'balance_loss': total_balance_loss / max(len(train_loader), 1),
        'recon_loss': total_recon_loss / max(len(train_loader), 1),
        'active_acc': total_active_correct / max(total_active_samples, 1),
    }


def write_training_history_csv(history_rows, output_csv):
    if not history_rows:
        return
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(output_path, index=False, encoding='utf-8-sig')


def build_artifact_paths(config, run_name: str) -> dict:
    """为当前运行生成独立文件名，避免覆盖既有通用结果文件。"""
    tag = (run_name or getattr(config, 'output_tag', '') or "process_diagnosis").strip()
    return {
        'tag': tag,
        'checkpoint': Path(config.output_dir) / f"best_model_{tag}.pth",
        'history': Path(config.output_dir) / f"training_history_{tag}.csv",
        'by_scenario_test': Path(config.output_dir) / f"by_scenario_test_{tag}.csv",
        'by_scenario_val': Path(config.output_dir) / f"by_scenario_val_{tag}.csv",
        'metrics': Path(config.reports_dir) / f"last_run_metrics_{tag}.json",
    }


def load_dataset_manifest(node_timeseries_file: str) -> dict:
    data_path = Path(node_timeseries_file)
    manifest_candidates = [
        data_path.parent / "dataset_manifest.json",
        data_path.parent / "repair_manifest_monitor_only.json",
    ]
    for manifest_path in manifest_candidates:
        if manifest_path.exists():
            try:
                import json
                with manifest_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def build_config_snapshot(config, split_diag: dict) -> dict:
    dataset_manifest = load_dataset_manifest(config.node_timeseries_file)
    snapshot = {
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'dropout': config.dropout,
        'patience': config.patience,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'sequence_length': config.sequence_length,
        'window_stride': config.window_stride,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'soft_label_sigma': config.soft_label_sigma,
        'train_with_full_graph': getattr(config, 'train_with_full_graph', False),
        'use_time_pos_encoding': getattr(config, 'use_time_pos_encoding', True),
        'use_trend_feature': getattr(config, 'use_trend_feature', False),
        'dataset_active_overlap_threshold': getattr(config, 'dataset_active_overlap_threshold', 0.5),
        'dataset_transition_overlap_threshold': getattr(config, 'dataset_transition_overlap_threshold', 0.0),
        'dataset_active_signal_quantile': getattr(config, 'dataset_active_signal_quantile', 0.95),
        'dataset_transition_signal_quantile': getattr(config, 'dataset_transition_signal_quantile', 0.90),
        'lambda_type': getattr(config, 'lambda_type', 0.0),
        'lambda_loc': getattr(config, 'lambda_loc', 1.0),
        'loc_loss_mode': getattr(config, 'loc_loss_mode', 'soft'),
        'loc_soft_weight': getattr(config, 'loc_soft_weight', 0.5),
        'lambda_rank': getattr(config, 'lambda_rank', 0.0),
        'rank_margin': getattr(config, 'rank_margin', 0.1),
        'lambda_balance': getattr(config, 'lambda_balance', 0.0),
        'hydraulic_attention_max_hops': getattr(config, 'hydraulic_attention_max_hops', 0),
        'hydraulic_observed_source_only': getattr(config, 'hydraulic_observed_source_only', False),
        'model_type': config.model_type,
        'time_hidden_dim': config.time_hidden_dim,
        'spatial_hidden_dim': config.spatial_hidden_dim,
        'num_time_layers': config.num_time_layers,
        'num_spatial_layers': config.num_spatial_layers,
        'n_features': len(config.selected_features),
        'candidate_nodes_file': os.path.basename(config.candidate_nodes_file),
        'monitor_nodes_file_config': os.path.basename(config.monitor_nodes_file),
        'monitor_nodes_file_dataset': dataset_manifest.get('monitor_nodes_file', ''),
        'monitor_node_count_dataset': dataset_manifest.get('monitor_node_count', None),
        'strict_monitor_only_dataset': dataset_manifest.get('strict_monitor_only', None),
        'training_data_subdir': getattr(config, 'training_data_subdir', ''),
        'node_timeseries_file': config.node_timeseries_file,
        'split_mode': getattr(config, 'dataset_split_mode', 'scenario'),
    }
    if getattr(config, 'node_static_feature_names', None):
        snapshot['node_static_feature_names'] = list(config.node_static_feature_names)
        snapshot['node_static_feature_dim'] = len(config.node_static_feature_names)
    if dataset_manifest:
        snapshot['dataset_manifest'] = dataset_manifest
    return snapshot


def main():
    """主训练流程"""
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--seed", type=int, default=None, help="覆盖 config.random_seed，用于三种子等重复实验")
    _parser.add_argument("--subdir", type=str, default=None, help="覆盖 config.training_data_subdir，如 time_gated / time_gated_betweenness")
    _parser.add_argument("--defect-csv", type=str, default=None, help="覆盖 config.defect_matrix_file（B 组实验用）")
    _parser.add_argument("--run-name", type=str, default=None, help="实验标识，用于输出文件命名（如 degree_seed42）")
    _parser.add_argument("--raw-features", action="store_true", help="仅用原始特征（去掉残差列），用于 B3 对比")
    _parser.add_argument("--features", type=str, default=None,
                         help="逗号分隔的特征列表，覆盖 config.selected_features。"
                              "例：--features depth_residual,lateral_inflow_residual,total_inflow_residual,"
                              "pollut_BODf_residual,pollut_NH4_residual,pollut_DO_residual")
    _parser.add_argument("--model-type", type=str, default=None, help="覆盖 config.model_type，如 gru_gcn / hydraulic_inverse")
    _parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=["scenario", "node_holdout", "scenario_nodecovered", "scenario_seen", "scenario_traincovers"],
        help="数据划分: scenario / node_holdout / scenario_nodecovered(训练覆盖测试缺陷点)",
    )
    _parser.add_argument("--n-holdout-nodes", type=int, default=None, help="node_holdout 时测试集保留的缺陷节点数，默认 10")
    _parser.add_argument("--lambda-loc", type=float, default=None, help="覆盖 config.lambda_loc，用于提高定位损失权重")
    _parser.add_argument("--lambda-type", type=float, default=None, help="覆盖 config.lambda_type，用于启用 I/E/P 类型辅助头")
    _parser.add_argument("--lambda-rank", type=float, default=None, help="覆盖 config.lambda_rank，用于启用 hard negative 排序损失")
    _parser.add_argument("--rank-margin", type=float, default=None, help="覆盖 config.rank_margin")
    _parser.add_argument("--lambda-balance", type=float, default=None, help="覆盖 config.lambda_balance，用于抑制候选分数塌缩到少数 attractor 点")
    _parser.add_argument("--attention-max-hops", type=int, default=None, help="hydraulic_inverse 局部聚合 hop 上限；不传或 0 表示全图")
    _parser.add_argument("--soft-label-sigma", type=float, default=None, help="覆盖 config.soft_label_sigma，用于控制定位软标签扩散范围")
    _parser.add_argument("--loc-loss-mode", type=str, default=None, choices=["soft", "hard", "hybrid"],
                         help="定位损失形式：soft=KL软标签，hard=候选CE，hybrid=两者加权")
    _parser.add_argument("--loc-soft-weight", type=float, default=None,
                         help="hybrid 定位损失中 soft 部分权重，剩余部分给 hard CE")
    _parser.add_argument("--train-with-full-graph", action="store_true",
                         help="训练阶段不过滤到候选集，定位损失在全图上计算")
    _parser.add_argument("--hydraulic-observed-source-only", action="store_true",
                         help="仅允许 hydraulic_inverse 从 observed nodes 汇聚信息")
    _parser.add_argument("--use-propagation-delay", action="store_true",
                         help="在 hydraulic_inverse 路径特征中追加传播延迟先验")
    _parser.add_argument("--learning-rate", type=float, default=None, help="覆盖 config.learning_rate")
    _parser.add_argument("--weight-decay", type=float, default=None, help="覆盖 config.weight_decay")
    _parser.add_argument("--dropout", type=float, default=None, help="覆盖 config.dropout")
    _parser.add_argument("--num-epochs", type=int, default=None, help="覆盖 config.num_epochs")
    _parser.add_argument("--lambda-recon", type=float, default=None, help="override config.lambda_recon for reconstruction loss")
    _parser.add_argument("--recon-mask-prob", type=float, default=None, help="override config.recon_mask_prob for reconstruction masking")
    _args = _parser.parse_args()

    # ========== 初始化配置 ==========
    config = Config()
    if _args.seed is not None:
        config.random_seed = _args.seed
        logger.info(f"使用命令行种子: {config.random_seed}")
    if _args.subdir is not None:
        config.training_data_subdir = _args.subdir.strip()
        import os
        _root = config.training_data_dir
        _sub = config.training_data_subdir
        _parquet_name = "node_timeseries.parquet" if _args.raw_features else "node_timeseries_with_residuals.parquet"
        config.node_timeseries_file = os.path.normpath(os.path.join(_root, _sub, _parquet_name))
        logger.info(f"使用数据子目录: {config.training_data_subdir} -> {config.node_timeseries_file}")
    if getattr(config, 'dataset_label_mode', 'auto') == 'auto' and 'full_injection' in str(getattr(config, 'training_data_subdir', '')).lower():
        config.dataset_label_mode = 'always_on'
        logger.info("妫€娴嬪埌 full_injection 瀛愮洰褰曪紝鑷姩灏? dataset_label_mode 鍒囨崲涓? always_on")
    if _args.defect_csv is not None:
        config.defect_matrix_file = os.path.normpath(_args.defect_csv)
        logger.info(f"使用自定义缺陷矩阵: {config.defect_matrix_file}")
    if _args.features is not None:
        config.selected_features = [f.strip() for f in _args.features.split(',') if f.strip()]
        logger.info(f"使用命令行指定特征 ({len(config.selected_features)}): {config.selected_features}")
    if _args.raw_features:
        config.selected_features = [f for f in config.selected_features if 'residual' not in f]
        logger.info(f"使用原始特征（无残差），特征数: {len(config.selected_features)}")
    if _args.model_type is not None:
        config.model_type = _args.model_type.strip()
        logger.info(f"使用命令行模型类型: {config.model_type}")
    if _args.split_mode is not None:
        config.dataset_split_mode = _args.split_mode.strip()
        logger.info(f"使用命令行划分方式: {config.dataset_split_mode}")
    if _args.n_holdout_nodes is not None:
        config.dataset_n_holdout_nodes = _args.n_holdout_nodes
        logger.info(f"node_holdout 节点数: {config.dataset_n_holdout_nodes}")
    if _args.lambda_loc is not None:
        config.lambda_loc = float(_args.lambda_loc)
        logger.info(f"使用命令行 lambda_loc: {config.lambda_loc}")
    if _args.attention_max_hops is not None:
        config.hydraulic_attention_max_hops = int(_args.attention_max_hops)
        logger.info(f"使用命令行 attention_max_hops: {config.hydraulic_attention_max_hops}")
    if _args.use_propagation_delay:
        config.use_propagation_delay = True
        logger.info("使用命令行 use_propagation_delay: True")
    if _args.soft_label_sigma is not None:
        config.soft_label_sigma = float(_args.soft_label_sigma)
        logger.info(f"使用命令行 soft_label_sigma: {config.soft_label_sigma}")
    if _args.loc_loss_mode is not None:
        config.loc_loss_mode = str(_args.loc_loss_mode).strip().lower()
        logger.info(f"使用命令行 loc_loss_mode: {config.loc_loss_mode}")
    if _args.loc_soft_weight is not None:
        config.loc_soft_weight = float(_args.loc_soft_weight)
        logger.info(f"使用命令行 loc_soft_weight: {config.loc_soft_weight}")
    if _args.train_with_full_graph:
        config.train_with_full_graph = True
        logger.info("使用命令行 train_with_full_graph: True")
    if _args.lambda_type is not None:
        config.lambda_type = float(_args.lambda_type)
        logger.info(f"使用命令行 lambda_type: {config.lambda_type}")
    if _args.lambda_rank is not None:
        config.lambda_rank = float(_args.lambda_rank)
        logger.info(f"使用命令行 lambda_rank: {config.lambda_rank}")
    if _args.rank_margin is not None:
        config.rank_margin = float(_args.rank_margin)
        logger.info(f"使用命令行 rank_margin: {config.rank_margin}")
    if _args.lambda_balance is not None:
        config.lambda_balance = float(_args.lambda_balance)
        logger.info(f"使用命令行 lambda_balance: {config.lambda_balance}")
    if _args.lambda_recon is not None:
        config.lambda_recon = float(_args.lambda_recon)
        logger.info(f"lambda_recon: {config.lambda_recon}")
    if _args.recon_mask_prob is not None:
        config.recon_mask_prob = float(_args.recon_mask_prob)
        logger.info(f"recon_mask_prob: {config.recon_mask_prob}")
    if _args.hydraulic_observed_source_only:
        config.hydraulic_observed_source_only = True
        logger.info("使用命令行 hydraulic_observed_source_only: True")
    if _args.learning_rate is not None:
        config.learning_rate = float(_args.learning_rate)
        logger.info(f"浣跨敤鍛戒护琛?learning_rate: {config.learning_rate}")
    if _args.weight_decay is not None:
        config.weight_decay = float(_args.weight_decay)
        logger.info(f"浣跨敤鍛戒护琛?weight_decay: {config.weight_decay}")
    if _args.dropout is not None:
        config.dropout = float(_args.dropout)
        logger.info(f"浣跨敤鍛戒护琛?dropout: {config.dropout}")
    if _args.num_epochs is not None:
        config.num_epochs = int(_args.num_epochs)
        logger.info(f"浣跨敤鍛戒护琛?num_epochs: {config.num_epochs}")
    run_name = (_args.run_name or "").strip()
    artifact_paths = build_artifact_paths(config, run_name)
    checkpoint_path = str(artifact_paths['checkpoint'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"输出标签: {artifact_paths['tag']}")

    logger.info("\n" + "=" * 80)
    logger.info("异常检测模型训练")
    logger.info("=" * 80)

    # ========== 步骤1：加载数据集 ==========
    logger.info("\n[/6] 加载数据集...")
    logger.info(f"  时序数据: {config.node_timeseries_file}")

    # 检查文件是否存在
    if not Path(config.node_timeseries_file).exists():
        logger.warning(f"文件不存在: {config.node_timeseries_file}")
        logger.warning("  正在尝试使用原始文件...")
        config.node_timeseries_file = config.node_timeseries_file.replace(
            '_with_residuals', ''
        )
        if not Path(config.node_timeseries_file).exists():
            raise FileNotFoundError(
                f"找不到时序数据文件！请先运行 25_baseline_feature_engineering.py"
            )
        logger.warning("  使用原始特征（没有残差），性能可能较差")

    create_dataloaders = load_data_module(config)

    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        node_timeseries_file=config.node_timeseries_file,
        adjacency_matrix_file=config.adjacency_matrix_file,
        node_list_file=config.node_list_file,
        defect_matrix_file=config.defect_matrix_file,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        window_stride=config.window_stride,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        normalize=True,
        random_seed=config.random_seed,
        feature_names=config.selected_features,
        candidate_nodes_file=config.candidate_nodes_file,
        soft_label_sigma=config.soft_label_sigma,
        use_full_graph=config.train_with_full_graph,  # ✅ P0 传递训练模式
        signal_threshold=config.dataset_signal_threshold,
        loc_ratio_clip=config.dataset_loc_ratio_clip,
        label_mode=config.dataset_label_mode,
        overlap_threshold=config.dataset_overlap_threshold,
        always_on_force_target=config.dataset_always_on_force_target,
        use_time_pos_encoding=getattr(config, 'use_time_pos_encoding', True),
        use_observed_mask_feature=getattr(config, 'use_observed_mask_feature', True),
        use_trend_feature=getattr(config, 'use_trend_feature', False),
        active_overlap_threshold=getattr(config, 'dataset_active_overlap_threshold', 0.5),
        transition_overlap_threshold=getattr(config, 'dataset_transition_overlap_threshold', 0.0),
        e_class_oversample_ratio=getattr(config, 'dataset_e_class_oversample_ratio', 1),
        split_mode=getattr(config, 'dataset_split_mode', 'scenario'),
        n_holdout_nodes=getattr(config, 'dataset_n_holdout_nodes', 10)
    )

    dataset_base, train_indices = _get_subset_base_and_indices(train_loader.dataset)
    active_signal_threshold, transition_signal_threshold = estimate_process_signal_thresholds(
        dataset_base,
        train_indices,
        active_quantile=getattr(config, 'dataset_active_signal_quantile', 0.95),
        transition_quantile=getattr(config, 'dataset_transition_signal_quantile', 0.90)
    )
    dataset_base.configure_process_labels(
        active_signal_threshold=active_signal_threshold,
        transition_signal_threshold=transition_signal_threshold
    )
    if active_signal_threshold is not None:
        logger.info(
            "过程诊断阈值: active_signal_threshold=%.4f transition_signal_threshold=%.4f",
            active_signal_threshold,
            transition_signal_threshold,
        )
    else:
        logger.info("过程诊断阈值: 当前训练集未触发 always_on 阈值估计，沿用 overlap 口径或兼容设置")

    # split 诊断（写入 last_run_metrics_{run_name}.json，避免仅靠终端日志）
    split_diag = compute_split_diagnostics(train_loader, val_loader, test_loader)

    input_dim = int(getattr(dataset, 'input_feature_dim', 0) or dataset.samples[0]['features'].shape[-1])
    num_nodes = len(dataset.node_list)

    logger.info(f"  特征维度: {input_dim}")
    logger.info(f"  节点数: {num_nodes}")
    logger.info(f"  训练样本: {len(train_loader.dataset)}")
    logger.info(f"  验证样本: {len(val_loader.dataset)}")
    logger.info(f"  测试样本: {len(test_loader.dataset)}")

    # Task4：训练集中出现过的缺陷节点（用于评估时区分见过/未见过）
    train_seen_node_indices = set()
    ds = train_loader.dataset
    base = getattr(ds, 'dataset', ds)  # Subset 时用 .dataset
    indices = getattr(ds, 'indices', range(len(ds)))
    if hasattr(base, 'samples'):
        for i in indices:
            s = base.samples[i]
            if s.get('target_node_idx', -1) >= 0:
                train_seen_node_indices.add(s['target_node_idx'])
    logger.info(f"  训练集缺陷节点数(用于评估拆分): {len(train_seen_node_indices)}")

    # ========== 步骤2：创建模型 ==========
    logger.info("\n[2/6] 创建模型...")

    model_kw = dict(
        model_type=config.model_type,
        input_dim=input_dim,
        time_hidden_dim=config.time_hidden_dim,
        spatial_hidden_dim=config.spatial_hidden_dim,
        num_time_layers=config.num_time_layers,
        num_spatial_layers=config.num_spatial_layers,
        num_nodes=num_nodes,
        dropout=config.dropout,
    )
    if config.model_type in {
        'hydraulic_inverse',
        'hydraulic_inverse_ctx',
        'hydraulic_inverse_ctxfix',
        'hydraulic_inverse_typeaware',
        'hydraulic_inverse_lstm',
        'hydraulic_inverse_static',
        'hydraulic_inverse_staticbias',
        'hydraulic_inverse_deepattn_static',
        'hydraulic_inverse_deepattn_staticbias',
        'hydraulic_inverse_deepattn_recon',
    }:
        graph_path_file = getattr(config, 'graph_path_features_file', '') or os.path.join(SCRIPT_ROOT, 'input_1', 'graph_path_features.npz')
        if not os.path.exists(graph_path_file):
            raise FileNotFoundError(
                f"hydraulic_inverse 模型需要图路径特征文件，请先运行: python prep/build_graph_features.py\n  缺失: {graph_path_file}"
            )
        npz = np.load(graph_path_file)
        graph_features_dict = {k: npz[k] for k in npz.files}
        model_kw['graph_features_dict'] = graph_features_dict
        logger.info(f"  已加载图路径特征: {graph_path_file}")
        model_kw['use_flow_direction'] = getattr(config, 'use_flow_direction', True)
        model_kw['use_propagation_delay'] = getattr(config, 'use_propagation_delay', False)
        model_kw['propagation_delay_velocity_mps'] = getattr(config, 'propagation_delay_velocity_mps', 0.5)
        model_kw['attention_max_hops'] = getattr(config, 'hydraulic_attention_max_hops', 0)
        model_kw['observed_source_only'] = getattr(config, 'hydraulic_observed_source_only', False)
        if config.model_type in {
            'hydraulic_inverse_static',
            'hydraulic_inverse_staticbias',
            'hydraulic_inverse_deepattn_static',
            'hydraulic_inverse_deepattn_staticbias',
        }:
            node_static_features, node_static_feature_names = build_node_static_features(config, dataset, graph_features_dict)
            model_kw['node_static_features'] = node_static_features
            config.node_static_feature_names = node_static_feature_names
            logger.info(f"  已构建节点静态先验: {node_static_feature_names}")

    if config.model_type == 'hydraulic_inverse_deepattn_recon':
        model_kw['reconstruction_target_dim'] = int(len(config.selected_features))
        logger.info(f"  reconstruction_target_dim: {model_kw['reconstruction_target_dim']}")

    model = create_model(**model_kw).to(device)

    logger.info(f"  模型类型: {config.model_type}")
    logger.info(f"  时间隐藏层: {config.time_hidden_dim}")
    logger.info(f"  空间隐藏层: {config.spatial_hidden_dim}")
    logger.info(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 步骤3：配置训练 ==========
    logger.info("\n[3/6] 配置训练...")

    # 损失函数
    # ✅ 改进：使用KL散度损失适配软标签
    criterion_loc = nn.KLDivLoss(reduction='batchmean')
    criterion_active = nn.CrossEntropyLoss(reduction='mean')
    lambda_type = getattr(config, 'lambda_type', 0.0)
    lambda_loc = getattr(config, 'lambda_loc', 1.0)
    loc_loss_mode = getattr(config, 'loc_loss_mode', 'soft')
    loc_soft_weight = getattr(config, 'loc_soft_weight', 0.5)
    lambda_rank = getattr(config, 'lambda_rank', 0.0)
    rank_margin = getattr(config, 'rank_margin', 0.1)
    lambda_balance = getattr(config, 'lambda_balance', 0.0)
    lambda_recon = getattr(config, 'lambda_recon', 0.0)
    recon_mask_prob = getattr(config, 'recon_mask_prob', 0.3)
    criterion_type = nn.CrossEntropyLoss(reduction='mean') if lambda_type > 0 else None

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )

    logger.info(f"  学习率: {config.learning_rate}")
    logger.info(f"  权重衰减: {config.weight_decay}")
    logger.info("  活跃期检测Loss: CrossEntropy")
    logger.info(f"  定位Loss: {loc_loss_mode} (soft_weight={loc_soft_weight:.2f})")
    logger.info(f"  缺陷类型辅助头: lambda_type={lambda_type} (I/E/P, 仅 has_defect=1)")
    logger.info(f"  定位损失权重 lambda_loc: {lambda_loc}")
    logger.info(f"  排序损失权重 lambda_rank: {lambda_rank} (margin={rank_margin})")
    logger.info(f"  候选平衡正则 lambda_balance: {lambda_balance}")
    logger.info(f"  早停耐心值: {config.patience}")
    
    # ✅ 新增：训练配置
    use_full_graph = getattr(config, 'train_with_full_graph', False)  # 训练时是否使用全图
    logger.info(f"  训练模式: {'全图(无候选限制)' if use_full_graph else '候选集限制'}")

    # 评估损失函数（训练与评估分离）
    criterion_loc_eval = nn.KLDivLoss(reduction='batchmean')
    criterion_active_eval = nn.CrossEntropyLoss(reduction='mean')

    # 如果仅评估（不训练），通过环境变量触发
    if os.environ.get('EVAL_ONLY') == '1':
        logger.info("\n[跳过训练] 仅评估并导出 by_scenario …")
        os.makedirs(config.model_save_dir, exist_ok=True)
        model_path = str(artifact_paths['checkpoint'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到最佳模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate_model(
            model, test_loader, criterion_loc_eval, device,
            criterion_active=criterion_active_eval,
            train_seen_node_indices=train_seen_node_indices
        )
        by_sc = test_metrics.get('by_scenario', {})
        if by_sc:
            rows = []
            for sc, d in by_sc.items():
                rows.append({
                    'defect_id': sc,
                    'n_windows': d.get('n_windows', 0),
                    'top1': d.get('top1', 0.0),
                    'top3': d.get('top3', 0.0),
                    'mrr': d.get('mrr', 0.0)
                })
            df_out = pd.DataFrame(rows)
            out_csv = artifact_paths['by_scenario_test']
            df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
            logger.info(f"[OK] 已导出测试 by_scenario: {out_csv}")
        # Task4：见过/未见过、按类型
        sn = test_metrics.get('seen_nodes', {})
        un = test_metrics.get('unseen_nodes', {})
        if sn or un:
            logger.info("  [Task4] 训练见过 vs 未见过缺陷节点:")
            if sn:
                logger.info(f"    见过: n={sn.get('n', 0)}, MRR={sn.get('mrr', 0):.4f}, Top-1={sn.get('top1', 0):.4f}")
            if un:
                logger.info(f"    未见过: n={un.get('n', 0)}, MRR={un.get('mrr', 0):.4f}, Top-1={un.get('top1', 0):.4f}")
        bt = test_metrics.get('by_defect_type', {})
        if bt:
            logger.info("  [Task4] 按缺陷类型 I/E/P:")
            for typ in ('I', 'E', 'P'):
                d = bt.get(typ, {})
                if d and d.get('n', 0) > 0:
                    logger.info(f"    {typ}: n={d.get('n', 0)}, MRR={d.get('mrr', 0):.4f}, Top-1={d.get('top1', 0):.4f}")

        # eval-only 也落盘 last_run_metrics，便于快速验证 split 诊断与结果读取链路
        _config_snapshot = build_config_snapshot(config, split_diag)
        last_metrics = {
            'mrr': float(test_metrics.get('mrr', 0.0)),
            'top1': float(test_metrics.get('topk_recall_1', 0.0)),
            'top3': float(test_metrics.get('topk_recall_3', 0.0)),
            'top5': float(test_metrics.get('topk_recall_5', 0.0)),
            'active_acc': float(test_metrics.get('active_acc', 0.0)),
            'active_period_recall': float(test_metrics.get('active_period_recall', 0.0)),
            'detection_latency_mean': float(test_metrics.get('detection_latency_mean', float('nan'))),
            'event_level_top1': float(test_metrics.get('event_level_top1', 0.0)),
            'event_level_top3': float(test_metrics.get('event_level_top3', 0.0)),
            'event_level_top5': float(test_metrics.get('event_level_top5', 0.0)),
            'seed': config.random_seed,
            'subdir': getattr(config, 'training_data_subdir', ''),
            'defect_csv': str(getattr(config, 'defect_matrix_file', '')),
            'split_diag': split_diag,
            'config_snapshot': _config_snapshot,
        }
        out_dir = Path(config.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(artifact_paths['metrics'], 'w', encoding='utf-8') as _f:
            _json.dump(last_metrics, _f, ensure_ascii=False, indent=2)
        logger.info(f"  [OK] 已写指标文件 -> {artifact_paths['metrics']}")
        return

    # ========== 步骤4：训练循环 ==========
    logger.info("\n[4/6] 开始训练...")
    logger.info("=" * 80)

    # 以“定位任务”的排序指标作为主准则选模型（更符合工程目标）
    primary_metric = 'mrr'  # 也可改成 'topk_recall_1'
    best_val_score = -float('inf')
    best_val_loss = float('inf')  # 仅用于同分时的次级比较
    patience_counter = 0

    history_rows = []
    for epoch in range(config.num_epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_loc, criterion_active, device, use_full_graph,
            lambda_type=lambda_type,
            criterion_type=criterion_type if lambda_type > 0 else None,
            lambda_loc=lambda_loc,
            loc_loss_mode=loc_loss_mode,
            loc_soft_weight=loc_soft_weight,
            lambda_rank=lambda_rank,
            rank_margin=rank_margin,
            lambda_balance=lambda_balance,
            lambda_recon=lambda_recon,
            recon_mask_prob=recon_mask_prob,
        )

        # 验证
        # 评估时使用 CrossEntropyLoss，因为评估需要硬标签计算准确率
        val_metrics = evaluate_model(
            model, val_loader, criterion_loc_eval, device,
            criterion_active=criterion_active_eval,
            train_seen_node_indices=train_seen_node_indices
        )

        # 学习率调度
        if scheduler:
            scheduler.step()

        # 打印指标
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"  训练Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  训练 ActiveLoss: {train_metrics['active_loss']:.4f}")
        logger.info(f"  训练 LocLoss: {train_metrics['loc_loss']:.4f}")
        if 'rank_loss' in train_metrics:
            logger.info(f"  训练 RankLoss: {train_metrics['rank_loss']:.4f}")
        if 'balance_loss' in train_metrics:
            logger.info(f"  训练 BalanceLoss: {train_metrics['balance_loss']:.6f}")
        logger.info(f"  训练 ActiveAcc: {train_metrics['active_acc']:.4f}")
        logger.info(f"  验证Loss: {val_metrics.get('loss', float('nan')):.4f}")
        logger.info(
            f"  验证 活跃期检测 - Acc: {val_metrics.get('active_acc', 0.0):.4f}, "
            f"Recall: {val_metrics.get('active_period_recall', 0.0):.4f}"
        )
        logger.info(f"  验证 定位指标 - MRR: {val_metrics.get('mrr', 0.0):.4f}")
        logger.info(
            f"  验证 Top-K召回率 - "
            f"Top-1: {val_metrics.get('topk_recall_1', 0.0):.4f}, "
            f"Top-3: {val_metrics.get('topk_recall_3', 0.0):.4f}, "
            f"Top-5: {val_metrics.get('topk_recall_5', 0.0):.4f}"
        )
        logger.info(
            f"  验证 过程指标 - Latency(mean,h): {val_metrics.get('detection_latency_mean', float('nan')):.4f}, "
            f"EventTop1: {val_metrics.get('event_level_top1', 0.0):.4f}, "
            f"EventTop3: {val_metrics.get('event_level_top3', 0.0):.4f}, "
            f"EventTop5: {val_metrics.get('event_level_top5', 0.0):.4f}"
        )
        # （可选）按 scenario_id 汇总窗口级别定位表现（便于定位“哪个缺陷场景/节点最难”）
        by_sc = val_metrics.get('by_scenario', {})
        if by_sc:
            # 仅打印 Top-1 最差的前 5 个 scenario（你也可以改成按 mrr 排序）
            worst = sorted(by_sc.items(), key=lambda kv: kv[1].get('top1', 0.0))[:5]
            logger.info("  验证 by_scenario (Top-1 最差5个):")
            for sc, d in worst:
                logger.info(f"    {sc}: n={d.get('n_windows', 0)}, top1={d.get('top1', 0.0):.4f}, top3={d.get('top3', 0.0):.4f}, mrr={d.get('mrr', 0.0):.4f}")


        # 早停和模型保存（可选：以「见过节点」验证 MRR 为主，更稳）
        val_loss = float(val_metrics.get('loss', float('nan')))
        if getattr(config, 'use_seen_mrr_for_early_stopping', False) and val_metrics.get('seen_nodes') and val_metrics['seen_nodes'].get('n', 0) > 0:
            val_score = float(val_metrics['seen_nodes'].get('mrr', 0.0))
            score_name = 'seen_mrr'
        else:
            val_score = float(val_metrics.get(primary_metric, val_metrics.get('topk_recall_1', 0.0)))
            score_name = primary_metric

        improved = (val_score > best_val_score + 1e-6) or (abs(val_score - best_val_score) <= 1e-6 and val_loss < best_val_loss)
        history_rows.append({
            'epoch': epoch + 1,
            'train_loss': float(train_metrics['loss']),
            'train_active_loss': float(train_metrics['active_loss']),
            'train_loc_loss': float(train_metrics['loc_loss']),
            'train_active_acc': float(train_metrics['active_acc']),
            'val_loss': float(val_metrics.get('loss', float('nan'))),
            'val_active_acc': float(val_metrics.get('active_acc', 0.0)),
            'val_active_recall': float(val_metrics.get('active_period_recall', 0.0)),
            'val_mrr': float(val_metrics.get('mrr', 0.0)),
            'val_top1': float(val_metrics.get('topk_recall_1', 0.0)),
            'val_top3': float(val_metrics.get('topk_recall_3', 0.0)),
            'val_top5': float(val_metrics.get('topk_recall_5', 0.0)),
            'val_detection_latency_mean': float(val_metrics.get('detection_latency_mean', float('nan'))),
            'val_event_top1': float(val_metrics.get('event_level_top1', 0.0)),
            'val_event_top3': float(val_metrics.get('event_level_top3', 0.0)),
            'val_event_top5': float(val_metrics.get('event_level_top5', 0.0)),
            'val_score_used_for_early_stop': float(val_score),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'improved': int(improved),
        })

        if improved:
            best_val_score = val_score
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            os.makedirs(config.output_dir, exist_ok=True)
            model_path = checkpoint_path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_score,
                'primary_metric': primary_metric,
                'val_loss': best_val_loss,
                'config.py': config.__dict__
            }, model_path)
            logger.info(f"  [OK] 保存最佳模型 (Val {score_name}: {best_val_score:.4f}, Loss: {best_val_loss:.4f})")

            # 导出当前最佳验证 by_scenario
            by_sc_val = val_metrics.get('by_scenario', {})
            if by_sc_val:
                rows = []
                for sc, d in by_sc_val.items():
                    rows.append({
                        'defect_id': sc,
                        'n_windows': d.get('n_windows', 0),
                        'top1': d.get('top1', 0.0),
                        'top3': d.get('top3', 0.0),
                        'mrr': d.get('mrr', 0.0)
                    })
                df_val = pd.DataFrame(rows)
                out_csv_val = artifact_paths['by_scenario_val']
                df_val.to_csv(out_csv_val, index=False, encoding='utf-8-sig')
                logger.info(f"  [OK] 已导出验证 by_scenario (best): {out_csv_val}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"\n早停：验证 {score_name} 连续{config.patience}轮未改善")
                break

    # ========== 步骤5：测试集评估 ==========
    logger.info("\n[5/6] 测试集评估...")

    # 加载最佳模型
    history_path = artifact_paths['history']
    write_training_history_csv(history_rows, history_path)
    logger.info(f"  [OK] Saved training history: {history_path}")

    model_path = checkpoint_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Current run checkpoint not found: {model_path}\n"
            f"Please confirm this run saved a best model before test-time loading."
        )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(
        model, test_loader, criterion_loc_eval, device,
        criterion_active=criterion_active_eval,
        train_seen_node_indices=train_seen_node_indices
    )

    logger.info("\n" + "=" * 80)
    logger.info("测试集最终评估")
    logger.info("=" * 80)
    logger.info(f"  测试Loss: {test_metrics['loss']:.4f}")
    logger.info(
        f"  活跃期检测 - Acc: {test_metrics.get('active_acc', 0.0):.4f}, "
        f"Recall: {test_metrics.get('active_period_recall', 0.0):.4f}"
    )
    mrr = test_metrics.get('mrr', 0.0)
    logger.info(f"  定位指标 - MRR: {mrr:.4f}")

    logger.info("\n  Top-K 召回率:")
    logger.info(f"     Top-1: {test_metrics.get('topk_recall_1', 0):.4f}")
    logger.info(f"     Top-3: {test_metrics.get('topk_recall_3', 0):.4f}")
    logger.info(f"     Top-5: {test_metrics.get('topk_recall_5', 0):.4f}")
    logger.info(
        "  过程级指标 - DetectionLatency(mean,h): %.4f, EventTop1: %.4f, EventTop3: %.4f, EventTop5: %.4f",
        float(test_metrics.get('detection_latency_mean', float('nan'))),
        float(test_metrics.get('event_level_top1', 0.0)),
        float(test_metrics.get('event_level_top3', 0.0)),
        float(test_metrics.get('event_level_top5', 0.0)),
    )

    # Task4：见过/未见过节点、按缺陷类型 I/E/P
    sn = test_metrics.get('seen_nodes', {})
    un = test_metrics.get('unseen_nodes', {})
    if sn and un:
        logger.info("\n  训练见过 vs 未见过缺陷节点:")
        logger.info(f"     见过节点: n={sn.get('n', 0)}, MRR={sn.get('mrr', 0):.4f}, Top-1={sn.get('top1', 0):.4f}, Top-3={sn.get('top3', 0):.4f}")
        logger.info(f"     未见过节点: n={un.get('n', 0)}, MRR={un.get('mrr', 0):.4f}, Top-1={un.get('top1', 0):.4f}, Top-3={un.get('top3', 0):.4f}")
    bt = test_metrics.get('by_defect_type', {})
    if bt:
        logger.info("\n  按缺陷类型 (I/E/P):")
        for typ in ('I', 'E', 'P'):
            d = bt.get(typ, {})
            if d and d.get('n', 0) > 0:
                logger.info(f"     {typ}: n={d.get('n', 0)}, MRR={d.get('mrr', 0):.4f}, Top-1={d.get('top1', 0):.4f}, Top-3={d.get('top3', 0):.4f}")

    logger.info("\n  k-hop邻域命中率:")
    logger.info(f"     k=0 (精确命中): {test_metrics['khop_recall_0']:.4f}")
    logger.info(f"     k= (-hop邻域): {test_metrics['khop_recall_1']:.4f}")
    logger.info(f"     k=2 (2-hop邻域): {test_metrics['khop_recall_2']:.4f}")

    logger.info("\n  预测距离分布:")
    dist_dist = test_metrics['distance_distribution']
    for dist_key in sorted(
            [k for k in dist_dist.keys() if k != '>10'],
            key=lambda x: int(x) if isinstance(x, int) else 999
    ):
        logger.info(f"     距离={dist_key}: {dist_dist[dist_key]} 样本")
    if '>10' in dist_dist:
        logger.info(f"     距离>10: {dist_dist['>10']} 样本")

    # （可选）按 scenario_id 输出测试集汇总
    by_sc = test_metrics.get('by_scenario', {})
    if by_sc:
        # 导出测试 by_scenario CSV
        rows = []
        for sc, d in by_sc.items():
            rows.append({
                'defect_id': sc,
                'n_windows': d.get('n_windows', 0),
                'top1': d.get('top1', 0.0),
                'top3': d.get('top3', 0.0),
                'mrr': d.get('mrr', 0.0)
            })
        df_out = pd.DataFrame(rows)
        out_csv = artifact_paths['by_scenario_test']
        df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"  [OK] 已导出测试 by_scenario: {out_csv}")
        best = sorted(by_sc.items(), key=lambda kv: kv[1].get('top1', 0.0), reverse=True)[:5]
        logger.info("\n  测试 by_scenario (Top-1 最好5个):")
        for sc, d in best:
            logger.info(f"    {sc}: n={d.get('n_windows', 0)}, top1={d.get('top1', 0.0):.4f}, top3={d.get('top3', 0.0):.4f}, mrr={d.get('mrr', 0.0):.4f}")

    # 供 run_chapter1.py 等脚本读取本次运行的指标
    _config_snapshot = build_config_snapshot(config, split_diag)
    last_metrics = {
        'mrr': float(test_metrics.get('mrr', 0.0)),
        'top1': float(test_metrics.get('topk_recall_1', 0.0)),
        'top3': float(test_metrics.get('topk_recall_3', 0.0)),
        'top5': float(test_metrics.get('topk_recall_5', 0.0)),
        'active_acc': float(test_metrics.get('active_acc', 0.0)),
        'active_period_recall': float(test_metrics.get('active_period_recall', 0.0)),
        'detection_latency_mean': float(test_metrics.get('detection_latency_mean', float('nan'))),
        'event_level_top1': float(test_metrics.get('event_level_top1', 0.0)),
        'event_level_top3': float(test_metrics.get('event_level_top3', 0.0)),
        'event_level_top5': float(test_metrics.get('event_level_top5', 0.0)),
        'seed': config.random_seed,
        'subdir': getattr(config, 'training_data_subdir', ''),
        'defect_csv': str(getattr(config, 'defect_matrix_file', '')),
        'split_diag': split_diag,
        'config_snapshot': _config_snapshot,
    }
    bt = test_metrics.get('by_defect_type', {})
    if bt:
        last_metrics['by_type_top1'] = {k: float(v.get('top1', 0.0)) for k, v in bt.items() if v and v.get('n', 0) > 0}
    sn = test_metrics.get('seen_nodes', {})
    un = test_metrics.get('unseen_nodes', {})
    if sn:
        last_metrics['seen_nodes'] = {'n': int(sn.get('n', 0)), 'mrr': float(sn.get('mrr', 0)), 'top1': float(sn.get('top1', 0)), 'top3': float(sn.get('top3', 0))}
    if un:
        last_metrics['unseen_nodes'] = {'n': int(un.get('n', 0)), 'mrr': float(un.get('mrr', 0)), 'top1': float(un.get('top1', 0)), 'top3': float(un.get('top3', 0))}
    out_dir = Path(config.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(artifact_paths['metrics'], 'w', encoding='utf-8') as _f:
        _json.dump(last_metrics, _f, ensure_ascii=False, indent=2)
    logger.info(f"  [OK] 已写指标文件 -> {artifact_paths['metrics']}")

    logger.info("=" * 80)

    # ========== 步骤6：可视化预测案例 ==========
    logger.info("\n[6/6] 预测案例分析...")
    analyze_predictions(model, test_loader, device, num_samples=5)

    logger.info("\n[OK] 训练完成")
    logger.info(f"   最佳模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
