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
import torch
import torch.nn as nn
import torch.optim as optim
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

# 配置日志
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


def train_epoch(model, train_loader, optimizer, criterion_node, device, use_full_graph=False,
                lambda_type=0.0, criterion_type=None):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion_node: 节点级损失函数
        device: 计算设备
        use_full_graph: 是否在训练时使用全图（不限制候选集）
        lambda_type: 缺陷类型辅助头损失权重（0 表示不计算）
        criterion_type: 类型分类 CE 损失函数

    Returns:
        metrics: 训练指标字典
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        x = batch['features'].to(device)

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

        # 前向传播（含缺陷类型头 logits_defect_type [B, 3]）
        logits_node, logits_has_defect, logits_defect_type = model(x, adj)  # [B, N, 2], [B, 2], [B, 3]

        # ✅ 改进：使用软标签 + KL散度损失
        scores = logits_node[:, :, 1]  # [B, N]

        # 训练时可以选择是否使用全图
        if candidate_mask.dim() == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand_as(scores)
        if not use_full_graph:
            scores = scores.masked_fill(~candidate_mask, -1e9)

        # 仅保留启用定位的样本
        loc_mask = (soft_labels.sum(dim=-1) > 1e-8)
        if loc_mask.sum() == 0:
            continue

        scores = scores[loc_mask]
        soft_labels = soft_labels[loc_mask]
        candidate_mask = candidate_mask[loc_mask]
        target_node_idx = target_node_idx[loc_mask]

        # KL散度损失需要预测值经过log_softmax，真实值是概率分布
        candidate_mask_bool = candidate_mask.bool()
        scores_masked = scores.masked_fill(~candidate_mask_bool, -1e9)
        soft_labels_masked = soft_labels.masked_fill(~candidate_mask_bool, 0.0)
        soft_sum = soft_labels_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        soft_labels_norm = soft_labels_masked / soft_sum

        log_probs = torch.log_softmax(scores_masked, dim=-1)
        loss_loc = criterion_node(log_probs, soft_labels_norm)
        loss = loss_loc

        # 缺陷类型辅助头：仅对 has_defect=1 的样本计算 I/E/P 分类损失
        if lambda_type > 0 and criterion_type is not None:
            has_defect = batch['has_defect'].to(device)  # [B]
            defect_type_str_batch = batch['defect_type_str']  # list of str or tensor
            B = has_defect.shape[0]
            type_ids = []
            for i in range(B):
                if has_defect[i].item() != 1:
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

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return {
        'loss': total_loss / len(train_loader)
    }


def main():
    """主训练流程"""
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--seed", type=int, default=None, help="覆盖 config.random_seed，用于三种子等重复实验")
    _parser.add_argument("--subdir", type=str, default=None, help="覆盖 config.training_data_subdir，如 time_gated / time_gated_betweenness")
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
        config.node_timeseries_file = os.path.normpath(os.path.join(_root, _sub, "node_timeseries_with_residuals.parquet"))
        logger.info(f"使用数据子目录: {config.training_data_subdir} -> {config.node_timeseries_file}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    logger.info("\n" + "=" * 80)
    logger.info("异常检测模型训练")
    logger.info("=" * 80)

    # ========== 步骤1：加载数据集 ==========
    logger.info("\n[/6] 加载数据集...")
    logger.info(f"  时序数据: {config.node_timeseries_file}")

    # 检查文件是否存在
    if not Path(config.node_timeseries_file).exists():
        logger.warning(f"⚠️  文件不存在: {config.node_timeseries_file}")
        logger.warning("  正在尝试使用原始文件...")
        config.node_timeseries_file = config.node_timeseries_file.replace(
            '_with_residuals', ''
        )
        if not Path(config.node_timeseries_file).exists():
            raise FileNotFoundError(
                f"找不到时序数据文件！请先运行 25_baseline_feature_engineering.py"
            )
        logger.warning("  ⚠️  使用原始特征（没有残差），性能可能较差")

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
        e_class_oversample_ratio=getattr(config, 'dataset_e_class_oversample_ratio', 1)
    )

    input_dim = len(dataset.feature_cols)
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

    model = create_model(
        model_type=config.model_type,
        input_dim=input_dim,
        time_hidden_dim=config.time_hidden_dim,
        spatial_hidden_dim=config.spatial_hidden_dim,
        num_time_layers=config.num_time_layers,
        num_spatial_layers=config.num_spatial_layers,
        num_nodes=num_nodes,
        dropout=config.dropout
    ).to(device)

    logger.info(f"  模型类型: {config.model_type}")
    logger.info(f"  时间隐藏层: {config.time_hidden_dim}")
    logger.info(f"  空间隐藏层: {config.spatial_hidden_dim}")
    logger.info(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 步骤3：配置训练 ==========
    logger.info("\n[3/6] 配置训练...")

    # 损失函数
    # ✅ 改进：使用KL散度损失适配软标签
    criterion_node = nn.KLDivLoss(reduction='batchmean')
    lambda_type = getattr(config, 'lambda_type', 0.0)
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
    logger.info("  定位Loss: KL Divergence (soft labels + spatial awareness)")
    logger.info(f"  缺陷类型辅助头: lambda_type={lambda_type} (I/E/P, 仅 has_defect=1)")
    logger.info(f"  早停耐心值: {config.patience}")
    
    # ✅ 新增：训练配置
    use_full_graph = getattr(config, 'train_with_full_graph', False)  # 训练时是否使用全图
    logger.info(f"  训练模式: {'全图(无候选限制)' if use_full_graph else '候选集限制'}")

    # 评估损失函数（训练与评估分离）
    criterion_eval = torch.nn.CrossEntropyLoss(reduction='mean')

    # 如果仅评估（不训练），通过环境变量触发
    if os.environ.get('EVAL_ONLY') == '1':
        logger.info("\n[跳过训练] 仅评估并导出 by_scenario …")
        os.makedirs(config.model_save_dir, exist_ok=True)
        model_path = os.path.join(config.model_save_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到最佳模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate_model(
            model, test_loader, criterion_eval, device,
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
            out_dir = Path(config.output_dir)
            out_csv = out_dir / 'by_scenario_test_latest.csv'
            df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
            # 兼容默认路径
            default_csv = Path(r"E:\11.16\input2\by_scenario_latest.csv")
            try:
                df_out.to_csv(default_csv, index=False, encoding='utf-8-sig')
            except Exception:
                pass
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
        return

    # ========== 步骤4：训练循环 ==========
    logger.info("\n[4/6] 开始训练...")
    logger.info("=" * 80)

    # 以“定位任务”的排序指标作为主准则选模型（更符合工程目标）
    primary_metric = 'mrr'  # 也可改成 'topk_recall_1'
    best_val_score = -float('inf')
    best_val_loss = float('inf')  # 仅用于同分时的次级比较
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_node, device, use_full_graph,
            lambda_type=lambda_type, criterion_type=criterion_type if lambda_type > 0 else None
        )

        # 验证
        # 评估时使用 CrossEntropyLoss，因为评估需要硬标签计算准确率
        val_metrics = evaluate_model(
            model, val_loader, criterion_eval, device,
            train_seen_node_indices=train_seen_node_indices
        )

        # 学习率调度
        if scheduler:
            scheduler.step()

        # 打印指标
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"  训练Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  验证Loss: {val_metrics.get('loss', float('nan')):.4f}")
        logger.info(f"  验证 定位指标 - MRR: {val_metrics.get('mrr', 0.0):.4f}")
        logger.info(
            f"  验证 Top-K召回率 - "
            f"Top-1: {val_metrics.get('topk_recall_1', 0.0):.4f}, "
            f"Top-3: {val_metrics.get('topk_recall_3', 0.0):.4f}, "
            f"Top-5: {val_metrics.get('topk_recall_5', 0.0):.4f}"
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

        if improved:
            best_val_score = val_score
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            os.makedirs(config.output_dir, exist_ok=True)
            model_path = os.path.join(config.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_score,
                'primary_metric': primary_metric,
                'val_loss': best_val_loss,
                'config.py': config.__dict__
            }, model_path)
            logger.info(f"  ✅ 保存最佳模型 (Val {score_name}: {best_val_score:.4f}, Loss: {best_val_loss:.4f})")

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
                out_csv_val = Path(config.output_dir) / 'by_scenario_val_best.csv'
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
    model_path = os.path.join(config.output_dir, "best_model.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(
        model, test_loader, criterion_eval, device,
        train_seen_node_indices=train_seen_node_indices
    )

    logger.info("\n" + "=" * 80)
    logger.info("测试集最终评估")
    logger.info("=" * 80)
    logger.info(f"  测试Loss: {test_metrics['loss']:.4f}")
    mrr = test_metrics.get('mrr', 0.0)
    logger.info(f"  定位指标 - MRR: {mrr:.4f}")

    logger.info(f"\n  ✅ Top-K 召回率:")
    logger.info(f"     Top-1: {test_metrics.get('topk_recall_1', 0):.4f}")
    logger.info(f"     Top-3: {test_metrics.get('topk_recall_3', 0):.4f}")
    logger.info(f"     Top-5: {test_metrics.get('topk_recall_5', 0):.4f}")

    # Task4：见过/未见过节点、按缺陷类型 I/E/P
    sn = test_metrics.get('seen_nodes', {})
    un = test_metrics.get('unseen_nodes', {})
    if sn and un:
        logger.info(f"\n  📌 训练见过 vs 未见过缺陷节点:")
        logger.info(f"     见过节点: n={sn.get('n', 0)}, MRR={sn.get('mrr', 0):.4f}, Top-1={sn.get('top1', 0):.4f}, Top-3={sn.get('top3', 0):.4f}")
        logger.info(f"     未见过节点: n={un.get('n', 0)}, MRR={un.get('mrr', 0):.4f}, Top-1={un.get('top1', 0):.4f}, Top-3={un.get('top3', 0):.4f}")
    bt = test_metrics.get('by_defect_type', {})
    if bt:
        logger.info(f"\n  📌 按缺陷类型 (I/E/P):")
        for typ in ('I', 'E', 'P'):
            d = bt.get(typ, {})
            if d and d.get('n', 0) > 0:
                logger.info(f"     {typ}: n={d.get('n', 0)}, MRR={d.get('mrr', 0):.4f}, Top-1={d.get('top1', 0):.4f}, Top-3={d.get('top3', 0):.4f}")

    logger.info(f"\n  ℹ️  k-hop邻域命中率:")
    logger.info(f"     k=0 (精确命中): {test_metrics['khop_recall_0']:.4f}")
    logger.info(f"     k= (-hop邻域): {test_metrics['khop_recall_1']:.4f}")
    logger.info(f"     k=2 (2-hop邻域): {test_metrics['khop_recall_2']:.4f}")

    logger.info(f"\n  📊 预测距离分布:")
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
        out_dir = Path(config.output_dir)
        out_csv = out_dir / 'by_scenario_test_latest.csv'
        df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
        # 兼容默认路径（供分析脚本默认寻找）
        default_csv = Path(r"E:\11.16\input2\by_scenario_latest.csv")
        try:
            df_out.to_csv(default_csv, index=False, encoding='utf-8-sig')
        except Exception:
            pass
        logger.info(f"  [OK] 已导出测试 by_scenario: {out_csv}")
        best = sorted(by_sc.items(), key=lambda kv: kv[1].get('top1', 0.0), reverse=True)[:5]
        logger.info("\n  测试 by_scenario (Top-1 最好5个):")
        for sc, d in best:
            logger.info(f"    {sc}: n={d.get('n_windows', 0)}, top1={d.get('top1', 0.0):.4f}, top3={d.get('top3', 0.0):.4f}, mrr={d.get('mrr', 0.0):.4f}")

    # 供 run_chapter1.py 等脚本读取本次运行的指标
    last_metrics = {
        'mrr': float(test_metrics.get('mrr', 0.0)),
        'top1': float(test_metrics.get('topk_recall_1', 0.0)),
        'top3': float(test_metrics.get('topk_recall_3', 0.0)),
        'top5': float(test_metrics.get('topk_recall_5', 0.0)),
        'seed': config.random_seed,
        'subdir': getattr(config, 'training_data_subdir', ''),
    }
    bt = test_metrics.get('by_defect_type', {})
    if bt:
        last_metrics['by_type_top1'] = {k: float(v.get('top1', 0.0)) for k, v in bt.items() if v and v.get('n', 0) > 0}
    out_dir = Path(config.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(out_dir / 'last_run_metrics.json', 'w', encoding='utf-8') as _f:
        _json.dump(last_metrics, _f, ensure_ascii=False, indent=2)
    logger.info(f"  [OK] 已写 last_run_metrics.json -> {out_dir / 'last_run_metrics.json'}")

    logger.info("=" * 80)

    # ========== 步骤6：可视化预测案例 ==========
    logger.info("\n[6/6] 预测案例分析...")
    analyze_predictions(model, test_loader, device, num_samples=5)

    logger.info("\n✅ 训练完成！")
    logger.info(f"   最佳模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
