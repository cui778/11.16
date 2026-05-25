# -*- coding: utf-8 -*-
"""
评估指标工具函数。来源：原 script2/utils/metrics_patched.py
包含图距离、k-hop 命中率、Top-K、MRR、距离分布等。
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_graph_distance(adj_matrix: np.ndarray,
                           source: int,
                           target: int,
                           max_hops: int = 5,
                           undirected: bool = False) -> int:
    """
    计算图中两个节点之间的最短距离（BFS）

    Args:
        adj_matrix: 邻接矩阵 [N, N]
        source: 源节点索引
        target: 目标节点索引
        max_hops: 最大搜索深度

    Returns:
        距离（边数），如果不连通返回 max_hops +

    Examples:
        >>> adj = np.array([[0, , 0], [, 0, ], [0, , 0]])
        >>> compute_graph_distance(adj, 0, 2, max_hops=5)
        2
    """
    if source == target:
        return 0

    if undirected:
        # 无向化：A 或 A^T 只要有边就认为可达
        adj_u = ((adj_matrix > 0) | (adj_matrix.T > 0)).astype(np.int32)
    else:
        adj_u = (adj_matrix > 0).astype(np.int32)

    visited = set([source])
    queue = [(source, 0)]

    while queue:
        node, dist = queue.pop(0)

        if dist >= max_hops:
            continue

        # 获取邻接节点
        neighbors = np.where(adj_u[node] > 0)[0]
        for neighbor in neighbors:
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return max_hops + 1

def compute_khop_hit_from_indices(pred_node_idx: int,
                                  true_node_idx: int,
                                  adj_matrix: np.ndarray,
                                  k_hops: int = 1,
                                  undirected: bool = False) -> float:
    """基于(预测节点索引,真实节点索引)计算k-hop命中(0/1)。"""
    dist = compute_graph_distance(
        adj_matrix,
        pred_node_idx,
        true_node_idx,
        max_hops=k_hops,
        undirected=undirected,
    )
    return 1.0 if dist <= k_hops else 0.0


def compute_distance_distribution_from_indices(pred_node_idx: int,
                                               true_node_idx: int,
                                               adj_matrix: np.ndarray,
                                               max_hops: int = 15,
                                               cap: int = 10,
                                               undirected: bool = False) -> Dict:
    """基于(预测节点索引,真实节点索引)统计距离分布bucket。"""
    dist = compute_graph_distance(
        adj_matrix,
        pred_node_idx,
        true_node_idx,
        max_hops=max_hops,
        undirected=undirected,
    )
    if dist <= cap:
        return {int(dist): 1}
    return {f'>{cap}': 1}




def compute_neighborhood_accuracy(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  adj_matrix: np.ndarray,
                                  k_hops: int = 1) -> float:
    """
    计算 k-hop 邻域命中率

    判断预测的缺陷节点是否在真实缺陷节点的 k-hop 邻域内

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_pred: 预测概率或标签 [N]，值越大表示越可能是缺陷
        adj_matrix: 邻接矩阵 [N, N]
        k_hops: 邻域范围（边数）

    Returns:
        命中率（0-）

    Examples:
        >>> y_true = np.array([0, 0, , 0])
        >>> y_pred = np.array([0., 0.8, 0.3, 0.2])  # 预测节点1
        >>> adj = np.array([[0,,0,0], [,0,,0], [0,,0,], [0,0,,0]])
        >>> compute_neighborhood_accuracy(y_true, y_pred, adj, k_hops=)
        .0  # 节点1是节点2的1-hop邻居
    """
    # 找到真实缺陷节点
    true_defect_nodes = np.where(y_true == 1)[0]

    if len(true_defect_nodes) == 0:
        return 1.0  # 没有缺陷节点，认为全部正确

    # 找到预测的缺陷节点（概率最高的那个）
    pred_defect_node = int(np.argmax(y_pred))

    # 检查预测节点是否在真实缺陷节点的 k-hop 邻域内
    for true_node in true_defect_nodes:
        if compute_khop_hit_from_indices(pred_defect_node, int(true_node), adj_matrix, k_hops=k_hops) >= 1.0:
            return 1.0

    return 0.0


def compute_distance_distribution(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  adj_matrix: np.ndarray) -> Dict:
    """
    计算预测节点到真实缺陷节点的距离分布

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_pred: 预测概率或标签 [N]
        adj_matrix: 邻接矩阵 [N, N]

    Returns:
        距离分布字典 {0: count, : count, ..., '>10': count}

    Examples:
        >>> y_true = np.array([0, 0, , 0])
        >>> y_pred = np.array([0., 0.8, 0.3, 0.2])
        >>> adj = np.array([[0,,0,0], [,0,,0], [0,,0,], [0,0,,0]])
        >>> compute_distance_distribution(y_true, y_pred, adj)
        {: }  # 预测节点1到真实缺陷节点2的距离是1
    """
    true_defect_nodes = np.where(y_true == 1)[0]

    if len(true_defect_nodes) == 0:
        return {0: 1}  # 没有缺陷节点

    pred_defect_node = int(np.argmax(y_pred))

    # 计算到最近真实缺陷节点的距离（若有多个真实节点，取最近）
    min_distance = float('inf')
    for true_node in true_defect_nodes:
        dist = compute_graph_distance(adj_matrix, pred_defect_node, int(true_node), max_hops=15)
        min_distance = min(min_distance, dist)

    # 分类距离（cap=10）
    if min_distance <= 10:
        return {int(min_distance): 1}
    else:
        return {'>10': 1}


def compute_topk_recall(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        k: int = 5) -> float:
    """
    计算 Top-K 召回率

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_prob: 预测概率 [N]
        k: 取前K个预测

    Returns:
        Top-K 召回率（0-）

    Examples:
        >>> y_true = np.array([0, 0, , 0, ])
        >>> y_prob = np.array([0., 0.3, 0.9, 0.2, 0.8])
        >>> compute_topk_recall(y_true, y_prob, k=2)
        .0  # Top-2预测是[2,4]，覆盖了所有真实缺陷[2,4]
    """
    # 找到真实缺陷节点
    true_defect_idx = np.where(y_true == 1)[0]

    if len(true_defect_idx) == 0:
        return 1.0  # 没有缺陷节点，认为全部正确

    # 找到概率最高的K个节点
    topk_idx = np.argsort(y_prob)[-k:]

    # 检查真实缺陷节点是否在Top-K中
    hits = np.sum(np.isin(true_defect_idx, topk_idx))
    recall = hits / len(true_defect_idx)

    return recall


def compute_topk_accuracy(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          k: int = 5) -> float:
    """
    计算 Top-K 准确率（是否至少命中一个真实缺陷）

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_prob: 预测概率 [N]
        k: 取前K个预测

    Returns:
        Top-K 准确率（0 or ）
    """
    true_defect_idx = np.where(y_true == 1)[0]

    if len(true_defect_idx) == 0:
        return 1.0

    topk_idx = np.argsort(y_prob)[-k:]

    # 只要有任何一个命中就算成功
    return 1.0 if np.any(np.isin(topk_idx, true_defect_idx)) else 0.0


def compute_precision_at_k(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           k: int = 5) -> float:
    """
    计算 Precision@K

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_prob: 预测概率 [N]
        k: 取前K个预测

    Returns:
        Precision@K（0-）
    """
    true_defect_idx = np.where(y_true == 1)[0]

    if len(true_defect_idx) == 0:
        return 1.0

    topk_idx = np.argsort(y_prob)[-k:]
    hits = np.sum(np.isin(topk_idx, true_defect_idx))

    return hits / k


def compute_mrr(y_true: np.ndarray,
                y_prob: np.ndarray) -> float:
    """
    计算平均倒数排名 (Mean Reciprocal Rank)

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_prob: 预测概率 [N]

    Returns:
        MRR分数（0-）

    Examples:
        >>> y_true = np.array([0, 0, , 0])
        >>> y_prob = np.array([0., 0.3, 0.9, 0.2])
        >>> compute_mrr(y_true, y_prob)
        .0  # 第一个缺陷排在第1位，MRR=/=.0
    """
    true_defect_idx = np.where(y_true == 1)[0]

    if len(true_defect_idx) == 0:
        return 1.0

    # 按概率降序排列
    sorted_idx = np.argsort(y_prob)[::-1]

    # 找到第一个真实缺陷的排名
    for rank, idx in enumerate(sorted_idx, 1):
        if idx in true_defect_idx:
            return 1.0 / rank

    return 0.0


def compute_ndcg(y_true: np.ndarray,
                 y_prob: np.ndarray,
                 k: int = 10) -> float:
    """
    计算归一化折损累积增益 (NDCG@K)

    Args:
        y_true: 真实标签 [N]，1表示缺陷节点
        y_prob: 预测概率 [N]
        k: 计算前K个

    Returns:
        NDCG@K分数（0-）
    """
    true_defect_idx = set(np.where(y_true == 1)[0])

    if len(true_defect_idx) == 0:
        return 1.0

    # 按概率降序排列
    topk_idx = np.argsort(y_prob)[::-1][:k]

    # 计算DCG
    dcg = 0.0
    for i, idx in enumerate(topk_idx, 1):
        if idx in true_defect_idx:
            dcg += 1.0 / np.log2(i + 1)

    # 计算IDCG (理想情况下的DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_defect_idx), k)))

    return dcg / idcg if idcg > 0 else 0.0