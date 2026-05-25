# -*- coding: utf-8 -*-
"""
异常检测模型定义。来源：原 script2/models/anomaly_detection_model.py
包含 GRU+GCN 等模型架构，便于后续扩展和对比实验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetectionModel(nn.Module):
    """
    节点级缺陷定位模型 - 时空图神经网络

    架构：
        . 时间编码器：GRU提取每个节点的时序特征
        2. 空间编码器：GCN聚合邻域信息
        3. 分类头：节点级二分类（缺陷/正常）

    Args:
        input_dim: 输入特征维度
        time_hidden_dim: 时间编码器隐藏层维度
        spatial_hidden_dim: 空间编码器隐藏层维度
        num_time_layers: GRU层数
        num_spatial_layers: GCN层数
        num_nodes: 节点数量
        dropout: Dropout比率
    """

    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,
                 num_nodes=None,
                 dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.time_hidden_dim = time_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.num_nodes = num_nodes

        # 时间编码器：GRU
        self.time_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=time_hidden_dim,
            num_layers=num_time_layers,
            batch_first=True,
            dropout=dropout if num_time_layers > 1 else 0
        )

        # 空间编码器：多层GCN
        self.spatial_layers = nn.ModuleList()
        for i in range(num_spatial_layers):
            in_dim = time_hidden_dim if i == 0 else spatial_hidden_dim
            self.spatial_layers.append(nn.Linear(in_dim, spatial_hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 节点级分类头：二分类（缺陷定位）
        self.node_head = nn.Linear(spatial_hidden_dim, 2)
        # 图级分类头：有/无缺陷（两阶段/多任务）
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        # 图级辅助头：缺陷类型 I/E/P（仅对有缺陷样本监督，辅助表征）
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)

    def forward(self, x, adj):
        """
        前向传播

        Args:
            x: 输入特征 [B, T, N, F]
            adj: 邻接矩阵 [N, N] 或 [B, N, N]

        Returns:
            logits_node: 节点级预测logits [B, N, 2]
            logits_has_defect: 图级有/无缺陷预测logits [B, 2]
        """
        B, T, N, F = x.shape

        # 时间编码：对每个节点的时序数据进行编码
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # [B*N, T, F]
        _, h_n = self.time_encoder(x_reshaped)
        x_encoded = h_n[-1].view(B, N, -1)  # [B, N, time_hidden_dim]

        # 空间编码：GCN聚合邻域信息（含自环，保留节点自身特征）
        x_spatial = x_encoded
        for i, layer in enumerate(self.spatial_layers):
            # 邻接矩阵 [B, N, N]
            if adj.dim() == 2:
                adj_norm = adj.unsqueeze(0).expand(B, -1, -1)
            else:
                adj_norm = adj.clone()

            # 加自环 A_hat = A + I，使缺陷节点自身残差信号得以保留
            N = adj_norm.shape[-1]
            I = torch.eye(N, device=adj_norm.device, dtype=adj_norm.dtype).unsqueeze(0).expand(B, -1, -1)
            adj_norm = adj_norm + I

            # 度归一化：D^{-1} A_hat
            D = adj_norm.sum(dim=2, keepdim=True) + 1e-8
            adj_norm = adj_norm / D

            # 图卷积：下游节点聚合上游特征。约定 A[u,v]=1 表示 u→v（水从u流向v），
            # (A @ x)[u] 会聚合下游 v 的特征；物理上应为下游 u 聚合上游 v，故用 A^T
            x_agg = torch.bmm(adj_norm.transpose(1, 2), x_spatial)  # [B, N, hidden_dim]
            x_spatial = layer(x_agg)

            # 激活和正则化（最后一层除外）
            if i < len(self.spatial_layers) - 1:
                x_spatial = self.relu(x_spatial)
                x_spatial = self.dropout(x_spatial)

        # 节点级分类
        logits_node = self.node_head(x_spatial)  # [B, N, 2]
        # 图级有/无缺陷分类（对节点表示做平均池化）
        graph_repr = x_spatial.mean(dim=1)  # [B, hidden]
        logits_has_defect = self.has_defect_head(graph_repr)  # [B, 2]
        logits_defect_type = self.defect_type_head(graph_repr)  # [B, 3] I/E/P

        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        """
        获取节点嵌入表示（用于可视化或下游任务）

        Args:
            x: 输入特征 [B, T, N, F]
            adj: 邻接矩阵 [N, N] 或 [B, N, N]

        Returns:
            embeddings: 节点嵌入 [B, N, spatial_hidden_dim]
        """
        B, T, N, F = x.shape

        # 时间编码
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        _, h_n = self.time_encoder(x_reshaped)
        x_encoded = h_n[-1].view(B, N, -1)

        # 空间编码（与 forward 一致：含自环）
        x_spatial = x_encoded
        for i, layer in enumerate(self.spatial_layers):
            if adj.dim() == 2:
                adj_norm = adj.unsqueeze(0).expand(B, -1, -1)
            else:
                adj_norm = adj.clone()
            N = adj_norm.shape[-1]
            I = torch.eye(N, device=adj_norm.device, dtype=adj_norm.dtype).unsqueeze(0).expand(B, -1, -1)
            adj_norm = adj_norm + I
            D = adj_norm.sum(dim=2, keepdim=True) + 1e-8
            adj_norm = adj_norm / D
            # 与 forward 一致：A^T 使下游节点聚合上游特征
            x_agg = torch.bmm(adj_norm.transpose(1, 2), x_spatial)
            x_spatial = layer(x_agg)

            if i < len(self.spatial_layers) - 1:
                x_spatial = self.relu(x_spatial)

        return x_spatial


# ============ 水力逆向注意力模型（路径特征 → 注意力） ============

class HydraulicInverseAttentionModel(nn.Module):
    """
    用图路径物理特征（最短距离、管道长度、高程差、流向）做逆向注意力，替代 GCN。
    时间编码器 GRU 不变；空间层：对每个节点 c，用 c→s 的路径特征得到 α(c,s)，再汇聚 h_c = Σ_s α(c,s)·W·h_s。
    """

    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,  # 忽略，兼容 create_model 参数
                 num_nodes=None,
                 dropout=0.2,
                 graph_features_path=None,
                 graph_features_dict=None,
                 use_flow_direction=True,
                 use_propagation_delay=False):
        super().__init__()
        self.input_dim = input_dim
        self.time_hidden_dim = time_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.num_nodes = num_nodes
        self.use_flow_direction = use_flow_direction
        self.use_propagation_delay = use_propagation_delay

        # 时间编码器：与 GRU-GCN 相同
        self.time_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=time_hidden_dim,
            num_layers=num_time_layers,
            batch_first=True,
            dropout=dropout if num_time_layers > 1 else 0
        )

        # 路径特征 → 注意力 logit：4 维 (shortest_dist, pipe_length, flow_direction, elevation_diff)
        path_dim = 4
        self.path_to_att = nn.Sequential(
            nn.Linear(path_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.proj = nn.Linear(time_hidden_dim, spatial_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 与 baseline 一致的输出头
        self.node_head = nn.Linear(spatial_hidden_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)

        # 图路径特征 [N, N, 4] 预加载为 buffer（不参与训练）
        if graph_features_dict is not None:
            self._register_graph_features_from_dict(graph_features_dict)
        elif graph_features_path is not None:
            self._register_graph_features_from_file(graph_features_path)

    def _register_graph_features_from_dict(self, d):
        import numpy as np
        shortest_dist = d["shortest_dist"]
        pipe_length_dist = d["pipe_length_dist"]
        flow_direction = d["flow_direction"]
        elevation_diff = d["elevation_diff"]
        N = shortest_dist.shape[0]
        path_feat = np.stack([shortest_dist, pipe_length_dist, flow_direction, elevation_diff], axis=-1).astype(np.float32)
        self.register_buffer("path_feat", torch.from_numpy(path_feat))
        self.num_nodes = N

    def _register_graph_features_from_file(self, path):
        import numpy as np
        data = np.load(path)
        self._register_graph_features_from_dict(dict(data))

    def forward(self, x, adj):
        B, T, N, nF = x.shape
        device = x.device

        # 时间编码
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, nF)
        _, h_n = self.time_encoder(x_reshaped)
        h = h_n[-1].view(B, N, -1)  # [B, N, time_hidden_dim]

        path_feat = self.path_feat.to(device)  # [N, N, 4]
        if path_feat.shape[0] != N:
            raise RuntimeError(f"path_feat nodes {path_feat.shape[0]} != input nodes {N}")

        # 无连接掩码：shortest_dist 为最大截断值(15) 且 flow_direction==0 视为无连接
        no_path = (path_feat[:, :, 0] >= 14.9) & (path_feat[:, :, 2] == 0)
        no_path = no_path.float() * -1e9  # [N, N]

        att_logits = self.path_to_att(path_feat).squeeze(-1)  # [N, N]
        att_logits = att_logits + no_path
        att_weights = F.softmax(att_logits, dim=-1)  # [N, N]，行和为 1

        # h_new[i] = sum_j att_weights[i,j] * h[:,j,:]  -> [B, N, time_hidden_dim]
        h_agg = torch.einsum("ij,bjd->bid", att_weights, h)
        x_spatial = self.relu(self.proj(h_agg))
        x_spatial = self.dropout(x_spatial)

        logits_node = self.node_head(x_spatial)
        graph_repr = x_spatial.mean(dim=1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        B, T, N, nF = x.shape
        device = x.device
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, nF)
        _, h_n = self.time_encoder(x_reshaped)
        h = h_n[-1].view(B, N, -1)
        path_feat = self.path_feat.to(device)
        no_path = (path_feat[:, :, 0] >= 14.9) & (path_feat[:, :, 2] == 0)
        no_path = no_path.float() * -1e9
        att_logits = self.path_to_att(path_feat).squeeze(-1) + no_path
        att_weights = F.softmax(att_logits, dim=-1)
        h_agg = torch.einsum("ij,bjd->bid", att_weights, h)
        return self.relu(self.proj(h_agg))


# ============ 其他模型变体（预留接口） ============

class LSTMGATModel(nn.Module):
    """
    LSTM + GAT 变体（预留，后续可扩展）
    """

    def __init__(self, input_dim, **kwargs):
        super().__init__()
        # TODO: 实现LSTM+GAT架构
        pass

    def forward(self, x, adj):
        raise NotImplementedError("LSTM+GAT模型待实现")


class TransformerGNNModel(nn.Module):
    """
    Transformer + GNN 变体（预留，后续可扩展）
    """

    def __init__(self, input_dim, **kwargs):
        super().__init__()
        # TODO: 实现Transformer+GNN架构
        pass

    def forward(self, x, adj):
        raise NotImplementedError("Transformer+GNN模型待实现")


# ============ 简单基线（用于判断任务是否真的需要图/复杂时序） ============

class GRUOnlyModel(nn.Module):
    """
    只用 GRU 做时间编码，不做任何空间聚合（不用邻接矩阵）。
    用于对比：图结构是否必要。
    """
    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,
                 num_nodes=None,
                 dropout=0.2):
        super().__init__()
        self.time_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=time_hidden_dim,
            num_layers=num_time_layers,
            batch_first=True,
            dropout=dropout if num_time_layers > 1 else 0
        )
        self.proj = nn.Linear(time_hidden_dim, spatial_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.node_head = nn.Linear(spatial_hidden_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)

    def forward(self, x, adj):
        B, T, N, nF = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, nF)
        _, h_n = self.time_encoder(x_reshaped)
        h = h_n[-1].view(B, N, -1)
        x_spatial = self.relu(self.proj(h))
        x_spatial = self.dropout(x_spatial)
        logits_node = self.node_head(x_spatial)
        graph_repr = x_spatial.mean(dim=1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        B, T, N, nF = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, nF)
        _, h_n = self.time_encoder(x_reshaped)
        h = h_n[-1].view(B, N, -1)
        return self.relu(self.proj(h))


class TimeMeanLinearModel(nn.Module):
    """
    最简单基线：对时间维做 mean，再一层线性 → 节点得分。
    不用 GRU，不用图。用于判断“时序+图”是否都有必要。
    """
    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,
                 num_nodes=None,
                 dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, spatial_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.node_head = nn.Linear(spatial_hidden_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)

    def forward(self, x, adj):
        B, T, N, nF = x.shape
        x_mean = x.mean(dim=1)  # [B, N, nF]
        x_spatial = self.relu(self.proj(x_mean))
        x_spatial = self.dropout(x_spatial)
        logits_node = self.node_head(x_spatial)
        graph_repr = x_spatial.mean(dim=1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        B, T, N, nF = x.shape
        x_mean = x.mean(dim=1)
        return self.relu(self.proj(x_mean))


# ============ 模型工厂函数 ============

class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x[:, :, -1]


class GraphSAGELayerWithEdge(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, adj):
        if adj.dim() == 2:
            adj_batch = adj.unsqueeze(0).expand(node_feat.size(0), -1, -1)
        else:
            adj_batch = adj
        adj_batch = adj_batch.float()
        incoming_adj = adj_batch.transpose(1, 2)
        degree = incoming_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_mean = torch.bmm(incoming_adj, node_feat) / degree

        src_feat = node_feat.unsqueeze(2).expand(-1, -1, node_feat.size(1), -1)
        dst_feat = node_feat.unsqueeze(1).expand(-1, node_feat.size(1), -1, -1)
        edge_input = torch.cat([src_feat, dst_feat, dst_feat - src_feat, src_feat * dst_feat], dim=-1)
        edge_feat = self.edge_mlp(edge_input)

        edge_mask = incoming_adj.unsqueeze(-1)
        edge_agg = (edge_feat * edge_mask).sum(dim=1) / degree

        out = torch.cat([node_feat, neighbor_mean, edge_agg], dim=-1)
        out = self.update(out)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + node_feat)


class TemporalGraphSAGEEdgeModel(nn.Module):
    def __init__(
        self,
        input_dim,
        time_hidden_dim=128,
        spatial_hidden_dim=256,
        num_time_layers=1,
        num_spatial_layers=2,
        num_nodes=None,
        dropout=0.2,
        temporal_encoder="lstm",
    ):
        super().__init__()
        self.temporal_encoder_name = temporal_encoder
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)

        if temporal_encoder == "lstm":
            self.time_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=time_hidden_dim,
                num_layers=num_time_layers,
                batch_first=True,
                dropout=dropout if num_time_layers > 1 else 0.0,
            )
        elif temporal_encoder == "tcn":
            self.time_encoder = TemporalConvEncoder(input_dim, time_hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")

        self.temporal_proj = nn.Linear(time_hidden_dim, spatial_hidden_dim)
        self.sage_layers = nn.ModuleList(
            [GraphSAGELayerWithEdge(spatial_hidden_dim, dropout=dropout) for _ in range(num_spatial_layers)]
        )
        self.node_head = nn.Linear(spatial_hidden_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)

    def _encode_time(self, x):
        bsz, steps, num_nodes, num_feat = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(bsz * num_nodes, steps, num_feat)
        if self.temporal_encoder_name == "lstm":
            _, (hidden, _) = self.time_encoder(x_reshaped)
            temporal_feat = hidden[-1]
        else:
            temporal_feat = self.time_encoder(x_reshaped)
        temporal_feat = temporal_feat.view(bsz, num_nodes, -1)
        return F.relu(self.temporal_proj(temporal_feat))

    def get_node_embeddings(self, x, adj):
        node_feat = self._encode_time(x)
        for layer in self.sage_layers:
            node_feat = layer(node_feat, adj)
        return node_feat

    def forward(self, x, adj):
        node_feat = self.get_node_embeddings(x, adj)
        node_feat = self.dropout(node_feat)
        logits_node = self.node_head(node_feat)
        graph_repr = node_feat.mean(dim=1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type


class LSTMGraphSAGEEdgeModel(TemporalGraphSAGEEdgeModel):
    def __init__(self, *args, **kwargs):
        kwargs["temporal_encoder"] = "lstm"
        super().__init__(*args, **kwargs)


class TCNGraphSAGEEdgeModel(TemporalGraphSAGEEdgeModel):
    def __init__(self, *args, **kwargs):
        kwargs["temporal_encoder"] = "tcn"
        super().__init__(*args, **kwargs)


def create_model(model_type='gru_gcn', **kwargs):
    """
    模型工厂函数，便于切换不同模型架构

    Args:
        model_type: 模型类型 ('gru_gcn', 'lstm_gat', 'transformer_gnn')
        **kwargs: 模型参数

    Returns:
        model: 实例化的模型
    """
    model_registry = {
        'gru_gcn': AnomalyDetectionModel,
        'hydraulic_inverse': HydraulicInverseAttentionModel,
        'gru_only': GRUOnlyModel,
        'time_mean_linear': TimeMeanLinearModel,
        'lstm_graphsage_edge': LSTMGraphSAGEEdgeModel,
        'tcn_graphsage_edge': TCNGraphSAGEEdgeModel,
        'lstm_gat': LSTMGATModel,
        'transformer_gnn': TransformerGNNModel,
    }

    if model_type not in model_registry:
        raise ValueError(f"未知模型类型: {model_type}. 可选: {list(model_registry.keys())}")

    # 仅水力逆向模型需要图路径特征；其余忽略 graph_features_*
    if model_type != 'hydraulic_inverse':
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('graph_features')}
    return model_registry[model_type](**kwargs)
