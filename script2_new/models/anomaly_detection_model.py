# -*- coding: utf-8 -*-
"""
异常检测模型定义。来源：原 script2/models/anomaly_detection_model.py
包含 GRU+GCN 等模型架构，便于后续扩展和对比实验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


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
                 dropout=0.2,
                 temporal_encoder="gru"):
        super().__init__()

        self.input_dim = input_dim
        self.time_hidden_dim = time_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.num_nodes = num_nodes
        self.temporal_encoder_name = temporal_encoder

        # 时间编码器：GRU
        if temporal_encoder == "gru":
            self.time_encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=time_hidden_dim,
                num_layers=num_time_layers,
                batch_first=True,
                dropout=dropout if num_time_layers > 1 else 0
            )
        elif temporal_encoder == "lstm":
            self.time_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=time_hidden_dim,
                num_layers=num_time_layers,
                batch_first=True,
                dropout=dropout if num_time_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")

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
        if self.temporal_encoder_name == "lstm":
            h_n = h_n[0]
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
        if self.temporal_encoder_name == "lstm":
            h_n = h_n[0]
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
                 temporal_encoder="gru",
                 graph_features_path=None,
                 graph_features_dict=None,
                 use_flow_direction=True,
                 path_prior_mode="full",
                 use_propagation_delay=False,
                 propagation_delay_velocity_mps=0.5,
                 attention_max_hops=None,
                 observed_source_only=False,
                 graph_context_head=False,
                 type_conditioned_head=False,
                 type_head_temperature=1.0,
                 node_static_features=None,
                 static_feature_scale=1.0,
                 static_logit_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.time_hidden_dim = time_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.num_nodes = num_nodes
        self.use_flow_direction = use_flow_direction
        self.path_prior_mode = str(path_prior_mode).strip().lower()
        if self.path_prior_mode not in {"full", "distance_only", "none"}:
            raise ValueError(
                "path_prior_mode must be one of: full, distance_only, none"
            )
        self.use_propagation_delay = use_propagation_delay
        self.propagation_delay_velocity_mps = max(float(propagation_delay_velocity_mps), 1e-6)
        self.attention_max_hops = None if attention_max_hops in (None, "", 0) else int(attention_max_hops)
        self.observed_source_only = bool(observed_source_only)
        self.graph_context_head = bool(graph_context_head)
        self.type_conditioned_head = bool(type_conditioned_head)
        self.type_head_temperature = float(type_head_temperature)
        self.static_feature_scale = float(static_feature_scale)
        self.static_logit_bias = bool(static_logit_bias)
        self.temporal_encoder_name = temporal_encoder
        self.att_hidden_dim = 32

        # 时间编码器：与 GRU-GCN 相同
        if temporal_encoder == "gru":
            self.time_encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=time_hidden_dim,
                num_layers=num_time_layers,
                batch_first=True,
                dropout=dropout if num_time_layers > 1 else 0
            )
        elif temporal_encoder == "lstm":
            self.time_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=time_hidden_dim,
                num_layers=num_time_layers,
                batch_first=True,
                dropout=dropout if num_time_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")

        # 动态路径注意力：同时结合当前样本表示与静态路径先验
        path_dim = 5 if self.use_propagation_delay else 4
        self.query_proj = nn.Linear(time_hidden_dim, self.att_hidden_dim)
        self.key_proj = nn.Linear(time_hidden_dim, self.att_hidden_dim)
        self.path_proj = nn.Linear(path_dim, self.att_hidden_dim)
        self.att_out = nn.Linear(self.att_hidden_dim, 1)
        self.self_proj = nn.Linear(time_hidden_dim, spatial_hidden_dim)
        self.neigh_proj = nn.Linear(time_hidden_dim, spatial_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 与 baseline 一致的输出头
        self.node_head = nn.Linear(spatial_hidden_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)

        # 图路径特征 [N, N, 4] 预加载为 buffer（不参与训练）
        node_head_in_dim = spatial_hidden_dim * 2 if self.graph_context_head else spatial_hidden_dim
        self.node_head = nn.Linear(node_head_in_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)
        if self.type_conditioned_head:
            self.node_type_heads = nn.ModuleList([
                nn.Linear(node_head_in_dim, 2) for _ in range(3)
            ])

        if graph_features_dict is not None:
            self._register_graph_features_from_dict(graph_features_dict)
        elif graph_features_path is not None:
            self._register_graph_features_from_file(graph_features_path)
        self.has_node_static = node_static_features is not None
        if self.has_node_static:
            static_tensor = torch.as_tensor(node_static_features, dtype=torch.float32)
            if static_tensor.dim() != 2:
                raise ValueError("node_static_features must have shape [N, D]")
            self.register_buffer("node_static_feat", static_tensor)
            self.static_proj = nn.Linear(static_tensor.shape[-1], spatial_hidden_dim)
            if self.static_logit_bias:
                self.static_node_head = nn.Linear(static_tensor.shape[-1], 2)

    def _register_graph_features_from_dict(self, d):
        import numpy as np
        shortest_dist = d["shortest_dist"]
        pipe_length_dist = d["pipe_length_dist"]
        flow_direction = d["flow_direction"]
        elevation_diff = d["elevation_diff"]
        if not self.use_flow_direction:
            flow_direction = np.zeros_like(flow_direction, dtype=np.float32)
        N = shortest_dist.shape[0]
        feat_parts = [shortest_dist, pipe_length_dist, flow_direction, elevation_diff]
        if self.use_propagation_delay:
            delay_hours = np.clip(pipe_length_dist / (self.propagation_delay_velocity_mps * 3600.0), 0.0, 6.0).astype(np.float32)
            feat_parts.append(np.log1p(delay_hours).astype(np.float32))
        path_feat = np.stack(feat_parts, axis=-1).astype(np.float32)
        self.register_buffer("path_feat", torch.from_numpy(path_feat))
        self.num_nodes = N

    def _register_graph_features_from_file(self, path):
        import numpy as np
        data = np.load(path)
        self._register_graph_features_from_dict(dict(data))

    def _encode_time(self, x):
        B, T, N, nF = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, nF)
        enc_out = self.time_encoder(x_reshaped)
        if self.temporal_encoder_name == "lstm":
            _, (h_n, _) = enc_out
        else:
            _, h_n = enc_out
        return h_n[-1].view(B, N, -1)

    def _path_prior_features(self, path_feat):
        """Return the path channels exposed to attention logits.

        The original path tensor is still used for reachability masking. This
        isolates the contribution of path-feature values without allowing
        disconnected node pairs into the attention graph.
        """
        if self.path_prior_mode == "full":
            return path_feat
        if self.path_prior_mode == "distance_only":
            prior = torch.zeros_like(path_feat)
            prior[..., :2] = path_feat[..., :2]
            return prior
        return torch.zeros_like(path_feat)

    def _get_static_embeddings(self, device, num_nodes, batch_size):
        if not self.has_node_static:
            return None, None
        static_feat = self.node_static_feat.to(device)
        if static_feat.shape[0] != num_nodes:
            raise RuntimeError(f"node_static_feat nodes {static_feat.shape[0]} != input nodes {num_nodes}")
        static_embed = self.static_proj(static_feat).unsqueeze(0).expand(batch_size, -1, -1)
        return static_feat, static_embed

    def _dynamic_attention(self, h, path_feat, observed_mask=None):
        q = self.query_proj(h).unsqueeze(2)        # [B, N, 1, A]
        k = self.key_proj(h).unsqueeze(1)          # [B, 1, N, A]
        p = self.path_proj(self._path_prior_features(path_feat)).unsqueeze(0) # [1, N, N, A]

        att_hidden = torch.tanh(q + k + p)
        att_logits = self.att_out(att_hidden).squeeze(-1)  # [B, N, N]

        no_path = (path_feat[:, :, 0] >= 14.9) & (path_feat[:, :, 2] == 0)
        if self.attention_max_hops is not None:
            local_mask = path_feat[:, :, 0] > float(self.attention_max_hops)
            no_path = no_path | local_mask
        batch_mask = no_path.unsqueeze(0).expand(h.shape[0], -1, -1)
        if self.observed_source_only and observed_mask is not None:
            source_mask = ~observed_mask.bool().unsqueeze(1).expand(-1, h.shape[1], -1)
            batch_mask = batch_mask | source_mask
        att_logits = att_logits.masked_fill(batch_mask, -1e9)
        att_weights = F.softmax(att_logits, dim=-1)
        return torch.einsum("bij,bjd->bid", att_weights, h)

    def forward(self, x, adj, defect_type_override=None):
        B, T, N, nF = x.shape
        device = x.device

        h = self._encode_time(x)  # [B, N, time_hidden_dim]
        path_feat = self.path_feat.to(device)  # [N, N, 4]
        if path_feat.shape[0] != N:
            raise RuntimeError(f"path_feat nodes {path_feat.shape[0]} != input nodes {N}")

        observed_mask = (x[:, 0, :, -1] > 0.5) if self.observed_source_only else None
        h_agg = self._dynamic_attention(h, path_feat, observed_mask=observed_mask)
        static_feat, static_embed = self._get_static_embeddings(device, N, B)
        fused = self.self_proj(h) + self.neigh_proj(h_agg)
        if static_embed is not None:
            fused = fused + self.static_feature_scale * static_embed
        x_spatial = self.relu(fused)
        x_spatial = self.dropout(x_spatial)

        graph_repr = x_spatial.mean(dim=1)
        if self.graph_context_head:
            graph_context = graph_repr.unsqueeze(1).expand(-1, N, -1)
            node_input = torch.cat([x_spatial, graph_context], dim=-1)
        else:
            node_input = x_spatial
        logits_node = self.node_head(node_input)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        if self.type_conditioned_head:
            if defect_type_override is not None:
                type_probs = torch.softmax(
                    logits_defect_type / max(self.type_head_temperature, 1e-6),
                    dim=-1
                )
                valid_mask = defect_type_override >= 0
                if valid_mask.any():
                    override_onehot = F.one_hot(
                        defect_type_override[valid_mask].long(),
                        num_classes=3
                    ).to(type_probs.dtype)
                    type_probs = type_probs.clone()
                    type_probs[valid_mask] = override_onehot
            else:
                type_probs = torch.softmax(
                    logits_defect_type / max(self.type_head_temperature, 1e-6),
                    dim=-1
                )
            expert_logits = torch.stack([head(node_input) for head in self.node_type_heads], dim=1)
            logits_node = logits_node + torch.einsum("bt,btnc->bnc", type_probs, expert_logits)
        if static_feat is not None and self.static_logit_bias:
            static_bias = self.static_node_head(static_feat).unsqueeze(0).expand(B, -1, -1)
            logits_node = logits_node + static_bias
        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        device = x.device
        h = self._encode_time(x)
        path_feat = self.path_feat.to(device)
        observed_mask = (x[:, 0, :, -1] > 0.5) if self.observed_source_only else None
        h_agg = self._dynamic_attention(h, path_feat, observed_mask=observed_mask)
        _, static_embed = self._get_static_embeddings(device, h.shape[1], h.shape[0])
        fused = self.self_proj(h) + self.neigh_proj(h_agg)
        if static_embed is not None:
            fused = fused + self.static_feature_scale * static_embed
        return self.relu(fused)


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


class LSTMGCNModel(nn.Module):
    """
    用 LSTM 替换时间编码器，其余 GCN 与双头结构保持一致。
    """

    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,
                 num_nodes=None,
                 dropout=0.2,
                 temporal_encoder="gru"):
        super().__init__()

        self.input_dim = input_dim
        self.time_hidden_dim = time_hidden_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.num_nodes = num_nodes
        self.temporal_encoder_name = temporal_encoder

        self.time_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=time_hidden_dim,
            num_layers=num_time_layers,
            batch_first=True,
            dropout=dropout if num_time_layers > 1 else 0
        )

        self.spatial_layers = nn.ModuleList()
        for i in range(num_spatial_layers):
            in_dim = time_hidden_dim if i == 0 else spatial_hidden_dim
            self.spatial_layers.append(nn.Linear(in_dim, spatial_hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        node_head_in_dim = spatial_hidden_dim * 2 if self.graph_context_head else spatial_hidden_dim
        self.node_head = nn.Linear(node_head_in_dim, 2)
        self.has_defect_head = nn.Linear(spatial_hidden_dim, 2)
        self.defect_type_head = nn.Linear(spatial_hidden_dim, 3)
        if self.type_conditioned_head:
            self.node_type_heads = nn.ModuleList([
                nn.Linear(node_head_in_dim, 2) for _ in range(3)
            ])

    def forward(self, x, adj):
        B, T, N, F = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        _, (h_n, _) = self.time_encoder(x_reshaped)
        x_encoded = h_n[-1].view(B, N, -1)

        x_spatial = x_encoded
        for i, layer in enumerate(self.spatial_layers):
            if adj.dim() == 2:
                adj_norm = adj.unsqueeze(0).expand(B, -1, -1)
            else:
                adj_norm = adj.clone()

            n_nodes = adj_norm.shape[-1]
            identity = torch.eye(n_nodes, device=adj_norm.device, dtype=adj_norm.dtype).unsqueeze(0).expand(B, -1, -1)
            adj_norm = adj_norm + identity
            degree = adj_norm.sum(dim=2, keepdim=True) + 1e-8
            adj_norm = adj_norm / degree

            x_agg = torch.bmm(adj_norm.transpose(1, 2), x_spatial)
            x_spatial = layer(x_agg)
            if i < len(self.spatial_layers) - 1:
                x_spatial = self.relu(x_spatial)
                x_spatial = self.dropout(x_spatial)

        logits_node = self.node_head(x_spatial)
        graph_repr = x_spatial.mean(dim=1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        B, T, N, F = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        _, (h_n, _) = self.time_encoder(x_reshaped)
        x_encoded = h_n[-1].view(B, N, -1)

        x_spatial = x_encoded
        for i, layer in enumerate(self.spatial_layers):
            if adj.dim() == 2:
                adj_norm = adj.unsqueeze(0).expand(B, -1, -1)
            else:
                adj_norm = adj.clone()
            n_nodes = adj_norm.shape[-1]
            identity = torch.eye(n_nodes, device=adj_norm.device, dtype=adj_norm.dtype).unsqueeze(0).expand(B, -1, -1)
            adj_norm = adj_norm + identity
            degree = adj_norm.sum(dim=2, keepdim=True) + 1e-8
            adj_norm = adj_norm / degree
            x_agg = torch.bmm(adj_norm.transpose(1, 2), x_spatial)
            x_spatial = layer(x_agg)
            if i < len(self.spatial_layers) - 1:
                x_spatial = self.relu(x_spatial)
        return x_spatial


class LSTMOnlyModel(nn.Module):
    """
    只用 LSTM 做时间编码，不做图聚合。
    """

    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,
                 num_nodes=None,
                 dropout=0.2,
                 temporal_encoder="gru"):
        super().__init__()
        self.time_encoder = nn.LSTM(
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
        _, (h_n, _) = self.time_encoder(x_reshaped)
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
        _, (h_n, _) = self.time_encoder(x_reshaped)
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
    def __init__(self,
                 input_dim,
                 time_hidden_dim=128,
                 spatial_hidden_dim=256,
                 num_time_layers=1,
                 num_spatial_layers=2,
                 num_nodes=None,
                 dropout=0.2,
                 temporal_encoder="lstm"):
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
        _, (hidden, _) = self.time_encoder(x_reshaped)
        temporal_feat = hidden[-1].view(bsz, num_nodes, -1)
        return F.relu(self.temporal_proj(temporal_feat))

    def get_node_embeddings(self, x, adj):
        node_feat = self._encode_time(x)
        for layer in self.sage_layers:
            node_feat = layer(node_feat, adj)
        return node_feat

    def forward(self, x, adj):
        node_feat = self.get_node_embeddings(x, adj)
        logits_node = self.node_head(node_feat)
        graph_repr = node_feat.mean(dim=1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type


# ============ 模型工厂函数 ============

class HydraulicInverseLSTMModel(HydraulicInverseAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("temporal_encoder", "lstm")
        super().__init__(*args, **kwargs)


class HydraulicInverseContextModel(HydraulicInverseAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("graph_context_head", True)
        super().__init__(*args, **kwargs)


class HydraulicInverseTypeAwareModel(HydraulicInverseAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("type_conditioned_head", True)
        super().__init__(*args, **kwargs)


class HydraulicInverseContextFixedModel(HydraulicInverseAttentionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_head = nn.Linear(self.spatial_hidden_dim * 2, 2)

    def forward(self, x, adj, defect_type_override=None):
        B, T, N, nF = x.shape
        device = x.device

        h = self._encode_time(x)
        path_feat = self.path_feat.to(device)
        if path_feat.shape[0] != N:
            raise RuntimeError(f"path_feat nodes {path_feat.shape[0]} != input nodes {N}")

        observed_mask = (x[:, 0, :, -1] > 0.5) if self.observed_source_only else None
        h_agg = self._dynamic_attention(h, path_feat, observed_mask=observed_mask)
        static_feat, static_embed = self._get_static_embeddings(device, N, B)
        fused = self.self_proj(h) + self.neigh_proj(h_agg)
        if static_embed is not None:
            fused = fused + self.static_feature_scale * static_embed
        x_spatial = self.relu(fused)
        x_spatial = self.dropout(x_spatial)

        graph_repr = x_spatial.mean(dim=1)
        graph_context = graph_repr.unsqueeze(1).expand(-1, N, -1)
        node_input = torch.cat([x_spatial, graph_context], dim=-1)
        logits_node = self.node_head(node_input)
        if static_feat is not None and self.static_logit_bias:
            static_bias = self.static_node_head(static_feat).unsqueeze(0).expand(B, -1, -1)
            logits_node = logits_node + static_bias
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type


class HydraulicInverseStaticModel(HydraulicInverseAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("static_logit_bias", False)
        super().__init__(*args, **kwargs)


class HydraulicInverseStaticBiasModel(HydraulicInverseAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("static_logit_bias", True)
        super().__init__(*args, **kwargs)


class HydraulicInverseRankHeadModel(HydraulicInverseAttentionModel):
    """
    Minimal task-aligned variant for localization ranking:
    - replace the 2-class node head with a scalar score head
    - keep graph-level heads unchanged
    - add lightweight post-aggregation refinement layers so num_spatial_layers
      is no longer effectively ignored
    """

    def __init__(self, *args, **kwargs):
        graph_features_path = kwargs.get("graph_features_path", None)
        num_spatial_layers = int(kwargs.get("num_spatial_layers", 2))
        super().__init__(*args, **kwargs)
        if graph_features_path is None:
            default_graph_path = Path(__file__).resolve().parent.parent / "input_1" / "graph_path_features.npz"
            graph_features_path = str(default_graph_path) if default_graph_path.exists() else None
        self.graph_features_path_ref = graph_features_path
        self.rank_num_spatial_layers = max(1, num_spatial_layers)

        node_head_in_dim = self.spatial_hidden_dim * 2 if self.graph_context_head else self.spatial_hidden_dim
        hidden_dim = max(32, node_head_in_dim // 2)
        self.node_score_head = nn.Sequential(
            nn.Linear(node_head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(hidden_dim, 1),
        )

        self.refine_layers = nn.ModuleList()
        self.refine_norms = nn.ModuleList()
        for _ in range(self.rank_num_spatial_layers - 1):
            self.refine_layers.append(nn.Linear(self.spatial_hidden_dim, self.spatial_hidden_dim))
            self.refine_norms.append(nn.LayerNorm(self.spatial_hidden_dim))

        if self.static_logit_bias and hasattr(self, "node_static_feat"):
            static_in_dim = self.node_static_feat.shape[-1]
            self.static_score_head = nn.Linear(static_in_dim, 1)

    def forward(self, x, adj, defect_type_override=None):
        B, T, N, nF = x.shape
        device = x.device

        h = self._encode_time(x)
        if not hasattr(self, "path_feat"):
            if self.graph_features_path_ref is None:
                raise RuntimeError("path_feat is missing and graph_features_path_ref is not available")
            self._register_graph_features_from_file(self.graph_features_path_ref)
        path_feat = self.path_feat.to(device)
        if path_feat.shape[0] != N:
            raise RuntimeError(f"path_feat nodes {path_feat.shape[0]} != input nodes {N}")

        observed_mask = (x[:, 0, :, -1] > 0.5) if self.observed_source_only else None
        h_agg = self._dynamic_attention(h, path_feat, observed_mask=observed_mask)
        static_feat, static_embed = self._get_static_embeddings(device, N, B)
        fused = self.self_proj(h) + self.neigh_proj(h_agg)
        if static_embed is not None:
            fused = fused + self.static_feature_scale * static_embed
        x_spatial = self.relu(fused)
        x_spatial = self.dropout(x_spatial)

        for layer, norm in zip(self.refine_layers, self.refine_norms):
            residual = x_spatial
            x_spatial = self.relu(layer(x_spatial))
            x_spatial = self.dropout(x_spatial)
            x_spatial = norm(x_spatial + residual)

        graph_repr = x_spatial.mean(dim=1)
        if self.graph_context_head:
            graph_context = graph_repr.unsqueeze(1).expand(-1, N, -1)
            node_input = torch.cat([x_spatial, graph_context], dim=-1)
        else:
            node_input = x_spatial

        node_scores = self.node_score_head(node_input).squeeze(-1)
        if static_feat is not None and self.static_logit_bias and hasattr(self, "static_score_head"):
            node_scores = node_scores + self.static_score_head(static_feat).squeeze(-1).unsqueeze(0).expand(B, -1)

        logits_node = torch.stack([torch.zeros_like(node_scores), node_scores], dim=-1)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        return logits_node, logits_has_defect, logits_defect_type


class HydraulicInverseDeepAttentionModel(HydraulicInverseAttentionModel):
    """
    A deeper hydraulic-inverse variant that repeats dynamic attention refinement
    multiple times so far-away candidates can accumulate more discriminative
    structural evidence before the localization head.
    """

    def __init__(self, *args, **kwargs):
        graph_features_path = kwargs.get("graph_features_path", None)
        requested_layers = int(kwargs.get("num_spatial_layers", 2))
        allow_shallow = bool(kwargs.pop("allow_shallow_deepattn", False))
        if requested_layers < 1:
            raise ValueError("num_spatial_layers must be >= 1")
        super().__init__(*args, **kwargs)
        self.deep_num_layers = requested_layers if allow_shallow else max(3, requested_layers)
        self.allow_shallow_deepattn = allow_shallow
        if graph_features_path is None:
            default_graph_path = Path(__file__).resolve().parent.parent / "input_1" / "graph_path_features.npz"
            graph_features_path = str(default_graph_path) if default_graph_path.exists() else None
        self.graph_features_path_ref = graph_features_path

        self.layer_query_projs = nn.ModuleList()
        self.layer_key_projs = nn.ModuleList()
        self.layer_path_projs = nn.ModuleList()
        self.layer_att_outs = nn.ModuleList()
        self.layer_self_projs = nn.ModuleList()
        self.layer_neigh_projs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(self.deep_num_layers):
            in_dim = self.time_hidden_dim if layer_idx == 0 else self.spatial_hidden_dim
            self.layer_query_projs.append(nn.Linear(in_dim, self.att_hidden_dim))
            self.layer_key_projs.append(nn.Linear(in_dim, self.att_hidden_dim))
            path_dim = 5 if self.use_propagation_delay else 4
            self.layer_path_projs.append(nn.Linear(path_dim, self.att_hidden_dim))
            self.layer_att_outs.append(nn.Linear(self.att_hidden_dim, 1))
            self.layer_self_projs.append(nn.Linear(in_dim, self.spatial_hidden_dim))
            self.layer_neigh_projs.append(nn.Linear(in_dim, self.spatial_hidden_dim))
            if layer_idx > 0:
                self.layer_norms.append(nn.LayerNorm(self.spatial_hidden_dim))

    def _dynamic_attention_layer(self, state, path_feat, layer_idx, observed_mask=None):
        q = self.layer_query_projs[layer_idx](state).unsqueeze(2)
        k = self.layer_key_projs[layer_idx](state).unsqueeze(1)
        prior_feat = self._path_prior_features(path_feat)
        p = self.layer_path_projs[layer_idx](prior_feat).unsqueeze(0)

        att_hidden = torch.tanh(q + k + p)
        att_logits = self.layer_att_outs[layer_idx](att_hidden).squeeze(-1)

        no_path = (path_feat[:, :, 0] >= 14.9) & (path_feat[:, :, 2] == 0)
        if self.attention_max_hops is not None:
            local_mask = path_feat[:, :, 0] > float(self.attention_max_hops)
            no_path = no_path | local_mask
        batch_mask = no_path.unsqueeze(0).expand(state.shape[0], -1, -1)
        if self.observed_source_only and observed_mask is not None:
            source_mask = ~observed_mask.bool().unsqueeze(1).expand(-1, state.shape[1], -1)
            batch_mask = batch_mask | source_mask
        att_logits = att_logits.masked_fill(batch_mask, -1e9)
        att_weights = F.softmax(att_logits, dim=-1)
        return torch.einsum("bij,bjd->bid", att_weights, state)

    def _run_deep_stack(self, x):
        B, T, N, _ = x.shape
        device = x.device
        if not hasattr(self, "path_feat"):
            if self.graph_features_path_ref is None:
                raise RuntimeError("path_feat is missing for HydraulicInverseDeepAttentionModel")
            self._register_graph_features_from_file(self.graph_features_path_ref)
        path_feat = self.path_feat.to(device)
        if path_feat.shape[0] != N:
            raise RuntimeError(f"path_feat nodes {path_feat.shape[0]} != input nodes {N}")

        observed_mask = (x[:, 0, :, -1] > 0.5) if self.observed_source_only else None
        _, static_embed = self._get_static_embeddings(device, N, B)

        state = self._encode_time(x)
        for layer_idx in range(self.deep_num_layers):
            h_agg = self._dynamic_attention_layer(state, path_feat, layer_idx, observed_mask=observed_mask)
            fused = self.layer_self_projs[layer_idx](state) + self.layer_neigh_projs[layer_idx](h_agg)
            if static_embed is not None:
                fused = fused + self.static_feature_scale * static_embed
            fused = self.relu(fused)
            fused = self.dropout(fused)
            if layer_idx > 0:
                fused = self.layer_norms[layer_idx - 1](fused + state)
            state = fused
        return state

    def forward(self, x, adj, defect_type_override=None):
        x_spatial = self._run_deep_stack(x)
        B, _, N, _ = x.shape
        device = x.device
        graph_repr = x_spatial.mean(dim=1)
        if self.graph_context_head:
            graph_context = graph_repr.unsqueeze(1).expand(-1, N, -1)
            node_input = torch.cat([x_spatial, graph_context], dim=-1)
        else:
            node_input = x_spatial
        logits_node = self.node_head(node_input)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        if self.type_conditioned_head:
            if defect_type_override is not None:
                type_probs = torch.softmax(
                    logits_defect_type / max(self.type_head_temperature, 1e-6),
                    dim=-1
                )
                valid_mask = defect_type_override >= 0
                if valid_mask.any():
                    override_onehot = F.one_hot(
                        defect_type_override[valid_mask].long(),
                        num_classes=3
                    ).to(type_probs.dtype)
                    type_probs = type_probs.clone()
                    type_probs[valid_mask] = override_onehot
            else:
                type_probs = torch.softmax(
                    logits_defect_type / max(self.type_head_temperature, 1e-6),
                    dim=-1
                )
            expert_logits = torch.stack([head(node_input) for head in self.node_type_heads], dim=1)
            logits_node = logits_node + torch.einsum("bt,btnc->bnc", type_probs, expert_logits)
        if self.has_node_static and self.static_logit_bias:
            static_feat = self.node_static_feat.to(device)
            static_bias = self.static_node_head(static_feat).unsqueeze(0).expand(B, -1, -1)
            logits_node = logits_node + static_bias
        return logits_node, logits_has_defect, logits_defect_type

    def get_node_embeddings(self, x, adj):
        return self._run_deep_stack(x)


class HydraulicInverseDeepAttentionStaticModel(HydraulicInverseDeepAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("static_logit_bias", False)
        super().__init__(*args, **kwargs)


class HydraulicInverseDeepAttentionStaticBiasModel(HydraulicInverseDeepAttentionModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("static_logit_bias", True)
        super().__init__(*args, **kwargs)


class HydraulicInverseDeepAttentionReconModel(HydraulicInverseDeepAttentionModel):
    def __init__(self, *args, reconstruction_target_dim=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_target_dim = int(reconstruction_target_dim)
        recon_hidden = max(32, self.spatial_hidden_dim // 2)
        self.recon_head = nn.Sequential(
            nn.Linear(self.spatial_hidden_dim, recon_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.2),
            nn.Linear(recon_hidden, self.reconstruction_target_dim),
        )
        recon_ctx_dim = max(16, self.reconstruction_target_dim * 2)
        self.recon_context_proj = nn.Linear(self.reconstruction_target_dim, recon_ctx_dim)
        node_head_in_dim = self.spatial_hidden_dim * 2 if self.graph_context_head else self.spatial_hidden_dim
        self.node_head = nn.Linear(node_head_in_dim + recon_ctx_dim, 2)

    def forward(self, x, adj, defect_type_override=None):
        x_spatial = self._run_deep_stack(x)
        B, _, N, _ = x.shape
        graph_repr = x_spatial.mean(dim=1)

        recon_pred = self.recon_head(x_spatial)
        recon_context = self.relu(self.recon_context_proj(recon_pred))

        if self.graph_context_head:
            graph_context = graph_repr.unsqueeze(1).expand(-1, N, -1)
            node_input = torch.cat([x_spatial, graph_context, recon_context], dim=-1)
        else:
            node_input = torch.cat([x_spatial, recon_context], dim=-1)

        logits_node = self.node_head(node_input)
        logits_has_defect = self.has_defect_head(graph_repr)
        logits_defect_type = self.defect_type_head(graph_repr)
        if self.type_conditioned_head:
            if defect_type_override is not None:
                type_probs = torch.softmax(
                    logits_defect_type / max(self.type_head_temperature, 1e-6),
                    dim=-1
                )
                valid_mask = defect_type_override >= 0
                if valid_mask.any():
                    override_onehot = F.one_hot(
                        defect_type_override[valid_mask].long(),
                        num_classes=3
                    ).to(type_probs.dtype)
                    type_probs = type_probs.clone()
                    type_probs[valid_mask] = override_onehot
            else:
                type_probs = torch.softmax(
                    logits_defect_type / max(self.type_head_temperature, 1e-6),
                    dim=-1
                )
            expert_logits = torch.stack([head(node_input) for head in self.node_type_heads], dim=1)
            logits_node = logits_node + torch.einsum("bt,btnc->bnc", type_probs, expert_logits)
        return logits_node, logits_has_defect, logits_defect_type, recon_pred


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
        'lstm_gcn': LSTMGCNModel,
        'hydraulic_inverse': HydraulicInverseAttentionModel,
        'hydraulic_inverse_ctx': HydraulicInverseContextModel,
        'hydraulic_inverse_ctxfix': HydraulicInverseContextFixedModel,
        'hydraulic_inverse_typeaware': HydraulicInverseTypeAwareModel,
        'hydraulic_inverse_lstm': HydraulicInverseLSTMModel,
        'hydraulic_inverse_static': HydraulicInverseStaticModel,
        'hydraulic_inverse_staticbias': HydraulicInverseStaticBiasModel,
        'hydraulic_inverse_deepattn_static': HydraulicInverseDeepAttentionStaticModel,
        'hydraulic_inverse_deepattn_staticbias': HydraulicInverseDeepAttentionStaticBiasModel,
        'hydraulic_inverse_deepattn_recon': HydraulicInverseDeepAttentionReconModel,
        'hydraulic_inverse_rankhead': HydraulicInverseRankHeadModel,
        'hydraulic_inverse_deepattn': HydraulicInverseDeepAttentionModel,
        'gru_only': GRUOnlyModel,
        'lstm_only': LSTMOnlyModel,
        'lstm_graphsage_edge': TemporalGraphSAGEEdgeModel,
        'time_mean_linear': TimeMeanLinearModel,
    }

    if model_type not in model_registry:
        raise ValueError(f"未知模型类型: {model_type}. 可选: {list(model_registry.keys())}")

    # 仅水力逆向模型需要图路径特征；其余忽略 graph_features_*
    if model_type not in {
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
        'hydraulic_inverse_rankhead',
        'hydraulic_inverse_deepattn',
    }:
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('graph_features')}
    return model_registry[model_type](**kwargs)
