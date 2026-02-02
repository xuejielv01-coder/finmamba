# -*- coding: utf-8 -*-
"""
Market-Aware Graph (MAG) 模块

基于论文: FinMamba (arXiv:2502.06707)

核心组件:
1. 静态图: 基于行业分类
2. 动态图: 基于价格序列相似性
3. 自适应剪枝: 根据大盘状态调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGraph(nn.Module):
    """
    静态图模块 - 基于行业分类
    
    股票之间的静态关联由行业分类决定
    """
    
    def __init__(self, n_industries: int = 30, d_model: int = 64):
        super().__init__()
        
        self.n_industries = n_industries
        self.d_model = d_model
        
        # 行业嵌入
        self.industry_embed = nn.Embedding(n_industries, d_model)
        
        # 行业关系学习
        self.industry_relation = nn.Parameter(
            torch.eye(n_industries) + 0.1 * torch.randn(n_industries, n_industries)
        )
    
    def forward(self, industry_ids):
        """
        Args:
            industry_ids: (N,) 每只股票的行业 ID
        
        Returns:
            adj_matrix: (N, N) 邻接矩阵
        """
        N = industry_ids.shape[0]
        
        # 获取行业关系
        relation = torch.sigmoid(self.industry_relation)  # (n_industries, n_industries)
        
        # 构建邻接矩阵
        adj_matrix = relation[industry_ids][:, industry_ids]  # (N, N)
        
        return adj_matrix


class DynamicGraph(nn.Module):
    """
    动态图模块 - 基于价格序列相似性
    
    使用注意力机制计算股票间的动态关联
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Query, Key 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, stock_features):
        """
        Args:
            stock_features: (N, d_model) 股票特征
        
        Returns:
            adj_matrix: (N, N) 动态邻接矩阵
        """
        N = stock_features.shape[0]
        
        # 计算 Q, K
        Q = self.q_proj(stock_features)  # (N, d_model)
        K = self.k_proj(stock_features)  # (N, d_model)
        
        # 多头注意力
        Q = Q.view(N, self.n_heads, self.head_dim)  # (N, h, d)
        K = K.view(N, self.n_heads, self.head_dim)  # (N, h, d)
        
        # 计算注意力分数
        attn = torch.einsum('nhd,mhd->hnm', Q, K) * self.scale  # (h, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 平均多头
        adj_matrix = attn.mean(dim=0)  # (N, N)
        
        return adj_matrix


class AdaptivePruning(nn.Module):
    """
    自适应剪枝模块 - 根据大盘状态调整图结构
    
    当大盘趋势明确时，保留更多关联
    当大盘震荡时，剪枝更多噪声连接
    """
    
    def __init__(self, d_model: int = 64):
        super().__init__()
        
        # 大盘状态编码器
        self.market_encoder = nn.Sequential(
            nn.Linear(5, 32),  # 大盘 5 个特征: open, high, low, close, volume
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 剪枝阈值调节
        self.threshold_base = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, adj_matrix, market_state=None):
        """
        Args:
            adj_matrix: (N, N) 原始邻接矩阵
            market_state: (5,) 大盘状态特征 [可选]
        
        Returns:
            pruned_adj: (N, N) 剪枝后的邻接矩阵
        """
        if market_state is not None:
            # 根据大盘状态计算剪枝强度
            market_factor = self.market_encoder(market_state)  # (1,)
            threshold = self.threshold_base * (1 + 0.5 * (market_factor - 0.5))
        else:
            threshold = self.threshold_base
        
        # 软剪枝
        mask = (adj_matrix > threshold).float()
        pruned_adj = adj_matrix * mask
        
        # 归一化
        row_sum = pruned_adj.sum(dim=1, keepdim=True) + 1e-8
        pruned_adj = pruned_adj / row_sum
        
        return pruned_adj


class GraphConvLayer(nn.Module):
    """
    图卷积层
    """
    
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: (N, d_in) 节点特征
            adj: (N, N) 邻接矩阵
        
        Returns:
            y: (N, d_out) 更新后的节点特征
        """
        # 邻居聚合
        h = torch.matmul(adj, x)  # (N, d_in)
        
        # 线性变换
        h = self.linear(h)  # (N, d_out)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h


class MarketAwareGraph(nn.Module):
    """
    Market-Aware Graph (MAG) 完整模块
    
    融合静态图、动态图和自适应剪枝
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_industries: int = 30,
        n_gcn_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_static: bool = True,
        use_dynamic: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_static = use_static
        self.use_dynamic = use_dynamic
        
        # 静态图
        if use_static:
            self.static_graph = StaticGraph(n_industries, d_model)
        
        # 动态图
        if use_dynamic:
            self.dynamic_graph = DynamicGraph(d_model, n_heads, dropout)
        
        # 自适应剪枝
        self.pruning = AdaptivePruning(d_model)
        
        # 图融合权重
        if use_static and use_dynamic:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # GCN 层
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(d_model, d_model, dropout)
            for _ in range(n_gcn_layers)
        ])
        
        # 输出层范数
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        stock_features,
        industry_ids=None,
        market_state=None
    ):
        """
        Args:
            stock_features: (N, d_model) 股票特征
            industry_ids: (N,) 行业 ID [可选]
            market_state: (5,) 大盘状态 [可选]
        
        Returns:
            enhanced_features: (N, d_model) 图增强后的特征
        """
        N = stock_features.shape[0]
        device = stock_features.device
        
        # 构建邻接矩阵
        if self.use_static and self.use_dynamic and industry_ids is not None:
            static_adj = self.static_graph(industry_ids)
            dynamic_adj = self.dynamic_graph(stock_features)
            
            # 融合
            alpha = torch.sigmoid(self.fusion_weight)
            adj = alpha * static_adj + (1 - alpha) * dynamic_adj
        elif self.use_static and industry_ids is not None:
            adj = self.static_graph(industry_ids)
        elif self.use_dynamic:
            adj = self.dynamic_graph(stock_features)
        else:
            # 默认使用单位矩阵
            adj = torch.eye(N, device=device)
        
        # 自适应剪枝
        adj = self.pruning(adj, market_state)
        
        # GCN 传播
        h = stock_features
        for gcn in self.gcn_layers:
            h = gcn(h, adj) + h  # 残差连接
        
        h = self.norm(h)
        
        return h
