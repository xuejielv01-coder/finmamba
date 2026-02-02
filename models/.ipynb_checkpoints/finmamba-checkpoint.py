# -*- coding: utf-8 -*-
"""
FinMamba 完整模型实现

基于论文: FinMamba: Market-Aware Graph Enhanced Mamba for Stock Movement Prediction

核心组件:
1. 特征嵌入层 - 将原始特征 + 行业代码投影到模型维度
2. Multi-Level Mamba - 多尺度时序建模 (日级/周级)
3. 行业嵌入 - 正确融入行业关系
4. 预测头 - 涨跌预测/排序分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mamba_block import MambaEncoder


class IndustryEmbedding(nn.Module):
    """
    行业嵌入层
    
    将行业代码转换为可学习的嵌入向量
    """
    
    def __init__(
        self,
        n_industries: int = 30,
        d_model: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_industries = n_industries
        self.d_model = d_model
        
        # 行业嵌入表
        self.embedding = nn.Embedding(n_industries, d_model)
        
        # 行业关系矩阵 (学习行业间相关性)
        self.industry_relation = nn.Parameter(
            torch.eye(n_industries) + 0.1 * torch.randn(n_industries, n_industries)
        )
        
        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, industry_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            industry_ids: (batch,) 行业ID，整数 [0, n_industries)
            
        Returns:
            industry_emb: (batch, d_model) 行业嵌入向量
        """
        # 获取行业嵌入
        emb = self.embedding(industry_ids)  # (B, d_model)
        
        # 融入行业关系信息
        relation_weights = F.softmax(self.industry_relation, dim=-1)
        weighted_emb = relation_weights[industry_ids]  # (B, n_industries)
        all_emb = self.embedding.weight  # (n_industries, d_model)
        relation_emb = torch.matmul(weighted_emb, all_emb)  # (B, d_model)
        
        # 融合原始嵌入和关系嵌入
        emb = emb + 0.3 * relation_emb
        
        return self.proj(emb)


class FeatureEmbedding(nn.Module):
    """
    特征嵌入层
    
    将原始特征投影到模型维度，并融合行业嵌入
    """
    
    def __init__(
        self,
        input_dim: int = 48,  # 原始特征维度
        d_model: int = 128,
        n_industries: int = 30,
        use_industry: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_industry = use_industry
        
        # 简化：减少特征投影层数，从2层减少到1层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 行业嵌入
        if use_industry:
            self.industry_emb = IndustryEmbedding(
                n_industries=n_industries,
                d_model=d_model,
                dropout=dropout
            )
        
        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, d_model))  # 支持最大256序列长度
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 融合层 (特征 + 行业)
        if use_industry:
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        industry_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 原始特征
            industry_ids: (batch,) 行业ID [可选]
            
        Returns:
            emb: (batch, seq_len, d_model) 嵌入后的特征
        """
        batch, seq_len, _ = x.shape
        
        # 特征投影
        h = self.feature_proj(x)  # (B, L, d_model)
        
        # 添加位置编码
        h = h + self.pos_embed[:, :seq_len, :]
        
        # 融合行业嵌入
        if self.use_industry and industry_ids is not None:
            industry_emb = self.industry_emb(industry_ids)  # (B, d_model)
            industry_emb = industry_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, d_model)
            
            # 拼接并融合
            h = torch.cat([h, industry_emb], dim=-1)  # (B, L, 2 * d_model)
            h = self.fusion(h)  # (B, L, d_model)
        
        return h


class MultiLevelMamba(nn.Module):
    """
    多层级 Mamba 模块
    
    同时处理不同时间尺度的特征:
    - 日级 (fine-grained): 捕捉短期波动
    - 周级 (medium-grained): 捕捉中期趋势
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        levels: Tuple[int, ...] = (1, 5),  # 日级/周级
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.levels = levels
        self.n_levels = len(levels)
        
        # 每个尺度的 Mamba 编码器
        self.encoders = nn.ModuleList([
            MambaEncoder(
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(self.n_levels)
        ])
        
        # 下采样层 (用于生成不同尺度)
        self.downsample = nn.ModuleList([
            nn.AvgPool1d(kernel_size=level, stride=level) if level > 1 else nn.Identity()
            for level in levels
        ])
        
        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * self.n_levels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            y: (batch, d_model) 融合后的特征
        """
        batch, seq_len, d_model = x.shape
        
        level_outputs = []
        
        for i, (level, encoder, downsample) in enumerate(
            zip(self.levels, self.encoders, self.downsample)
        ):
            # 下采样: (B, L, D) -> (B, D, L) -> pool -> (B, D, L') -> (B, L', D)
            if level > 1:
                x_level = x.transpose(1, 2)  # (B, D, L)
                x_level = downsample(x_level)  # (B, D, L')
                x_level = x_level.transpose(1, 2)  # (B, L', D)
            else:
                x_level = x
            
            # Mamba 编码
            h = encoder(x_level)  # (B, L', D)
            
            # 取最后一个时间步作为该尺度的表示
            h = h[:, -1, :]  # (B, D)
            
            level_outputs.append(h)
        
        # 拼接多尺度特征
        multi_scale = torch.cat(level_outputs, dim=-1)  # (B, D * n_levels)
        
        # 融合
        fused = self.fusion(multi_scale)  # (B, D)
        
        return fused


class PredictionHead(nn.Module):
    """
    预测头
    
    将特征映射到预测分数
    """
    
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 简化：减少预测头层数，从3层减少到1层
        self.head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
            
        Returns:
            pred: (batch, 1)
        """
        return self.head(x)


class FinMamba(nn.Module):
    """
    FinMamba 完整模型
    
    结合:
    1. 特征嵌入 (含行业嵌入)
    2. Multi-Level Mamba (多尺度时序建模)
    3. 预测头
    """
    
    def __init__(
        self,
        seq_len: int = 60,
        feature_dim: int = 48,
        d_model: int = 64,  # 简化：从128减少到64
        n_layers: int = 2,   # 简化：从3减少到2
        n_heads: int = 4,  # 保留兼容性，Mamba不使用
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        levels: Tuple[int, ...] = (1,),  # 简化：只保留日级，移除周级
        n_industries: int = 30,
        use_industry: bool = True,
        dropout: float = 0.1  # 简化：降低dropout比例
    ):
        """
        Args:
            seq_len: 序列长度 (回看天数)
            feature_dim: 输入特征维度
            d_model: 模型隐藏维度
            n_layers: Mamba 层数
            n_heads: 注意力头数 (保留兼容性)
            d_state: SSM 状态维度
            d_conv: SSM 卷积核大小
            expand: SSM 扩展因子
            levels: 多尺度级别 (天数)
            n_industries: 行业数量
            use_industry: 是否使用行业嵌入
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.use_industry = use_industry
        self.n_industries = n_industries
        
        # 特征嵌入 - 简化：减少投影层数量
        self.embedding = FeatureEmbedding(
            input_dim=feature_dim,
            d_model=d_model,
            n_industries=n_industries,
            use_industry=use_industry,
            dropout=dropout
        )
        
        # Multi-Level Mamba
        self.mamba = MultiLevelMamba(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            levels=levels,
            dropout=dropout
        )
        
        # 预测头 - 简化：减少层数
        self.head = PredictionHead(
            d_model=d_model,
            dropout=dropout
        )
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        industry_ids: Optional[torch.Tensor] = None,
        market_state: Optional[torch.Tensor] = None  # 保留兼容性
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim) 输入特征
            industry_ids: (batch,) 行业 ID [可选]
            market_state: (5,) 大盘状态 [可选，保留兼容性]
            
        Returns:
            pred: (batch, 1) 预测分数
        """
        # 特征嵌入 (含行业信息)
        h = self.embedding(x, industry_ids)  # (B, L, D)
        
        # Multi-Level Mamba
        h = self.mamba(h)  # (B, D)
        
        # 预测
        pred = self.head(h)  # (B, 1)
        
        return pred
    
    def get_embedding(self, x: torch.Tensor, industry_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取中间 embedding (用于相似股推荐)
        
        Returns:
            (B, d_model)
        """
        h = self.embedding(x, industry_ids)
        h = self.mamba(h)
        return h
    
    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 兼容性别名
AlphaModel = FinMamba


def create_model(**kwargs) -> FinMamba:
    """便捷函数: 创建模型"""
    return FinMamba(**kwargs)


if __name__ == "__main__":
    # 测试模型
    model = FinMamba(
        seq_len=60,
        feature_dim=48,
        d_model=96,
        n_layers=2,
        use_industry=True
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(4, 60, 48)
    industry_ids = torch.randint(0, 30, (4,))
    
    with torch.no_grad():
        y = model(x, industry_ids)
    
    print(f"Input shape: {x.shape}")
    print(f"Industry IDs: {industry_ids}")
    print(f"Output shape: {y.shape}")
    print(f"Output values: {y.squeeze()}")
