# -*- coding: utf-8 -*-
"""
Transformer 编码器实现

用于与 Mamba 混合架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    包含：
    1. 多头自注意力机制
    2. 前馈网络
    3. Layer Normalization
    4. Dropout
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        """
        Args:
            d_model: 模型隐藏维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏维度
            dropout: Dropout 比例
            layer_norm_eps: Layer Normalization epsilon
        """
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # 使用 batch_first=True 以便与 Mamba 输出兼容
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 输入特征
            attn_mask: (batch, seq_len, seq_len) 注意力掩码
            key_padding_mask: (batch, seq_len) 键填充掩码
            
        Returns:
            y: (batch, seq_len, d_model) 输出特征
        """
        # 自注意力子层
        attn_output, _ = self.self_attn(
            x, x, x,  # q, k, v
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    堆叠多个 TransformerEncoderLayer
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型隐藏维度
            n_layers: Transformer 层数
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏维度
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 输入特征
            attn_mask: (batch, seq_len, seq_len) 注意力掩码
            key_padding_mask: (batch, seq_len) 键填充掩码
            
        Returns:
            y: (batch, seq_len, d_model) 输出特征
        """
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        return x