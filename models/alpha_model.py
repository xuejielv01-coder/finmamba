# -*- coding: utf-8 -*-
"""
Alpha Model - 主入口模型

直接使用 FinMamba 作为核心模型
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from .finmamba import FinMamba


class AlphaModel(FinMamba):
    """
    Alpha 预测模型
    
    继承自 FinMamba，使用配置文件中的默认参数
    """
    
    def __init__(
        self,
        seq_len: int = None,
        feature_dim: int = None,
        d_model: int = None,
        n_heads: int = None,
        n_layers: int = None,
        d_state: int = None,
        d_conv: int = None,
        expand: int = None,
        n_industries: int = None,
        levels: tuple = None,
        patch_len: int = None,  # 保留兼容性
        stride: int = None,  # 保留兼容性
        dropout: float = None,
        use_gat: bool = False,  # 保留兼容性
        use_industry: bool = True
    ):
        # 使用配置或默认值
        super().__init__(
            seq_len=seq_len or Config.SEQ_LEN,
            feature_dim=feature_dim or Config.FEATURE_DIM,
            d_model=d_model or Config.D_MODEL,
            n_layers=n_layers or Config.N_LAYERS,
            n_heads=n_heads or Config.N_HEADS,
            d_state=d_state or Config.D_STATE,
            d_conv=d_conv or Config.D_CONV,
            expand=expand or Config.EXPAND,
            levels=levels or Config.MAMBA_LEVELS,
            n_industries=n_industries or Config.N_INDUSTRIES,
            use_industry=use_industry and Config.USE_GRAPH,  # 行业嵌入开关
            dropout=dropout or Config.DROPOUT
        )


def create_model(**kwargs) -> AlphaModel:
    """便捷函数: 创建模型"""
    return AlphaModel(**kwargs)


if __name__ == "__main__":
    # 测试模型
    model = AlphaModel()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(4, Config.SEQ_LEN, Config.FEATURE_DIM)
    industry_ids = torch.randint(0, Config.N_INDUSTRIES, (4,))
    
    with torch.no_grad():
        y = model(x, industry_ids)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 测试 GPU
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        industry_ids = industry_ids.cuda()
        
        import time
        model.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = model(x, industry_ids)
            
            # 计时
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = model(x, industry_ids)
            torch.cuda.synchronize()
            elapsed = time.time() - start
        
        print(f"GPU test passed")
        print(f"Throughput: {100 * 4 / elapsed:.0f} samples/s")
        
        # 显存占用
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory: {allocated:.2f} MB")
