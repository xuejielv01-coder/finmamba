# -*- coding: utf-8 -*-
"""
随机种子控制器
确保实验可复现性
"""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    设置所有随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    获取最佳可用设备
    
    Returns:
        torch.device: CUDA 设备 (如可用) 或 CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[System] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[System] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[System] Using CPU")
    return device


def clear_gpu_memory():
    """清理 GPU 显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
