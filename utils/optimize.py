# -*- coding: utf-8 -*-
"""
PyTorch 性能优化
针对 GTX 1660 + 13代 i5 + 32GB 内存
"""

import torch
import torch.backends.cudnn as cudnn
import os
from pathlib import Path

import sys
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger("Optimize")


def optimize_pytorch():
    """
    优化 PyTorch 设置
    
    针对硬件配置:
    - GPU: NVIDIA GeForce GTX 1660 (6GB VRAM)
    - CPU: 13代 i5 (6核 12线程)
    - RAM: 32GB
    """
    logger.info("=" * 60)
    logger.info("PyTorch 性能优化")
    logger.info("=" * 60)
    
    # 1. CUDA 优化
    if torch.cuda.is_available():
        logger.info("\n[1/5] CUDA 优化")
        
        # 启用 cudnn benchmark（自动寻找最优算法）
        cudnn.benchmark = True
        logger.info("  ✓ 启用 cudnn.benchmark")
        
        # 启用 cudnn deterministic（可复现性，但可能略慢）
        # cudnn.deterministic = True  # 注释掉以获得更好性能
        
        # 设置 CUDA 内存分配策略
        # GTX 1660 有 6GB VRAM，使用合理的缓存
        torch.cuda.empty_cache()
        logger.info("  ✓ 清理 CUDA 缓存")
        
        # 设置 CUDA 设备
        device = torch.device('cuda')
        logger.info(f"  ✓ CUDA 设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 启用 TF32（TensorFloat-32）加速
        # GTX 1660 支持 TF32，可以加速矩阵运算
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("  ✓ 启用 TF32 加速")
        
    else:
        logger.warning("CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
    
    # 2. CPU 优化
    logger.info("\n[2/5] CPU 优化")
    
    # 设置 PyTorch 线程数（13代 i5 通常是 6核 12线程）
    # 使用所有物理核心
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    torch.set_num_threads(cpu_count)
    logger.info(f"  ✓ PyTorch 线程数: {cpu_count}")
    
    # 3. 内存优化
    logger.info("\n[3/5] 内存优化")
    
    # 设置内存分配器
    # GTX 1660 有 6GB VRAM，使用合理的缓存
    if torch.cuda.is_available():
        # 设置 CUDA 内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        logger.info("  ✓ CUDA 内存分配策略: max_split_size_mb=512")
    
    # 4. 数据加载优化
    logger.info("\n[4/5] 数据加载优化")
    
    # 设置 Windows 多进程启动方法
    if os.name == 'nt':
        os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
        logger.info("  ✓ Windows 多进程设置")
    
    # 5. 其他优化
    logger.info("\n[5/5] 其他优化")
    
    # 禁用警告（可选）
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    logger.info("  ✓ 禁用用户警告")
    
    logger.info(f"\n{'=' * 60}")
    logger.info("PyTorch 优化完成！")
    logger.info(f"{'=' * 60}")
    
    return device


def get_optimal_batch_size(model, input_shape=(60, 48), device='cuda'):
    """
    自动确定最优批次大小
    
    针对 GTX 1660 (6GB VRAM)
    """
    logger.info("\n自动确定最优批次大小...")
    
    # GTX 1660 的 VRAM 限制
    # 模型参数: ~6M
    # 每个样本: 60 * 48 * 4 bytes = ~11.5 KB
    # 梯度: ~6M * 4 bytes = ~24 MB
    # 优化器状态: ~6M * 4 bytes * 2 = ~48 MB
    
    # 尝试不同的批次大小
    batch_sizes = [256, 512, 768, 1024, 1280, 1536]
    optimal_batch_size = 512
    
    for batch_size in batch_sizes:
        try:
            # 创建测试张量
            x = torch.randn(batch_size, *input_shape, device=device)
            y = torch.randn(batch_size, 1, device=device)
            
            # 前向传播
            pred = model(x)
            
            # 计算损失
            loss = (pred - y).mean()
            
            # 反向传播
            loss.backward()
            
            # 清理
            del x, y, pred, loss
            torch.cuda.empty_cache()
            
            logger.info(f"  ✓ Batch size {batch_size}: OK")
            optimal_batch_size = batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info(f"  ✗ Batch size {batch_size}: OOM")
                break
            else:
                raise
    
    logger.info(f"\n最优批次大小: {optimal_batch_size}")
    return optimal_batch_size


def optimize_dataloader_settings():
    """
    优化 DataLoader 设置
    
    针对硬件配置:
    - CPU: 13代 i5 (6核 12线程)
    - RAM: 32GB
    """
    logger.info("\n优化 DataLoader 设置...")
    
    import multiprocessing
    
    # CPU 核心数
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"CPU 核心数: {cpu_count}")
    
    # 训练集 workers（使用所有核心）
    train_workers = min(6, cpu_count)  # Windows 下使用 6 workers
    logger.info(f"训练集 num_workers: {train_workers}")
    
    # 验证集 workers（使用较少的核心）
    val_workers = min(4, cpu_count)
    logger.info(f"验证集 num_workers: {val_workers}")
    
    # 测试集 workers（使用较少的核心）
    test_workers = min(4, cpu_count)
    logger.info(f"测试集 num_workers: {test_workers}")
    
    # 其他优化参数
    settings = {
        'pin_memory': True,  # 启用 pin_memory（加速 CPU→GPU 传输）
        'persistent_workers': True,  # 持久化 workers（减少进程创建开销）
        'prefetch_factor': 2,  # 预取 2 个 batch
        'drop_last': True,  # 丢弃最后一个不完整的 batch
    }
    
    logger.info(f"DataLoader 优化参数:")
    for key, value in settings.items():
        logger.info(f"  {key}: {value}")
    
    return {
        'train_workers': train_workers,
        'val_workers': val_workers,
        'test_workers': test_workers,
        **settings
    }


if __name__ == "__main__":
    # 测试优化
    device = optimize_pytorch()
    dataloader_settings = optimize_dataloader_settings()
    
    print(f"\n优化设备: {device}")
    print(f"DataLoader 设置: {dataloader_settings}")
