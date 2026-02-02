# -*- coding: utf-8 -*-
"""
模型量化工具

支持INT8量化，减小模型大小和推理延迟
"""

import torch
import torch.nn as nn
from typing import Optional


def quantize_model(
    model: nn.Module,
    input_example: torch.Tensor,
    industry_ids: Optional[torch.Tensor] = None,
    quantization_method: str = "dynamic"
) -> nn.Module:
    """
    对模型进行量化
    
    Args:
        model: 待量化的模型
        input_example: 输入示例，用于动态量化或静态量化校准
        industry_ids: 行业ID示例，用于模型前向传播
        quantization_method: 量化方法，可选：
            - "dynamic": 动态量化（仅权重量化，运行时激活量化）
            - "static": 静态量化（权重和激活均量化，需要校准）
            - "qat": 量化感知训练（需要重新训练）
    
    Returns:
        量化后的模型
    """
    model.eval()
    
    if quantization_method == "dynamic":
        # 动态量化 - 适用于CPU推理，仅量化权重
        logger.info("Performing dynamic quantization...")
        
        # 定义量化配置
        quantization_config = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.default_weight_observer.with_args(dtype=torch.qint8)
        )
        
        # 准备模型
        model_prepared = torch.quantization.prepare(model)
        
        # 动态量化
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    elif quantization_method == "static":
        # 静态量化 - 适用于CPU推理，需要校准
        logger.info("Performing static quantization...")
        
        # 定义量化配置
        quantization_config = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.default_weight_observer.with_args(dtype=torch.qint8)
        )
        
        # 设置量化配置
        model.qconfig = quantization_config
        
        # 准备模型
        model_prepared = torch.quantization.prepare(model)
        
        # 校准模型 - 使用输入示例
        with torch.no_grad():
            model_prepared(input_example, industry_ids)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    elif quantization_method == "qat":
        # 量化感知训练 - 需要重新训练
        logger.info("Setting up quantization-aware training...")
        
        # 定义量化配置
        quantization_config = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.default_weight_observer.with_args(dtype=torch.qint8)
        )
        
        # 设置量化配置
        model.qconfig = quantization_config
        
        # 准备模型进行量化感知训练
        model_qat = torch.quantization.prepare_qat(model)
        
        return model_qat
    
    else:
        raise ValueError(f"Unknown quantization method: {quantization_method}")


def save_quantized_model(
    model: nn.Module,
    save_path: str,
    input_example: torch.Tensor,
    industry_ids: Optional[torch.Tensor] = None
) -> None:
    """
    保存量化模型
    
    Args:
        model: 量化后的模型
        save_path: 保存路径
        input_example: 输入示例，用于导出TorchScript
        industry_ids: 行业ID示例
    """
    # 转换为TorchScript以便部署
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, (input_example, industry_ids))
    
    # 保存模型
    torch.jit.save(scripted_model, save_path)
    logger.info(f"Quantized model saved to {save_path}")


def load_quantized_model(load_path: str) -> torch.jit.ScriptModule:
    """
    加载量化模型
    
    Args:
        load_path: 模型加载路径
    
    Returns:
        加载的量化模型
    """
    model = torch.jit.load(load_path)
    logger.info(f"Quantized model loaded from {load_path}")
    return model


def get_model_size(model: nn.Module) -> float:
    """
    获取模型大小（MB）
    
    Args:
        model: 模型
    
    Returns:
        模型大小（MB）
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    return size_mb


# 配置日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Quantization")


if __name__ == "__main__":
    """
    量化工具命令行示例
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="模型量化工具")
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument("--output_path", type=str, required=True, help="量化模型输出路径")
    parser.add_argument("--quantization_method", type=str, default="dynamic", 
                        choices=["dynamic", "static", "qat"], help="量化方法")
    parser.add_argument("--batch_size", type=int, default=1, help="输入示例批次大小")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导入模型创建函数
    from finmamba.models.finmamba import create_model
    
    # 创建模型
    model = create_model()
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path))
    logger.info(f"Original model loaded from {args.model_path}")
    logger.info(f"Original model size: {get_model_size(model):.2f} MB")
    
    # 创建输入示例
    seq_len = 60
    feature_dim = 66
    input_example = torch.randn(args.batch_size, seq_len, feature_dim)
    industry_ids = torch.randint(0, 111, (args.batch_size,))
    
    # 量化模型
    quantized_model = quantize_model(
        model,
        input_example,
        industry_ids,
        args.quantization_method
    )
    
    logger.info(f"Quantized model size: {get_model_size(quantized_model):.2f} MB")
    
    # 保存量化模型
    save_quantized_model(
        quantized_model,
        args.output_path,
        input_example,
        industry_ids
    )
    
    logger.info("Quantization completed successfully!")