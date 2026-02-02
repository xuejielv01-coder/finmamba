# -*- coding: utf-8 -*-
"""
模型量化模块
改进22：实现模型剪枝、知识蒸馏和INT8量化

功能:
- 动态量化 (Dynamic Quantization)
- 静态量化 (Static Quantization)
- 模型剪枝 (Pruning)
- 知识蒸馏 (Knowledge Distillation)
- 量化效果评估
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import time
import copy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.quantization import quantize_dynamic, prepare, convert
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("ModelQuantization")


class DynamicQuantizer:
    """动态量化器"""
    
    def __init__(self):
        """初始化动态量化器"""
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
    
    def quantize(self, 
                model: 'nn.Module',
                dtype: 'torch.dtype' = None) -> 'nn.Module':
        """
        对模型进行动态量化
        
        动态量化对激活值在运行时进行量化，
        适用于LSTM、Transformer等序列模型
        
        Args:
            model: PyTorch模型
            dtype: 量化数据类型 (默认qint8)
            
        Returns:
            量化后的模型
        """
        if dtype is None:
            dtype = torch.qint8
        
        # 动态量化主要针对Linear层
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        
        logger.info("Model dynamically quantized to INT8")
        
        return quantized_model
    
    def compare_size(self, 
                     original_model: 'nn.Module',
                     quantized_model: 'nn.Module') -> Dict:
        """比较量化前后的模型大小"""
        def get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return param_size + buffer_size
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        return {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0,
            'size_reduction_pct': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0,
        }


class ModelPruner:
    """模型剪枝器"""
    
    def __init__(self):
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
    
    def magnitude_prune(self,
                        model: 'nn.Module',
                        sparsity: float = 0.3,
                        layer_types: Tuple = None) -> 'nn.Module':
        """
        基于幅度的剪枝
        
        移除权重绝对值最小的参数
        
        Args:
            model: PyTorch模型
            sparsity: 稀疏度 (0-1)，表示要剪枝的参数比例
            layer_types: 要剪枝的层类型
            
        Returns:
            剪枝后的模型
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            logger.error("Pruning requires PyTorch 1.4+")
            return model
        
        if layer_types is None:
            layer_types = (nn.Linear, nn.Conv1d, nn.Conv2d)
        
        pruned_model = copy.deepcopy(model)
        
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, layer_types):
                if hasattr(module, 'weight'):
                    parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            logger.warning("No layers found for pruning")
            return pruned_model
        
        # 全局非结构化剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        
        # 使剪枝永久化
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        logger.info(f"Model pruned with {sparsity*100:.1f}% sparsity")
        
        return pruned_model
    
    def structured_prune(self,
                        model: 'nn.Module',
                        layer_name: str,
                        n_channels: int) -> 'nn.Module':
        """
        结构化剪枝（移除整个通道/神经元）
        
        Args:
            model: PyTorch模型
            layer_name: 层名称
            n_channels: 要剪枝的通道数
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            return model
        
        pruned_model = copy.deepcopy(model)
        
        # 找到指定层
        for name, module in pruned_model.named_modules():
            if name == layer_name and hasattr(module, 'weight'):
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=n_channels, 
                    n=2, 
                    dim=0
                )
                prune.remove(module, 'weight')
                logger.info(f"Pruned {n_channels} channels from {layer_name}")
                break
        
        return pruned_model
    
    def calculate_sparsity(self, model: 'nn.Module') -> Dict:
        """计算模型的稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'nonzero_params': total_params - zero_params,
            'sparsity': sparsity,
            'sparsity_pct': sparsity * 100,
        }


class KnowledgeDistiller:
    """知识蒸馏器"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        初始化知识蒸馏器
        
        Args:
            temperature: 蒸馏温度（越高，软标签越平滑）
            alpha: 蒸馏损失权重 (1-alpha为硬标签权重)
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self,
                         student_logits: 'torch.Tensor',
                         teacher_logits: 'torch.Tensor',
                         targets: 'torch.Tensor' = None,
                         hard_loss_fn: Callable = None) -> 'torch.Tensor':
        """
        计算蒸馏损失
        
        L = α * L_soft + (1-α) * L_hard
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            targets: 真实标签（用于硬损失）
            hard_loss_fn: 硬损失函数
            
        Returns:
            蒸馏损失
        """
        T = self.temperature
        
        # 软损失：学生模仿教师的软标签
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        
        # 硬损失：学生直接从标签学习
        if targets is not None and hard_loss_fn is not None:
            hard_loss = hard_loss_fn(student_logits, targets)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss
        
        return total_loss
    
    def distill(self,
               teacher_model: 'nn.Module',
               student_model: 'nn.Module',
               train_loader: 'torch.utils.data.DataLoader',
               optimizer: 'torch.optim.Optimizer',
               epochs: int = 10,
               device: str = 'cuda') -> 'nn.Module':
        """
        执行知识蒸馏训练
        
        Args:
            teacher_model: 教师模型（大模型）
            student_model: 学生模型（小模型）
            train_loader: 训练数据加载器
            optimizer: 优化器
            epochs: 训练轮数
            device: 设备
            
        Returns:
            训练后的学生模型
        """
        teacher_model.eval()
        teacher_model.to(device)
        student_model.train()
        student_model.to(device)
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device) if len(batch) > 1 else None
                else:
                    inputs = batch.to(device)
                    targets = None
                
                # 获取教师和学生的输出
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                
                student_logits = student_model(inputs)
                
                # 计算蒸馏损失
                loss = self.distillation_loss(
                    student_logits, 
                    teacher_logits,
                    targets,
                    hard_loss_fn=nn.MSELoss() if targets is not None else None
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            logger.info(f"Distillation Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return student_model


class QuantizationEvaluator:
    """量化效果评估器"""
    
    def __init__(self):
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
    
    def benchmark_inference(self,
                           model: 'nn.Module',
                           input_shape: Tuple,
                           n_runs: int = 100,
                           device: str = 'cpu') -> Dict:
        """
        基准测试推理速度
        
        Args:
            model: 模型
            input_shape: 输入形状
            n_runs: 运行次数
            device: 设备
            
        Returns:
            基准测试结果
        """
        model.eval()
        model.to(device)
        
        dummy_input = torch.randn(*input_shape).to(device)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # 计时
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        return {
            'total_time_s': elapsed,
            'avg_time_ms': elapsed * 1000 / n_runs,
            'throughput': n_runs / elapsed,
            'n_runs': n_runs,
            'device': device,
        }
    
    def compare_accuracy(self,
                        original_model: 'nn.Module',
                        optimized_model: 'nn.Module',
                        test_loader: 'torch.utils.data.DataLoader',
                        device: str = 'cpu') -> Dict:
        """
        比较精度差异
        
        Args:
            original_model: 原始模型
            optimized_model: 优化后模型
            test_loader: 测试数据
            device: 设备
            
        Returns:
            精度比较结果
        """
        original_model.eval()
        optimized_model.eval()
        original_model.to(device)
        optimized_model.to(device)
        
        original_outputs = []
        optimized_outputs = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                
                orig_out = original_model(inputs).cpu().numpy()
                opt_out = optimized_model(inputs).cpu().numpy()
                
                original_outputs.append(orig_out)
                optimized_outputs.append(opt_out)
        
        original_outputs = np.concatenate(original_outputs, axis=0)
        optimized_outputs = np.concatenate(optimized_outputs, axis=0)
        
        # 计算差异
        mae = np.mean(np.abs(original_outputs - optimized_outputs))
        mse = np.mean((original_outputs - optimized_outputs) ** 2)
        max_diff = np.max(np.abs(original_outputs - optimized_outputs))
        
        # 相关性
        correlation = np.corrcoef(original_outputs.flatten(), optimized_outputs.flatten())[0, 1]
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'max_diff': float(max_diff),
            'correlation': float(correlation),
            'n_samples': len(original_outputs),
        }
    
    def full_evaluation(self,
                       original_model: 'nn.Module',
                       optimized_model: 'nn.Module',
                       input_shape: Tuple,
                       test_loader: 'torch.utils.data.DataLoader' = None,
                       device: str = 'cpu') -> Dict:
        """
        完整评估
        """
        result = {
            'timestamp': datetime.now().isoformat(),
        }
        
        # 模型大小
        quantizer = DynamicQuantizer()
        result['size_comparison'] = quantizer.compare_size(original_model, optimized_model)
        
        # 推理速度
        result['original_speed'] = self.benchmark_inference(original_model, input_shape, device=device)
        result['optimized_speed'] = self.benchmark_inference(optimized_model, input_shape, device=device)
        result['speedup'] = result['original_speed']['avg_time_ms'] / result['optimized_speed']['avg_time_ms']
        
        # 精度（如果提供测试数据）
        if test_loader is not None:
            result['accuracy_comparison'] = self.compare_accuracy(
                original_model, optimized_model, test_loader, device
            )
        
        # 稀疏度
        pruner = ModelPruner()
        result['original_sparsity'] = pruner.calculate_sparsity(original_model)
        result['optimized_sparsity'] = pruner.calculate_sparsity(optimized_model)
        
        return result


class ModelOptimizer:
    """模型优化器（综合量化、剪枝）"""
    
    def __init__(self, storage_dir: Path = None):
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.storage_dir = storage_dir or Config.MODEL_DIR / "optimized"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.quantizer = DynamicQuantizer()
        self.pruner = ModelPruner()
        self.evaluator = QuantizationEvaluator()
    
    def optimize(self,
                model: 'nn.Module',
                method: str = 'quantize',
                sparsity: float = 0.3,
                **kwargs) -> 'nn.Module':
        """
        优化模型
        
        Args:
            model: 原始模型
            method: 优化方法 ('quantize', 'prune', 'both')
            sparsity: 剪枝稀疏度
            
        Returns:
            优化后的模型
        """
        optimized = model
        
        if method in ['prune', 'both']:
            logger.info(f"Applying pruning with {sparsity*100:.1f}% sparsity...")
            optimized = self.pruner.magnitude_prune(optimized, sparsity=sparsity)
        
        if method in ['quantize', 'both']:
            logger.info("Applying dynamic quantization...")
            optimized = self.quantizer.quantize(optimized)
        
        return optimized
    
    def save_optimized(self, model: 'nn.Module', name: str):
        """保存优化后的模型"""
        path = self.storage_dir / f"{name}.pt"
        torch.save(model.state_dict(), path)
        logger.info(f"Optimized model saved to {path}")
    
    def generate_report(self,
                       original_model: 'nn.Module',
                       optimized_model: 'nn.Module',
                       input_shape: Tuple) -> str:
        """生成优化报告"""
        eval_result = self.evaluator.full_evaluation(
            original_model, optimized_model, input_shape
        )
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    模型优化报告                                ║
╠══════════════════════════════════════════════════════════════╣
║ 生成时间: {eval_result['timestamp'][:19]:<41}║
╠══════════════════════════════════════════════════════════════╣
║ 模型大小:                                                    ║
║   原始: {eval_result['size_comparison']['original_size_mb']:.2f} MB{' ' * 45}║
║   优化后: {eval_result['size_comparison']['quantized_size_mb']:.2f} MB{' ' * 42}║
║   压缩比: {eval_result['size_comparison']['compression_ratio']:.2f}x{' ' * 45}║
║   减少: {eval_result['size_comparison']['size_reduction_pct']:.1f}%{' ' * 48}║
╠══════════════════════════════════════════════════════════════╣
║ 推理速度:                                                    ║
║   原始: {eval_result['original_speed']['avg_time_ms']:.2f} ms{' ' * 44}║
║   优化后: {eval_result['optimized_speed']['avg_time_ms']:.2f} ms{' ' * 41}║
║   加速: {eval_result['speedup']:.2f}x{' ' * 49}║
╚══════════════════════════════════════════════════════════════╝
"""
        return report


if __name__ == "__main__":
    print("模型量化模块测试")
    print("="*50)
    
    if not HAS_TORCH:
        print("需要安装PyTorch才能运行测试")
    else:
        # 创建测试模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(100, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        model = SimpleModel()
        
        # 测试动态量化
        print("\n1. 动态量化测试")
        quantizer = DynamicQuantizer()
        quantized_model = quantizer.quantize(model)
        size_comp = quantizer.compare_size(model, quantized_model)
        print(f"  原始大小: {size_comp['original_size_mb']:.4f} MB")
        print(f"  量化后: {size_comp['quantized_size_mb']:.4f} MB")
        print(f"  压缩比: {size_comp['compression_ratio']:.2f}x")
        
        # 测试剪枝
        print("\n2. 模型剪枝测试")
        pruner = ModelPruner()
        pruned_model = pruner.magnitude_prune(model, sparsity=0.5)
        sparsity_info = pruner.calculate_sparsity(pruned_model)
        print(f"  稀疏度: {sparsity_info['sparsity_pct']:.1f}%")
        print(f"  非零参数: {sparsity_info['nonzero_params']}")
        
        # 测试推理速度
        print("\n3. 推理速度基准测试")
        evaluator = QuantizationEvaluator()
        original_speed = evaluator.benchmark_inference(model, (1, 100), n_runs=100)
        quantized_speed = evaluator.benchmark_inference(quantized_model, (1, 100), n_runs=100)
        print(f"  原始: {original_speed['avg_time_ms']:.3f} ms")
        print(f"  量化后: {quantized_speed['avg_time_ms']:.3f} ms")
        print(f"  加速: {original_speed['avg_time_ms'] / quantized_speed['avg_time_ms']:.2f}x")
        
        # 综合优化
        print("\n4. 综合优化（剪枝+量化）")
        optimizer = ModelOptimizer()
        optimized = optimizer.optimize(model, method='both', sparsity=0.3)
        print(optimizer.generate_report(model, optimized, (1, 100)))
        
        print("\n模型量化模块测试完成!")
