# -*- coding: utf-8 -*-
"""
模型不确定性量化模块
改进7：实现MC Dropout和Ensemble方法

功能:
- MC Dropout (蒙特卡洛Dropout)
- 模型集成 (Ensemble)
- 预测区间估计
- 不确定性分解 (认知/偶然)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("UncertaintyQuantification")


class MCDropout:
    """蒙特卡洛Dropout不确定性估计"""
    
    def __init__(self, model: 'nn.Module', n_samples: int = 30):
        """
        初始化MC Dropout
        
        Args:
            model: PyTorch模型（必须包含Dropout层）
            n_samples: MC采样次数
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.model = model
        self.n_samples = n_samples
        
        # 检查模型是否有Dropout层
        self._has_dropout = self._check_dropout()
        if not self._has_dropout:
            logger.warning("Model does not contain Dropout layers. MC Dropout may not work properly.")
    
    def _check_dropout(self) -> bool:
        """检查模型是否包含Dropout层"""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                return True
        return False
    
    def _enable_dropout(self):
        """启用模型中的Dropout层（即使在eval模式下）"""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
    
    def predict_with_uncertainty(self, 
                                  x: 'torch.Tensor',
                                  return_samples: bool = False) -> Dict:
        """
        使用MC Dropout进行不确定性预测
        
        Args:
            x: 输入张量
            return_samples: 是否返回所有采样结果
            
        Returns:
            包含均值、标准差和置信区间的字典
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(x)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_samples, batch_size, ...)
        
        # 计算统计量
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # 计算置信区间
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)
        
        lower_80 = np.percentile(predictions, 10, axis=0)
        upper_80 = np.percentile(predictions, 90, axis=0)
        
        result = {
            'mean': mean,
            'std': std,
            'epistemic_uncertainty': std,  # MC Dropout主要捕获认知不确定性
            'confidence_interval_95': (lower_95, upper_95),
            'confidence_interval_80': (lower_80, upper_80),
        }
        
        if return_samples:
            result['samples'] = predictions
        
        return result
    
    def get_prediction_interval(self, 
                                x: 'torch.Tensor',
                                confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取预测区间
        
        Args:
            x: 输入张量
            confidence: 置信水平
            
        Returns:
            (下界, 上界) 元组
        """
        result = self.predict_with_uncertainty(x, return_samples=True)
        samples = result['samples']
        
        alpha = (1 - confidence) / 2
        lower = np.percentile(samples, alpha * 100, axis=0)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=0)
        
        return lower, upper


class DeepEnsemble:
    """深度集成模型"""
    
    def __init__(self, models: List['nn.Module'] = None, n_models: int = 5):
        """
        初始化深度集成
        
        Args:
            models: 预训练的模型列表
            n_models: 集成模型数量（如果models为None）
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.models = models or []
        self.n_models = n_models if models is None else len(models)
        self.is_trained = len(self.models) > 0
    
    def add_model(self, model: 'nn.Module'):
        """添加模型到集成"""
        self.models.append(model)
        self.is_trained = True
    
    def predict_with_uncertainty(self, x: 'torch.Tensor') -> Dict:
        """
        使用集成模型进行不确定性预测
        
        Args:
            x: 输入张量
            
        Returns:
            包含均值、标准差和不确定性分解的字典
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(x)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_models, batch_size, ...)
        
        # 计算统计量
        mean = np.mean(predictions, axis=0)
        
        # 认知不确定性：模型间的差异
        epistemic = np.std(predictions, axis=0)
        
        # 总不确定性
        total_uncertainty = epistemic
        
        # 置信区间
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean,
            'std': epistemic,
            'epistemic_uncertainty': epistemic,
            'total_uncertainty': total_uncertainty,
            'confidence_interval_95': (lower_95, upper_95),
            'n_models': len(self.models),
            'model_predictions': predictions,
        }
    
    def get_model_agreement(self, x: 'torch.Tensor') -> float:
        """
        计算模型一致性得分
        
        Args:
            x: 输入张量
            
        Returns:
            一致性得分 (0-1)
        """
        result = self.predict_with_uncertainty(x)
        predictions = result['model_predictions']
        
        # 计算模型间的相关性
        n_models = predictions.shape[0]
        correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[i].flatten(), predictions[j].flatten())[0, 1]
                correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 1.0


class UncertaintyDecomposition:
    """不确定性分解"""
    
    @staticmethod
    def decompose_uncertainty(predictions: np.ndarray,
                              targets: np.ndarray = None) -> Dict:
        """
        分解总不确定性为认知和偶然不确定性
        
        对于回归问题:
        - 认知不确定性 (Epistemic): 模型不确定性，可通过更多数据减少
        - 偶然不确定性 (Aleatoric): 数据固有噪声，无法减少
        
        Args:
            predictions: 多次预测结果 (n_samples, n_points)
            targets: 真实目标值（可选）
            
        Returns:
            不确定性分解结果
        """
        # 认知不确定性：预测的方差
        epistemic = np.var(predictions, axis=0)
        
        # 如果有目标值，估计偶然不确定性
        if targets is not None:
            mean_pred = np.mean(predictions, axis=0)
            residuals = targets - mean_pred
            aleatoric = np.var(residuals)
        else:
            # 使用预测的均值方差作为偶然不确定性的代理
            mean_pred = np.mean(predictions, axis=0)
            aleatoric = np.mean(epistemic)  # 简化估计
        
        # 总不确定性
        total = epistemic.mean() + aleatoric
        
        return {
            'epistemic_mean': float(np.mean(epistemic)),
            'epistemic_std': float(np.std(epistemic)),
            'aleatoric': float(aleatoric),
            'total': float(total),
            'epistemic_ratio': float(np.mean(epistemic) / total) if total > 0 else 0,
        }


class UncertaintyCalibration:
    """不确定性校准"""
    
    @staticmethod
    def calculate_calibration_error(predicted_std: np.ndarray,
                                    predictions: np.ndarray,
                                    targets: np.ndarray,
                                    n_bins: int = 10) -> Dict:
        """
        计算不确定性校准误差
        
        理想情况下，预测的标准差应该与实际误差成正比
        
        Args:
            predicted_std: 预测的标准差
            predictions: 预测均值
            targets: 真实值
            n_bins: 分箱数量
            
        Returns:
            校准指标
        """
        errors = np.abs(predictions - targets)
        
        # 按预测不确定性分箱
        bin_edges = np.percentile(predicted_std, np.linspace(0, 100, n_bins + 1))
        
        bin_errors = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (predicted_std >= bin_edges[i]) & (predicted_std < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_errors.append(np.mean(errors[mask]))
                bin_stds.append(np.mean(predicted_std[mask]))
        
        bin_errors = np.array(bin_errors)
        bin_stds = np.array(bin_stds)
        
        # 计算相关性
        if len(bin_errors) > 2:
            correlation = np.corrcoef(bin_stds, bin_errors)[0, 1]
        else:
            correlation = 0.0
        
        # 计算校准误差
        calibration_error = np.mean(np.abs(bin_stds - bin_errors))
        
        return {
            'correlation': float(correlation),
            'calibration_error': float(calibration_error),
            'bin_stds': bin_stds.tolist(),
            'bin_errors': bin_errors.tolist(),
            'is_calibrated': correlation > 0.7,
        }
    
    @staticmethod
    def calculate_coverage(predictions: np.ndarray,
                           lower: np.ndarray,
                           upper: np.ndarray,
                           targets: np.ndarray) -> Dict:
        """
        计算预测区间覆盖率
        
        Args:
            predictions: 预测均值
            lower: 下界
            upper: 上界
            targets: 真实值
            
        Returns:
            覆盖率指标
        """
        in_interval = (targets >= lower) & (targets <= upper)
        coverage = np.mean(in_interval)
        
        # 计算区间宽度
        width = upper - lower
        mean_width = np.mean(width)
        
        return {
            'coverage': float(coverage),
            'mean_interval_width': float(mean_width),
            'n_samples': len(targets),
        }


class PredictionWithUncertainty:
    """带不确定性的预测结果"""
    
    def __init__(self, 
                 mean: float,
                 std: float,
                 lower_95: float = None,
                 upper_95: float = None,
                 confidence: float = None,
                 uncertainty_type: str = 'epistemic'):
        """
        初始化预测结果
        
        Args:
            mean: 预测均值
            std: 预测标准差
            lower_95: 95%置信区间下界
            upper_95: 95%置信区间上界
            confidence: 置信度得分
            uncertainty_type: 不确定性类型
        """
        self.mean = mean
        self.std = std
        self.lower_95 = lower_95 if lower_95 is not None else mean - 1.96 * std
        self.upper_95 = upper_95 if upper_95 is not None else mean + 1.96 * std
        self.confidence = confidence if confidence is not None else self._calculate_confidence()
        self.uncertainty_type = uncertainty_type
    
    def _calculate_confidence(self) -> float:
        """根据不确定性计算置信度"""
        # 标准差越小，置信度越高
        # 使用sigmoid变换
        normalized_std = self.std / (abs(self.mean) + 1e-10)
        confidence = 1.0 / (1.0 + np.exp(normalized_std * 5 - 2))
        return float(confidence)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'mean': self.mean,
            'std': self.std,
            'lower_95': self.lower_95,
            'upper_95': self.upper_95,
            'confidence': self.confidence,
            'uncertainty_type': self.uncertainty_type,
        }
    
    def __repr__(self):
        return f"Prediction(mean={self.mean:.4f}, std={self.std:.4f}, conf={self.confidence:.2f})"


class UncertaintyAwarePredictor:
    """不确定性感知预测器"""
    
    def __init__(self, 
                 model: 'nn.Module',
                 method: str = 'mc_dropout',
                 n_samples: int = 30):
        """
        初始化预测器
        
        Args:
            model: PyTorch模型
            method: 不确定性估计方法 ('mc_dropout', 'ensemble')
            n_samples: 采样次数
        """
        self.model = model
        self.method = method
        self.n_samples = n_samples
        
        if method == 'mc_dropout':
            self.uncertainty_estimator = MCDropout(model, n_samples)
        elif method == 'ensemble':
            self.uncertainty_estimator = DeepEnsemble([model])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict(self, x: 'torch.Tensor') -> List[PredictionWithUncertainty]:
        """
        进行带不确定性的预测
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果列表
        """
        result = self.uncertainty_estimator.predict_with_uncertainty(x)
        
        predictions = []
        mean = result['mean'].flatten()
        std = result['std'].flatten()
        lower_95, upper_95 = result['confidence_interval_95']
        lower_95 = lower_95.flatten()
        upper_95 = upper_95.flatten()
        
        for i in range(len(mean)):
            pred = PredictionWithUncertainty(
                mean=float(mean[i]),
                std=float(std[i]),
                lower_95=float(lower_95[i]),
                upper_95=float(upper_95[i]),
                uncertainty_type='epistemic'
            )
            predictions.append(pred)
        
        return predictions
    
    def predict_with_rejection(self, 
                               x: 'torch.Tensor',
                               confidence_threshold: float = 0.5) -> Tuple[List, List[int]]:
        """
        带拒绝的预测（低置信度样本被拒绝）
        
        Args:
            x: 输入张量
            confidence_threshold: 置信度阈值
            
        Returns:
            (预测结果, 拒绝的索引)
        """
        predictions = self.predict(x)
        
        accepted = []
        rejected_indices = []
        
        for i, pred in enumerate(predictions):
            if pred.confidence >= confidence_threshold:
                accepted.append(pred)
            else:
                rejected_indices.append(i)
        
        return accepted, rejected_indices


if __name__ == "__main__":
    print("模型不确定性量化模块测试")
    print("="*50)
    
    if not HAS_TORCH:
        print("需要安装PyTorch才能运行测试")
    else:
        # 创建一个简单的测试模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 32)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(32, 1)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        # 测试MC Dropout
        print("\n1. MC Dropout 测试")
        model = SimpleModel()
        mc_dropout = MCDropout(model, n_samples=20)
        
        x = torch.randn(5, 10)
        result = mc_dropout.predict_with_uncertainty(x)
        
        print(f"  预测均值形状: {result['mean'].shape}")
        print(f"  预测标准差形状: {result['std'].shape}")
        print(f"  95%置信区间: [{result['confidence_interval_95'][0][0]:.4f}, "
              f"{result['confidence_interval_95'][1][0]:.4f}]")
        
        # 测试不确定性感知预测器
        print("\n2. 不确定性感知预测器测试")
        predictor = UncertaintyAwarePredictor(model, method='mc_dropout')
        predictions = predictor.predict(x)
        
        for i, pred in enumerate(predictions[:3]):
            print(f"  样本{i}: {pred}")
        
        # 测试带拒绝的预测
        print("\n3. 带拒绝的预测测试")
        accepted, rejected = predictor.predict_with_rejection(x, confidence_threshold=0.3)
        print(f"  接受: {len(accepted)}, 拒绝: {len(rejected)}")
        
        print("\n不确定性量化模块测试完成!")
