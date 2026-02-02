# -*- coding: utf-8 -*-
"""
模型解释性模块
改进8：实现SHAP值计算和注意力可视化

功能:
- 特征重要性分析
- SHAP值近似计算
- 注意力权重提取
- 特征贡献度分析
- 预测解释报告
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("ModelInterpretability")


class PermutationImportance:
    """排列特征重要性"""
    
    def __init__(self, model: 'nn.Module', 
                 loss_fn: Callable = None,
                 n_repeats: int = 10):
        """
        初始化排列重要性计算器
        
        Args:
            model: PyTorch模型
            loss_fn: 损失函数（用于评估预测质量）
            n_repeats: 每个特征的重复次数
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.model = model
        self.loss_fn = loss_fn or nn.MSELoss()
        self.n_repeats = n_repeats
    
    def calculate(self,
                  X: 'torch.Tensor',
                  y: 'torch.Tensor',
                  feature_names: List[str] = None) -> pd.DataFrame:
        """
        计算特征重要性
        
        通过打乱每个特征并测量预测质量下降来评估重要性
        
        Args:
            X: 输入数据 (batch, seq_len, features) 或 (batch, features)
            y: 真实标签
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        self.model.eval()
        
        # 获取基准性能
        with torch.no_grad():
            base_pred = self.model(X)
            base_loss = self.loss_fn(base_pred, y).item()
        
        n_features = X.shape[-1]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        importances = []
        
        for feat_idx in range(n_features):
            feat_losses = []
            
            for _ in range(self.n_repeats):
                # 打乱该特征
                X_permuted = X.clone()
                perm_idx = torch.randperm(X.shape[0])
                
                if X.dim() == 3:  # (batch, seq, features)
                    X_permuted[:, :, feat_idx] = X[perm_idx, :, feat_idx]
                else:  # (batch, features)
                    X_permuted[:, feat_idx] = X[perm_idx, feat_idx]
                
                # 计算打乱后的损失
                with torch.no_grad():
                    perm_pred = self.model(X_permuted)
                    perm_loss = self.loss_fn(perm_pred, y).item()
                
                feat_losses.append(perm_loss)
            
            # 重要性 = 打乱后损失 - 基准损失
            importance_mean = np.mean(feat_losses) - base_loss
            importance_std = np.std(feat_losses)
            
            importances.append({
                'feature': feature_names[feat_idx],
                'importance': importance_mean,
                'importance_std': importance_std,
                'base_loss': base_loss,
            })
        
        df = pd.DataFrame(importances)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # 归一化
        total_importance = df['importance'].sum()
        if total_importance > 0:
            df['importance_pct'] = df['importance'] / total_importance * 100
        else:
            df['importance_pct'] = 0
        
        return df


class GradientBasedSaliency:
    """基于梯度的显著性分析"""
    
    def __init__(self, model: 'nn.Module'):
        """
        初始化梯度显著性分析器
        
        Args:
            model: PyTorch模型
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.model = model
    
    def calculate_saliency(self, X: 'torch.Tensor') -> np.ndarray:
        """
        计算输入梯度（显著性图）
        
        Args:
            X: 输入张量
            
        Returns:
            显著性图
        """
        self.model.eval()
        X.requires_grad = True
        
        output = self.model(X)
        
        # 对输出求和并反向传播
        if output.dim() > 1:
            output = output.sum(dim=1)
        output = output.sum()
        
        output.backward()
        
        # 梯度绝对值作为显著性
        saliency = X.grad.abs().detach().cpu().numpy()
        
        return saliency
    
    def calculate_integrated_gradients(self,
                                        X: 'torch.Tensor',
                                        baseline: 'torch.Tensor' = None,
                                        n_steps: int = 50) -> np.ndarray:
        """
        计算积分梯度（更准确的归因方法）
        
        Args:
            X: 输入张量
            baseline: 基线输入（默认为零）
            n_steps: 积分步数
            
        Returns:
            积分梯度
        """
        if baseline is None:
            baseline = torch.zeros_like(X)
        
        # 生成从baseline到X的路径
        scaled_inputs = []
        for step in range(n_steps + 1):
            alpha = step / n_steps
            scaled = baseline + alpha * (X - baseline)
            scaled_inputs.append(scaled)
        
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad = True
        
        self.model.eval()
        outputs = self.model(scaled_inputs)
        
        if outputs.dim() > 1:
            outputs = outputs.sum(dim=1)
        outputs = outputs.sum()
        
        outputs.backward()
        
        grads = scaled_inputs.grad.view(n_steps + 1, *X.shape)
        
        # 梯形积分
        avg_grads = (grads[:-1] + grads[1:]) / 2
        integrated_grads = avg_grads.mean(dim=0)
        
        # 乘以输入差异
        ig = (X - baseline) * integrated_grads
        
        return ig.detach().cpu().numpy()


class AttentionExtractor:
    """注意力权重提取器"""
    
    def __init__(self, model: 'nn.Module'):
        """
        初始化注意力提取器
        
        Args:
            model: PyTorch模型（需包含注意力层）
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.model = model
        self.attention_weights = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子以捕获注意力权重"""
        
        def get_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    # MultiheadAttention返回(output, attention_weights)
                    if output[1] is not None:
                        self.attention_weights[name] = output[1].detach().cpu()
                        return
                
                # 尝试从模块属性获取
                if hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach().cpu()
            
            return hook
        
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(get_attention_hook(name))
    
    def extract(self, X: 'torch.Tensor') -> Dict[str, np.ndarray]:
        """
        提取注意力权重
        
        Args:
            X: 输入张量
            
        Returns:
            注意力权重字典
        """
        self.attention_weights.clear()
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(X)
        
        # 转换为numpy
        result = {k: v.numpy() for k, v in self.attention_weights.items()}
        
        return result
    
    def visualize_attention(self, 
                           attention: np.ndarray,
                           tokens: List[str] = None) -> str:
        """
        生成注意力可视化的文本表示
        
        Args:
            attention: 注意力权重 (seq_len, seq_len)
            tokens: 标记列表
            
        Returns:
            文本表示
        """
        if attention.ndim > 2:
            attention = attention.mean(axis=0)  # 平均多头
        
        if attention.ndim > 2:
            attention = attention[0]  # 取第一个样本
        
        seq_len = attention.shape[0]
        
        if tokens is None:
            tokens = [f't{i}' for i in range(seq_len)]
        
        lines = []
        lines.append("Attention Heatmap:")
        lines.append("-" * (seq_len * 8 + 5))
        
        # 表头
        header = "     " + " ".join([f"{t:>6}" for t in tokens[:10]])
        lines.append(header)
        
        # 数据行
        for i in range(min(10, seq_len)):
            row = f"{tokens[i]:>4} " + " ".join([f"{attention[i,j]:>6.3f}" for j in range(min(10, seq_len))])
            lines.append(row)
        
        return "\n".join(lines)


class FeatureContribution:
    """特征贡献度分析"""
    
    def __init__(self, model: 'nn.Module'):
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        self.model = model
    
    def analyze_single_prediction(self,
                                   X: 'torch.Tensor',
                                   feature_names: List[str] = None,
                                   n_samples: int = 100) -> Dict:
        """
        分析单个预测的特征贡献
        
        使用近似SHAP：通过对比添加/移除特征的预测变化
        
        Args:
            X: 单个样本输入
            feature_names: 特征名称
            n_samples: 采样次数
            
        Returns:
            贡献度分析结果
        """
        self.model.eval()
        
        X = X.unsqueeze(0) if X.dim() == 1 else X
        n_features = X.shape[-1]
        
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(n_features)]
        
        # 获取完整预测
        with torch.no_grad():
            full_pred = self.model(X).item()
        
        contributions = {}
        
        for feat_idx in range(n_features):
            # 通过遮蔽该特征估计贡献
            masked_preds = []
            
            for _ in range(n_samples):
                X_masked = X.clone()
                # 用随机抽样替换该特征
                random_value = torch.randn_like(X_masked[..., feat_idx])
                X_masked[..., feat_idx] = random_value
                
                with torch.no_grad():
                    masked_pred = self.model(X_masked).item()
                masked_preds.append(masked_pred)
            
            # 贡献 = 完整预测 - 遮蔽后平均预测
            contribution = full_pred - np.mean(masked_preds)
            
            contributions[feature_names[feat_idx]] = {
                'contribution': float(contribution),
                'masked_mean': float(np.mean(masked_preds)),
                'masked_std': float(np.std(masked_preds)),
            }
        
        # 排序
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]['contribution']), 
            reverse=True
        )
        
        return {
            'prediction': full_pred,
            'contributions': dict(sorted_contributions),
            'top_positive': [(k, v['contribution']) for k, v in sorted_contributions if v['contribution'] > 0][:5],
            'top_negative': [(k, v['contribution']) for k, v in sorted_contributions if v['contribution'] < 0][:5],
        }


class PredictionExplainer:
    """预测解释器"""
    
    def __init__(self, model: 'nn.Module', feature_names: List[str] = None):
        """
        初始化预测解释器
        
        Args:
            model: PyTorch模型
            feature_names: 特征名称列表
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.model = model
        self.feature_names = feature_names
        
        self.permutation_importance = PermutationImportance(model)
        self.saliency = GradientBasedSaliency(model)
        self.contribution = FeatureContribution(model)
    
    def explain_prediction(self,
                          X: 'torch.Tensor',
                          sample_idx: int = 0) -> Dict:
        """
        解释单个预测
        
        Args:
            X: 输入数据
            sample_idx: 样本索引
            
        Returns:
            解释结果
        """
        sample = X[sample_idx:sample_idx+1] if X.dim() > 1 else X.unsqueeze(0)
        
        self.model.eval()
        
        # 获取预测
        with torch.no_grad():
            prediction = self.model(sample).item()
        
        # 梯度显著性
        try:
            saliency = self.saliency.calculate_saliency(sample.clone())
            saliency_scores = saliency.flatten()
        except Exception as e:
            logger.warning(f"Saliency calculation failed: {e}")
            saliency_scores = None
        
        # 特征贡献
        try:
            contributions = self.contribution.analyze_single_prediction(
                sample.squeeze(0), self.feature_names
            )
        except Exception as e:
            logger.warning(f"Contribution analysis failed: {e}")
            contributions = {}
        
        return {
            'prediction': prediction,
            'saliency_scores': saliency_scores.tolist() if saliency_scores is not None else None,
            'contributions': contributions,
            'sample_shape': list(sample.shape),
        }
    
    def generate_report(self, X: 'torch.Tensor', y: 'torch.Tensor' = None) -> str:
        """
        生成解释报告
        
        Args:
            X: 输入数据
            y: 标签（可选）
            
        Returns:
            文本报告
        """
        report = []
        report.append("=" * 60)
        report.append("             模型预测解释报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().isoformat()}")
        report.append(f"样本数: {X.shape[0]}")
        report.append(f"特征数: {X.shape[-1]}")
        report.append("")
        
        # 特征重要性
        if y is not None:
            report.append("📊 特征重要性排名（基于排列重要性）:")
            report.append("-" * 40)
            
            try:
                importance_df = self.permutation_importance.calculate(X, y, self.feature_names)
                for i, row in importance_df.head(10).iterrows():
                    bar = "█" * int(row['importance_pct'] / 5)
                    report.append(f"  {row['feature']:<20} {row['importance_pct']:>6.2f}% {bar}")
            except Exception as e:
                report.append(f"  计算失败: {e}")
        
        report.append("")
        
        # 样本解释示例
        report.append("📝 样本预测解释（第一个样本）:")
        report.append("-" * 40)
        
        try:
            explanation = self.explain_prediction(X, 0)
            report.append(f"  预测值: {explanation['prediction']:.4f}")
            
            if explanation.get('contributions'):
                report.append("  主要正向贡献:")
                for feat, contrib in explanation['contributions'].get('top_positive', [])[:3]:
                    report.append(f"    + {feat}: {contrib:+.4f}")
                
                report.append("  主要负向贡献:")
                for feat, contrib in explanation['contributions'].get('top_negative', [])[:3]:
                    report.append(f"    - {feat}: {contrib:+.4f}")
        except Exception as e:
            report.append(f"  解释失败: {e}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    print("模型解释性模块测试")
    print("="*50)
    
    if not HAS_TORCH:
        print("需要安装PyTorch才能运行测试")
    else:
        # 创建测试模型
        class SimpleModel(nn.Module):
            def __init__(self, n_features=10):
                super().__init__()
                self.fc1 = nn.Linear(n_features, 32)
                self.fc2 = nn.Linear(32, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel(n_features=10)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # 生成测试数据
        X = torch.randn(50, 10)
        y = torch.randn(50, 1)
        
        # 测试排列重要性
        print("\n1. 排列特征重要性测试")
        perm_imp = PermutationImportance(model, n_repeats=5)
        importance_df = perm_imp.calculate(X, y, feature_names)
        print(importance_df.head().to_string(index=False))
        
        # 测试梯度显著性
        print("\n2. 梯度显著性测试")
        saliency = GradientBasedSaliency(model)
        sal_map = saliency.calculate_saliency(X[:1].clone())
        print(f"  显著性图形状: {sal_map.shape}")
        print(f"  最重要特征: feature_{np.argmax(sal_map)}")
        
        # 测试特征贡献
        print("\n3. 特征贡献分析测试")
        contrib = FeatureContribution(model)
        result = contrib.analyze_single_prediction(X[0], feature_names, n_samples=20)
        print(f"  预测值: {result['prediction']:.4f}")
        print("  Top正向贡献:", result['top_positive'][:3])
        
        # 测试预测解释器
        print("\n4. 预测解释报告")
        explainer = PredictionExplainer(model, feature_names)
        report = explainer.generate_report(X, y)
        print(report)
        
        print("\n模型解释性模块测试完成!")
