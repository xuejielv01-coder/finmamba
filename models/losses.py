# -*- coding: utf-8 -*-
"""
损失函数模块

包含:
- MSELoss: 预测误差
- RankLoss: 排序损失
- CombinedLoss: 组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class RankLoss(nn.Module):
    """
    排序损失 (Pairwise Ranking Loss)
    
    优化目标: 正确排序股票收益
    """
    
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1) 预测分数
            target: (B,) 真实标签 (排名或收益)
        
        Returns:
            Ranking loss
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 确保至少是 1D 张量
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        B = pred.size(0)
        if B < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # 创建配对比较
        # diff_pred[i, j] = pred[i] - pred[j]
        diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)  # (B, B)
        diff_target = target.unsqueeze(1) - target.unsqueeze(0)  # (B, B)
        
        # 符号比较: target_i > target_j 时，应有 pred_i > pred_j
        sign = torch.sign(diff_target)  # (B, B)
        
        # Hinge loss: max(0, margin - sign * diff_pred)
        loss = F.relu(self.margin - sign * diff_pred)
        
        # 只计算上三角 (避免重复)
        mask = torch.triu(torch.ones(B, B, device=pred.device), diagonal=1).bool()
        loss = loss[mask].mean()
        
        return loss


class ICLoss(nn.Module):
    """
    IC (Information Coefficient) 损失
    
    最大化预测与真实值的相关性
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1) 预测分数
            target: (B,) 真实标签
        
        Returns:
            -IC (负相关系数，用于最小化)
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 确保至少是 1D 张量
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        # 标准化
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_std = pred.std() + 1e-8
        target_std = target.std() + 1e-8
        
        pred_norm = (pred - pred_mean) / pred_std
        target_norm = (target - target_mean) / target_std
        
        # 皮尔逊相关系数
        ic = (pred_norm * target_norm).mean()
        
        # 返回负值 (最小化损失 = 最大化 IC)
        return -ic



class DirectionLoss(nn.Module):
    """
    方向预测损失
    
    优化目标: 预测符号与真实符号一致
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1) 预测分数 [-1, 1]
            target: (B,) 真实收益
        
        Returns:
            Direction loss
        """
        # Hinge Loss: relu( -pred * sign(target) )
        # 如果符号一致，loss为0
        # 如果符号不一致，loss为 |pred|
        return torch.mean(F.relu(-pred * torch.sign(target)))


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合 MSE 和 RankLoss
    支持多任务学习：同时预测多个时间尺度
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        rank_weight: float = 1.0,
        ic_weight: float = 0.5,
        dir_weight: float = 0.5,  # 新增方向损失权重
        task_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.ic_weight = ic_weight
        self.dir_weight = dir_weight
        self.task_weights = task_weights
        
        self.mse_loss = nn.MSELoss()
        self.rank_loss = RankLoss()
        self.ic_loss = ICLoss()
        self.dir_loss = DirectionLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> tuple:
        """
        Args:
            pred: (B, 1) 或 (B, n_horizons) 预测分数
            target: (B,) 或 (B, n_horizons) 真实标签
        
        Returns:
            (total_loss, loss_dict)
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 确保至少是 2D 张量
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        
        n_horizons = pred.size(1)
        
        # 如果只有一个目标列，扩展到与预测相同的列数
        if target.size(1) == 1 and n_horizons > 1:
            target = target.expand(-1, n_horizons)
        
        total_loss = 0.0
        loss_dict = {
            'total': 0.0,
            'mse': 0.0,
            'rank': 0.0,
            'ic': 0.0,
            'dir': 0.0
        }
        
        # 为每个预测期限计算损失
        for i in range(n_horizons):
            pred_i = pred[:, i]
            target_i = target[:, i]
            
            # 各项损失
            mse = self.mse_loss(pred_i, target_i)
            rank = self.rank_loss(pred_i, target_i)
            ic = self.ic_loss(pred_i, target_i)
            direction = self.dir_loss(pred_i, target_i)
            
            # 计算当前任务的权重
            task_weight = self.task_weights[i] if self.task_weights else 1.0
            
            # 累加损失
            task_loss = (
                self.mse_weight * mse +
                self.rank_weight * rank +
                self.ic_weight * ic +
                self.dir_weight * direction
            ) * task_weight
            
            total_loss += task_loss
            
            # 记录各期限的损失
            loss_dict[f'mse_{i+1}d'] = mse.item()
            loss_dict[f'rank_{i+1}d'] = rank.item()
            loss_dict[f'ic_{i+1}d'] = -ic.item()
            loss_dict[f'dir_{i+1}d'] = direction.item()
        
        # 计算平均损失
        total_loss = total_loss / n_horizons
        loss_dict['total'] = total_loss.item()
        loss_dict['mse'] = sum(loss_dict[f'mse_{i+1}d'] for i in range(n_horizons)) / n_horizons
        loss_dict['rank'] = sum(loss_dict[f'rank_{i+1}d'] for i in range(n_horizons)) / n_horizons
        loss_dict['ic'] = sum(loss_dict[f'ic_{i+1}d'] for i in range(n_horizons)) / n_horizons
        loss_dict['dir'] = sum(loss_dict[f'dir_{i+1}d'] for i in range(n_horizons)) / n_horizons
        loss_dict['dir'] = sum(loss_dict[f'dir_{i+1}d'] for i in range(n_horizons)) / n_horizons
        
        return total_loss, loss_dict


import numpy as np


def calculate_ic(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 IC (Information Coefficient)
    
    Args:
        pred: 预测值
        target: 真实值
    
    Returns:
        IC 值
    """
    pred = pred.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()
    
    from scipy.stats import pearsonr
    try:
        ic, _ = pearsonr(pred, target)
        return ic if not np.isnan(ic) else 0.0
    except:
        return 0.0


def calculate_rank_ic(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 RankIC (Spearman Correlation)
    
    Args:
        pred: 预测值
        target: 真实值
    
    Returns:
        RankIC 值
    """
    pred = pred.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()
    
    from scipy.stats import spearmanr
    try:
        # 检查输入是否为常量
        if len(np.unique(pred)) == 1 or len(np.unique(target)) == 1:
            return 0.0
        rank_ic, _ = spearmanr(pred, target)
        return rank_ic if not np.isnan(rank_ic) else 0.0
    except:
        return 0.0
