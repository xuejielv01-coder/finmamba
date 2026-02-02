# -*- coding: utf-8 -*-
"""
SOTA 指标计算
PRD 5.2 实现

特性:
- RankIC (Spearman)
- ICIR
- 分组单调性
- 多头超额收益
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("Metrics")


class SOTAMetrics:
    """
    SOTA 指标计算器
    
    计算:
    - Rank IC
    - ICIR
    - 分组单调性
    - 多头超额
    """
    
    def __init__(self, n_groups: int = 10):
        """
        初始化
        
        Args:
            n_groups: 分组数量
        """
        self.n_groups = n_groups
        self.results: Dict = {}
    
    def calculate_rank_ic(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        计算 Rank IC (Spearman Correlation)
        
        Args:
            pred: 预测值
            target: 真实收益
        
        Returns:
            Rank IC
        """
        try:
            ic, _ = spearmanr(pred, target)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    def calculate_ic(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        计算 IC (Pearson Correlation)
        """
        try:
            ic, _ = pearsonr(pred, target)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    def calculate_icir(
        self,
        daily_ics: List[float]
    ) -> float:
        """
        计算 ICIR (IC / IC标准差)
        
        Args:
            daily_ics: 每日 IC 列表
        
        Returns:
            ICIR
        """
        if not daily_ics:
            return 0.0
        
        daily_ics = np.array(daily_ics)
        mean_ic = np.mean(daily_ics)
        std_ic = np.std(daily_ics)
        
        if std_ic < 1e-8:
            return 0.0
        
        return mean_ic / std_ic
    
    def calculate_group_returns(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        n_groups: int = None
    ) -> Dict:
        """
        计算分组收益 (分位数)
        
        Args:
            pred: 预测值
            target: 真实收益
            n_groups: 分组数
        
        Returns:
            分组收益字典
        """
        n_groups = n_groups or self.n_groups
        
        # 按预测值分组
        try:
            groups = pd.qcut(pred, q=n_groups, labels=False, duplicates='drop')
        except:
            return {}
        
        group_returns = {}
        for g in range(n_groups):
            mask = groups == g
            if mask.sum() > 0:
                group_returns[f'G{g+1}'] = np.mean(target[mask])
        
        return group_returns
    
    def check_monotonicity(
        self,
        group_returns: Dict
    ) -> Tuple[bool, str]:
        """
        检查分组单调性
        
        理想情况: G1 (Top) > G2 > ... > G10 (Bottom)
        
        Args:
            group_returns: 分组收益
        
        Returns:
            (是否单调, 警告信息)
        """
        if len(group_returns) < 2:
            return True, ""
        
        values = list(group_returns.values())
        
        # 检查严格递减
        is_monotonic = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        message = ""
        if not is_monotonic:
            # 找出违反单调性的位置
            violations = []
            for i in range(len(values)-1):
                if values[i] < values[i+1]:
                    violations.append(f"G{i+1} < G{i+2}")
            message = f"Non-monotonic: {', '.join(violations)}"
            logger.warning(f"Model Overfitting: {message}")
        
        return is_monotonic, message
    
    def calculate_long_excess(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        top_pct: float = 0.1
    ) -> float:
        """
        计算多头超额收益  
        
        Top 10% 股票相对于全体的超额收益
        
        Args:
            pred: 预测值
            target: 真实收益
            top_pct: 头部比例
        
        Returns:
            超额收益
        """
        if len(pred) == 0: return 0.0
        
        # 找到 top 10% 的阈值
        try:
            threshold = np.percentile(pred, 100 * (1 - top_pct))
        except:
            return 0.0
        
        # 计算头部收益
        top_mask = pred >= threshold
        if np.sum(top_mask) == 0: return 0.0
        
        top_return = np.mean(target[top_mask])
        
        # 全体平均收益
        avg_return = np.mean(target)
        
        # 超额
        excess = top_return - avg_return
        
        return excess

    def calculate_long_short_spread(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        quantile: float = 0.1
    ) -> float:
        """
        计算多空收益差 (Long-Short Spread)
        Top 10% - Bottom 10%
        """
        if len(pred) == 0: return 0.0
        
        try:
            top_thresh = np.percentile(pred, 100 * (1 - quantile))
            bot_thresh = np.percentile(pred, 100 * quantile)
        except:
            return 0.0
            
        top_mask = pred >= top_thresh
        bot_mask = pred <= bot_thresh
        
        if np.sum(top_mask) == 0 or np.sum(bot_mask) == 0: return 0.0
        
        top_ret = np.mean(target[top_mask])
        bot_ret = np.mean(target[bot_mask])
        
        return top_ret - bot_ret

    def calculate_classification_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> Dict:
        """
        计算分类指标 (Accuracy, Precision, Recall, F1)
        注意: 假设 pred > 0 为预测上涨, target > 0 为实际上涨
        """
        if len(pred) == 0: return {}
        
        # 转换为二分类
        pred_label = (pred > 0).astype(int)
        target_label = (target > 0).astype(int)
        
        # 混淆矩阵
        tp = np.sum((pred_label == 1) & (target_label == 1))
        tn = np.sum((pred_label == 0) & (target_label == 0))
        fp = np.sum((pred_label == 1) & (target_label == 0))
        fn = np.sum((pred_label == 0) & (target_label == 1))
        
        total = len(pred)
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
        }
    
    def evaluate(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame = None,
        date_col: str = 'date',
        pred_col: str = 'score',
        target_col: str = 'return'
    ) -> Dict:
        """
        完整评估
        
        Args:
            predictions: 预测 DataFrame (含 date, ts_code, score)
            actuals: 真实收益 DataFrame (含 date, ts_code, return)
            date_col: 日期列名
            pred_col: 预测列名
            target_col: 收益列名
        
        Returns:
            评估结果
        """
        # 合并预测和实际值
        if actuals is not None:
            data = predictions.merge(actuals, on=['ts_code', date_col], how='inner')
        else:
            data = predictions
            if target_col not in data.columns:
                logger.error("Missing target column")
                return {}
        
        # 按日期计算 IC
        daily_ics = []
        daily_rank_ics = []
        daily_group_returns = []
        
        for date, group in data.groupby(date_col):
            pred = group[pred_col].values
            target = group[target_col].values
            
            if len(pred) < 10:
                continue
            
            # IC
            ic = self.calculate_ic(pred, target)
            rank_ic = self.calculate_rank_ic(pred, target)
            
            daily_ics.append(ic)
            daily_rank_ics.append(rank_ic)
            
            # 分组收益
            gr = self.calculate_group_returns(pred, target)
            if gr:
                daily_group_returns.append(gr)
        
        # 汇总结果
        self.results = {
            'mean_ic': np.mean(daily_ics) if daily_ics else 0.0,
            'mean_rank_ic': np.mean(daily_rank_ics) if daily_rank_ics else 0.0,
            'ic_std': np.std(daily_ics) if daily_ics else 0.0,
            'rank_ic_std': np.std(daily_rank_ics) if daily_rank_ics else 0.0,
            'icir': self.calculate_icir(daily_rank_ics),
            'n_days': len(daily_ics)
        }
        
        # 平均分组收益
        if daily_group_returns:
            avg_group_returns = {}
            for key in daily_group_returns[0].keys():
                values = [gr.get(key, 0) for gr in daily_group_returns]
                avg_group_returns[key] = np.mean(values)
            
            self.results['group_returns'] = avg_group_returns
            
            # 单调性检查
            is_mono, msg = self.check_monotonicity(avg_group_returns)
            self.results['is_monotonic'] = is_mono
            self.results['monotonicity_msg'] = msg
        
        # 多头超额
        pred = data[pred_col].values
        target = data[target_col].values
        self.results['long_excess'] = self.calculate_long_excess(pred, target)
        
        # 多空收益差
        self.results['long_short_spread'] = self.calculate_long_short_spread(pred, target)
        
        # 分类指标
        cls_metrics = self.calculate_classification_metrics(pred, target)
        self.results.update(cls_metrics)
        
        # SOTA 验收
        self._check_sota_thresholds()
        
        return self.results
    
    def _check_sota_thresholds(self):
        """检查是否达到 SOTA 标准"""
        warnings = []
        
        if self.results.get('mean_rank_ic', 0) < Config.SOTA_TARGET_IC:
            warnings.append(f"Rank IC ({self.results['mean_rank_ic']:.4f}) < Target ({Config.SOTA_TARGET_IC})")
        
        if self.results.get('icir', 0) < Config.SOTA_TARGET_ICIR:
            warnings.append(f"ICIR ({self.results['icir']:.4f}) < Target ({Config.SOTA_TARGET_ICIR})")
        
        if not self.results.get('is_monotonic', True):
            warnings.append("Non-monotonic group returns detected")
        
        self.results['sota_passed'] = len(warnings) == 0
        self.results['sota_warnings'] = warnings
        
        for w in warnings:
            logger.warning(f"SOTA: {w}")
    
    def generate_report(self) -> str:
        """生成评估报告"""
        r = self.results
        
        report = f"""
═══════════════════════════════════════════════
        DeepAlpha SOTA Metrics Report
═══════════════════════════════════════════════

📊 Information Coefficient
─────────────────────────
Mean IC:          {r.get('mean_ic', 0):>10.4f}
Mean Rank IC:     {r.get('mean_rank_ic', 0):>10.4f}
IC Std:           {r.get('ic_std', 0):>10.4f}
ICIR:             {r.get('icir', 0):>10.4f}

📈 Group Returns
────────────────
"""
        if 'group_returns' in r:
            for g, ret in r['group_returns'].items():
                report += f"{g}:              {ret*100:>10.2f}%\n"
        
        report += f"""
📉 Risk Metrics
────────────────
Long Excess:      {r.get('long_excess', 0)*100:>10.2f}%
Long-Short Spread:{r.get('long_short_spread', 0)*100:>10.2f}%
Monotonicity:     {'✓ PASS' if r.get('is_monotonic', True) else '✗ FAIL'}

🎯 Classification Metrics (Accuracy)
────────────────────────────────────
Accuracy:         {r.get('accuracy', 0)*100:>10.2f}%
Precision:        {r.get('precision', 0)*100:>10.2f}%
Recall:           {r.get('recall', 0)*100:>10.2f}%
F1-Score:         {r.get('f1_score', 0):>10.4f}

🎯 SOTA Verification
────────────────────
Status:           {'✓ PASS' if r.get('sota_passed', False) else '✗ FAIL'}
"""
        if r.get('sota_warnings'):
            report += "Warnings:\n"
            for w in r['sota_warnings']:
                report += f"  • {w}\n"
        
        report += "═══════════════════════════════════════════════\n"
        
        return report


def calculate_metrics(predictions: pd.DataFrame, **kwargs) -> Dict:
    """便捷函数：计算指标"""
    calculator = SOTAMetrics()
    return calculator.evaluate(predictions, **kwargs)
