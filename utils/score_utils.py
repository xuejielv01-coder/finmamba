# -*- coding: utf-8 -*-
"""
分数处理工具模块
用于提升预测区分度和动态方向判断

特性:
- 截面分数标准化 (Cross-Sectional Normalization)
- 动态方向分类
- 基于分位数的置信度计算
- 分数分布追踪
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
from pathlib import Path


class CrossSectionalNormalizer:
    """
    截面分数标准化器
    
    将原始模型分数转换为相对排名分位数，提升区分度
    """
    
    def __init__(self, history_size: int = 30):
        """
        Args:
            history_size: 历史分布保留天数
        """
        self.history_size = history_size
        self._score_history: deque = deque(maxlen=history_size)
    
    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        拟合并转换分数为百分位排名
        
        Args:
            scores: 原始分数数组 (N,)
            
        Returns:
            百分位排名数组 (0-1)
        """
        if len(scores) == 0:
            return np.array([])
        
        # 计算百分位排名 (使用 scipy 风格的排名)
        n = len(scores)
        ranks = np.argsort(np.argsort(scores)) + 1  # 1-based rank
        percentiles = ranks / n
        
        # 记录历史分布
        self._score_history.append({
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q50': float(np.percentile(scores, 50)),
            'q75': float(np.percentile(scores, 75)),
        })
        
        return percentiles
    
    def transform_single(self, score: float, reference_scores: np.ndarray) -> float:
        """
        将单个分数转换为相对于参考分数的百分位
        
        Args:
            score: 单个分数
            reference_scores: 参考分数数组
            
        Returns:
            百分位 (0-1)
        """
        if len(reference_scores) == 0:
            return 0.5
        
        # 计算百分位
        percentile = np.sum(reference_scores < score) / len(reference_scores)
        return float(percentile)
    
    def get_distribution_stats(self) -> Dict:
        """获取历史分布统计"""
        if not self._score_history:
            return {}
        
        recent = list(self._score_history)[-1]
        return recent


class DirectionClassifier:
    """
    动态方向分类器
    
    基于分位数而非硬编码阈值进行方向判断
    """
    
    # 默认分位数阈值 (调整为更合理的标准)
    # 目的: 让各个方向类别都有合理的分布，避免全部集中在"中性"
    DEFAULT_THRESHOLDS = {
        'very_bullish': 0.90,  # Top 10% - 强势看涨
        'bullish': 0.75,       # Top 25% - 看涨
        'neutral_high': 0.60,  # Top 40% - 中性偏多
        'neutral_low': 0.40,   # Bottom 40% - 中性偏空
        'bearish': 0.25,       # Bottom 25% - 看跌
        'very_bearish': 0.10,  # Bottom 10% - 强势看跌
    }
    
    # 方向标签
    DIRECTION_LABELS = {
        'very_bullish': '强势看涨',
        'bullish': '看涨',
        'neutral_high': '中性偏多',
        'neutral': '中性',
        'neutral_low': '中性偏空',
        'bearish': '看跌',
        'very_bearish': '强势看跌',
    }
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Args:
            thresholds: 自定义阈值字典
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
    
    def classify(self, percentile: float) -> Tuple[str, str]:
        """
        根据百分位分类方向
        
        Args:
            percentile: 百分位排名 (0-1)
            
        Returns:
            (direction_key, direction_label)
        """
        if percentile >= self.thresholds['very_bullish']:
            key = 'very_bullish'
        elif percentile >= self.thresholds['bullish']:
            key = 'bullish'
        elif percentile >= self.thresholds['neutral_high']:
            key = 'neutral_high'
        elif percentile >= self.thresholds['neutral_low']:
            key = 'neutral'
        elif percentile >= self.thresholds['bearish']:
            key = 'neutral_low'
        elif percentile >= self.thresholds['very_bearish']:
            key = 'bearish'
        else:
            key = 'very_bearish'
        
        return key, self.DIRECTION_LABELS[key]
    
    def get_risk_level(self, percentile: float) -> str:
        """
        根据百分位获取风险等级
        
        Args:
            percentile: 百分位排名
            
        Returns:
            风险等级: 'low', 'medium', 'high'
        """
        if percentile >= 0.7:
            return 'low'
        elif percentile >= 0.3:
            return 'medium'
        else:
            return 'high'


class ConfidenceCalculator:
    """
    置信度计算器
    
    基于分位数离散度计算置信度
    """
    
    def __init__(self, extreme_threshold: float = 0.1):
        """
        Args:
            extreme_threshold: 极端分位阈值 (距离 0 或 1 的距离)
        """
        self.extreme_threshold = extreme_threshold
    
    def calculate(
        self, 
        percentile: float, 
        score_std: float = None,
        historical_accuracy: float = None
    ) -> Tuple[float, str]:
        """
        计算置信度
        
        Args:
            percentile: 百分位排名
            score_std: 分数标准差 (可选)
            historical_accuracy: 历史准确率 (可选)
            
        Returns:
            (confidence, reason)
        """
        # 基础置信度：距离 0.5 越远越可信
        base_confidence = 2 * abs(percentile - 0.5)
        
        # 极端值加成
        if percentile <= self.extreme_threshold or percentile >= (1 - self.extreme_threshold):
            extreme_bonus = 0.2
            reason = "极端分位，信号较强"
        else:
            extreme_bonus = 0
            reason = "中等分位"
        
        # 分布稳定性调整
        stability_factor = 1.0
        if score_std is not None:
            # 标准差越小，分数越集中，置信度应该降低
            if score_std < 0.01:
                stability_factor = 0.7
                reason = "分数分布过于集中，区分度低"
            elif score_std > 0.1:
                stability_factor = 1.1
                reason = "分数分布合理，区分度高"
        
        # 历史准确率调整
        if historical_accuracy is not None:
            accuracy_factor = 0.5 + 0.5 * historical_accuracy
        else:
            accuracy_factor = 1.0
        
        # 最终置信度
        confidence = min(1.0, (base_confidence + extreme_bonus) * stability_factor * accuracy_factor)
        
        return float(confidence), reason


class ScoreDistributionTracker:
    """
    分数分布追踪器
    
    记录历史分数分布，用于动态阈值调整
    """
    
    def __init__(self, cache_dir: Path = None, max_days: int = 60):
        """
        Args:
            cache_dir: 缓存目录
            max_days: 最大保留天数
        """
        self.cache_dir = cache_dir or Path("data/score_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_days = max_days
        self._current_distribution: Optional[Dict] = None
    
    def record_daily_distribution(self, date: str, scores: np.ndarray, ts_codes: List[str]):
        """
        记录每日分数分布
        
        Args:
            date: 日期 YYYYMMDD
            scores: 分数数组
            ts_codes: 股票代码列表
        """
        if len(scores) == 0:
            return
        
        distribution = {
            'date': date,
            'n_stocks': len(scores),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'q10': float(np.percentile(scores, 10)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
            'q90': float(np.percentile(scores, 90)),
            # 存储 Top/Bottom 股票用于参考
            'top_10': self._get_top_stocks(scores, ts_codes, 10),
            'bottom_10': self._get_bottom_stocks(scores, ts_codes, 10),
        }
        
        self._current_distribution = distribution
        
        # 保存到文件
        cache_file = self.cache_dir / f"score_dist_{date}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(distribution, f, ensure_ascii=False, indent=2)
    
    def _get_top_stocks(self, scores: np.ndarray, ts_codes: List[str], n: int) -> List[Dict]:
        """获取 Top N 股票"""
        indices = np.argsort(scores)[-n:][::-1]
        return [
            {'ts_code': ts_codes[i], 'score': float(scores[i])}
            for i in indices if i < len(ts_codes)
        ]
    
    def _get_bottom_stocks(self, scores: np.ndarray, ts_codes: List[str], n: int) -> List[Dict]:
        """获取 Bottom N 股票"""
        indices = np.argsort(scores)[:n]
        return [
            {'ts_code': ts_codes[i], 'score': float(scores[i])}
            for i in indices if i < len(ts_codes)
        ]
    
    def load_distribution(self, date: str) -> Optional[Dict]:
        """加载指定日期的分布"""
        cache_file = self.cache_dir / f"score_dist_{date}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_current_distribution(self) -> Optional[Dict]:
        """获取当前分布"""
        return self._current_distribution
    
    def get_recent_distributions(self, n_days: int = 7) -> List[Dict]:
        """获取最近 N 天的分布"""
        import glob
        files = sorted(glob.glob(str(self.cache_dir / "score_dist_*.json")))[-n_days:]
        distributions = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as fp:
                distributions.append(json.load(fp))
        return distributions


class ScoreUtils:
    """
    分数处理工具集成类
    
    组合所有分数处理功能
    """
    
    def __init__(self, cache_dir: Path = None):
        self.normalizer = CrossSectionalNormalizer()
        self.classifier = DirectionClassifier()
        self.confidence_calc = ConfidenceCalculator()
        self.tracker = ScoreDistributionTracker(cache_dir)
        
        # 当前批次的参考分数
        self._reference_scores: Optional[np.ndarray] = None
        self._reference_codes: Optional[List[str]] = None
    
    def set_reference_scores(self, scores: np.ndarray, ts_codes: List[str] = None):
        """
        设置参考分数集 (用于单股百分位计算)
        
        Args:
            scores: 全市场分数
            ts_codes: 对应的股票代码
        """
        self._reference_scores = scores
        self._reference_codes = ts_codes
    
    def process_single_score(
        self, 
        score: float, 
        ts_code: str = None
    ) -> Dict:
        """
        处理单个分数，返回完整诊断
        
        Args:
            score: 原始分数
            ts_code: 股票代码
            
        Returns:
            包含百分位、方向、置信度的字典
        """
        # 计算百分位
        if self._reference_scores is not None and len(self._reference_scores) > 0:
            percentile = self.normalizer.transform_single(score, self._reference_scores)
            score_std = float(np.std(self._reference_scores))
        else:
            # 没有参考分数时使用默认值
            percentile = 0.5
            score_std = None
        
        # 分类方向
        direction_key, direction_label = self.classifier.classify(percentile)
        
        # 计算风险等级
        risk_level = self.classifier.get_risk_level(percentile)
        
        # 计算置信度
        confidence, confidence_reason = self.confidence_calc.calculate(
            percentile, score_std
        )
        
        # 计算预期收益范围 (基于历史分布)
        expected_return = self._estimate_expected_return(percentile)
        
        return {
            'raw_score': float(score),
            'rank_percentile': float(percentile),
            'direction': direction_label,
            'direction_key': direction_key,
            'risk_level': risk_level,
            'confidence': float(confidence),
            'confidence_reason': confidence_reason,
            'expected_return_range': expected_return,
            'market_context': self._get_market_context(),
        }
    
    def process_batch_scores(
        self, 
        scores: np.ndarray, 
        ts_codes: List[str],
        date: str = None
    ) -> List[Dict]:
        """
        批量处理分数
        
        Args:
            scores: 分数数组
            ts_codes: 股票代码列表
            date: 日期
            
        Returns:
            处理结果列表
        """
        # 设置参考分数
        self.set_reference_scores(scores, ts_codes)
        
        # 记录分布
        if date:
            self.tracker.record_daily_distribution(date, scores, ts_codes)
        
        # 计算百分位
        percentiles = self.normalizer.fit_transform(scores)
        
        results = []
        for i, (score, percentile, code) in enumerate(zip(scores, percentiles, ts_codes)):
            direction_key, direction_label = self.classifier.classify(percentile)
            risk_level = self.classifier.get_risk_level(percentile)
            confidence, reason = self.confidence_calc.calculate(percentile)
            
            results.append({
                'ts_code': code,
                'score': float(score),
                'rank': i + 1 if i < len(scores) else len(scores),
                'rank_percentile': float(percentile),
                'direction': direction_label,
                'direction_key': direction_key,
                'risk_level': risk_level,
                'confidence': float(confidence),
            })
        
        # 按分数排序并更新排名
        results.sort(key=lambda x: x['score'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    def _estimate_expected_return(self, percentile: float) -> Dict:
        """
        估算预期收益范围 (动态计算版本)
        
        基于百分位和历史分布特征动态计算预期收益
        不再使用固定阈值，而是根据实际数据特征调整
        """
        # 获取当前分布统计
        dist = self.tracker.get_current_distribution()
        
        # 动态计算基准收益倍数 (基于分数分布的离散度)
        if dist:
            score_std = dist.get('std', 0.02)
            score_range = dist.get('max', 0.05) - dist.get('min', -0.05)
            
            # 分数离散度越大，预期收益波动越大
            volatility_factor = min(2.0, score_range / 0.10 + score_std / 0.02)
        else:
            volatility_factor = 1.0
        
        # 基于百分位的动态收益估计
        # 使用连续函数而非离散阈值
        # 中心化百分位: -0.5 到 0.5
        centered_percentile = percentile - 0.5
        
        # 基础日收益 (中性百分位期望为0)
        # 非线性映射: 极端分位有更高的预期收益/损失
        base_return = centered_percentile * 0.02 * (1 + abs(centered_percentile))
        
        # 应用波动率因子
        adjusted_return = base_return * volatility_factor
        
        # 计算收益区间
        uncertainty = 0.01 * volatility_factor * (1 - abs(centered_percentile) * 0.5)
        
        low = adjusted_return - uncertainty
        mid = adjusted_return
        high = adjusted_return + uncertainty
        
        # 生成人类可读标签
        mid_pct = mid * 100
        range_pct = uncertainty * 100
        sign = '+' if mid_pct > 0 else ''
        label = f"{sign}{mid_pct:.2f}%±{range_pct:.2f}%"
        
        return {
            'low': float(low),
            'mid': float(mid),
            'high': float(high),
            'label': label,
            'volatility_factor': volatility_factor,
        }
    
    def _get_market_context(self) -> Dict:
        """获取市场上下文"""
        dist = self.tracker.get_current_distribution()
        if dist:
            return {
                'total_stocks': dist.get('n_stocks', 0),
                'score_range': f"{dist.get('min', 0):.4f} ~ {dist.get('max', 0):.4f}",
                'score_median': dist.get('median', 0),
            }
        return {}


# 便捷函数
_global_utils: ScoreUtils = None

def get_score_utils() -> ScoreUtils:
    """获取全局分数工具实例"""
    global _global_utils
    if _global_utils is None:
        _global_utils = ScoreUtils()
    return _global_utils
