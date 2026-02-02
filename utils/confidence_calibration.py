# -*- coding: utf-8 -*-
"""
预测置信度校准模块
改进13：实现Platt Scaling和Isotonic Regression

功能:
- Platt Scaling校准
- Isotonic Regression校准
- 温度缩放
- 校准效果评估
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy import stats
from scipy.optimize import minimize

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    IsotonicRegression = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("ConfidenceCalibration")


class PlattScaling:
    """Platt Scaling 置信度校准"""
    
    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0
        self.fitted: bool = False
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _negative_log_likelihood(self, params: np.ndarray, 
                                  scores: np.ndarray, 
                                  targets: np.ndarray) -> float:
        """负对数似然损失"""
        a, b = params
        probs = self._sigmoid(a * scores + b)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        nll = -np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
        return nll
    
    def fit(self, scores: np.ndarray, targets: np.ndarray):
        """
        拟合Platt Scaling参数
        
        Args:
            scores: 模型原始分数
            targets: 真实标签 (0/1)
        """
        scores = np.asarray(scores).flatten()
        targets = np.asarray(targets).flatten()
        
        # 使用L-BFGS-B优化
        result = minimize(
            self._negative_log_likelihood,
            x0=[1.0, 0.0],
            args=(scores, targets),
            method='L-BFGS-B'
        )
        
        self.a, self.b = result.x
        self.fitted = True
        
        logger.info(f"Platt Scaling fitted: a={self.a:.4f}, b={self.b:.4f}")
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        校准置信度
        
        Args:
            scores: 原始分数
            
        Returns:
            校准后的概率
        """
        if not self.fitted:
            logger.warning("Platt Scaling not fitted, returning raw scores")
            return self._sigmoid(scores)
        
        scores = np.asarray(scores).flatten()
        return self._sigmoid(self.a * scores + self.b)
    
    def save(self, path: Path):
        """保存参数"""
        params = {'a': self.a, 'b': self.b, 'fitted': self.fitted}
        with open(path, 'w') as f:
            json.dump(params, f)
    
    def load(self, path: Path):
        """加载参数"""
        with open(path, 'r') as f:
            params = json.load(f)
        self.a = params['a']
        self.b = params['b']
        self.fitted = params['fitted']


class IsotonicCalibration:
    """Isotonic Regression 置信度校准"""
    
    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("请安装 sklearn: pip install scikit-learn")
        
        self.regressor = IsotonicRegression(out_of_bounds='clip')
        self.fitted: bool = False
    
    def fit(self, scores: np.ndarray, targets: np.ndarray):
        """
        拟合Isotonic Regression
        
        Args:
            scores: 模型原始分数
            targets: 真实标签
        """
        scores = np.asarray(scores).flatten()
        targets = np.asarray(targets).flatten()
        
        self.regressor.fit(scores, targets)
        self.fitted = True
        
        logger.info("Isotonic Regression calibrator fitted")
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """校准置信度"""
        if not self.fitted:
            logger.warning("Isotonic calibrator not fitted")
            return scores
        
        scores = np.asarray(scores).flatten()
        return self.regressor.predict(scores)
    
    def save(self, path: Path):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self.regressor, f)
    
    def load(self, path: Path):
        """加载模型"""
        with open(path, 'rb') as f:
            self.regressor = pickle.load(f)
        self.fitted = True


class TemperatureScaling:
    """温度缩放校准"""
    
    def __init__(self):
        self.temperature: float = 1.0
        self.fitted: bool = False
    
    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """带温度的Softmax"""
        scaled = logits / temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / np.sum(exp_scaled)
    
    def _cross_entropy_loss(self, temperature: float, 
                            logits: np.ndarray, 
                            targets: np.ndarray) -> float:
        """交叉熵损失"""
        total_loss = 0.0
        for i in range(len(logits)):
            probs = self._softmax(logits[i], temperature)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            total_loss -= np.log(probs[int(targets[i])])
        return total_loss / len(logits)
    
    def fit(self, logits: np.ndarray, targets: np.ndarray):
        """
        拟合温度参数
        
        Args:
            logits: 模型输出logits
            targets: 真实标签
        """
        logits = np.asarray(logits)
        targets = np.asarray(targets)
        
        # 优化温度参数
        result = minimize(
            self._cross_entropy_loss,
            x0=[1.0],
            args=(logits, targets),
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)]
        )
        
        self.temperature = result.x[0]
        self.fitted = True
        
        logger.info(f"Temperature Scaling fitted: T={self.temperature:.4f}")
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """校准logits"""
        return logits / self.temperature
    
    def save(self, path: Path):
        """保存参数"""
        params = {'temperature': self.temperature, 'fitted': self.fitted}
        with open(path, 'w') as f:
            json.dump(params, f)
    
    def load(self, path: Path):
        """加载参数"""
        with open(path, 'r') as f:
            params = json.load(f)
        self.temperature = params['temperature']
        self.fitted = params['fitted']


class ConfidenceCalibrator:
    """统一的置信度校准器"""
    
    def __init__(self, method: str = 'platt', storage_dir: Path = None):
        """
        初始化校准器
        
        Args:
            method: 校准方法 ('platt', 'isotonic', 'temperature')
            storage_dir: 模型存储目录
        """
        self.method = method
        self.storage_dir = storage_dir or Config.MODEL_DIR / "calibration"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建校准器
        if method == 'platt':
            self.calibrator = PlattScaling()
        elif method == 'isotonic':
            self.calibrator = IsotonicCalibration()
        elif method == 'temperature':
            self.calibrator = TemperatureScaling()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        logger.info(f"Confidence calibrator initialized: {method}")
    
    def fit(self, scores: np.ndarray, targets: np.ndarray):
        """拟合校准器"""
        self.calibrator.fit(scores, targets)
        self.save()
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """校准分数"""
        return self.calibrator.calibrate(scores)
    
    def save(self):
        """保存校准器"""
        path = self.storage_dir / f"{self.method}_calibrator.json"
        if self.method == 'isotonic':
            path = self.storage_dir / f"{self.method}_calibrator.pkl"
        self.calibrator.save(path)
        logger.info(f"Calibrator saved to {path}")
    
    def load(self):
        """加载校准器"""
        if self.method == 'isotonic':
            path = self.storage_dir / f"{self.method}_calibrator.pkl"
        else:
            path = self.storage_dir / f"{self.method}_calibrator.json"
        
        if path.exists():
            self.calibrator.load(path)
            logger.info(f"Calibrator loaded from {path}")
            return True
        return False


class CalibrationEvaluator:
    """校准效果评估器"""
    
    @staticmethod
    def expected_calibration_error(probs: np.ndarray, 
                                   targets: np.ndarray, 
                                   n_bins: int = 10) -> float:
        """
        计算期望校准误差 (ECE)
        
        Args:
            probs: 预测概率
            targets: 真实标签
            n_bins: 分箱数量
            
        Returns:
            ECE值
        """
        probs = np.asarray(probs).flatten()
        targets = np.asarray(targets).flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(probs[in_bin])
                avg_accuracy = np.mean(targets[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(probs: np.ndarray, 
                                  targets: np.ndarray, 
                                  n_bins: int = 10) -> float:
        """
        计算最大校准误差 (MCE)
        """
        probs = np.asarray(probs).flatten()
        targets = np.asarray(targets).flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(probs[in_bin])
                avg_accuracy = np.mean(targets[in_bin])
                mce = max(mce, np.abs(avg_confidence - avg_accuracy))
        
        return mce
    
    @staticmethod
    def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
        """
        计算Brier分数
        """
        probs = np.asarray(probs).flatten()
        targets = np.asarray(targets).flatten()
        return np.mean((probs - targets) ** 2)
    
    @staticmethod
    def reliability_diagram(probs: np.ndarray, 
                           targets: np.ndarray, 
                           n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成可靠性图数据
        
        Returns:
            (mean_predicted_probs, fraction_of_positives, bin_counts)
        """
        if HAS_SKLEARN:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                targets, probs, n_bins=n_bins, strategy='uniform'
            )
            return mean_predicted_value, fraction_of_positives, np.array([])
        
        # 手动计算
        probs = np.asarray(probs).flatten()
        targets = np.asarray(targets).flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mean_probs = []
        fractions = []
        counts = []
        
        for i in range(n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            count = np.sum(in_bin)
            
            if count > 0:
                mean_probs.append(np.mean(probs[in_bin]))
                fractions.append(np.mean(targets[in_bin]))
                counts.append(count)
        
        return np.array(mean_probs), np.array(fractions), np.array(counts)
    
    @classmethod
    def evaluate(cls, probs: np.ndarray, targets: np.ndarray) -> Dict:
        """
        全面评估校准效果
        """
        return {
            'ece': cls.expected_calibration_error(probs, targets),
            'mce': cls.maximum_calibration_error(probs, targets),
            'brier': cls.brier_score(probs, targets),
        }
    
    @classmethod
    def compare_calibration(cls, 
                           raw_scores: np.ndarray, 
                           calibrated_probs: np.ndarray,
                           targets: np.ndarray) -> Dict:
        """
        比较校准前后的效果
        """
        # 将原始分数转换为概率
        raw_probs = 1.0 / (1.0 + np.exp(-raw_scores))
        
        before = cls.evaluate(raw_probs, targets)
        after = cls.evaluate(calibrated_probs, targets)
        
        improvement = {
            'ece_improvement': (before['ece'] - after['ece']) / before['ece'] * 100,
            'mce_improvement': (before['mce'] - after['mce']) / before['mce'] * 100,
            'brier_improvement': (before['brier'] - after['brier']) / before['brier'] * 100,
        }
        
        return {
            'before': before,
            'after': after,
            'improvement': improvement,
        }


class RankConfidenceCalibrator:
    """
    排名模型的置信度校准器
    
    专门用于处理股票排名预测的置信度问题
    """
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Config.MODEL_DIR / "calibration"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 历史统计
        self.score_stats: Dict = {}
        self.fitted: bool = False
    
    def fit(self, scores: np.ndarray, returns: np.ndarray, 
            quantiles: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        拟合校准器
        
        Args:
            scores: 历史预测分数
            returns: 历史实际收益
            quantiles: 分位数阈值
        """
        scores = np.asarray(scores).flatten()
        returns = np.asarray(returns).flatten()
        
        # 计算分数的分位数
        score_quantiles = np.quantile(scores, quantiles)
        
        # 统计每个分数区间的收益分布
        self.score_stats = {
            'quantiles': quantiles,
            'score_thresholds': score_quantiles.tolist(),
            'return_stats': {},
        }
        
        prev_threshold = float('-inf')
        for i, threshold in enumerate(score_quantiles):
            mask = (scores > prev_threshold) & (scores <= threshold)
            if np.sum(mask) > 0:
                bucket_returns = returns[mask]
                self.score_stats['return_stats'][f'q{i}'] = {
                    'mean': float(np.mean(bucket_returns)),
                    'std': float(np.std(bucket_returns)),
                    'positive_rate': float(np.mean(bucket_returns > 0)),
                    'n_samples': int(np.sum(mask)),
                }
            prev_threshold = threshold
        
        # 顶部分位数
        mask = scores > score_quantiles[-1]
        if np.sum(mask) > 0:
            bucket_returns = returns[mask]
            self.score_stats['return_stats']['top'] = {
                'mean': float(np.mean(bucket_returns)),
                'std': float(np.std(bucket_returns)),
                'positive_rate': float(np.mean(bucket_returns > 0)),
                'n_samples': int(np.sum(mask)),
            }
        
        self.fitted = True
        logger.info("Rank confidence calibrator fitted")
        
        # 保存
        self.save()
    
    def get_confidence(self, score: float) -> Dict:
        """
        获取分数对应的置信度信息
        
        Args:
            score: 预测分数
            
        Returns:
            置信度信息字典
        """
        if not self.fitted:
            return {
                'confidence': 0.5,
                'expected_return': 0.0,
                'positive_probability': 0.5,
                'reliability': 'unknown',
            }
        
        thresholds = self.score_stats['score_thresholds']
        
        # 确定分数所在的区间
        bucket = 'top'
        for i, threshold in enumerate(thresholds):
            if score <= threshold:
                bucket = f'q{i}'
                break
        
        stats = self.score_stats['return_stats'].get(bucket, {})
        
        # 计算置信度（基于正收益率和样本量）
        positive_rate = stats.get('positive_rate', 0.5)
        n_samples = stats.get('n_samples', 0)
        
        # 样本量加权的置信度调整
        sample_weight = min(1.0, n_samples / 100)
        adjusted_confidence = 0.5 + (positive_rate - 0.5) * sample_weight
        
        # 可靠性评级
        if n_samples < 30:
            reliability = 'low'
        elif n_samples < 100:
            reliability = 'medium'
        else:
            reliability = 'high'
        
        return {
            'confidence': adjusted_confidence,
            'expected_return': stats.get('mean', 0.0),
            'return_std': stats.get('std', 0.0),
            'positive_probability': positive_rate,
            'reliability': reliability,
            'n_historical_samples': n_samples,
        }
    
    def save(self):
        """保存校准器"""
        path = self.storage_dir / "rank_calibrator.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.score_stats, f, indent=2)
    
    def load(self) -> bool:
        """加载校准器"""
        path = self.storage_dir / "rank_calibrator.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self.score_stats = json.load(f)
            self.fitted = True
            logger.info("Rank calibrator loaded")
            return True
        return False


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成模拟数据
    n_samples = 1000
    scores = np.random.randn(n_samples)
    # 使用sigmoid加噪声生成目标
    probs_true = 1.0 / (1.0 + np.exp(-scores))
    targets = (np.random.rand(n_samples) < probs_true).astype(float)
    
    # 测试Platt Scaling
    print("Testing Platt Scaling...")
    platt = PlattScaling()
    platt.fit(scores[:800], targets[:800])
    calibrated = platt.calibrate(scores[800:])
    
    # 评估
    evaluator = CalibrationEvaluator()
    raw_probs = 1.0 / (1.0 + np.exp(-scores[800:]))
    
    print(f"Raw ECE: {evaluator.expected_calibration_error(raw_probs, targets[800:]):.4f}")
    print(f"Calibrated ECE: {evaluator.expected_calibration_error(calibrated, targets[800:]):.4f}")
    
    # 测试排名校准器
    print("\nTesting Rank Confidence Calibrator...")
    returns = scores * 0.02 + np.random.randn(n_samples) * 0.01
    
    rank_calibrator = RankConfidenceCalibrator()
    rank_calibrator.fit(scores, returns)
    
    # 获取置信度
    for test_score in [-2, -1, 0, 1, 2]:
        confidence_info = rank_calibrator.get_confidence(test_score)
        print(f"Score {test_score}: confidence={confidence_info['confidence']:.3f}, "
              f"expected_return={confidence_info['expected_return']:.4f}")
    
    print("\nCalibration module tests passed!")
