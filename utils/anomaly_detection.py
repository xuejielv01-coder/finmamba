# -*- coding: utf-8 -*-
"""
异常检测模块
改进14：实现预测异常检测和数据异常识别

功能:
- 预测异常检测（识别异常预测）
- 数据异常检测（异常样本识别）
- 分布外检测 (OOD Detection)
- 多种异常检测方法
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

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

logger = get_logger("AnomalyDetection")


class StatisticalAnomalyDetector:
    """基于统计的异常检测器"""
    
    def __init__(self, 
                 method: str = 'zscore',
                 threshold: float = 3.0):
        """
        初始化统计异常检测器
        
        Args:
            method: 检测方法 ('zscore', 'iqr', 'mad')
            threshold: 异常阈值
        """
        self.method = method
        self.threshold = threshold
        
        # 拟合参数
        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.mad_ = None
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """
        拟合检测器
        
        Args:
            X: 训练数据
        """
        X = np.asarray(X).flatten()
        
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        self.median_ = np.median(X)
        self.mad_ = np.median(np.abs(X - self.median_))
        self.q1_ = np.percentile(X, 25)
        self.q3_ = np.percentile(X, 75)
        self.iqr_ = self.q3_ - self.q1_
        
        self.fitted = True
        logger.info(f"Statistical detector fitted with {self.method} method")
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测异常
        
        Args:
            X: 待检测数据
            
        Returns:
            (异常标记数组, 异常分数数组)
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        X = np.asarray(X).flatten()
        
        if self.method == 'zscore':
            scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
            is_anomaly = scores > self.threshold
            
        elif self.method == 'iqr':
            lower = self.q1_ - self.threshold * self.iqr_
            upper = self.q3_ + self.threshold * self.iqr_
            is_anomaly = (X < lower) | (X > upper)
            
            # 归一化分数
            distance_from_bound = np.maximum(
                lower - X, 
                X - upper
            )
            scores = np.maximum(0, distance_from_bound / (self.iqr_ + 1e-10))
            
        elif self.method == 'mad':
            # Modified Z-score using MAD
            scores = 0.6745 * np.abs(X - self.median_) / (self.mad_ + 1e-10)
            is_anomaly = scores > self.threshold
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return is_anomaly, scores
    
    def fit_detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并检测"""
        self.fit(X)
        return self.detect(X)


class IsolationForestDetector:
    """隔离森林异常检测器（简化实现）"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_samples: int = 256,
                 contamination: float = 0.1):
        """
        初始化隔离森林
        
        Args:
            n_estimators: 树的数量
            max_samples: 每棵树的采样数
            contamination: 预期异常比例
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        
        self.trees: List[Dict] = []
        self.threshold_ = None
        self.fitted = False
    
    def _build_tree(self, X: np.ndarray, depth: int = 0, max_depth: int = 10) -> Dict:
        """构建一棵隔离树"""
        n_samples = X.shape[0]
        
        # 终止条件
        if depth >= max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
        
        # 随机选择特征和分割点
        feature_idx = np.random.randint(0, X.shape[1])
        feature_values = X[:, feature_idx]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}
        
        split_value = np.random.uniform(min_val, max_val)
        
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature_idx,
            'split': split_value,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth),
        }
    
    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """计算样本在树中的路径长度"""
        if tree['type'] == 'leaf':
            size = tree['size']
            if size <= 1:
                return depth
            # 使用平均未成功搜索长度估计
            return depth + 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size
        
        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)
    
    def fit(self, X: np.ndarray):
        """拟合隔离森林"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        sample_size = min(self.max_samples, n_samples)
        max_depth = int(np.ceil(np.log2(max(sample_size, 2))))
        
        self.trees = []
        for _ in range(self.n_estimators):
            # 随机采样
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[indices]
            
            tree = self._build_tree(X_sample, max_depth=max_depth)
            self.trees.append(tree)
        
        # 计算阈值
        scores = self._anomaly_score(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        
        self.fitted = True
        logger.info(f"Isolation Forest fitted with {self.n_estimators} trees")
    
    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """计算异常分数"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 理论平均路径长度
        n = self.max_samples
        c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        
        scores = []
        for x in X:
            path_lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            
            # 异常分数
            score = 2 ** (-avg_path_length / c_n)
            scores.append(score)
        
        return np.array(scores)
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测异常"""
        if not self.fitted:
            raise ValueError("Detector not fitted")
        
        scores = self._anomaly_score(X)
        is_anomaly = scores > self.threshold_
        
        return is_anomaly, scores
    
    def fit_detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并检测"""
        self.fit(X)
        return self.detect(X)


class LocalOutlierFactor:
    """局部离群因子检测（简化实现）"""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        """
        初始化LOF检测器
        
        Args:
            n_neighbors: 邻居数量
            contamination: 预期异常比例
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        self.X_train_ = None
        self.threshold_ = None
        self.fitted = False
    
    def _k_nearest_neighbors(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """找到k个最近邻"""
        distances = np.linalg.norm(X - x, axis=1)
        k_indices = np.argsort(distances)[1:self.n_neighbors + 1]  # 排除自己
        return k_indices, distances[k_indices]
    
    def _reachability_distance(self, X: np.ndarray, a: np.ndarray, b_idx: int) -> float:
        """计算可达距离"""
        b = X[b_idx]
        _, k_distances = self._k_nearest_neighbors(X, b)
        k_distance = k_distances[-1] if len(k_distances) > 0 else 0
        
        dist = np.linalg.norm(a - b)
        return max(k_distance, dist)
    
    def _local_reachability_density(self, X: np.ndarray, idx: int) -> float:
        """计算局部可达密度"""
        x = X[idx]
        k_indices, _ = self._k_nearest_neighbors(X, x)
        
        if len(k_indices) == 0:
            return 0
        
        reach_distances = [
            self._reachability_distance(X, x, neighbor_idx) 
            for neighbor_idx in k_indices
        ]
        
        avg_reach_dist = np.mean(reach_distances)
        return 1.0 / (avg_reach_dist + 1e-10)
    
    def fit(self, X: np.ndarray):
        """拟合LOF"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.X_train_ = X.copy()
        
        # 计算训练数据的LOF
        lof_scores = self._lof_scores(X)
        self.threshold_ = np.percentile(lof_scores, 100 * (1 - self.contamination))
        
        self.fitted = True
        logger.info(f"LOF detector fitted with {self.n_neighbors} neighbors")
    
    def _lof_scores(self, X: np.ndarray) -> np.ndarray:
        """计算LOF分数"""
        n_samples = X.shape[0]
        
        # 简化计算：使用相邻样本的比值
        lrd = np.array([self._local_reachability_density(X, i) for i in range(n_samples)])
        
        lof_scores = []
        for i in range(n_samples):
            k_indices, _ = self._k_nearest_neighbors(X, X[i])
            
            if len(k_indices) == 0 or lrd[i] == 0:
                lof_scores.append(1.0)
            else:
                neighbor_lrd = lrd[k_indices]
                lof = np.mean(neighbor_lrd) / lrd[i]
                lof_scores.append(lof)
        
        return np.array(lof_scores)
    
    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测异常"""
        if not self.fitted:
            raise ValueError("Detector not fitted")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 对新样本使用训练数据计算LOF
        combined = np.vstack([self.X_train_, X])
        all_scores = self._lof_scores(combined)
        
        scores = all_scores[len(self.X_train_):]
        is_anomaly = scores > self.threshold_
        
        return is_anomaly, scores


class NeuralNetworkAnomalyDetector:
    """基于神经网络的异常检测器"""
    
    def __init__(self, model: 'nn.Module', threshold_percentile: float = 95):
        """
        初始化神经网络异常检测器
        
        使用模型的重构误差或预测不确定性来检测异常
        
        Args:
            model: PyTorch模型
            threshold_percentile: 异常阈值百分位
        """
        if not HAS_TORCH:
            raise ImportError("请安装PyTorch")
        
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None
        self.fitted = False
    
    def fit(self, X: 'torch.Tensor', method: str = 'reconstruction'):
        """
        拟合检测器
        
        Args:
            X: 训练数据
            method: 检测方法 ('reconstruction', 'prediction_variance')
        """
        self.method = method
        self.model.eval()
        
        with torch.no_grad():
            if method == 'reconstruction':
                # 假设模型是自编码器
                reconstructed = self.model(X)
                errors = torch.mean((X - reconstructed) ** 2, dim=-1).cpu().numpy()
            else:
                # 使用预测方差（需要MC Dropout）
                predictions = []
                self.model.train()  # 启用dropout
                for _ in range(30):
                    pred = self.model(X)
                    predictions.append(pred.cpu().numpy())
                predictions = np.array(predictions)
                errors = np.var(predictions, axis=0).mean(axis=-1)
        
        self.threshold_ = np.percentile(errors, self.threshold_percentile)
        self.fitted = True
        
        logger.info(f"Neural network detector fitted with {method} method")
    
    def detect(self, X: 'torch.Tensor') -> Tuple[np.ndarray, np.ndarray]:
        """检测异常"""
        if not self.fitted:
            raise ValueError("Detector not fitted")
        
        self.model.eval()
        
        with torch.no_grad():
            if self.method == 'reconstruction':
                reconstructed = self.model(X)
                errors = torch.mean((X - reconstructed) ** 2, dim=-1).cpu().numpy()
            else:
                predictions = []
                self.model.train()
                for _ in range(30):
                    pred = self.model(X)
                    predictions.append(pred.cpu().numpy())
                predictions = np.array(predictions)
                errors = np.var(predictions, axis=0).mean(axis=-1)
        
        is_anomaly = errors > self.threshold_
        return is_anomaly, errors


class PredictionAnomalyMonitor:
    """预测异常监控器"""
    
    def __init__(self):
        """初始化预测异常监控器"""
        self.history = []
        self.detector = StatisticalAnomalyDetector(method='zscore', threshold=3.0)
        self.fitted = False
    
    def update(self, predictions: np.ndarray, actuals: np.ndarray = None):
        """
        更新监控数据
        
        Args:
            predictions: 预测值
            actuals: 实际值（可选）
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions.tolist(),
            'n_predictions': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
        }
        
        if actuals is not None:
            errors = predictions - actuals
            entry['errors'] = errors.tolist()
            entry['mse'] = float(np.mean(errors ** 2))
            entry['mae'] = float(np.mean(np.abs(errors)))
        
        self.history.append(entry)
        
        # 动态更新检测器
        if len(self.history) >= 10:
            all_predictions = np.concatenate([
                np.array(h['predictions']) for h in self.history[-100:]
            ])
            self.detector.fit(all_predictions)
            self.fitted = True
    
    def check_anomaly(self, predictions: np.ndarray) -> Dict:
        """
        检查预测是否异常
        
        Args:
            predictions: 待检查的预测
            
        Returns:
            异常检测结果
        """
        if not self.fitted:
            return {
                'is_anomaly': False,
                'message': 'Detector not fitted yet',
                'n_anomalies': 0,
            }
        
        is_anomaly, scores = self.detector.detect(predictions)
        
        anomaly_indices = np.where(is_anomaly)[0]
        
        return {
            'is_anomaly': np.any(is_anomaly),
            'n_anomalies': int(np.sum(is_anomaly)),
            'anomaly_ratio': float(np.mean(is_anomaly)),
            'anomaly_indices': anomaly_indices.tolist(),
            'max_score': float(np.max(scores)),
            'mean_score': float(np.mean(scores)),
        }
    
    def get_summary(self) -> Dict:
        """获取监控摘要"""
        if not self.history:
            return {}
        
        recent = self.history[-10:]
        
        return {
            'n_checks': len(self.history),
            'avg_predictions_per_check': np.mean([h['n_predictions'] for h in recent]),
            'prediction_mean_trend': [h['mean'] for h in recent],
            'prediction_std_trend': [h['std'] for h in recent],
        }


class AnomalyDetectorFactory:
    """异常检测器工厂"""
    
    @staticmethod
    def create(method: str, **kwargs) -> Any:
        """
        创建异常检测器
        
        Args:
            method: 检测方法
            **kwargs: 方法参数
            
        Returns:
            异常检测器实例
        """
        if method == 'zscore':
            return StatisticalAnomalyDetector(method='zscore', **kwargs)
        elif method == 'iqr':
            return StatisticalAnomalyDetector(method='iqr', **kwargs)
        elif method == 'mad':
            return StatisticalAnomalyDetector(method='mad', **kwargs)
        elif method == 'isolation_forest':
            return IsolationForestDetector(**kwargs)
        elif method == 'lof':
            return LocalOutlierFactor(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("异常检测模块测试")
    print("="*50)
    
    # 生成测试数据（包含异常点）
    np.random.seed(42)
    normal_data = np.random.randn(100)
    anomalies = np.array([10, -8, 15, -12])  # 异常点
    test_data = np.concatenate([normal_data, anomalies])
    
    # 测试统计方法
    print("\n1. Z-Score异常检测")
    detector = StatisticalAnomalyDetector(method='zscore', threshold=3.0)
    is_anomaly, scores = detector.fit_detect(test_data)
    print(f"  检测到 {np.sum(is_anomaly)} 个异常")
    print(f"  异常索引: {np.where(is_anomaly)[0].tolist()}")
    
    # 测试IQR方法
    print("\n2. IQR异常检测")
    detector_iqr = StatisticalAnomalyDetector(method='iqr', threshold=1.5)
    is_anomaly, scores = detector_iqr.fit_detect(test_data)
    print(f"  检测到 {np.sum(is_anomaly)} 个异常")
    
    # 测试隔离森林
    print("\n3. 隔离森林异常检测")
    test_2d = np.column_stack([test_data, test_data + np.random.randn(len(test_data)) * 0.5])
    iso_forest = IsolationForestDetector(n_estimators=50, contamination=0.05)
    is_anomaly, scores = iso_forest.fit_detect(test_2d)
    print(f"  检测到 {np.sum(is_anomaly)} 个异常")
    
    # 测试预测监控
    print("\n4. 预测异常监控")
    monitor = PredictionAnomalyMonitor()
    
    # 模拟多次预测
    for _ in range(15):
        preds = np.random.randn(20) * 0.1
        monitor.update(preds)
    
    # 检查异常预测
    anomaly_preds = np.array([5.0, 0.1, -4.0, 0.05])
    result = monitor.check_anomaly(anomaly_preds)
    print(f"  异常检测结果: {result['n_anomalies']} 个异常")
    print(f"  最大异常分数: {result['max_score']:.2f}")
    
    # 工厂模式
    print("\n5. 检测器工厂")
    for method in ['zscore', 'iqr', 'isolation_forest']:
        detector = AnomalyDetectorFactory.create(method)
        print(f"  创建 {method} 检测器: {type(detector).__name__}")
    
    print("\n异常检测模块测试完成!")
