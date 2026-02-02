# -*- coding: utf-8 -*-
from .logger import get_logger, logger, QTextEditHandler
from .seeder import seed_everything, get_device, clear_gpu_memory
from .tools import rate_limit, retry, timer, is_main_board, is_st_stock, robust_zscore

# 高优先级改进模块
from .data_quality import DataQualityMonitor, DataQualityMetrics, DataDriftDetector
from .training_monitor import TrainingMonitor, MetricsTracker
from .confidence_calibration import ConfidenceCalibrator, PlattScaling, RankConfidenceCalibrator
from .portfolio_risk import PortfolioRiskManager, VaRCalculator, CVaRCalculator, DrawdownAnalyzer

# 中优先级改进模块
from .uncertainty_quantification import MCDropout, DeepEnsemble, UncertaintyAwarePredictor
from .slippage_model import DynamicSlippage, VolumeWeightedSlippage, TransactionCostCalculator
from .lookahead_bias_detector import LookaheadBiasReport, TemporalLeakageDetector
from .model_quantization import ModelOptimizer, DynamicQuantizer, ModelPruner

# 低优先级改进模块
from .model_interpretability import PredictionExplainer, PermutationImportance, FeatureContribution
from .anomaly_detection import StatisticalAnomalyDetector, IsolationForestDetector, PredictionAnomalyMonitor
from .stress_testing import StressTester, StressTestReport, HistoricalScenario

# 可选：超参数调优（需要optuna）
try:
    from .hyperparam_tuner import HyperParamTuner, HyperParamSpace
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    HyperParamTuner = None
    HyperParamSpace = None

__all__ = [
    # 原有模块
    'get_logger', 'logger', 'QTextEditHandler',
    'seed_everything', 'get_device', 'clear_gpu_memory',
    'rate_limit', 'retry', 'timer', 'is_main_board', 'is_st_stock', 'robust_zscore',
    # 数据质量监控
    'DataQualityMonitor', 'DataQualityMetrics', 'DataDriftDetector',
    # 训练监控
    'TrainingMonitor', 'MetricsTracker',
    # 置信度校准
    'ConfidenceCalibrator', 'PlattScaling', 'RankConfidenceCalibrator',
    # 组合风险
    'PortfolioRiskManager', 'VaRCalculator', 'CVaRCalculator', 'DrawdownAnalyzer',
    # 超参数调优
    'HyperParamTuner', 'HyperParamSpace', 'HAS_OPTUNA',
    # 不确定性量化
    'MCDropout', 'DeepEnsemble', 'UncertaintyAwarePredictor',
    # 滑点模型
    'DynamicSlippage', 'VolumeWeightedSlippage', 'TransactionCostCalculator',
    # 前瞻偏差检测
    'LookaheadBiasReport', 'TemporalLeakageDetector',
    # 模型量化
    'ModelOptimizer', 'DynamicQuantizer', 'ModelPruner',
    # 模型解释性
    'PredictionExplainer', 'PermutationImportance', 'FeatureContribution',
    # 异常检测
    'StatisticalAnomalyDetector', 'IsolationForestDetector', 'PredictionAnomalyMonitor',
    # 压力测试
    'StressTester', 'StressTestReport', 'HistoricalScenario',
]


