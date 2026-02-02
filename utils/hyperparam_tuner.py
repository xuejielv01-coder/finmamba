# -*- coding: utf-8 -*-
"""
超参数自动调优模块
改进10：集成Optuna实现自动超参数优化

功能:
- 贝叶斯优化搜索超参数
- 多目标优化（IC/Sharpe）
- 剪枝机制加速搜索
- 可视化优化过程
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None
    Trial = None
    MedianPruner = None
    HyperbandPruner = None
    TPESampler = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("HyperParamTuner")


class HyperParamSpace:
    """超参数搜索空间定义"""
    
    # 模型架构参数
    MODEL_PARAMS = {
        'd_model': {'type': 'categorical', 'choices': [64, 128, 256, 512]},
        'd_state': {'type': 'int', 'low': 8, 'high': 64, 'step': 8},
        'n_layers': {'type': 'int', 'low': 2, 'high': 8},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
        'use_industry_embed': {'type': 'categorical', 'choices': [True, False]},
    }
    
    # 训练参数
    TRAINING_PARAMS = {
        'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
        'weight_decay': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-2},
        'warmup_epochs': {'type': 'int', 'low': 1, 'high': 10},
        'max_epochs': {'type': 'int', 'low': 30, 'high': 100},
        'patience': {'type': 'int', 'low': 5, 'high': 20},
    }
    
    # 损失函数参数
    LOSS_PARAMS = {
        'rank_loss_weight': {'type': 'float', 'low': 0.1, 'high': 1.0, 'step': 0.1},
        'ic_loss_weight': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.1},
    }
    
    # 数据参数
    DATA_PARAMS = {
        'lookback': {'type': 'categorical', 'choices': [10, 20, 30, 60]},
        'train_ratio': {'type': 'float', 'low': 0.6, 'high': 0.8, 'step': 0.05},
    }
    
    @classmethod
    def get_all_params(cls) -> Dict:
        """获取所有参数空间"""
        all_params = {}
        all_params.update(cls.MODEL_PARAMS)
        all_params.update(cls.TRAINING_PARAMS)
        all_params.update(cls.LOSS_PARAMS)
        all_params.update(cls.DATA_PARAMS)
        return all_params
    
    @classmethod
    def suggest_param(cls, trial: 'Trial', name: str, config: Dict) -> Any:
        """根据配置建议参数值"""
        param_type = config['type']
        
        if param_type == 'int':
            return trial.suggest_int(name, config['low'], config['high'], 
                                      step=config.get('step', 1))
        elif param_type == 'float':
            return trial.suggest_float(name, config['low'], config['high'],
                                        step=config.get('step'))
        elif param_type == 'loguniform':
            return trial.suggest_float(name, config['low'], config['high'], log=True)
        elif param_type == 'categorical':
            return trial.suggest_categorical(name, config['choices'])
        else:
            raise ValueError(f"Unknown param type: {param_type}")


class ObjectiveFunction:
    """优化目标函数"""
    
    def __init__(self, 
                 train_fn: Callable,
                 eval_fn: Callable,
                 metric: str = 'ic',
                 param_space: Dict = None,
                 fast_mode: bool = True):
        """
        初始化目标函数
        
        Args:
            train_fn: 训练函数，接收超参数字典，返回训练后的模型
            eval_fn: 评估函数，接收模型，返回指标字典
            metric: 优化目标指标（'ic', 'sharpe', 'combined'）
            param_space: 自定义参数空间
            fast_mode: 快速模式（减少epoch数加速搜索）
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.metric = metric
        self.param_space = param_space or HyperParamSpace.get_all_params()
        self.fast_mode = fast_mode
        
        self.best_params: Dict = {}
        self.best_value: float = float('-inf')
        self.history: List[Dict] = []
    
    def __call__(self, trial: 'Trial') -> float:
        """执行一次试验"""
        # 采样超参数
        params = {}
        for name, config in self.param_space.items():
            params[name] = HyperParamSpace.suggest_param(trial, name, config)
        
        # 快速模式下减少训练epoch
        if self.fast_mode:
            params['max_epochs'] = min(params.get('max_epochs', 30), 15)
            params['patience'] = min(params.get('patience', 10), 5)
        
        try:
            # 训练模型
            logger.info(f"Trial {trial.number}: Training with params {params}")
            model = self.train_fn(params)
            
            # 评估模型
            metrics = self.eval_fn(model)
            
            # 计算目标值
            if self.metric == 'ic':
                value = metrics.get('ic', 0.0)
            elif self.metric == 'sharpe':
                value = metrics.get('sharpe_ratio', 0.0)
            elif self.metric == 'combined':
                # 多目标：IC和Sharpe的加权组合
                ic = metrics.get('ic', 0.0)
                sharpe = metrics.get('sharpe_ratio', 0.0)
                value = 0.5 * ic + 0.5 * (sharpe / 3.0)  # 标准化Sharpe
            else:
                value = metrics.get(self.metric, 0.0)
            
            # 记录结果
            result = {
                'trial': trial.number,
                'params': params,
                'metrics': metrics,
                'value': value,
                'timestamp': datetime.now().isoformat(),
            }
            self.history.append(result)
            
            # 更新最佳结果
            if value > self.best_value:
                self.best_value = value
                self.best_params = params.copy()
                logger.info(f"New best: {value:.4f}")
            
            # 报告中间结果用于剪枝
            trial.report(value, step=1)
            
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned")
                raise optuna.TrialPruned()
            
            return value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('-inf')


class HyperParamTuner:
    """超参数调优器"""
    
    def __init__(self, 
                 storage_dir: Path = None,
                 study_name: str = "finmamba_hpo"):
        """
        初始化调优器
        
        Args:
            storage_dir: 结果存储目录
            study_name: 研究名称
        """
        if not HAS_OPTUNA:
            raise ImportError("请安装 optuna: pip install optuna")
        
        self.storage_dir = storage_dir or Config.MODEL_DIR / "hpo"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.study_name = study_name
        
        self.study: Optional['optuna.Study'] = None
        self.objective: Optional[ObjectiveFunction] = None
        
        logger.info(f"HyperParamTuner initialized, storage: {self.storage_dir}")
    
    def create_study(self, 
                    direction: str = 'maximize',
                    pruner: str = 'hyperband',
                    load_if_exists: bool = True) -> 'optuna.Study':
        """
        创建或加载研究
        
        Args:
            direction: 优化方向 ('maximize' 或 'minimize')
            pruner: 剪枝器类型 ('median', 'hyperband')
            load_if_exists: 如果存在则加载
        """
        if not HAS_OPTUNA:
            raise ImportError("请安装 optuna: pip install optuna")
        
        # 设置存储
        storage = f"sqlite:///{self.storage_dir / 'optuna.db'}"
        
        # 创建剪枝器
        if pruner == 'median' and MedianPruner is not None:
            pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        elif pruner == 'hyperband' and HyperbandPruner is not None:
            pruner_obj = HyperbandPruner(min_resource=1, max_resource=10)
        else:
            pruner_obj = None
        
        # 创建采样器
        sampler = TPESampler(seed=Config.SEED, n_startup_trials=10) if TPESampler is not None else None
        
        # 创建研究
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner_obj,
            load_if_exists=load_if_exists
        )
        
        logger.info(f"Study created/loaded: {self.study_name}")
        return self.study
    
    def optimize(self,
                train_fn: Callable,
                eval_fn: Callable,
                n_trials: int = 50,
                timeout: int = None,
                metric: str = 'ic',
                param_space: Dict = None,
                fast_mode: bool = True,
                callbacks: List[Callable] = None) -> Dict:
        """
        执行超参数优化
        
        Args:
            train_fn: 训练函数
            eval_fn: 评估函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            metric: 优化目标
            param_space: 参数空间
            fast_mode: 快速模式
            callbacks: 回调函数列表
            
        Returns:
            最佳参数和结果
        """
        if self.study is None:
            self.create_study()
        
        # 创建目标函数
        self.objective = ObjectiveFunction(
            train_fn=train_fn,
            eval_fn=eval_fn,
            metric=metric,
            param_space=param_space,
            fast_mode=fast_mode
        )
        
        logger.info(f"Starting optimization: {n_trials} trials, metric={metric}")
        
        # 执行优化
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # 获取最佳结果
        best_trial = self.study.best_trial
        
        result = {
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'n_trials': len(self.study.trials),
            'n_complete': len([t for t in self.study.trials 
                               if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in self.study.trials 
                             if t.state == optuna.trial.TrialState.PRUNED]),
            'history': self.objective.history,
        }
        
        # 保存结果
        self._save_results(result)
        
        logger.info(f"Optimization complete. Best {metric}: {best_trial.value:.4f}")
        
        return result
    
    def _save_results(self, result: Dict):
        """保存优化结果"""
        result_file = self.storage_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 转换为可序列化格式
        serializable_result = {
            'best_params': result['best_params'],
            'best_value': float(result['best_value']),
            'n_trials': result['n_trials'],
            'n_complete': result['n_complete'],
            'n_pruned': result['n_pruned'],
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {result_file}")
    
    def get_optimization_history(self) -> pd.DataFrame:
        """获取优化历史"""
        if self.study is None:
            return pd.DataFrame()
        
        trials_df = self.study.trials_dataframe()
        return trials_df
    
    def get_param_importance(self) -> Dict[str, float]:
        """获取参数重要性"""
        if self.study is None or len(self.study.trials) < 10:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Failed to compute param importance: {e}")
            return {}
    
    def suggest_next_params(self) -> Dict:
        """建议下一组参数（用于手动调整）"""
        if self.study is None:
            return {}
        
        # 创建一个虚拟试验来获取建议
        trial = self.study.ask()
        
        suggested = {}
        param_space = HyperParamSpace.get_all_params()
        
        for name, config in param_space.items():
            suggested[name] = HyperParamSpace.suggest_param(trial, name, config)
        
        return suggested
    
    def get_best_params(self) -> Dict:
        """获取当前最佳参数"""
        if self.study is None or len(self.study.trials) == 0:
            return {}
        
        return self.study.best_params
    
    def generate_report(self) -> str:
        """生成优化报告"""
        if self.study is None:
            return "No study available."
        
        trials = self.study.trials
        n_complete = len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║               超参数优化报告                                    ║
╠══════════════════════════════════════════════════════════════╣
║ 研究名称: {self.study_name:<50}║
║ 总试验数: {len(trials):<50}║
║ 完成试验: {n_complete:<50}║
║ 剪枝试验: {n_pruned:<50}║
║ 最佳值:   {self.study.best_value:.4f:<50}║
╠══════════════════════════════════════════════════════════════╣
║ 最佳参数:                                                     ║
"""
        
        for name, value in self.study.best_params.items():
            report += f"║   {name}: {value:<52}║\n"
        
        # 参数重要性
        importance = self.get_param_importance()
        if importance:
            report += "╠══════════════════════════════════════════════════════════════╣\n"
            report += "║ 参数重要性:                                                   ║\n"
            for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
                report += f"║   {name}: {imp:.4f:<48}║\n"
        
        report += "╚══════════════════════════════════════════════════════════════╝"
        
        return report


# 便捷函数
def quick_tune(train_fn: Callable, 
               eval_fn: Callable, 
               n_trials: int = 20,
               metric: str = 'ic') -> Dict:
    """
    快速超参数调优
    
    Args:
        train_fn: 训练函数
        eval_fn: 评估函数
        n_trials: 试验次数
        metric: 优化目标
        
    Returns:
        最佳参数
    """
    tuner = HyperParamTuner()
    result = tuner.optimize(
        train_fn=train_fn,
        eval_fn=eval_fn,
        n_trials=n_trials,
        metric=metric,
        fast_mode=True
    )
    
    print(tuner.generate_report())
    
    return result['best_params']


if __name__ == "__main__":
    # 测试代码
    if not HAS_OPTUNA:
        print("请先安装 optuna: pip install optuna")
    else:
        print("Optuna 已安装，超参数调优模块就绪")
        
        # 示例：使用模拟的训练和评估函数
        def mock_train(params):
            return params  # 返回参数作为"模型"
        
        def mock_eval(model):
            # 模拟评估，返回随机指标
            import random
            return {
                'ic': random.uniform(0.02, 0.08),
                'sharpe_ratio': random.uniform(0.5, 2.0),
            }
        
        tuner = HyperParamTuner()
        tuner.create_study()
        
        # 只运行少量试验作为测试
        result = tuner.optimize(
            train_fn=mock_train,
            eval_fn=mock_eval,
            n_trials=5,
            metric='ic',
            fast_mode=True
        )
        
        print(tuner.generate_report())
