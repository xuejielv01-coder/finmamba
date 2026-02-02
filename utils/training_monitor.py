# -*- coding: utf-8 -*-
"""
训练监控模块
改进11：实现TensorBoard/WandB集成

功能:
- TensorBoard实时监控
- 训练指标可视化
- 模型结构可视化
- 训练过程回放
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("TrainingMonitor")


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.steps: Dict[str, List[int]] = {}
        self.timestamps: Dict[str, List[str]] = {}
    
    def add(self, name: str, value: float, step: int):
        """添加指标值"""
        if name not in self.metrics:
            self.metrics[name] = []
            self.steps[name] = []
            self.timestamps[name] = []
        
        self.metrics[name].append(float(value))
        self.steps[name].append(step)
        self.timestamps[name].append(datetime.now().isoformat())
    
    def get(self, name: str) -> List[float]:
        """获取指标历史"""
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        """获取最新值"""
        values = self.metrics.get(name, [])
        return values[-1] if values else None
    
    def get_best(self, name: str, mode: str = 'max') -> Optional[float]:
        """获取最佳值"""
        values = self.metrics.get(name, [])
        if not values:
            return None
        return max(values) if mode == 'max' else min(values)
    
    def get_summary(self) -> Dict[str, Dict]:
        """获取所有指标的汇总"""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'latest': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'n_values': len(values),
                }
        return summary
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'metrics': self.metrics,
            'steps': self.steps,
            'timestamps': self.timestamps,
        }
    
    def clear(self):
        """清空数据"""
        self.metrics.clear()
        self.steps.clear()
        self.timestamps.clear()


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, 
                 log_dir: Path = None,
                 experiment_name: str = None,
                 enable_tensorboard: bool = True,
                 enable_file_logging: bool = True):
        """
        初始化训练监控器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            enable_tensorboard: 是否启用TensorBoard
            enable_file_logging: 是否启用文件日志
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir or Config.MODEL_DIR / "runs" / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_tensorboard = enable_tensorboard and HAS_TENSORBOARD
        self.enable_file_logging = enable_file_logging
        
        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None
        if self.enable_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoard writer initialized: {self.log_dir}")
        
        # 指标追踪器
        self.tracker = MetricsTracker()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.start_time = None
        self.config: Dict = {}
        
        # 回调函数
        self.callbacks: List[callable] = []
        
        logger.info(f"Training monitor initialized: {self.experiment_name}")
    
    def start_training(self, config: Dict = None):
        """开始训练"""
        self.start_time = datetime.now()
        self.config = config or {}
        
        # 记录配置
        if self.enable_tensorboard and self.writer:
            config_str = json.dumps(self.config, indent=2, ensure_ascii=False)
            self.writer.add_text("config", f"```json\n{config_str}\n```", 0)
        
        if self.enable_file_logging:
            config_file = self.log_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training started: {self.experiment_name}")
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """记录标量值"""
        step = step or self.global_step
        
        # 追踪指标
        self.tracker.add(tag, value, step)
        
        # TensorBoard
        if self.enable_tensorboard and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int = None):
        """记录多个标量值"""
        step = step or self.global_step
        
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            self.tracker.add(full_tag, value, step)
        
        if self.enable_tensorboard and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: Union[np.ndarray, 'torch.Tensor'], step: int = None):
        """记录直方图"""
        if not self.enable_tensorboard or not self.writer:
            return
        
        step = step or self.global_step
        
        if HAS_TORCH and isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor: 'torch.Tensor', step: int = None):
        """记录图像"""
        if not self.enable_tensorboard or not self.writer:
            return
        
        step = step or self.global_step
        self.writer.add_image(tag, img_tensor, step)
    
    def log_model_graph(self, model: 'torch.nn.Module', input_sample: 'torch.Tensor'):
        """记录模型结构图"""
        if not self.enable_tensorboard or not self.writer or not HAS_TORCH:
            return
        
        try:
            self.writer.add_graph(model, input_sample)
            logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def log_embedding(self, tag: str, embeddings: 'torch.Tensor', 
                      metadata: List[str] = None, step: int = None):
        """记录嵌入向量"""
        if not self.enable_tensorboard or not self.writer:
            return
        
        step = step or self.global_step
        self.writer.add_embedding(embeddings, metadata=metadata, tag=tag, global_step=step)
    
    def log_hyperparams(self, hparams: Dict, metrics: Dict):
        """记录超参数和对应指标"""
        if not self.enable_tensorboard or not self.writer:
            return
        
        # TensorBoard要求参数值为基本类型
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            else:
                clean_hparams[k] = str(v)
        
        self.writer.add_hparams(clean_hparams, metrics)
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """记录epoch级别的指标"""
        self.current_epoch = epoch
        
        # 训练指标
        for name, value in train_metrics.items():
            self.log_scalar(f"train/{name}", value, epoch)
        
        # 验证指标
        if val_metrics:
            for name, value in val_metrics.items():
                self.log_scalar(f"val/{name}", value, epoch)
        
        # 文件日志
        if self.enable_file_logging:
            log_entry = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'train': train_metrics,
                'val': val_metrics,
            }
            
            log_file = self.log_dir / "training_log.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_batch(self, batch_idx: int, loss: float, **kwargs):
        """记录batch级别的指标"""
        self.global_step += 1
        
        self.log_scalar("batch/loss", loss, self.global_step)
        
        for name, value in kwargs.items():
            self.log_scalar(f"batch/{name}", value, self.global_step)
    
    def log_learning_rate(self, lr: float, step: int = None):
        """记录学习率"""
        self.log_scalar("train/learning_rate", lr, step)
    
    def log_gradient_norm(self, grad_norm: float, step: int = None):
        """记录梯度范数"""
        self.log_scalar("train/grad_norm", grad_norm, step)
    
    def log_weight_distribution(self, model: 'torch.nn.Module', step: int = None):
        """记录权重分布"""
        if not HAS_TORCH or not self.enable_tensorboard or not self.writer:
            return
        
        step = step or self.global_step
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"weights/{name}", param.data, step)
                if param.grad is not None:
                    self.log_histogram(f"gradients/{name}", param.grad, step)
    
    def add_callback(self, callback: callable):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def trigger_callbacks(self, event: str, **kwargs):
        """触发回调"""
        for callback in self.callbacks:
            try:
                callback(event, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'experiment_name': self.experiment_name,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'elapsed_seconds': elapsed,
            'metrics_summary': self.tracker.get_summary(),
            'config': self.config,
        }
    
    def save_checkpoint_info(self, checkpoint_path: str, metrics: Dict):
        """保存检查点信息"""
        info = {
            'checkpoint_path': checkpoint_path,
            'epoch': self.current_epoch,
            'step': self.global_step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        info_file = self.log_dir / "checkpoints.jsonl"
        with open(info_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')
    
    def finish(self, final_metrics: Dict = None):
        """结束训练"""
        # 记录最终指标
        if final_metrics:
            if self.enable_file_logging:
                final_file = self.log_dir / "final_metrics.json"
                with open(final_file, 'w', encoding='utf-8') as f:
                    json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        
        # 保存完整历史
        if self.enable_file_logging:
            history_file = self.log_dir / "metrics_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracker.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()
        
        # 计算训练时间
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Training finished. Total time: {elapsed/3600:.2f} hours")
        
        logger.info(f"Training monitor closed: {self.experiment_name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class TrainingDashboard:
    """训练仪表盘（用于GUI显示）"""
    
    def __init__(self, monitor: TrainingMonitor = None):
        self.monitor = monitor
    
    def get_realtime_data(self) -> Dict:
        """获取实时数据"""
        if not self.monitor:
            return {}
        
        return {
            'current_epoch': self.monitor.current_epoch,
            'global_step': self.monitor.global_step,
            'latest_metrics': {
                name: self.monitor.tracker.get_latest(name)
                for name in self.monitor.tracker.metrics.keys()
            },
            'best_metrics': {
                'train/loss': self.monitor.tracker.get_best('train/loss', 'min'),
                'val/ic': self.monitor.tracker.get_best('val/ic', 'max'),
            },
        }
    
    def get_chart_data(self, metric_name: str) -> Dict:
        """获取图表数据"""
        if not self.monitor:
            return {'x': [], 'y': []}
        
        return {
            'x': self.monitor.tracker.steps.get(metric_name, []),
            'y': self.monitor.tracker.metrics.get(metric_name, []),
        }


def launch_tensorboard(log_dir: Path = None, port: int = 6006):
    """启动TensorBoard"""
    log_dir = log_dir or Config.MODEL_DIR / "runs"
    
    import subprocess
    import webbrowser
    
    cmd = f"tensorboard --logdir={log_dir} --port={port}"
    logger.info(f"Starting TensorBoard: {cmd}")
    
    try:
        # 在后台启动TensorBoard
        subprocess.Popen(cmd, shell=True)
        
        # 打开浏览器
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        logger.info(f"TensorBoard launched at {url}")
        
    except Exception as e:
        logger.error(f"Failed to launch TensorBoard: {e}")


if __name__ == "__main__":
    # 测试代码
    monitor = TrainingMonitor(experiment_name="test_experiment")
    monitor.start_training({'learning_rate': 0.001, 'batch_size': 32})
    
    # 模拟训练过程
    for epoch in range(5):
        train_loss = 1.0 / (epoch + 1)
        val_ic = 0.02 + 0.01 * epoch
        
        monitor.log_epoch(
            epoch, 
            train_metrics={'loss': train_loss, 'ic': val_ic * 0.8},
            val_metrics={'loss': train_loss * 1.1, 'ic': val_ic}
        )
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_ic={val_ic:.4f}")
    
    # 获取摘要
    summary = monitor.get_summary()
    print("\nTraining Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    monitor.finish({'final_ic': 0.06, 'final_sharpe': 1.5})
    
    print(f"\nLogs saved to: {monitor.log_dir}")
    print("Run 'tensorboard --logdir=<log_dir>' to visualize")
