# -*- coding: utf-8 -*-
"""
SOTA 训练器
PRD 3.2 实现

特性:
- AdamW 优化器
- ReduceLROnPlateau 调度器
- 熔断机制
- 动态 IC 监控
- Checkpoint 保存
"""

import time
import os
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

# 使用新的 torch.amp API (避免弃用警告)
from torch.amp import autocast, GradScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from utils.seeder import seed_everything, get_device, clear_gpu_memory
from models.finmamba import FinMamba, AlphaModel
from models.losses import CombinedLoss, calculate_rank_ic

logger = get_logger("Trainer")

# PyTorch 性能优化
def _optimize_pytorch():
    """优化 PyTorch 设置"""
    if torch.cuda.is_available():
        # 启用 cudnn benchmark（自动寻找最优算法）
        cudnn.benchmark = True
        logger.info("✓ 启用 cudnn.benchmark")
        
        # 启用 TF32（TensorFloat-32）加速
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ 启用 TF32 加速")
        
        # 设置 CUDA 内存分配策略（使用新的环境变量名）
        os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512'
        logger.info("✓ CUDA 内存分配策略: max_split_size_mb=512")
        
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()
        logger.info("✓ 清理 CUDA 缓存")
    
    # 设置 PyTorch 线程数
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    torch.set_num_threads(cpu_count)
    logger.info(f"✓ PyTorch 线程数: {cpu_count}")

# 初始化优化
_optimize_pytorch()


class StopTrainingError(Exception):
    """训练熔断异常"""
    pass


class Trainer:
    """
    SOTA 训练器
    
    特性:
    - IC 监控与早停
    - 动态学习率调整
    - 混合精度训练
    - Checkpoint 管理
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        lr: float = None,
        weight_decay: float = None,
        max_epochs: int = None,
        patience: int = None,
        use_amp: bool = True,
        device: torch.device = None
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            lr: 学习率
            weight_decay: 权重衰减
            max_epochs: 最大训练轮数
            patience: 早停耐心值
            use_amp: 是否使用混合精度
            device: 计算设备
        """
        # 设置随机种子
        seed_everything(Config.SEED)
        
        # 设备
        self.device = device or get_device()
        
        # 启用cudnn benchmark加速（针对固定输入大小优化卷积/矩阵运算）
        torch.backends.cudnn.benchmark = True
        
        # 模型
        if model is None:
            from models.finmamba import AlphaModel
            self.model = AlphaModel(
                feature_dim=Config.FEATURE_DIM,
                d_model=Config.D_MODEL,
                n_layers=Config.N_LAYERS,
                n_transformer_layers=Config.N_TRANSFORMER_LAYERS,
                n_heads=Config.N_HEADS,
                d_state=Config.D_STATE,
                levels=Config.MAMBA_LEVELS,
                n_industries=Config.N_INDUSTRIES,
                use_industry=Config.USE_GRAPH,
                dropout=Config.DROPOUT
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # 启用 torch.compile (A800 建议开启)
        if getattr(Config, 'ENABLE_COMPILE', False) and torch.__version__ >= '2.0.0' and self.device.type == 'cuda':
            try:
                logger.info("✓ 启用 torch.compile 模型加速")
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                logger.warning(f"torch.compile 启用失败: {e}，将使用普通模式")
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 超参数
        self.lr = lr or Config.LR_INIT
        self.weight_decay = weight_decay or Config.WEIGHT_DECAY
        self.max_epochs = max_epochs or Config.MAX_EPOCHS
        self.patience = patience or Config.PATIENCE
        self.use_amp = getattr(Config, 'USE_AMP', True) and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if getattr(Config, 'USE_BF16', False) else torch.float16
        
        # 优化器: AdamW (启用 fused=True 以在 A800 上获得最佳性能)
        optimizer_kwargs = {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
        # 检查是否支持 fused 参数 (PyTorch 2.0+)
        import inspect
        if 'fused' in inspect.signature(AdamW).parameters:
            optimizer_kwargs['fused'] = True
            logger.info("✓ 启用 Fused AdamW 优化器")
            
        self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        
        # 调度器: CosineAnnealingLR + 学习率预热
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
        from torch.optim.lr_scheduler import SequentialLR
        
        # 学习率预热阶段
        warmup_epochs = 3
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # 余弦退火阶段
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs - warmup_epochs,
            eta_min=5e-7
        )
        
        # 组合调度器
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        # 上一次的学习率（用于检测学习率变化）
        self._last_lr = self.lr
        
        # 损失函数 - 重新平衡权重，更关注排序和IC
        self.criterion = CombinedLoss(
            mse_weight=0.3,    # 降低MSE权重，减少对绝对值的过度关注
            rank_weight=1.0,   # 提高排序权重，优化相对排名
            ic_weight=0.7      # 提高IC权重，增强预测与真实值的相关性
        )
        
        # 混合精度 - 使用新的 API，并启用优化器步长融合
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        
        # A10 GPU优化：启用梯度累积步长融合
        self.grad_accum_steps = Config.GRAD_ACCUM_STEPS if hasattr(Config, 'GRAD_ACCUM_STEPS') else 1
        
        # A10 GPU优化：设置合适的内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存（符合用户要求）
        torch.cuda.empty_cache()  # 清理初始缓存
        
        # 训练状态
        self.best_ic = -float('inf')
        self.best_epoch = 0
        self.bad_epoch_count = 0
        self.current_epoch = 0
        
        # 历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_ic': [],
            'lr': []
        }
        
        # 回调函数
        self.callbacks = {
            'on_epoch_end': None,
            'on_batch_end': None,
            'on_train_end': None
        }
        
        # 停止标志
        self._stop_flag = False
        
        logger.info("Trainer initialized")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}")
    
    def set_callback(self, name: str, callback: Callable):
        """设置回调函数"""
        if name in self.callbacks:
            self.callbacks[name] = callback
    
    def train_epoch(self) -> Dict:
        """
        训练一个 epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        total_batches = len(self.train_loader)
        
        # 打印 epoch 开始信息
        print(f"[Train Epoch {self.current_epoch+1}] Total batches: {total_batches}", flush=True)
        
        import time
        epoch_batch_start = time.time()
        first_batch_wait_start = epoch_batch_start
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            if self._stop_flag:
                break

            if batch_idx == 0 and getattr(Config, 'LOG_FIRST_BATCH_TIMING', False):
                logger.info(f"First batch fetched in {time.time() - first_batch_wait_start:.1f}s")
            
            # 解包批次数据 (支持新旧格式)
            if len(batch_data) == 4:
                X, y, industry_ids, info = batch_data
                industry_ids = industry_ids.to(self.device, non_blocking=True)
            else:
                X, y, info = batch_data
                industry_ids = None
            
            # 确保输入数据类型正确，转换为float32
            X = X.float().to(self.device, non_blocking=True)  # 显式转换为float32
            y = y.float().to(self.device, non_blocking=True)  # 显式转换为float32
            
            # 梯度累积：只在第一步或累积步结束时清零梯度
            if batch_idx % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更高效
            
            # A10 GPU优化：使用统一的autocast上下文
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                first_forward_start = time.time() if (batch_idx == 0 and getattr(Config, 'LOG_FIRST_BATCH_TIMING', False)) else None
                pred = self.model(X, industry_ids)  # 传入行业ID
                loss, loss_dict = self.criterion(pred, y)
                if first_forward_start is not None:
                    logger.info(f"First forward+loss took {time.time() - first_forward_start:.1f}s")
                
                # 梯度累积：缩放损失
                loss = loss / self.grad_accum_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 梯度累积：只在累积步结束时进行梯度裁剪和优化器步骤
                if (batch_idx + 1) % self.grad_accum_steps == 0 or batch_idx == total_batches - 1:
                    # 启用梯度裁剪以减少过拟合和提高稳定性
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # 添加梯度裁剪
                    
                    # A10 GPU优化：启用优化器步长融合
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                
                # 梯度累积：只在累积步结束时进行梯度裁剪和优化器步骤
                if (batch_idx + 1) % self.grad_accum_steps == 0 or batch_idx == total_batches - 1:
                    # 启用梯度裁剪以减少过拟合
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    # A10 GPU优化：使用torch.no_grad()包裹权重更新
                    with torch.no_grad():
                        self.optimizer.step()
            
            total_loss += loss_dict['total']
            n_batches += 1
            
            # 每 100 个 batch 显示一次进度 (减少频率以提升速度)
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0 or batch_idx == total_batches - 1:
                progress_pct = 100.0 * (batch_idx + 1) / total_batches
                avg_loss_so_far = total_loss / n_batches
                
                # 计算速度
                elapsed = time.time() - epoch_batch_start
                samples_processed = (batch_idx + 1) * self.train_loader.batch_size
                speed = samples_processed / elapsed if elapsed > 0 else 0
                
                # 估计剩余时间
                remaining_batches = total_batches - batch_idx - 1
                eta_seconds = (remaining_batches / (batch_idx + 1)) * elapsed if batch_idx > 0 else 0
                
                msg = (
                    f"  Batch {batch_idx+1}/{total_batches} ({progress_pct:.1f}%) | "
                    f"Loss: {avg_loss_so_far:.4f} | "
                    f"Speed: {speed:.0f} samples/s | "
                    f"ETA: {eta_seconds:.0f}s"
                )
                print(msg, flush=True)
                # 减少日志输出频率（只在每 200 个 batch 记录日志）
                if (batch_idx + 1) % 200 == 0:
                    logger.info(msg)
            
            # Batch 回调
            if self.callbacks['on_batch_end']:
                self.callbacks['on_batch_end'](batch_idx, total_batches, loss_dict)
        
        # 计算平均值
        avg_loss = total_loss / max(n_batches, 1)
        
        return {
            'loss': avg_loss
        }
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """
        验证
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_dates = []  # 收集样本日期信息
        n_batches = 0
        
        # A10 GPU优化：验证时禁用梯度计算
        torch.autograd.set_grad_enabled(False)
        
        for batch_data in self.val_loader:
            # 解包批次数据 (支持新旧格式)
            if len(batch_data) == 4:
                X, y, industry_ids, info = batch_data
                industry_ids = industry_ids.to(self.device, non_blocking=True)
            else:
                X, y, info = batch_data
                industry_ids = None
            
            # 确保输入数据类型正确，转换为float32
            X = X.float().to(self.device, non_blocking=True)  # 显式转换为float32
            y = y.float().to(self.device, non_blocking=True)  # 显式转换为float32
            
            # A10 GPU优化：验证时也使用混合精度
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                pred = self.model(X, industry_ids)  # 传入行业ID
                loss, loss_dict = self.criterion(pred, y)
            
            total_loss += loss_dict['total']
            
            # A10 GPU优化：在GPU上累积预测和目标，减少CPU-GPU数据传输
            all_preds.append(pred)
            all_targets.append(y)
            n_batches += 1
            
            # 收集日期信息
            # 修复：处理info可能是张量的情况
            if info is not None and not isinstance(info, torch.Tensor):
                if isinstance(info, dict):
                    # 使用默认collate_fn时，info是字典，键为特征名，值为批次列表
                    dates = info.get('date', [])
                    all_dates.extend(dates)
                elif isinstance(info, list):
                    # 使用自定义collate_fn时，info是列表
                    for sample_info in info:
                        if isinstance(sample_info, dict):
                            all_dates.append(sample_info.get('date', ''))
                        else:
                            all_dates.append('')
                # 忽略其他类型的info，如张量
        
        # 计算 IC 和 Accuracy
        # 修复：避免修改只读张量，先将张量转换为numpy数组再处理
        all_preds_np = []
        all_targets_np = []
        
        # 逐批次处理，避免直接操作大张量
        if len(all_preds) > 0:
            for i in range(len(all_preds)):
                # 转换为float32并转移到CPU
                batch_preds = all_preds[i].float().detach().cpu().numpy()
                batch_targets = all_targets[i].float().detach().cpu().numpy()
                
                # 多期限输出处理
                if batch_preds.ndim == 2:
                    if batch_preds.shape[1] > 1:
                        batch_preds = batch_preds[:, 0]
                    else:
                        batch_preds = batch_preds.squeeze()
                
                batch_targets = batch_targets.flatten()
                batch_preds = batch_preds.flatten()
                
                if batch_preds.size > 0:
                    all_preds_np.append(batch_preds)
                    all_targets_np.append(batch_targets)
            
            if all_preds_np:
                all_preds_np = np.concatenate(all_preds_np)
                all_targets_np = np.concatenate(all_targets_np)
            else:
                all_preds_np = np.array([])
                all_targets_np = np.array([])
        
        # 计算全局 IC 和 Accuracy
        from models.losses import calculate_accuracy
        val_ic = calculate_rank_ic(torch.tensor(all_preds_np), torch.tensor(all_targets_np))
        val_acc = calculate_accuracy(torch.tensor(all_preds_np), torch.tensor(all_targets_np))
        
        avg_loss = total_loss / max(n_batches, 1)
        
        return {
            'loss': avg_loss,
            'ic': val_ic,
            'accuracy': val_acc,
            'samples': len(all_preds_np)
        }
    
    def train(self) -> Dict:
        """
        完整训练流程
        
        Returns:
            训练历史记录
        """
        logger.info(f"Starting training: {self.max_epochs} epochs")
        start_time = time.time()
        
        # 用于计算训练速度
        epoch_times = []

        if getattr(Config, 'COMPILE_WARMUP', False) and getattr(Config, 'ENABLE_COMPILE', False) and self.device.type == 'cuda':
            try:
                import time as _time
                warmup_start = _time.time()
                batch_data = next(iter(self.train_loader))
                if len(batch_data) == 4:
                    X, y, industry_ids, _ = batch_data
                    industry_ids = industry_ids.to(self.device, non_blocking=True)
                else:
                    X, y, _ = batch_data
                    industry_ids = None

                X = X.float().to(self.device, non_blocking=True)
                y = y.float().to(self.device, non_blocking=True)

                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    pred = self.model(X, industry_ids)
                    loss, _ = self.criterion(pred, y)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.zero_grad(set_to_none=True)

                logger.info(f"Compile warmup finished in {_time.time() - warmup_start:.1f}s")
            except Exception as e:
                logger.warning(f"Compile warmup failed: {e}")
        
        try:
            for epoch in range(self.max_epochs):
                if self._stop_flag:
                    logger.info("Training stopped by user")
                    break
                
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # 训练
                train_metrics = self.train_epoch()
                
                # 验证
                val_metrics = self.validate()
                
                # 更新学习率
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 检测学习率是否发生变化
                if current_lr != self._last_lr:
                    logger.info(f"Learning rate changed: {self._last_lr:.2e} -> {current_lr:.2e}")
                    self._last_lr = current_lr
                
                # 记录历史
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_ic'].append(val_metrics['ic'])
                self.history['lr'].append(current_lr)
                
                # 计算时间统计
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                # 计算训练速度 (samples/sec)
                train_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else len(self.train_loader) * 64
                samples_per_sec = train_samples / epoch_time if epoch_time > 0 else 0
                
                # 计算已用时间和剩余时间
                elapsed_time = time.time() - start_time
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = self.max_epochs - (epoch + 1)
                eta_seconds = avg_epoch_time * remaining_epochs
                
                # 格式化时间
                elapsed_str = self._format_time(elapsed_time)
                eta_str = self._format_time(eta_seconds)
                
                # 日志
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Val IC: {val_metrics['ic']:.4f} | "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | "
                    f"Speed: {samples_per_sec:.0f} samples/s | "
                    f"Time: {epoch_time:.1f}s | "
                    f"Elapsed: {elapsed_str} | "
                    f"ETA: {eta_str}"
                )
                
                # 检查是否为最佳模型
                if val_metrics['ic'] > self.best_ic:
                    if val_metrics['ic'] > Config.SOTA_MIN_IC_SAVE:
                        self.best_ic = val_metrics['ic']
                        self.best_epoch = epoch
                        self.save_checkpoint('best_model.pth')
                        logger.info(f"New best model saved! IC: {self.best_ic:.4f}")
                    self.bad_epoch_count = 0
                else:
                    self.bad_epoch_count += 1
                
                # 检查熔断条件
                if val_metrics['ic'] < Config.SOTA_MIN_IC:
                    self.bad_epoch_count += 1
                    logger.warning(f"Low IC warning: {val_metrics['ic']:.4f} < {Config.SOTA_MIN_IC}")
                
                if self.bad_epoch_count >= Config.BAD_EPOCH_LIMIT:
                    raise StopTrainingError(
                        f"Training stopped: IC too low for {self.bad_epoch_count} epochs"
                    )
                
                # 早停检查 - 更严格的早停逻辑
                if self.bad_epoch_count >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}, bad_epoch_count: {self.bad_epoch_count}")
                    break
                
                # 额外的早停条件：如果当前IC比最佳IC低很多，提前终止
                if self.best_ic > 0.05 and (self.best_ic - val_metrics['ic']) > 0.03:
                    logger.warning(f"IC dropped significantly: {val_metrics['ic']:.4f} < best_ic - 0.03, stopping early")
                    break
                
                # Epoch 回调 - 传递额外的时间信息
                if self.callbacks['on_epoch_end']:
                    self.callbacks['on_epoch_end'](
                        epoch,
                        train_metrics,
                        val_metrics,
                        current_lr,
                        {
                            'epoch_time': epoch_time,
                            'samples_per_sec': samples_per_sec,
                            'elapsed': elapsed_time,
                            'elapsed_str': elapsed_str,
                            'eta': eta_seconds,
                            'eta_str': eta_str
                        }
                    )
        
        except StopTrainingError as e:
            logger.error(str(e))
        
        # 训练结束
        total_time = time.time() - start_time
        logger.info(f"Training completed in {self._format_time(total_time)}")
        logger.info(f"Best IC: {self.best_ic:.4f} at epoch {self.best_epoch+1}")
        
        # 训练结束回调
        if self.callbacks['on_train_end']:
            self.callbacks['on_train_end'](self.history)
        
        return self.history
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """保存检查点"""
        save_path = Config.MODEL_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_ic': self.best_ic,
            'history': self.history
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pth'):
        """加载检查点"""
        load_path = Config.MODEL_DIR / filename
        
        if not load_path.exists():
            logger.warning(f"Checkpoint not found: {load_path}")
            return
        
        checkpoint = torch.load(load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_ic = checkpoint['best_ic']
        self.best_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {load_path}")
