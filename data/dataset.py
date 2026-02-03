# -*- coding: utf-8 -*-
"""
高性能数据集 - 完全重写版
使用 numpy memmap 和预计算索引
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

import sys
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from data.preprocessor import Preprocessor

logger = get_logger("Dataset")


class FastAlphaDataset(Dataset):
    """
    高性能 Alpha 数据集 - 完全优化版
    
    关键优化:
    1. 预计算所有样本索引到 numpy 数组
    2. 使用连续内存存储
    3. 避免所有 pandas 操作
    4. 零拷贝数据访问
    """
    
    def __init__(
        self,
        mode: str = 'train',
        train_ratio: float = 0.5,
        val_ratio: float = 0.3,
        force_rebuild: bool = False,
        cache_dir: Path = None,
        date_range: Tuple[str, str] = None
    ):
        """
        初始化数据集
        
        Args:
            mode: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            force_rebuild: 是否强制重建缓存
            cache_dir: 自定义缓存目录（可选）
            date_range: 自定义日期范围 ('2020-01-01', '2022-01-01')，若提供了此参数， train_ratio/val_ratio 逻辑会被覆盖用于该 mode
        """
        self.mode = mode
        self.date_range = date_range
        self.seq_len = Config.SEQ_LEN
        self.feature_cols = Preprocessor.FEATURES
        self.n_features = len(self.feature_cols)
        # 添加数据增强标志
        self.use_data_augmentation = mode == 'train'
        
        # 调试：打印特征列信息
        logger.info(f"Feature columns ({len(self.feature_cols)}): {self.feature_cols}")
        logger.info(f"Expected feature dim: {Config.FEATURE_DIM}")
        logger.info(f"Actual feature dim: {self.n_features}")
        
        # 缓存文件路径
        if cache_dir is None:
            cache_dir = Config.DATA_ROOT / 'fast_cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f'{mode}_cache.pkl'  # 保留用于兼容
        
        # 检查 numpy 缓存文件是否存在
        features_file = self.cache_dir / f'{mode}_features.npy'
        labels_file = self.cache_dir / f'{mode}_labels.npy'
        industry_file = self.cache_dir / f'{mode}_industry.npy'  # 行业代码文件
        samples_file = self.cache_dir / f'{mode}_samples.pkl'
        cache_exists = features_file.exists() and labels_file.exists() and samples_file.exists()
        
        # 尝试加载缓存
        if not force_rebuild and cache_exists:
            logger.info(f"Loading cached dataset for {mode}...")
            self._load_cache()
        else:
            logger.info(f"Building fast dataset for {mode}...")
            self._build_dataset(train_ratio, val_ratio)
            self._save_cache()
        
        logger.info(f"FastAlphaDataset [{mode}]: {len(self.samples)} samples")
    
    @staticmethod
    def get_stock_industry_map() -> dict:
        """Load stock code to industry ID mapping from stock_basic.parquet.

        Returns:
            dict: Mapping from ts_code (str) to industry ID (int). Missing or unknown industries map to 0.
        """
        stock_basic_path = Config.DATA_ROOT / 'stock_basic.parquet'
        if not stock_basic_path.exists():
            logger.warning(f"stock_basic.parquet not found at {stock_basic_path}, returning empty industry map")
            return {}
        try:
            df = pd.read_parquet(stock_basic_path, columns=['ts_code', 'industry'])
        except Exception as e:
            logger.error(f"Failed to read stock_basic.parquet: {e}")
            return {}
        unique_inds = df['industry'].dropna().unique()
        ind_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted(unique_inds))}
        industry_map = {}
        for _, row in df.iterrows():
            ts_code = row['ts_code']
            ind_name = row['industry']
            if pd.isna(ind_name):
                industry_map[ts_code] = 0
            else:
                # 限制行业ID不超过N_INDUSTRIES-1
                industry_id = ind_name_to_id.get(ind_name, 0)
                # 确保行业ID在模型支持的范围内
                industry_id = min(industry_id, Config.N_INDUSTRIES - 1)
                industry_map[ts_code] = industry_id
        return industry_map
    
    def _build_dataset(self, train_ratio: float, val_ratio: float):
        """构建数据集 - 使用分块读取避免内存溢出"""
        import gc
        from datetime import datetime, timedelta
        
        # 加载原始数据
        data_path = Config.PROCESSED_DATA_DIR / "all_features.parquet"
        if not data_path.exists():
            logger.error("Processed data not found")
            
            # 尝试自动处理数据
            try:
                logger.info("尝试自动处理数据...")
                
                # 1. 检查原始数据是否存在
                raw_data_dir = Config.DATA_ROOT / "raw"
                if not any(raw_data_dir.iterdir()):
                    logger.info("原始数据不存在，尝试自动下载...")
                    # 自动下载数据
                    from data.downloader import TushareDownloader
                    downloader = TushareDownloader()
                    downloader.download_all(force_update=False)
                    downloader.download_index_data()
                
                # 2. 处理原始数据
                logger.info("处理原始数据...")
                from data.preprocessor import Preprocessor
                preprocessor = Preprocessor()
                preprocessor.process_all_data()
                
                # 3. 重新检查处理后的数据
                if data_path.exists():
                    logger.info("数据处理成功，继续创建数据集...")
                else:
                    raise FileNotFoundError("数据处理后仍然不存在处理过的数据")
                
            except Exception as e:
                logger.error(f"自动数据处理失败: {e}")
                # 创建一个最小化的非空数据集，避免数据加载器创建失败
                logger.warning("Creating fallback dataset with dummy data...")
                
                # 创建虚拟数据
                dummy_seq_len = self.seq_len
                dummy_features = np.zeros((1, dummy_seq_len, self.n_features), dtype=np.float32)
                dummy_labels = np.zeros(1, dtype=np.float32)
                dummy_industry = np.zeros(1, dtype=np.int64)
                dummy_sample = [('dummy', 0, '2020-01-01')]
                
                self.features = dummy_features
                self.labels = dummy_labels
                self.industry_ids = dummy_industry
                self.samples = dummy_sample
                
                logger.info(f"Fallback: Created 1 dummy sample to avoid empty dataset")
                return
        
        # 先读取日期列，进行时间划分
        logger.info("Reading date information...")
        df_dates = pd.read_parquet(data_path, columns=['trade_date'])
        df_dates['trade_date'] = pd.to_datetime(df_dates['trade_date'])
        
        all_dates = sorted(df_dates['trade_date'].unique())
        
        # 优先使用外部传入的日期范围
        if self.date_range is not None:
            start_d, end_d = pd.to_datetime(self.date_range[0]), pd.to_datetime(self.date_range[1])
            date_range = [d for d in all_dates if start_d <= d <= end_d]
            logger.info(f"Using custom date range for {self.mode}: {start_d} to {end_d}")
            logger.info(f"  Found {len(date_range)} dates in range")
        else:
            # ============ 默认日期划分：最近两年到前半年数据，70%训练集，30%验证集 ============
            # 获取最新日期
            latest_date = all_dates[-1]
            
            # 计算前半年的结束日期（即验证集的结束日期）
            val_end_date = latest_date - pd.Timedelta(days=180)  # 前半年
            
            # 计算两年前的开始日期
            train_start_date = val_end_date - pd.Timedelta(days=730)  # 最近两年
            
            # 提取时间范围内的所有日期
            date_range = [d for d in all_dates if train_start_date <= d <= val_end_date]
            
            # 显示完整的划分信息 (仅在 train 模式显示一次)
            if self.mode == 'train':
                logger.info(f"--- Data Split Plan ---")
                logger.info(f"  Total dates in range: {len(date_range)}")
                logger.info(f"  Train start: {train_start_date.strftime('%Y-%m-%d')}")
                logger.info(f"  Val end: {val_end_date.strftime('%Y-%m-%d')}")
                logger.info(f"  Latest date: {latest_date.strftime('%Y-%m-%d')}")
            
            # 计算训练集和验证集的划分点（70%/30%）
            split_point = int(len(date_range) * 0.7)
            
            # 设置训练集和验证集的日期范围
            if self.mode == 'train':
                date_range = date_range[:split_point]
            elif self.mode == 'val':
                date_range = date_range[split_point:]
            else:  # test
                # 测试集使用前半年的数据
                test_dates = [d for d in all_dates if d > val_end_date]
                date_range = test_dates
        
        # 确保日期范围有效
        if len(date_range) == 0:
            # 调整日期范围，使用所有可用日期的最后10%作为当前模式的数据
            logger.warning(f"No dates found for {self.mode} mode with original range! Using fallback strategy.")
            if len(all_dates) > 0:
                # 使用所有可用日期的最后10%作为当前模式的数据
                fallback_split = int(len(all_dates) * 0.9)
                if self.mode == 'train':
                    date_range = all_dates[:fallback_split]
                else:
                    date_range = all_dates[fallback_split:]
                logger.info(f"  Fallback: Using {len(date_range)} dates for {self.mode} mode")
            else:
                logger.error(f"No dates available at all! Please check your data.")
                # 至少返回一个空数据集，避免崩溃
                self.features = np.array([], dtype=np.float32)
                self.labels = np.array([], dtype=np.float32)
                self.industry_ids = np.array([], dtype=np.int64)
                self.samples = []
                return
        
        # 清晰的日志输出
        logger.info(f"=== {self.mode.upper()} Dataset ===")
        logger.info(f"  Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')}")
        logger.info(f"  Total trading days: {len(date_range)}")
        
        # 分块读取数据 - 充分利用 96GB 内存，显著增大分块大小
        chunk_size = 50000  # 大幅增加分块大小 (10000 -> 50000)
        features_list = []
        labels_list = []
        industry_list = []
        samples = []
        
        # 内存优化：针对 96GB 内存减少清理频率
        MEMORY_CLEAR_INTERVAL = 50  # 增加清理间隔 (20 -> 50)
        
        # 行业代码映射 (股票代码 -> 行业ID)
        industry_map = self.get_stock_industry_map()
        
        # 获取所有股票代码 (如果map中没有，使用默认值)
        all_ts_codes = pd.read_parquet(data_path, columns=['ts_code'])['ts_code'].unique()
        n_stocks = len(all_ts_codes)
        
        # 补全缺失的映射 (默认行业ID=0)
        for ts_code in all_ts_codes:
            if ts_code not in industry_map:
                industry_map[ts_code] = 0
                
        logger.info(f"Industry map ready for {len(industry_map)} stocks")
        
        logger.info(f"Processing {n_stocks} stocks in chunks...")
        
        # 滑动窗口参数 - 从配置中读取步长，平衡样本量和训练速度
        use_sliding_window = True
        slide_step = Config.SLIDE_STEP  # 从配置中读取滑动窗口步长
        
        logger.info(f"Using sliding window: {use_sliding_window}, slide_step: {slide_step}")
        
        # 调试：打印日期范围
        logger.info(f"Date range: {date_range[0]} to {date_range[-1]}")
        logger.info(f"Number of stocks: {n_stocks}")
        logger.info(f"Chunk size: {chunk_size}")
        
        for batch_idx in range(0, n_stocks, chunk_size):
            ts_codes_batch = all_ts_codes[batch_idx:batch_idx + chunk_size]
            
            # 调试：打印批次信息
            logger.info(f"Processing batch {batch_idx//chunk_size + 1}/{(n_stocks + chunk_size - 1)//chunk_size}")
            logger.info(f"Stocks in batch: {len(ts_codes_batch)}")
            
            # 读取该批次的数据
            df_batch = pd.read_parquet(
                data_path,
                filters=[('ts_code', 'in', ts_codes_batch)]
            )
            
            # 调试：打印原始trade_date格式
            logger.debug(f"Original trade_date type: {type(df_batch['trade_date'].iloc[0]) if len(df_batch) > 0 else 'N/A'}")
            logger.debug(f"Original trade_date sample: {df_batch['trade_date'].iloc[0] if len(df_batch) > 0 else 'N/A'}")
            
            # 确保trade_date是datetime类型
            if len(df_batch) > 0:
                if isinstance(df_batch['trade_date'].iloc[0], str):
                    df_batch['trade_date'] = pd.to_datetime(df_batch['trade_date'])
                    logger.debug("Converted trade_date to datetime")
            
            # 过滤日期
            df_batch = df_batch[df_batch['trade_date'].isin(date_range)].copy()
            
            logger.info(f"Batch data shape after date filter: {df_batch.shape}")
            
            if len(df_batch) == 0:
                logger.warning(f"No data for batch {batch_idx//chunk_size + 1}")
                continue
            
            # 计算超额收益标签 (股票收益 - 市场平均收益)
            # 使用 ret_5d 优化一周持仓
            if 'ret_5d' not in df_batch.columns:
                if 'close' in df_batch.columns:
                    df_batch['ret_5d'] = df_batch.groupby('ts_code')['close'].pct_change(5) * 100
                else:
                    df_batch['ret_5d'] = 0
            
            # 计算 5 日后的未来收益
            df_batch['ret_5d_future'] = df_batch.groupby('ts_code')['ret_5d'].shift(-5)
            
            # 计算每日市场平均收益
            market_mean = df_batch.groupby('trade_date')['ret_5d_future'].transform('mean')
            
            # 超额收益 = 个股收益 - 市场平均收益
            df_batch['excess_return'] = df_batch['ret_5d_future'] - market_mean
            
            # 归一化到 [-1, 1] 范围 (除以标准差)
            excess_std = df_batch.groupby('trade_date')['excess_return'].transform('std')
            df_batch['ret_rank'] = df_batch['excess_return'] / (excess_std + 1e-8)
            df_batch['ret_rank'] = df_batch['ret_rank'].clip(-3, 3) / 3  # 裁剪并缩放到 [-1, 1]
            df_batch['ret_rank'] = (df_batch['ret_rank'] + 1) / 2  # 转换到 [0, 1]
            df_batch['ret_rank'] = df_batch['ret_rank'].fillna(0.5)
            
            # ========== 板块情绪特征计算 ==========
            # 需要 industry 字段，如果没有则使用默认值
            if 'industry' not in df_batch.columns:
                logger.warning("No 'industry' column found, using default sector features")
                df_batch['sector_up_ratio'] = 0.5
                df_batch['sector_avg_ret'] = 0.0
                df_batch['sector_momentum'] = 0.5
                df_batch['sector_volatility'] = 1.0
                df_batch['sector_leader'] = 0.0
            else:
                # 按日期和行业分组计算板块情绪
                group_key = ['trade_date', 'industry']
                
                # 1. 板块上涨家数比 (当日该板块上涨股票比例)
                if 'pct_chg' in df_batch.columns:
                    df_batch['sector_up_ratio'] = df_batch.groupby(group_key)['pct_chg'].transform(
                        lambda x: (x > 0).mean()
                    )
                else:
                    df_batch['sector_up_ratio'] = 0.5
                
                # 2. 板块平均涨幅
                if 'pct_chg' in df_batch.columns:
                    df_batch['sector_avg_ret'] = df_batch.groupby(group_key)['pct_chg'].transform('mean')
                else:
                    df_batch['sector_avg_ret'] = 0.0
                
                # 3. 板块动量 (板块在全市场的涨幅排名)
                if 'pct_chg' in df_batch.columns:
                    sector_daily_ret = df_batch.groupby(group_key)['pct_chg'].mean().reset_index()
                    sector_daily_ret['sector_momentum'] = sector_daily_ret.groupby('trade_date')['pct_chg'].rank(pct=True)
                    sector_daily_ret = sector_daily_ret.rename(columns={'pct_chg': '_sector_pct'})
                    df_batch = df_batch.merge(
                        sector_daily_ret[['trade_date', 'industry', 'sector_momentum']], 
                        on=['trade_date', 'industry'], 
                        how='left'
                    )
                    df_batch['sector_momentum'] = df_batch['sector_momentum'].fillna(0.5)
                else:
                    df_batch['sector_momentum'] = 0.5
                
                # 4. 板块波动率 (板块内股票涨跌幅标准差)
                if 'pct_chg' in df_batch.columns:
                    df_batch['sector_volatility'] = df_batch.groupby(group_key)['pct_chg'].transform('std')
                    df_batch['sector_volatility'] = df_batch['sector_volatility'].fillna(1.0)
                else:
                    df_batch['sector_volatility'] = 1.0
                
                # 5. 是否板块龙头 (当日涨幅在板块前 10%)
                if 'pct_chg' in df_batch.columns:
                    df_batch['sector_rank'] = df_batch.groupby(group_key)['pct_chg'].rank(pct=True)
                    df_batch['sector_leader'] = (df_batch['sector_rank'] >= 0.9).astype(float)
                    df_batch.drop(columns=['sector_rank'], inplace=True, errors='ignore')
                else:
                    df_batch['sector_leader'] = 0.0
            
            # 填充 NaN
            for col in ['sector_up_ratio', 'sector_avg_ret', 'sector_momentum', 'sector_volatility', 'sector_leader']:
                df_batch[col] = df_batch[col].fillna(0.0 if 'ret' in col else 0.5)
            
            logger.info(f"Added sector sentiment features")
            # ========== 板块情绪特征计算结束 ==========
            
            # 按股票分组
            stock_groups = df_batch.groupby('ts_code')
            logger.info(f"Number of stocks in batch with data: {len(stock_groups)}")
            
            for ts_code, group in stock_groups:
                group = group.sort_values('trade_date').reset_index(drop=True)
                n_rows = len(group)
                
                logger.debug(f"Processing stock {ts_code}, rows: {n_rows}, seq_len: {self.seq_len}")
                
                if n_rows < self.seq_len:
                    logger.debug(f"Skipping {ts_code}: not enough data ({n_rows} < {self.seq_len})")
                    continue
                
                # 修复：检查并添加缺失的特征列
                for feat in self.feature_cols:
                    if feat not in group.columns:
                        logger.debug(f"Adding missing feature: {feat}")
                        group[feat] = 0.0
                    # 确保没有NaN值
                    group[feat] = group[feat].fillna(0.0)
                
                # 提取特征矩阵
                feature_matrix = group[self.feature_cols].values.astype(np.float32)
                labels = group['ret_rank'].values.astype(np.float32)
                
                logger.debug(f"Feature matrix shape: {feature_matrix.shape}")
                logger.debug(f"Labels shape: {labels.shape}")
                
                # 创建样本索引 - 使用滑动窗口
                if use_sliding_window:
                    # 滑动窗口方式：增加训练数据量，提高泛化能力
                    logger.debug(f"Sliding window range: start={self.seq_len - 1}, end={n_rows - 1}, step={slide_step}")
                    
                    for i in range(self.seq_len - 1, n_rows - 1, slide_step):  # -1确保有未来收益
                        # 滑动窗口结束位置
                        end_idx = i + 1
                        # 窗口开始位置
                        start_idx = end_idx - self.seq_len
                        
                        # 确保窗口长度正确
                        if end_idx - start_idx != self.seq_len:
                            logger.debug(f"Skipping i={i}: window length incorrect")
                            continue
                        
                        # 提取序列
                        seq = feature_matrix[start_idx:end_idx]  # (seq_len, n_features)
                        # 使用未来一天的收益作为标签（i+1），避免数据泄漏
                        if i + 1 < n_rows:  # 确保有未来收益
                            label = labels[i + 1]
                        else:
                            logger.debug(f"Skipping i={i}: no future label")
                            continue  # 跳过最后一个样本，没有未来收益
                        
                        # 获取行业ID
                        ind_id = industry_map.get(ts_code, 0)
                        
                        # 获取当前样本对应的日期
                        current_date = group['trade_date'].iloc[i].strftime('%Y-%m-%d')
                        
                        features_list.append(seq)
                        labels_list.append(label)
                        industry_list.append(ind_id)
                        samples.append((ts_code, i, current_date))  # 添加日期信息
                        
                        # 调试：每生成10个样本打印一次
                        if len(samples) % 10 == 0:
                            logger.info(f"Generated {len(samples)} samples so far...")
                else:
                    # 传统方式：每个样本对应一个时间点
                    for i in range(self.seq_len - 1, n_rows - 1):  # -1确保有未来收益
                        start_idx = i - self.seq_len + 1
                        
                        # 提取序列
                        seq = feature_matrix[start_idx:i+1]  # (seq_len, n_features)
                        # 使用未来一天的收益作为标签（i+1），避免数据泄漏
                        if i + 1 < n_rows:  # 确保有未来收益
                            label = labels[i + 1]
                        else:
                            continue  # 跳过最后一个样本，没有未来收益
                        
                        # 获取行业ID
                        ind_id = industry_map.get(ts_code, 0)
                        
                        # 获取当前样本对应的日期
                        current_date = group['trade_date'].iloc[i].strftime('%Y-%m-%d')
                        
                        features_list.append(seq)
                        labels_list.append(label)
                        industry_list.append(ind_id)
                        samples.append((ts_code, i, current_date))  # 添加日期信息
            
            # 清理内存 - 立即清理每个批次的数据
            del df_batch
            gc.collect()
            
            # 每处理一定数量的块，更彻底地清理内存
            if (batch_idx + 1) % MEMORY_CLEAR_INTERVAL == 0:
                # 清理列表中的临时数据，只保留必要的信息
                logger.info(f"Processed {batch_idx + 1} batches, {len(samples)} samples so far...")
                logger.info("Performing deep memory cleanup...")
                # 强制释放未使用的内存
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 转换为连续数组 - 修复索引错误
        if len(features_list) > 0:
            self.samples = samples
            
            logger.info(f"Creating numpy arrays for {len(features_list)} samples...")
            
            # 使用更简单、更可靠的方式创建numpy数组
            # 直接将列表转换为numpy数组，避免分块填充的复杂性
            self.features = np.array(features_list, dtype=np.float32)  # (N, seq_len, n_features)
            self.labels = np.array(labels_list, dtype=np.float32)  # (N,)
            self.industry_ids = np.array(industry_list, dtype=np.int64)  # (N,) 行业ID
            
            # 清理所有临时列表，释放内存
            del features_list
            del labels_list
            del industry_list
            gc.collect()
        else:
            # 修复：如果没有生成任何样本，使用调试信息并返回非空数据集
            logger.warning(f"No samples generated for {self.mode} mode! Please check your data and parameters.")
            logger.warning(f"  Total dates: {len(date_range)}")
            logger.warning(f"  Total stocks: {n_stocks}")
            logger.warning(f"  Seq_len: {self.seq_len}")
            
            # 创建一个最小化的非空数据集，避免后续训练崩溃
            # 使用一个虚拟样本，避免空数组导致的错误
            dummy_feature = np.zeros((1, self.seq_len, self.n_features), dtype=np.float32)
            dummy_label = np.zeros(1, dtype=np.float32)
            dummy_industry = np.zeros(1, dtype=np.int64)
            dummy_sample = [('dummy', 0, '2020-01-01')]
            
            self.features = dummy_feature
            self.labels = dummy_label
            self.industry_ids = dummy_industry
            self.samples = dummy_sample
            
            logger.info(f"  Fallback: Created 1 dummy sample to avoid empty dataset")
        
        logger.info(f"Built {len(self.samples)} samples, features shape: {self.features.shape}")
    
    def _save_cache(self):
        """保存缓存 - 使用 numpy 格式避免内存问题"""
        try:
            # 保存为 numpy 文件（更高效）
            cache_dir = self.cache_file.parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Helper to safely save numpy file
            def safe_save(path, arr):
                if path.exists():
                    try:
                        path.unlink()  # Try to delete existing file first
                    except Exception:
                        pass  # If delete fails (e.g. locked), try saving anyway (might fail)
                np.save(path, arr)

            safe_save(cache_dir / f'{self.mode}_features.npy', self.features)
            safe_save(cache_dir / f'{self.mode}_labels.npy', self.labels)
            safe_save(cache_dir / f'{self.mode}_industry.npy', self.industry_ids)
            
            # 保存样本索引（小文件，用 pickle）
            with open(cache_dir / f'{self.mode}_samples.pkl', 'wb') as f:
                pickle.dump(self.samples, f)
            
            logger.info(f"Saved cache to {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """
        加载缓存 - 使用 numpy 格式，优化内存使用
        """
        try:
            cache_dir = self.cache_file.parent
            
            logger.info(f"Loading cache from {cache_dir}...")
            
            # 1. 首先加载samples，获取样本数量
            with open(cache_dir / f'{self.mode}_samples.pkl', 'rb') as f:
                self.samples = pickle.load(f)
            
            # 对于32GB内存，直接加载数据到内存中，提高访问速度
            # 32GB内存的80%为25.6GB，足够同时加载所有数据集
            logger.info(f"Directly loading data into memory for faster access...")
            self.features = np.load(
                cache_dir / f'{self.mode}_features.npy',
                mmap_mode=None  # 直接加载到内存
            )
            
            # labels 也直接加载到内存
            self.labels = np.load(
                cache_dir / f'{self.mode}_labels.npy',
                mmap_mode=None  # 直接加载到内存
            )
            
            # 加载行业ID (如果存在)
            industry_file = cache_dir / f'{self.mode}_industry.npy'
            if industry_file.exists():
                self.industry_ids = np.load(industry_file, mmap_mode=None)
            else:
                # 兼容旧缓存：生成默认行业ID，直接加载到内存
                n_samples = len(self.samples)
                self.industry_ids = np.zeros(n_samples, dtype=np.int64)
                logger.warning("Industry IDs not found in cache, using default (0)")
            
            logger.info(f"Loaded cache from {cache_dir} (mmap mode) - features shape: {self.features.shape}")
            
            # 检查缓存数据是否为空
            if len(self.samples) == 0 or len(self.features) == 0:
                logger.warning(f"Cache is empty for {self.mode} mode! Creating fallback dataset...")
                
                # 创建虚拟数据
                dummy_seq_len = self.seq_len
                dummy_features = np.zeros((1, dummy_seq_len, self.n_features), dtype=np.float32)
                dummy_labels = np.zeros(1, dtype=np.float32)
                dummy_industry = np.zeros(1, dtype=np.int64)
                dummy_sample = [('dummy', 0, '2020-01-01')]
                
                self.features = dummy_features
                self.labels = dummy_labels
                self.industry_ids = dummy_industry
                self.samples = dummy_sample
                
                logger.info(f"Fallback: Created 1 dummy sample to avoid empty dataset")
                
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        获取单个样本 - 包含行业ID
        
        从预计算的 numpy 数组读取（支持 mmap）
        
        Returns:
            X: (seq_len, n_features) 特征序列
            y: () 标签
            industry_id: () 行业ID
            info: 元数据字典
        """
        # 从 mmap 数组读取，避免一次性加载所有数据
        features = self.features[idx]  # 直接从内存映射读取，不复制到内存
        label = self.labels[idx]  # 直接从内存映射读取
        industry_id = int(self.industry_ids[idx])  # 行业ID
        
        # 解决负步长问题：确保features是连续数组
        # PyTorch不支持负步长的numpy数组，需要先复制
        features = np.ascontiguousarray(features)  # 确保步长为正
        
        # 数据增强（仅训练集）
        if self.use_data_augmentation:
            # 1. 添加少量随机噪声
            noise = np.random.normal(0, 0.02, features.shape)
            features = features + noise
            
            # 2. 随机缩放（±5%）
            scale = np.random.uniform(0.95, 1.05)
            features = features * scale
            
            # 3. 时间序列平移（±1步）
            if np.random.random() < 0.2:
                shift = np.random.choice([-1, 1])
                features = np.roll(features, shift, axis=0)
                # 填充缺失值
                if shift > 0:
                    features[:shift] = 0
                else:
                    features[shift:] = 0
            
            # 4. 随机翻转（时间轴）
            if np.random.random() < 0.1:
                features = np.flip(features, axis=0)
                features = np.ascontiguousarray(features)  # 确保步长为正
            
            # 5. 随机缺失（随机遮盖一些特征值）
            if np.random.random() < 0.2:
                mask = np.random.random(features.shape) < 0.05
                features[mask] = 0
            
            # 6. 随机交换特征（特征维度）
            if np.random.random() < 0.1:
                # 随机选择两个特征进行交换
                if features.shape[1] > 1:
                    idx1, idx2 = np.random.choice(features.shape[1], 2, replace=False)
                    features[:, [idx1, idx2]] = features[:, [idx2, idx1]]
                    features = np.ascontiguousarray(features)  # 确保步长为正
        
        # 转换为 tensor，显式转换为 float32 以匹配模型权重类型
        X = torch.from_numpy(features).float()  # 转换为 float32，避免Double类型
        y = torch.tensor(label, dtype=torch.float32)
        ind = torch.tensor(industry_id, dtype=torch.long)  # 行业ID
        
        # 从samples中获取日期信息
        if len(self.samples[idx]) == 3:
            ts_code, _, date = self.samples[idx]
        else:
            # 兼容旧格式
            ts_code, _ = self.samples[idx]
            date = ''
        info = {'ts_code': ts_code, 'date': date, 'future_ret': 0.0}
        
        return X, y, ind, info


class AlphaDataModule:
    """
    数据模块管理器 - 优化版
    """
    
    def __init__(
        self,
        data: pd.DataFrame = None,
        batch_size: int = None,
        num_workers: int = None,
        force_rebuild: bool = False,
        auto_clear_cache: bool = True,  # 新增：自动清除缓存
        train_range: Tuple[str, str] = None,
        val_range: Tuple[str, str] = None,
        memory_optimized: bool = True,  # 新增：内存优化模式
        **kwargs
    ):
        """
        初始化数据模块
        
        Args:
            data: 输入数据（忽略，使用缓存）
            batch_size: 批次大小
            num_workers: 数据加载线程数
            force_rebuild: 是否强制重建缓存
            auto_clear_cache: 是否自动清除缓存（默认True）
            train_range: 自定义训练集日期范围
            val_range: 自定义验证集日期范围
            memory_optimized: 是否启用内存优化模式（默认True）
        """
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.num_workers = num_workers
        self.force_rebuild = force_rebuild
        self.memory_optimized = memory_optimized
        
        # 自动清除缓存，确保每次训练使用最新数据
        # 使用基于时间戳的会话目录，彻底解决 Windows 文件锁定问题
        self.session_cache_dir = None
        if auto_clear_cache:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_cache_dir = Config.DATA_ROOT / 'fast_cache' / timestamp
            self._clear_old_caches() # 尝试清理旧缓存
            force_rebuild = True
        
        logger.info(f"Using session cache dir: {self.session_cache_dir}")
        self.train_range = train_range
        self.val_range = val_range
        
        # 内存优化：延迟创建数据集，只在需要时创建
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 创建所有数据集，提高训练过程中的验证效率
        # 32G内存的80%为25.6G，足够同时创建三个数据集
        logger.info(f"Creating train dataset... (Range: {train_range})")
        self.train_dataset = FastAlphaDataset(mode='train', force_rebuild=force_rebuild, cache_dir=self.session_cache_dir, date_range=train_range)
        
        logger.info(f"Creating validation dataset... (Range: {val_range})")
        self.val_dataset = FastAlphaDataset(mode='val', force_rebuild=force_rebuild, cache_dir=self.session_cache_dir, date_range=val_range)
        
        logger.info("Creating test dataset...")
        # 测试集暂时不提供自定义范围，或者可以使用默认逻辑
        self.test_dataset = FastAlphaDataset(mode='test', force_rebuild=force_rebuild, cache_dir=self.session_cache_dir)
    
    def _clear_old_caches(self):
        """尝试清理旧的缓存目录"""
        import shutil
        import time
        root_cache_dir = Config.DATA_ROOT / 'fast_cache'
        if not root_cache_dir.exists():
            return
            
        try:
            # 清理超过 24 小时的旧目录
            current_time = time.time()
            for path in root_cache_dir.iterdir():
                if path.is_dir():
                    try:
                        # 如果目录名是日期格式，或者修改时间很久以前
                        stat = path.stat()
                        if current_time - stat.st_mtime > 3600: # 1小时前的都可以清理
                            shutil.rmtree(path)
                            logger.info(f"Cleaned up old cache: {path}")
                    except Exception:
                        pass # 忽略清理失败（可能被占用）
        except Exception:
            pass

    def _clear_cache(self):
        """已弃用 - 由 session_cache_dir 替代"""
        pass
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器 - 超优化版 (适配 A800 & 96GB RAM)"""
        import os
        import multiprocessing
        
        # 优化 num_workers：在 96GB 内存下，Windows 也可以尝试开启少量 workers
        if self.num_workers is None:
            cpu_count = multiprocessing.cpu_count() or 6
            if os.name == 'nt':  # Windows
                # Windows 下开启 4 个 worker，配合 persistent_workers
                self.num_workers = min(4, cpu_count)
            else:  # Linux/Mac
                self.num_workers = min(8, cpu_count)
        
        # 硬件加速配置
        pin_memory = getattr(Config, 'PIN_MEMORY', True)
        
        logger.info(f"Train dataloader: num_workers={self.num_workers}, batch_size={self.batch_size}, pin_memory={pin_memory}")
        
        # A800 优化：启用 persistent_workers 和 prefetch_factor
        # 注意：Windows 下使用 num_workers > 0 必须在 main 中运行
        persistent_workers = True if self.num_workers > 0 else False
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if persistent_workers else None,
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器 - 优化版"""
        import os
        import multiprocessing
        
        if self.num_workers is None:
            cpu_count = multiprocessing.cpu_count() or 6
            if os.name == 'nt':
                self.num_workers = min(2, cpu_count)
            else:
                self.num_workers = min(4, cpu_count)
        
        pin_memory = getattr(Config, 'PIN_MEMORY', True)
        persistent_workers = True if self.num_workers > 0 else False
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if persistent_workers else None,
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        测试数据加载器
        """
        import os
        import multiprocessing
        
        # 测试集使用较少的 workers
        if self.num_workers is None:
            if os.name == 'nt':  # Windows
                # Windows 下使用 0 workers（避免内存复制问题）
                self.num_workers = 0
            else:  # Linux/Mac
                cpu_count = multiprocessing.cpu_count() or 6
                self.num_workers = min(6, cpu_count)
        
        logger.info(f"Test dataloader: num_workers={self.num_workers}, batch_size={self.batch_size}")
        
        # Windows 下禁用 persistent_workers（避免 pickle 问题）
        persistent_workers = False if os.name == 'nt' else True
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,  # Windows 下禁用
            prefetch_factor=2 if persistent_workers else None,  # Windows 下禁用
        )
    
    @staticmethod
    def _collate_fn(batch):
        """自定义 collate 函数 - 包含行业ID"""
        X_list, y_list, ind_list, info_list = zip(*batch)
        X = torch.stack(X_list)
        y = torch.stack(y_list)
        ind = torch.stack(ind_list)
        return X, y, ind, info_list


# 向后兼容
AlphaDataset = FastAlphaDataset


def create_dataloaders(
    data: pd.DataFrame = None,
    batch_size: int = None,
    force_rebuild: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    便捷函数: 创建数据加载器
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_module = AlphaDataModule(data, batch_size, force_rebuild=force_rebuild)
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )
