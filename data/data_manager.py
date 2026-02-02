# -*- coding: utf-8 -*-
"""
统一数据管理模块
功能:
- 统一数据存储路径管理
- 智能缓存机制
- 数据有效期管理
- 自动清理过期数据
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from collections import OrderedDict
import threading

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("DataManager")


class TradingCalendar:
    """
    交易日历工具类
    
    用于判断是否是交易日，避免在非交易日触发无效下载
    """
    
    # 中国股市周末休市
    WEEKEND_DAYS = {5, 6}  # 周六=5, 周日=6
    
    # 已知的固定节假日 (月-日格式)
    FIXED_HOLIDAYS = {
        '01-01',  # 元旦
        '05-01', '05-02', '05-03',  # 劳动节
        '10-01', '10-02', '10-03', '10-04', '10-05', '10-06', '10-07',  # 国庆节
    }
    
    # 缓存最近的交易日
    _last_trading_date_cache = {}
    
    @classmethod
    def is_weekend(cls, date: datetime) -> bool:
        """判断是否是周末"""
        return date.weekday() in cls.WEEKEND_DAYS
    
    @classmethod
    def is_likely_holiday(cls, date: datetime) -> bool:
        """
        判断是否可能是节假日
        
        注意：这是一个简化的判断，无法覆盖所有情况（如调休）
        建议使用缓存的最近交易日作为后备
        """
        month_day = date.strftime('%m-%d')
        return month_day in cls.FIXED_HOLIDAYS
    
    @classmethod
    def is_trading_day(cls, date_str: str) -> bool:
        """
        判断指定日期是否是交易日
        
        Args:
            date_str: 日期字符串 (YYYYMMDD格式)
            
        Returns:
            是否是交易日
        """
        try:
            date = datetime.strptime(date_str, '%Y%m%d')
            
            # 周末一定不是交易日
            if cls.is_weekend(date):
                return False
            
            # 已知固定节假日
            if cls.is_likely_holiday(date):
                return False
            
            return True
        except Exception:
            return True  # 解析失败时假设是交易日
    
    @classmethod
    def get_last_trading_date(cls, date_str: str = None) -> str:
        """
        获取最近的交易日
        
        如果指定日期不是交易日，则向前查找最近的交易日
        
        Args:
            date_str: 日期字符串 (YYYYMMDD格式)，None表示今天
            
        Returns:
            最近的交易日 (YYYYMMDD格式)
        """
        if date_str is None:
            date = datetime.now()
            date_str = date.strftime('%Y%m%d')
        else:
            date = datetime.strptime(date_str, '%Y%m%d')
        
        # 检查缓存
        if date_str in cls._last_trading_date_cache:
            return cls._last_trading_date_cache[date_str]
        
        # 最多向前查找30天
        for i in range(30):
            check_date = date - timedelta(days=i)
            check_str = check_date.strftime('%Y%m%d')
            
            if cls.is_trading_day(check_str):
                # 缓存结果
                cls._last_trading_date_cache[date_str] = check_str
                return check_str
        
        # 如果找不到（不应该发生），返回原日期
        return date_str
    
    @classmethod
    def is_market_open_time(cls) -> bool:
        """
        判断当前是否在交易时间
        
        A股交易时间: 9:30-11:30, 13:00-15:00
        """
        now = datetime.now()
        
        # 先判断是否是交易日
        if not cls.is_trading_day(now.strftime('%Y%m%d')):
            return False
        
        # 判断时间
        current_time = now.hour * 100 + now.minute
        
        morning_open = 930
        morning_close = 1130
        afternoon_open = 1300
        afternoon_close = 1500
        
        return (morning_open <= current_time <= morning_close or 
                afternoon_open <= current_time <= afternoon_close)
    
    @classmethod
    def should_update_data(cls, date_str: str = None) -> tuple:
        """
        判断是否应该更新数据
        
        Args:
            date_str: 目标日期
            
        Returns:
            (should_update, actual_date): 是否应该更新，实际应该使用的日期
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
        
        # 如果是交易日，正常更新
        if cls.is_trading_day(date_str):
            return True, date_str
        
        # 如果不是交易日，使用最近的交易日
        last_trading = cls.get_last_trading_date(date_str)
        logger.info(f"Date {date_str} is not a trading day, using last trading date: {last_trading}")
        
        return False, last_trading




class DataPaths:
    """统一数据路径管理"""
    
    def __init__(self, base_dir: Path = None):
        """
        初始化路径管理器
        
        Args:
            base_dir: 数据根目录
        """
        self.base_dir = base_dir or Config.DATA_ROOT
        
        # 核心数据目录
        self.raw_dir = self.base_dir / "raw"                    # 原始股票数据
        self.processed_dir = self.base_dir / "processed"        # 处理后的数据
        self.stats_dir = self.base_dir / "stats"                # 统计量缓存
        self.cache_dir = self.base_dir / "cache"                # 通用缓存
        
        # 功能专用目录
        self.score_cache_dir = self.cache_dir / "scores"        # 分数分布缓存
        self.kline_cache_dir = self.cache_dir / "kline"         # K线数据缓存
        self.indicator_cache_dir = self.cache_dir / "indicators" # 技术指标缓存
        self.model_cache_dir = self.cache_dir / "models"        # 模型相关缓存
        self.quality_dir = self.base_dir / "quality_reports"    # 数据质量报告
        
        # 组合数据目录
        self.portfolio_dir = self.base_dir / "portfolio"        # 持仓数据
        
        # 确保所有目录存在
        self._ensure_dirs()
        
        logger.info(f"DataPaths initialized with base: {self.base_dir}")
    
    def _ensure_dirs(self):
        """确保所有目录存在"""
        dirs = [
            self.raw_dir,
            self.processed_dir,
            self.stats_dir,
            self.cache_dir,
            self.score_cache_dir,
            self.kline_cache_dir,
            self.indicator_cache_dir,
            self.model_cache_dir,
            self.quality_dir,
            self.portfolio_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_stock_data_path(self, ts_code: str) -> Path:
        """获取股票数据文件路径"""
        filename = ts_code.replace('.', '_') + '.parquet'
        return self.raw_dir / filename
    
    def get_processed_data_path(self, ts_code: str) -> Path:
        """获取处理后的股票数据路径"""
        filename = ts_code.replace('.', '_') + '_processed.parquet'
        return self.processed_dir / filename
    
    def get_stats_cache_path(self, date: str) -> Path:
        """获取统计量缓存路径"""
        return self.stats_dir / f"{date}.json"
    
    def get_score_cache_path(self, date: str) -> Path:
        """获取分数分布缓存路径"""
        return self.score_cache_dir / f"score_dist_{date}.json"
    
    def get_kline_cache_path(self, ts_code: str, period: str = 'daily') -> Path:
        """获取K线数据缓存路径"""
        filename = f"{ts_code.replace('.', '_')}_{period}.parquet"
        return self.kline_cache_dir / filename
    
    def get_index_data_path(self, index_code: str) -> Path:
        """获取指数数据路径"""
        filename = f"index_{index_code.replace('.', '_')}.parquet"
        return self.raw_dir / filename
    
    def get_industry_map_path(self) -> Path:
        """获取行业映射文件路径"""
        return self.cache_dir / "industry_map.json"
    
    def get_stock_list_path(self) -> Path:
        """获取股票列表缓存路径"""
        return self.cache_dir / "stock_list.parquet"


class DataCache:
    """通用数据缓存"""
    
    DEFAULT_EXPIRY_HOURS = {
        'stock_data': 24,       # 股票数据24小时过期
        'kline': 12,            # K线数据12小时过期
        'stats': 48,            # 统计量48小时过期
        'stock_list': 168,      # 股票列表一周过期
        'industry_map': 720,    # 行业映射一个月过期
        'index_data': 24,       # 指数数据24小时过期
    }
    
    def __init__(self, paths: DataPaths = None):
        """
        初始化缓存管理器
        
        Args:
            paths: 路径管理器
        """
        self.paths = paths or DataPaths()
        self.meta_file = self.paths.cache_dir / "cache_meta.json"
        self._load_meta()
        self._lock = threading.Lock()
        
        # 内存缓存 (LRU)
        self._memory_cache: OrderedDict = OrderedDict()
        self._memory_cache_limit = 100
    
    def _load_meta(self):
        """加载缓存元数据"""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    self.meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache meta: {e}")
                self.meta = {}
        else:
            self.meta = {}
    
    def _save_meta(self):
        """保存缓存元数据"""
        try:
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache meta: {e}")
    
    def _get_cache_key(self, category: str, identifier: str) -> str:
        """生成缓存键"""
        return f"{category}:{identifier}"
    
    def is_valid(self, cache_path: Path, category: str = 'stock_data') -> bool:
        """
        检查缓存是否有效（未过期）
        
        Args:
            cache_path: 缓存文件路径
            category: 数据类别
            
        Returns:
            是否有效
        """
        if not cache_path.exists():
            return False
        
        # 获取过期时间
        expiry_hours = self.DEFAULT_EXPIRY_HOURS.get(category, 24)
        
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = mtime + timedelta(hours=expiry_hours)
        
        return datetime.now() < expiry_time
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """从内存缓存获取"""
        with self._lock:
            if key in self._memory_cache:
                # 移到末尾（最近访问）
                self._memory_cache.move_to_end(key)
                return self._memory_cache[key]
        return None
    
    def set_to_memory(self, key: str, value: Any):
        """设置内存缓存"""
        with self._lock:
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
            else:
                if len(self._memory_cache) >= self._memory_cache_limit:
                    # 移除最老的项
                    self._memory_cache.popitem(last=False)
            self._memory_cache[key] = value
    
    def update_meta(self, cache_path: Path, category: str):
        """更新缓存元数据"""
        key = str(cache_path)
        self.meta[key] = {
            'category': category,
            'updated_at': datetime.now().isoformat(),
            'size': cache_path.stat().st_size if cache_path.exists() else 0,
        }
        self._save_meta()
    
    def clean_expired(self, category: str = None):
        """
        清理过期缓存
        
        Args:
            category: 指定类别，None表示全部
        """
        cleaned = 0
        
        for key, info in list(self.meta.items()):
            if category and info.get('category') != category:
                continue
            
            cache_path = Path(key)
            if cache_path.exists():
                if not self.is_valid(cache_path, info.get('category', 'stock_data')):
                    try:
                        cache_path.unlink()
                        del self.meta[key]
                        cleaned += 1
                    except Exception as e:
                        logger.error(f"Failed to clean cache {key}: {e}")
        
        if cleaned > 0:
            self._save_meta()
            logger.info(f"Cleaned {cleaned} expired cache files")
        
        return cleaned


class UnifiedDataManager:
    """统一数据管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, base_dir: Path = None):
        """
        初始化统一数据管理器
        
        Args:
            base_dir: 数据根目录
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.paths = DataPaths(base_dir)
        self.cache = DataCache(self.paths)
        self._downloader = None
        self._initialized = True
        
        logger.info("UnifiedDataManager initialized")
    
    @property
    def downloader(self):
        """懒加载下载器"""
        if self._downloader is None:
            from data.downloader import TushareDownloader
            self._downloader = TushareDownloader()
        return self._downloader
    
    def _should_update_cache_based_on_trading_day(self, df: pd.DataFrame, ts_code: str, cache_path: Path = None) -> bool:
        """
        基于交易日判断是否需要更新缓存
        
        Args:
            df: 缓存中的数据
            ts_code: 股票代码
            cache_path: 缓存文件路径
            
        Returns:
            是否需要更新缓存
        """
        if df is None or df.empty:
            return True
        
        if 'trade_date' not in df.columns:
            return True
        
        # 获取缓存中的最新日期
        latest_date = df['trade_date'].max()
        # 统一强制转换为字符串，处理 Timestamp, datetime, np.datetime64 等各种情况
        if hasattr(latest_date, 'strftime'):
            latest_date = latest_date.strftime('%Y%m%d')
        else:
            latest_date = str(latest_date).replace('-', '').replace('/', '')[:8]
        
        # 获取最近交易日
        today = datetime.now().strftime('%Y%m%d')
        last_trading_day = TradingCalendar.get_last_trading_date(today)
        
        # 检查缓存文件的修改时间
        if cache_path and cache_path.exists():
            try:
                # 获取缓存文件的修改时间
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                cache_age_hours = (datetime.now() - mtime).total_seconds() / 3600
                
                # 如果缓存文件超过24小时，强制更新
                if cache_age_hours > 24:
                    logger.info(f"Cache file too old for {ts_code}: age={cache_age_hours:.1f} hours, will download fresh data")
                    return True
            except Exception as e:
                logger.warning(f"Failed to check cache file age for {ts_code}: {e}")
        
        # 如果缓存中的最新日期不是最近交易日，需要更新
        if latest_date < last_trading_day:
            logger.info(f"Cache stale for {ts_code}: latest={latest_date}, expected={last_trading_day}")
            return True
        
        return False
    
    def get_stock_data(self, 
                       ts_code,
                       start_date: str = None,
                       end_date: str = None,
                       use_cache: bool = True,
                       force_download: bool = False):
        """
        获取股票数据（优先使用缓存）
        
        关键逻辑：
        - 非交易日（周末/节假日）不触发下载，直接使用本地数据
        - 交易日且缓存过期才触发下载
        - 基于交易日的智能缓存更新：确保缓存包含最近交易日的数据
        
        Args:
            ts_code: 股票代码或股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            force_download: 是否强制下载（忽略缓存）
            
        Returns:
            单个股票代码：返回DataFrame
            股票代码列表：返回包含所有股票数据的列表
        """
        # 处理批量请求
        if isinstance(ts_code, list):
            return self._get_batch_stock_data(
                ts_code, start_date, end_date, use_cache, force_download
            )
        
        # 单个股票代码的处理逻辑
        cache_path = self.paths.get_stock_data_path(ts_code)
        cache_key = f"stock:{ts_code}"
        
        # 1. 尝试内存缓存
        if use_cache and not force_download:
            cached = self.cache.get_from_memory(cache_key)
            if cached is not None:
                logger.debug(f"Memory cache hit for {ts_code}")
                return self._filter_by_date(cached, start_date, end_date)
        
        # 2. 尝试文件缓存
        if use_cache and not force_download and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self.cache.set_to_memory(cache_key, df)
                logger.debug(f"File cache hit for {ts_code}")
                return self._filter_by_date(df, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to read cache for {ts_code}: {e}")
        
        # 3. 如果使用缓存且不强制下载，直接返回缓存数据
        if use_cache and not force_download and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self.cache.set_to_memory(cache_key, df)
                logger.debug(f"Using cached data for {ts_code} (force_download=False)")
                return self._filter_by_date(df, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to read cache for {ts_code}: {e}")
        
        # 4. 只有当force_download=True或缓存不存在时才检查数据新鲜度
        today = datetime.now().strftime('%Y%m%d')
        is_trading_day = TradingCalendar.is_trading_day(today)
        last_trading_day = TradingCalendar.get_last_trading_date(today)
        
        # 即使不是交易日，如果本地数据陈旧（没有包含到最近的交易日），也需要更新
        need_update = False
        
        if cache_path.exists():
            try:
                # 预读数据检查最新日期
                df = pd.read_parquet(cache_path)
                
                # 检查最新日期
                if 'trade_date' in df.columns:
                    latest_date = df['trade_date'].max()
                    # 确保格式统一
                    if isinstance(latest_date, (pd.Timestamp, datetime)):
                        latest_date = latest_date.strftime('%Y%m%d')
                    
                    if latest_date < last_trading_day:
                        # 数据过期，需要更新
                        need_update = True
                        logger.info(f"Data stale for {ts_code}: latest={latest_date}, expected={last_trading_day}. Update required.")
                    elif not is_trading_day:
                        # 数据够新且今天非交易日 -> 直接使用
                        self.cache.set_to_memory(cache_key, df)
                        logger.debug(f"Data up-to-date ({latest_date}) on non-trading day. Using cache.")
                        return self._filter_by_date(df, start_date, end_date)
                    elif is_trading_day:
                        # 交易日，数据已是最新的 -> 检查是否收盘
                        if latest_date == today:
                             # 已经有今天的数据了
                             self.cache.set_to_memory(cache_key, df)
                             return self._filter_by_date(df, start_date, end_date)
                        else:
                             # 交易日，但最新数据不是今天 -> 可能盘中更新 -> 视缓存策略而定
                             # 这里简单起见，如果缓存有效（Cache-Is-Valid判断过期时间）则用，否则更新
                             if self.cache.is_valid(cache_path, 'stock_data'):
                                 self.cache.set_to_memory(cache_key, df)
                                 return self._filter_by_date(df, start_date, end_date)
                             need_update = True
                else:
                    need_update = True # 格式不对，重新下载
            except Exception:
                need_update = True # 读取失败，重新下载
        else:
            need_update = True # 无文件，需要下载
        
        # 4. 执行下载（如果需要）
        # 只有在 need_update 为 True 时才走到这里
        # 特别保护：如果是非交易日且 need_update=True，说明你在补数据，这是允许的

        logger.info(f"Downloading data for {ts_code}...")
        try:
            df = self.downloader.download_stock_data(ts_code, start_date, end_date)
            
            if df is not None and not df.empty:
                # 合并现有数据（增量更新）
                if cache_path.exists() and not force_download:
                    existing_df = pd.read_parquet(cache_path)
                    df = self._merge_data(existing_df, df)
                
                # 保存到文件缓存
                df = df.copy()
                if 'trade_date' in df.columns:
                    df['trade_date'] = df['trade_date'].astype(str)
                df.to_parquet(cache_path, index=False)
                self.cache.update_meta(cache_path, 'stock_data')
                
                # 保存到内存缓存
                self.cache.set_to_memory(cache_key, df)
                
                logger.info(f"Downloaded and cached data for {ts_code}, shape: {df.shape}")
                return self._filter_by_date(df, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to download data for {ts_code}: {e}")
        
        # 5. 下载失败时尝试使用本地旧数据
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self.cache.set_to_memory(cache_key, df)
                logger.warning(f"Download failed, using old cache for {ts_code}")
                return self._filter_by_date(df, start_date, end_date)
            except Exception:
                pass
        
        return None
    
    def _get_batch_stock_data(self, 
                            ts_codes: List[str],
                            start_date: str = None,
                            end_date: str = None,
                            use_cache: bool = True,
                            force_download: bool = False) -> List[pd.DataFrame]:
        """
        批量获取股票数据（使用并发下载）
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            force_download: 是否强制下载（忽略缓存）
            
        Returns:
            包含所有股票数据的列表
        """
        all_data = []
        
        if not ts_codes:
            return all_data
        
        logger.info(f"Batch processing {len(ts_codes)} stocks...")
        
        # 1. 批量检查缓存状态
        stocks_to_download = []
        cached_data = []
        
        for ts_code in ts_codes:
            try:
                cache_path = self.paths.get_stock_data_path(ts_code)
                cache_key = f"stock:{ts_code}"
                
                # 尝试内存缓存
                if use_cache and not force_download:
                    cached = self.cache.get_from_memory(cache_key)
                    if cached is not None:
                        if not self._should_update_cache_based_on_trading_day(cached, ts_code, cache_path):
                            filtered_df = self._filter_by_date(cached, start_date, end_date)
                            cached_data.append(filtered_df)
                            continue
                
                # 尝试文件缓存
                if use_cache and not force_download and cache_path.exists():
                    df = pd.read_parquet(cache_path)
                    if not self._should_update_cache_based_on_trading_day(df, ts_code, cache_path):
                        filtered_df = self._filter_by_date(df, start_date, end_date)
                        cached_data.append(filtered_df)
                        continue
                
                # 需要下载
                stocks_to_download.append(ts_code)
            except Exception:
                stocks_to_download.append(ts_code)
        
        logger.info(f"Cached data: {len(cached_data)}, Need download: {len(stocks_to_download)}")
        
        # 2. 并发下载需要的数据
        downloaded_data = []
        if stocks_to_download:
            logger.info(f"Concurrent downloading {len(stocks_to_download)} stocks...")
            try:
                # 使用下载器的并发下载功能
                result = self.downloader.download_with_thread_pool(stocks_to_download, force_update=force_download)
                logger.info(f"Concurrent download completed: {result}")
                
                # 重新加载下载的数据
                for ts_code in stocks_to_download:
                    try:
                        df = self.get_stock_data(ts_code, start_date, end_date, use_cache, force_download=False)
                        if df is not None and not df.empty:
                            downloaded_data.append(df)
                    except Exception as e:
                        logger.debug(f"Failed to reload {ts_code}: {e}")
            except Exception as e:
                logger.error(f"Concurrent download failed: {e}")
        
        # 3. 合并所有数据
        all_data.extend(cached_data)
        all_data.extend(downloaded_data)
        
        logger.info(f"Batch processing completed: {len(all_data)} stocks loaded")
        return all_data
    
    def _filter_by_date(self, 
                        df: pd.DataFrame, 
                        start_date: str = None, 
                        end_date: str = None) -> pd.DataFrame:
        """按日期筛选数据 (增强健壮性)"""
        if df is None or df.empty:
            return df
        
        try:
            df = df.copy()
            
            # 确保日期格式统一
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
                # 移除无法解析日期的行
                df = df.dropna(subset=['trade_date'])
            else:
                return df
            
            if start_date:
                try:
                    start = pd.to_datetime(start_date)
                    df = df[df['trade_date'] >= start]
                except Exception:
                    pass
            
            if end_date:
                try:
                    end = pd.to_datetime(end_date)
                    df = df[df['trade_date'] <= end]
                except Exception:
                    pass
            
            return df
        except Exception as e:
            logger.error(f"Error filtering data by date: {e}")
            return df
    
    def _merge_data(self, 
                    existing: pd.DataFrame, 
                    new: pd.DataFrame) -> pd.DataFrame:
        """合并现有数据和新数据（去重）"""
        if existing is None or existing.empty:
            return new
        if new is None or new.empty:
            return existing
        
        try:
            # 统一日期格式为字符串，避免类型混合问题
            if 'trade_date' in existing.columns:
                existing = existing.copy()
                existing['trade_date'] = existing['trade_date'].astype(str).str.replace('-', '').str[:8]
            if 'trade_date' in new.columns:
                new = new.copy()
                new['trade_date'] = new['trade_date'].astype(str).str.replace('-', '').str[:8]
            
            # 合并并去重
            merged = pd.concat([existing, new], ignore_index=True)
            merged = merged.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
            merged = merged.sort_values('trade_date').reset_index(drop=True)
            
            return merged
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return new if new is not None and not new.empty else existing
    
    def get_kline_data(self,
                       ts_code: str,
                       period: str = 'daily',
                       n_bars: int = 200,
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取K线数据
        
        Args:
            ts_code: 股票代码
            period: K线周期 (daily, weekly, monthly)
            n_bars: 获取的K线数量
            use_cache: 是否使用缓存
            
        Returns:
            K线数据DataFrame
        """
        cache_path = self.paths.get_kline_cache_path(ts_code, period)
        cache_key = f"kline:{ts_code}:{period}"
        
        # 尝试缓存
        if use_cache:
            cached = self.cache.get_from_memory(cache_key)
            if cached is not None:
                return cached.tail(n_bars)
            
            if cache_path.exists() and self.cache.is_valid(cache_path, 'kline'):
                try:
                    df = pd.read_parquet(cache_path)
                    self.cache.set_to_memory(cache_key, df)
                    return df.tail(n_bars)
                except Exception:
                    pass
        
        # 下载数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=n_bars * 2)).strftime('%Y%m%d')
        
        df = self.get_stock_data(ts_code, start_date, end_date)
        
        if df is not None and not df.empty:
            # 保存K线缓存
            df = df.copy()
            if 'trade_date' in df.columns:
                df['trade_date'] = df['trade_date'].astype(str)
            df.to_parquet(cache_path, index=False)
            self.cache.update_meta(cache_path, 'kline')
            self.cache.set_to_memory(cache_key, df)
            return df.tail(n_bars)
        
        return None
    
    def get_stock_list(self, 
                       use_cache: bool = True,
                       force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取主板股票列表
        
        非交易日不刷新，直接使用本地缓存
        
        Args:
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新
            
        Returns:
            股票列表DataFrame
        """
        cache_path = self.paths.get_stock_list_path()
        cache_key = "stock_list:mainboard"
        
        # 尝试缓存
        if use_cache and not force_refresh:
            cached = self.cache.get_from_memory(cache_key)
            if cached is not None:
                return cached
            
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    self.cache.set_to_memory(cache_key, df)
                    logger.debug(f"Loaded stock list from cache: {len(df)} stocks")
                    return df
                except Exception:
                    pass
        
        # 检查是否是交易日 - 非交易日不下载
        today = datetime.now().strftime('%Y%m%d')
        if not TradingCalendar.is_trading_day(today) and not force_refresh:
            # 非交易日：如果本地有缓存就用，否则返回None
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    self.cache.set_to_memory(cache_key, df)
                    logger.debug(f"Non-trading day, using existing stock list: {len(df)} stocks")
                    return df
                except Exception:
                    pass
            logger.debug("Non-trading day and no local stock list, skipping download")
            return None
        
        # 交易日下载新列表
        try:
            logger.info("Downloading stock list...")
            df = self.downloader.get_main_board_stocks()
            if df is not None and not df.empty:
                df = df.copy()
                if 'trade_date' in df.columns:
                    df['trade_date'] = df['trade_date'].astype(str)
                df.to_parquet(cache_path, index=False)
                self.cache.update_meta(cache_path, 'stock_list')
                self.cache.set_to_memory(cache_key, df)
                logger.info(f"Downloaded and cached stock list: {len(df)} stocks")
                return df
        except Exception as e:
            logger.error(f"Failed to get stock list: {e}")
        
        # 下载失败时使用本地旧缓存
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                self.cache.set_to_memory(cache_key, df)
                logger.warning(f"Download failed, using old stock list: {len(df)} stocks")
                return df
            except Exception:
                pass
        
        return None
    
    def get_stats_cache(self, date: str) -> Optional[Dict]:
        """获取统计量缓存"""
        cache_path = self.paths.get_stats_cache_path(date)
        cache_key = f"stats:{date}"
        
        # 内存缓存
        cached = self.cache.get_from_memory(cache_key)
        if cached is not None:
            return cached
        
        # 文件缓存
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                self.cache.set_to_memory(cache_key, stats)
                return stats
            except Exception as e:
                logger.error(f"Failed to load stats cache: {e}")
        
        return None
    
    def save_stats_cache(self, date: str, stats: Dict):
        """保存统计量缓存"""
        cache_path = self.paths.get_stats_cache_path(date)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False)
        
        self.cache.update_meta(cache_path, 'stats')
        self.cache.set_to_memory(f"stats:{date}", stats)
    
    def batch_get_stock_data(self,
                              ts_codes: List[str],
                              start_date: str = None,
                              end_date: str = None,
                              progress_callback: callable = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取股票数据
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调函数
            
        Returns:
            {ts_code: DataFrame} 字典
        """
        results = {}
        total = len(ts_codes)
        
        for i, ts_code in enumerate(ts_codes):
            df = self.get_stock_data(ts_code, start_date, end_date)
            if df is not None and not df.empty:
                results[ts_code] = df
            
            if progress_callback:
                progress_callback(i + 1, total, ts_code)
        
        logger.info(f"Batch loaded {len(results)}/{total} stocks")
        return results
    
    def clean_cache(self, category: str = None, older_than_days: int = None):
        """
        清理缓存
        
        Args:
            category: 清理指定类别
            older_than_days: 清理N天前的缓存
        """
        if older_than_days:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            
            for dir_path in [self.paths.cache_dir, self.paths.kline_cache_dir]:
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff:
                            file_path.unlink()
                            logger.debug(f"Cleaned old cache: {file_path}")
        
        # 清理过期缓存
        self.cache.clean_expired(category)
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        stats = {
            'raw_files': len(list(self.paths.raw_dir.glob("*.parquet"))),
            'processed_files': len(list(self.paths.processed_dir.glob("*.parquet"))),
            'stats_files': len(list(self.paths.stats_dir.glob("*.json"))),
            'kline_files': len(list(self.paths.kline_cache_dir.glob("*.parquet"))),
            'score_files': len(list(self.paths.score_cache_dir.glob("*.json"))),
            'memory_cache_size': len(self.cache._memory_cache),
            'total_meta_entries': len(self.cache.meta),
        }
        
        # 计算磁盘占用
        total_size = 0
        for dir_path in [self.paths.raw_dir, self.paths.processed_dir, 
                        self.paths.cache_dir, self.paths.stats_dir]:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats


# 全局实例
_data_manager: Optional[UnifiedDataManager] = None


def get_data_manager() -> UnifiedDataManager:
    """获取数据管理器单例"""
    global _data_manager
    if _data_manager is None:
        _data_manager = UnifiedDataManager()
    return _data_manager


if __name__ == "__main__":
    print("统一数据管理器测试")
    print("="*50)
    
    manager = get_data_manager()
    
    # 显示路径配置
    print("\n数据路径配置:")
    print(f"  基础目录: {manager.paths.base_dir}")
    print(f"  原始数据: {manager.paths.raw_dir}")
    print(f"  处理数据: {manager.paths.processed_dir}")
    print(f"  缓存目录: {manager.paths.cache_dir}")
    print(f"  K线缓存: {manager.paths.kline_cache_dir}")
    print(f"  分数缓存: {manager.paths.score_cache_dir}")
    
    # 显示缓存统计
    print("\n缓存统计:")
    stats = manager.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n统一数据管理器测试完成!")
