# -*- coding: utf-8 -*-
"""
智能下载器 (YahooDownloader)
只使用 Yahoo Finance 作为数据源
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

import sys
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("Downloader")

class YahooDownloader:
    """
    Yahoo Finance 数据下载器
    
    特性:
    - 只使用 Yahoo Finance 获取 A 股数据
    - 兼容原有 TushareDownloader 接口
    - 原子化写入
    - 断点续传
    - 健壮的错误处理和重试机制
    """

    def __init__(self, token: str = None):
        """初始化下载器
        
        Args:
            token: 兼容参数，AkShare 不需要 token
        """
        # 确保目录存在
        Config.ensure_dirs()
        
        # 加载 manifest
        self.manifest = self._load_manifest()
        
        # 信号量控制并发
        # AkShare 是爬虫类接口，并发过高容易被封 IP，保守设置
        self._api_semaphore = threading.Semaphore(4)
        self._file_semaphore = threading.Semaphore(8)
        
        # 停止标志
        self._stop_flag = False
        
        # 进度回调
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
        
        # 并发下载设置 (使用 Config 中定义的值，但受限于 AkShare 特性可能需要调整)
        self.concurrent_workers = Config.CONCURRENT_WORKERS
        
        # 强制禁用系统代理，避免 ProxyError
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['ALL_PROXY'] = ''
        os.environ['all_proxy'] = ''
        
        # 禁用 AkShare 内部可能的代理设置
        import requests
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        
        # 创建一个无代理的会话
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # 确保会话不使用任何代理
        self.session.trust_env = False
        
        # Monkey patch requests库，确保所有请求都不使用代理
        import requests
        original_request = requests.request
        
        def patched_request(method, url, **kwargs):
            # 强制禁用代理
            kwargs['proxies'] = {}
            kwargs['verify'] = True
            return original_request(method, url, **kwargs)
        
        requests.request = patched_request
        
        # 同样patch Session类
        from requests.sessions import Session
        original_session_request = Session.request
        
        def patched_session_request(self, method, url, **kwargs):
            # 强制禁用代理
            kwargs['proxies'] = {}
            kwargs['verify'] = True
            return original_session_request(self, method, url, **kwargs)
        
        Session.request = patched_session_request
        
        logger.info("Monkey patched requests library to disable proxies")
        
        logger.info(f"YahooDownloader initialized with {self.concurrent_workers} concurrent workers (Proxies disabled)")

    def _load_manifest(self) -> Dict:
        """加载下载清单"""
        manifest_file = Config.MANIFEST_FILE
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """保存下载清单"""
        manifest_file = Config.MANIFEST_FILE
        with self._file_semaphore:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _atomic_write(self, df: pd.DataFrame, filepath: Path):
        """原子化写入 Parquet 文件"""
        import uuid
        temp_file = filepath.with_suffix(f'.{uuid.uuid4().hex}.temp.parquet')
        try:
            df = df.copy()
            if 'trade_date' in df.columns:
                df['trade_date'] = df['trade_date'].astype(str)
            
            with self._file_semaphore:
                df.to_parquet(
                    temp_file, 
                    index=False,
                    compression='snappy',
                    engine='pyarrow'
                )
                
                if filepath.exists():
                    filepath.unlink(missing_ok=True)
                
                try:
                    os.replace(temp_file, filepath)
                except PermissionError:
                    time.sleep(0.1)
                    if filepath.exists():
                        filepath.unlink(missing_ok=True)
                    os.replace(temp_file, filepath)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink(missing_ok=True)
                except:
                    pass
            raise e

    def get_main_board_stocks(self) -> pd.DataFrame:
        """获取主板股票列表"""
        logger.info("获取 A 股股票列表...")
        
        # 尝试从本地缓存加载股票列表
        cache_file = Config.CACHE_DIR / "stock_list_cache.parquet"
        if cache_file.exists():
            try:
                logger.info(f"从缓存加载股票列表: {cache_file}")
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    count = len(df)
                    logger.info(f"从缓存加载了 {count} 只股票")
                    return df
            except Exception as e:
                logger.warning(f"从缓存加载股票列表失败: {e}")
        
        # 使用内置的股票列表
        logger.info("使用内置股票列表")
        
        # 创建一个完整的股票列表（包含更多股票）
        built_in_stocks = [
            "600000.SH", "600519.SH", "000001.SZ", "000858.SZ", "000333.SZ",
            "601318.SH", "601888.SH", "600276.SH", "601166.SH", "600036.SH",
            "601288.SH", "600031.SH", "600887.SH", "601668.SH", "601628.SH",
            "601857.SH", "600028.SH", "600104.SH", "600703.SH", "601398.SH",
            "600016.SH", "601988.SH", "601169.SH", "600009.SH", "600585.SH",
            "002594.SZ", "000651.SZ", "002415.SZ", "000338.SZ", "002507.SZ",
            "002475.SZ", "002714.SZ", "002460.SZ", "002142.SZ", "002236.SZ",
            "000895.SZ", "002352.SZ", "000661.SZ", "002027.SZ", "002241.SZ"
        ]
        
        stock_names = [
            "浦发银行", "贵州茅台", "平安银行", "五粮液", "美的集团",
            "中国平安", "中国中免", "恒瑞医药", "兴业银行", "招商银行",
            "农业银行", "三一重工", "伊利股份", "中国建筑", "中国人寿",
            "中国石油", "中国石化", "上汽集团", "三安光电", "工商银行",
            "民生银行", "中国银行", "北京银行", "上海机场", "海螺水泥",
            "比亚迪", "格力电器", "海康威视", "潍柴动力", "涪陵榨菜",
            "立讯精密", "牧原股份", "赣锋锂业", "宁波银行", "大华股份",
            "双汇发展", "顺丰控股", "长春高新", "分众传媒", "歌尔股份"
        ]
        
        # 创建DataFrame
        df = pd.DataFrame({
            'ts_code': built_in_stocks,
            'code': [s.split('.')[0] for s in built_in_stocks],
            'name': stock_names,
            'symbol': [s.split('.')[0] for s in built_in_stocks],
            'area': 'CN',
            'industry': 'Unknown',
            'market': 'Main',
            'list_date': '20000101'
        })
        
        # 保存到缓存
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        logger.info(f"内置股票列表已缓存到 {cache_file}")
        
        count = len(df)
        logger.info(f"获取到 {count} 只主板股票")
        
        return df

    def download_stock_data(
        self,
        ts_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """下载单只股票的数据 (仅使用 Yahoo Finance)"""
        # 生成默认日期
        current_date = datetime.now()
        if end_date is None:
            end_date = current_date.strftime('%Y%m%d')
        if start_date is None:
            # 默认下载配置年限
            start_date = (current_date - timedelta(days=365*Config.DOWNLOAD_YEARS)).strftime('%Y%m%d')
        
        # 验证日期格式和范围
        try:
            # 验证开始日期
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            # 验证结束日期
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            # 验证日期范围
            min_date = datetime(1990, 1, 1)
            max_date = current_date
            
            if start_dt < min_date:
                logger.warning(f"Start date {start_date} is too early, using minimum date: 19900101")
                start_date = "19900101"
            elif start_dt > max_date:
                logger.warning(f"Start date {start_date} is in the future, using current date")
                start_date = max_date.strftime('%Y%m%d')
            
            if end_dt < min_date:
                logger.warning(f"End date {end_date} is too early, using minimum date: 19900101")
                end_date = "19900101"
            elif end_dt > max_date:
                logger.warning(f"End date {end_date} is in the future, using current date")
                end_date = max_date.strftime('%Y%m%d')
            
            # 验证开始日期不晚于结束日期
            if start_dt > end_dt:
                logger.warning(f"Start date {start_date} is after end date {end_date}, swapping dates")
                start_date, end_date = end_date, start_date
                
        except Exception as e:
            logger.error(f"Date validation failed: {e}")
            # 使用默认日期
            if start_date is None:
                start_date = (current_date - timedelta(days=365*Config.DOWNLOAD_YEARS)).strftime('%Y%m%d')
            if end_date is None:
                end_date = current_date.strftime('%Y%m%d')
        
        save_path = Config.RAW_DATA_DIR / f"{ts_code.replace('.', '_')}.parquet"
        
        # 检查本地缓存
        cached_df = None
        if save_path.exists():
            try:
                # logger.info(f"Loading cached data for {ts_code}")
                cached_df = pd.read_parquet(save_path)
                if not cached_df.empty:
                    # 检查缓存数据的时间范围
                    min_cache_date = cached_df['trade_date'].min()
                    max_cache_date = cached_df['trade_date'].max()
                    
                    # 如果缓存数据覆盖了请求的时间范围，直接返回缓存数据
                    if min_cache_date <= start_date and max_cache_date >= end_date:
                        # logger.info(f"Cached data covers requested range, using cache for {ts_code}")
                        return cached_df
                    # 如果缓存数据部分覆盖，只下载缺失的部分
                    elif min_cache_date <= end_date and max_cache_date >= start_date:
                        # logger.info(f"Cached data partially covers requested range, downloading missing data for {ts_code}")
                        # 计算需要下载的日期范围
                        if start_date < min_cache_date:
                            start_date = start_date
                        else:
                            start_date = max_cache_date
            except Exception as e:
                logger.warning(f"Failed to load cached data for {ts_code}: {e}")
        
        # Yahoo Finance 逻辑
        # logger.info(f"Downloading data for {ts_code} via Yahoo: {start_date} to {end_date}")
        
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                # 信号量控制 (可选)
                # with self._api_semaphore:
                
                # 为雅虎财经准备正确的股票代码格式
                # Expect ts_code to be XXXXXX.SH or XXXXXX.SZ
                # Yahoo format: XXXXXX.SS (Shanghai), XXXXXX.SZ (Shenzhen)
                
                parts = ts_code.split('.')
                symbol_part = parts[0]
                
                # Ensure symbol is pure digits
                import re
                symbol_part_digits = re.sub(r'\D', '', symbol_part)
                
                # Determine suffix for Yahoo
                yahoo_suffix = ""
                if len(parts) > 1:
                    if parts[1] == 'SH':
                        yahoo_suffix = ".SS"
                    elif parts[1] == 'SZ':
                        yahoo_suffix = ".SZ"
                
                # Fallback if suffix missing or unknown, infer from code
                if not yahoo_suffix:
                    if symbol_part_digits.startswith('6'):
                        yahoo_suffix = ".SS"
                    else:
                        yahoo_suffix = ".SZ" # Default to SZ for 0xxxxx, 3xxxxx
                
                yahoo_symbol = f"{symbol_part_digits}{yahoo_suffix}"
                
                # 只使用雅虎财经作为唯一数据源
                data_sources = [
                    {
                        'name': 'Yahoo Finance',
                        'func': lambda: self._get_yahoo_stock_data(yahoo_symbol, start_date, end_date)
                    }
                ]
                
                df_hist = None
                for source in data_sources:
                    try:
                        # logger.info(f"Trying {source['name']} API for {ts_code} (Yahoo Symbol: {yahoo_symbol})")
                        df_hist = source['func']()
                        if df_hist is not None and not df_hist.empty:
                            break # Successfully got data from this source
                    except Exception as e:
                        logger.warning(f"Error fetching data from {source['name']} for {ts_code}: {e}")
                
                if df_hist is None or df_hist.empty:
                    logger.warning(f"Yahoo Finance returned empty data for {ts_code}")
                    # 尝试使用缓存数据
                    if cached_df is not None:
                        return cached_df
                    return None

                # 补充字段
                df_hist['ts_code'] = ts_code
                
                # 填充缺失值
                df_hist = df_hist.fillna(0)
                
                # 合并缓存数据和新下载的数据
                if cached_df is not None:
                    # logger.info(f"Merging cached data with new data for {ts_code}")
                    # 合并数据
                    combined_df = pd.concat([cached_df, df_hist]).drop_duplicates(subset='trade_date', keep='last')
                    # 排序
                    combined_df = combined_df.sort_values('trade_date').reset_index(drop=True)
                    df_hist = combined_df
                
                # 排序
                df_hist = df_hist.sort_values('trade_date').reset_index(drop=True)
                
                # 保存到本地缓存
                try:
                    self._atomic_write(df_hist, save_path)
                    # logger.info(f"Saved data for {ts_code} to cache")
                except Exception as e:
                    logger.warning(f"Failed to save data to cache: {e}")
                
                return df_hist
                    
            except Exception as e:
                logger.error(f"{ts_code}: Download failed (attempt {retry+1}/{max_retries}) - {e}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        # 失败时尝试返回缓存
        if cached_df is not None:
            return cached_df
            
        return None

    def download_index_data(self, index_code: str = None) -> Optional[pd.DataFrame]:
        """下载指数数据 (使用 Yahoo Finance)"""
        if index_code is None:
            index_code = Config.INDEX_CODE  # e.g., 000905.SH
            
        symbol = index_code.split('.')[0]
        logger.info(f"下载指数数据: {index_code}...")
        
        try:
            # 为 Yahoo Finance 准备指数代码
            # 沪深300: 000300.SH -> 000300.SS
            # 中证500: 000905.SH -> 000905.SS
            # 创业板指: 399006.SZ -> 399006.SZ
            
            if index_code.endswith('.SH'):
                yahoo_symbol = f"{symbol}.SS"
            elif index_code.endswith('.SZ'):
                yahoo_symbol = f"{symbol}.SZ"
            else:
                yahoo_symbol = symbol
            
            logger.info(f"使用 Yahoo Finance 下载指数: {yahoo_symbol}")
            
            # 使用 yfinance 获取指数数据
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period="max")
            
            if df is not None and not df.empty:
                # 重置索引，将Date作为列
                df = df.reset_index()
                
                # 重命名列以匹配Tushare格式
                rename_map = {
                    'Date': 'trade_date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'vol'
                }
                df = df.rename(columns=rename_map)
                
                # 转换日期格式为YYYYMMDD
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
                
                # 计算涨跌幅
                if 'close' in df.columns:
                    df['pct_chg'] = df['close'].pct_change() * 100
                    df['pct_chg'] = df['pct_chg'].fillna(0)
                
                # 计算成交额
                if 'vol' in df.columns and 'close' in df.columns:
                    df['amount'] = df['vol'] * df['close']
                
                save_path = Config.RAW_DATA_DIR / f"index_{index_code.replace('.', '_')}.parquet"
                self._atomic_write(df, save_path)
                logger.info(f"指数 {index_code} 数据已保存，共 {len(df)} 条记录")
                return df
                
        except Exception as e:
            logger.error(f"下载指数 {index_code} 失败: {e}")
            return None
        return None

    def _validate_and_convert_date(self, date_str: str) -> str:
        """
        验证并转换日期格式
        
        Args:
            date_str: 日期字符串，支持 YYYYMMDD、YYYY-MM-DD 等格式
            
        Returns:
            转换后的日期字符串，YYYY-MM-DD 格式
        """
        try:
            # 处理不同格式的日期字符串
            if isinstance(date_str, str):
                # 移除可能的分隔符
                date_str = date_str.replace('-', '').replace('/', '').replace('.', '')
                
                # 验证日期格式
                if len(date_str) != 8 or not date_str.isdigit():
                    raise ValueError(f"Invalid date format: {date_str}")
                
                # 提取年月日
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:]
                
                # 验证日期有效性
                from datetime import datetime
                valid_date = datetime(int(year), int(month), int(day))
                
                # 验证日期范围（不早于1990年，不晚于当前日期）
                min_date = datetime(1990, 1, 1)
                max_date = datetime.now()
                if valid_date < min_date:
                    logger.warning(f"Date {date_str} is too early, using minimum date: 1990-01-01")
                    return "1990-01-01"
                elif valid_date > max_date:
                    logger.warning(f"Date {date_str} is in the future, using current date")
                    return max_date.strftime("%Y-%m-%d")
                
                return f"{year}-{month}-{day}"
            else:
                raise ValueError(f"Invalid date type: {type(date_str)}")
        except Exception as e:
            logger.error(f"Date validation failed: {e}")
            # 使用默认日期
            return "2020-01-01"
    
    def _get_yahoo_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从雅虎财经获取股票数据"""
        logger.info(f"Fetching data from Yahoo Finance for {symbol}")
        
        # 验证并转换日期格式
        start = self._validate_and_convert_date(start_date)
        end = self._validate_and_convert_date(end_date)
        
        logger.info(f"Date range: {start} to {end}")
        
        # 使用yfinance获取数据
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        
        if df.empty:
            logger.warning(f"Yahoo Finance returned empty data for {symbol}")
            return df
        
        # 重置索引，将Date作为列
        df = df.reset_index()
        
        # 重命名列以匹配Tushare格式
        rename_map = {
            'Date': 'trade_date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'vol',
            'Dividends': 'dividend',
            'Stock Splits': 'split'
        }
        df = df.rename(columns=rename_map)
        
        # 转换日期格式为YYYYMMDD
        try:
            # 确保 trade_date 列存在
            if 'trade_date' not in df.columns:
                logger.warning("No trade_date column found, generating default dates")
                # 生成默认日期
                from datetime import datetime, timedelta
                dates = [datetime.now() - timedelta(days=i) for i in range(len(df))]
                df['trade_date'] = [date.strftime('%Y%m%d') for date in dates]
            else:
                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
                
                # 处理转换失败的日期
                if df['trade_date'].isnull().any():
                    null_count = df['trade_date'].isnull().sum()
                    logger.warning(f"Found {null_count} null trade_date values, replacing with default dates")
                    # 生成默认日期序列
                    from datetime import datetime, timedelta
                    default_dates = [datetime.now() - timedelta(days=i) for i in range(len(df))]
                    df['trade_date'] = df['trade_date'].fillna(pd.Series(default_dates))
                
                # 转换为 YYYYMMDD 格式
                df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
        except Exception as e:
            logger.error(f"Failed to convert trade_date: {e}")
            # 生成默认日期
            from datetime import datetime, timedelta
            dates = [datetime.now() - timedelta(days=i) for i in range(len(df))]
            df['trade_date'] = [date.strftime('%Y%m%d') for date in dates]
        
        # 验证日期范围
        if 'trade_date' in df.columns:
            min_date = df['trade_date'].min()
            max_date = df['trade_date'].max()
            logger.info(f"Data date range: {min_date} to {max_date}")
            
            # 验证日期格式一致性
            invalid_dates = []
            for date_str in df['trade_date']:
                if len(date_str) != 8 or not date_str.isdigit():
                    invalid_dates.append(date_str)
            
            if invalid_dates:
                logger.warning(f"Found {len(invalid_dates)} invalid trade_date formats, replacing with default dates")
                # 重新生成默认日期
                from datetime import datetime, timedelta
                dates = [datetime.now() - timedelta(days=i) for i in range(len(df))]
                df['trade_date'] = [date.strftime('%Y%m%d') for date in dates]
        
        # 计算涨跌幅
        if 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0)
        
        # 计算成交额
        if 'vol' in df.columns and 'close' in df.columns:
            df['amount'] = df['vol'] * df['close']
        
        logger.info(f"Successfully fetched {len(df)} rows from Yahoo Finance for {symbol}")
        return df

    def _download_stock_task(self, ts_code: str, start_date: str = None) -> Tuple[str, Optional[pd.DataFrame]]:
        """并发任务包装"""
        if self._stop_flag:
            return ts_code, None
        try:
            # AkShare 内部可能有并发限制，增加一点随机延时避免瞬间并发过高
            time.sleep(0.01) 
            df = self.download_stock_data(ts_code, start_date=start_date)
            return ts_code, df
        except Exception as e:
            return ts_code, None

    def download_with_thread_pool(self, stock_list: List[str], force_update: bool = False, start_date: str = None, end_date: str = None) -> Dict[str, int]:
        """并发下载 (使用 Yahoo Finance)"""
        logger.info(f"Starting Yahoo Finance concurrent download for {len(stock_list)} stocks...")
        
        success = 0
        failed = 0
        skipped = 0
        
        # 准备任务
        tasks_args = []
        for ts_code in stock_list:
            if self._stop_flag: break
            
            # 严格跳过已缓存逻辑
            save_path = Config.RAW_DATA_DIR / f"{ts_code.replace('.', '_')}.parquet"
            if not force_update and save_path.exists():
                try:
                     # 简单检查文件是否有效（非空）
                     # 如果文件存在且非空，则通过
                    if save_path.stat().st_size > 1000: # 假设 > 1KB 为有效文件
                         skipped += 1
                         # logger.info(f"Skipping {ts_code}, already in cache")
                         continue
                except Exception:
                    pass
            
            # 如果文件不存在，或者强制更新，则加入任务
            tasks_args.append((ts_code, start_date, end_date))
            
        total_tasks = len(tasks_args) + skipped
        logger.info(f"Total stocks: {len(stock_list)}, Skipped: {skipped}, To Download: {len(tasks_args)}")
        
        # 如果没有任务，直接返回
        if not tasks_args:
             logger.info("All stocks skipped (already cached).")
             return {"success": success, "failed": failed, "skipped": skipped}
            
        total_tasks = len(tasks_args)
        
        # Yahoo 可以并发，设置 4-8 个 workers
        max_workers = 8 
        logger.info(f"Using ThreadPoolExecutor with {max_workers} workers")
        
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(self._download_stock_task, code, s, e): code 
                for (code, s, e) in tasks_args
            }
            
            for future in as_completed(future_to_stock):
                if self._stop_flag:
                    # 取消剩余任务? Executor比较难取消，只能break loop
                    break
                
                stock_code = future_to_stock[future]
                processed_count += 1
                
                try:
                    res_code, df = future.result()
                    if df is not None and not df.empty:
                        success += 1
                        status_msg = "Success"
                    else:
                        failed += 1
                        status_msg = "Failed/Empty"
                except Exception as e:
                    failed += 1
                    status_msg = f"Error: {str(e)}"
                
                # 进度回调
                if self.progress_callback:
                    percentage = int((processed_count / total_tasks) * 100)
                    msg = f"[{processed_count}/{total_tasks}] {stock_code}: {status_msg}"
                    self.progress_callback(percentage, msg)
                    
                # 简单的日志采样，避免刷屏
                if processed_count % 10 == 0:
                    logger.info(f"Progress: {processed_count}/{total_tasks} - Success: {success}, Failed: {failed}")

        return {"success": success, "failed": failed}

    def _download_stock_task(self, ts_code: str, start_date: str = None, end_date: str = None) -> Tuple[str, Optional[pd.DataFrame]]:
        """并发任务包装"""
        if self._stop_flag:
            return ts_code, None
        try:
            # 轻微随机延迟，避免完全同步触发风控
            import random
            time.sleep(random.uniform(0.1, 0.5))
            df = self.download_stock_data(ts_code, start_date=start_date, end_date=end_date)
            return ts_code, df
        except Exception as e:
            return ts_code, None

        return {"success": success, "failed": failed}

    def download_all(self, stock_list=None, force_update=False, start_date=None, end_date=None):
        """下载所有"""
        if stock_list is None:
            df = self.get_main_board_stocks()
            if df is None or df.empty:
                logger.error("Failed to get stock list, cannot proceed with download")
                return
            stock_list = df['ts_code'].tolist()
        if not stock_list:
            logger.error("Empty stock list, cannot proceed with download")
            return
        
        # 过滤出需要下载的股票（缓存中没有的）
        stocks_to_download = []
        logger.info("Checking existing files to skip (Resume Mode)...")
        
        for ts_code in stock_list:
            save_path = Config.RAW_DATA_DIR / f"{ts_code.replace('.', '_')}.parquet"
            should_download = True
            
            if not force_update and save_path.exists():
                try:
                    # Check file size to ensure it's not empty/corrupt (e.g. > 1KB)
                    # reading parquet is too slow for 3000+ files just to check existence
                    if save_path.stat().st_size > 1000:
                        should_download = False
                except Exception:
                    pass
            
            if should_download:
                stocks_to_download.append(ts_code)

        logger.info(f"Resume check complete. Found {len(stocks_to_download)} stocks missing or needing update out of {len(stock_list)}.")
        
        if not stocks_to_download:
            logger.info("All stocks are already in cache, no need to download")
            return
        
        logger.info(f"Starting download for {len(stocks_to_download)} stocks")
        self.download_with_thread_pool(stocks_to_download, force_update, start_date, end_date)

    def stop(self):
        self._stop_flag = True

    def reset(self):
        self._stop_flag = False

# 兼容性命名
TushareDownloader = YahooDownloader
