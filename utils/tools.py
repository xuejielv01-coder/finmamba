# -*- coding: utf-8 -*-
"""
工具函数集合
"""

import time
import functools
from typing import Callable, Any
from datetime import datetime, timedelta
import pandas as pd


def rate_limit(interval: float = 0.3):
    """
    API 请求限流装饰器
    
    Args:
        interval: 两次请求之间的最小间隔(秒)
    """
    def decorator(func: Callable) -> Callable:
        last_call = [0.0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_call[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result
        
        return wrapper
    return decorator


def retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟(秒)
        backoff: 延迟增长倍数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator


def timer(func: Callable) -> Callable:
    """
    计时装饰器
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[Timer] {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def get_trade_dates(start_date: str, end_date: str) -> list:
    """
    获取交易日列表 (简化版，实际应从交易所日历获取)
    
    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
    
    Returns:
        交易日列表
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    return [d.strftime('%Y%m%d') for d in dates]


def is_main_board(ts_code: str) -> bool:
    """
    判断是否为主板股票
    
    Args:
        ts_code: 股票代码 (如 600000.SH)
    
    Returns:
        是否为主板股票
    """
    code = ts_code[:2]
    return code in ('60', '00')


def is_st_stock(name: str) -> bool:
    """
    判断是否为 ST 股票
    
    Args:
        name: 股票名称
    
    Returns:
        是否为 ST 股票
    """
    if name is None:
        return False
    return 'ST' in name.upper() or '*ST' in name.upper()


def format_number(value: float, precision: int = 2) -> str:
    """
    格式化数字显示
    
    Args:
        value: 数值
        precision: 小数位数
    
    Returns:
        格式化后的字符串
    """
    if abs(value) >= 1e8:
        return f"{value/1e8:.{precision}f}亿"
    elif abs(value) >= 1e4:
        return f"{value/1e4:.{precision}f}万"
    else:
        return f"{value:.{precision}f}"


def calculate_return(price_series: pd.Series, periods: int = 1) -> pd.Series:
    """
    计算收益率
    
    Args:
        price_series: 价格序列
        periods: 收益计算周期
    
    Returns:
        收益率序列
    """
    return price_series.pct_change(periods=periods)


def robust_zscore(x: pd.Series) -> pd.Series:
    """
    稳健 Z-Score 标准化 (使用 Median 和 MAD)
    
    Args:
        x: 输入序列
    
    Returns:
        标准化后的序列
    """
    median = x.median()
    mad = (x - median).abs().median()
    # 避免除零
    mad = max(mad, 1e-8)
    return (x - median) / (1.4826 * mad)
