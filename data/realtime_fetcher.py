# -*- coding: utf-8 -*-
"""
实时行情获取器 (RealtimeFetcher) - AkShare版
使用 AkShare 获取盘中实时行情
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import akshare as ak

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("RealtimeFetcher")


class RealtimeFetcher:
    """
    实时行情获取器 (AkShare版)
    """
    
    # 交易时段
    MORNING_START = (9, 30)
    MORNING_END = (11, 30)
    AFTERNOON_START = (13, 0)
    AFTERNOON_END = (15, 0)
    
    def __init__(self, token: str = None):
        """初始化"""
        # AkShare 不需要 token
        self._realtime_cache: Dict[str, dict] = {}
        logger.info("RealtimeFetcher (AkShare) initialized")
    
    def is_trading_hours(self) -> bool:
        """判断当前是否为交易时间"""
        now = datetime.now()
        current_time = (now.hour, now.minute)
        
        # 简单判断周末
        if now.weekday() >= 5:
            return False
            
        # 上午交易时段
        if self.MORNING_START <= current_time <= self.MORNING_END:
            return True
        
        # 下午交易时段
        if self.AFTERNOON_START <= current_time <= self.AFTERNOON_END:
            return True
        
        return False
    
    def is_trading_day(self, date: str = None) -> bool:
        """判断是否为交易日"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
            
        try:
            date_dt = datetime.strptime(date, '%Y%m%d')
            # 简单判断周末
            if date_dt.weekday() >= 5:
                return False
            
            # 使用 akshare 工具接口 (可选，较慢则跳过)
            # tool_trade_date_hist_sina 返回所有交易日
            # 简化策略: 默认周一至周五是交易日 (忽略节假日以提高速度，或者维护一个本地日历)
            return True
        except Exception:
            return True
    
    def get_realtime_quotes(self, ts_codes: List[str] = None) -> pd.DataFrame:
        """获取实时行情"""
        try:
            # ak.stock_zh_a_spot_em() 获取所有A股实时行情
            # 这比 Tushare 的 batch 接口可能更重，但包含了所有数据
            df = ak.stock_zh_a_spot_em()
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 重命名列
            # 序号, 代码, 名称, 最新价, 涨跌幅, 涨跌额, ...
            rename_map = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'pct_chg',
                '成交量': 'volume',
                '成交额': 'amount',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'pre_close',
            }
            df = df.rename(columns=rename_map)
            
            # 构造 ts_code
            def get_ts_code(code):
                c = str(code)
                if c.startswith('6'):
                    return f"{c}.SH"
                elif c.startswith('0') or c.startswith('3'):
                    return f"{c}.SZ"
                return f"{c}.BJ"
            
            df['ts_code'] = df['symbol'].apply(get_ts_code)
            
            # 如果指定了代码，则过滤
            if ts_codes:
                df = df[df['ts_code'].isin(ts_codes)]
            
            return df
                
        except Exception as e:
            logger.error(f"Failed to get realtime quotes: {e}")
            return pd.DataFrame()
    
    def get_today_data(self, ts_code: str, history_df: pd.DataFrame = None) -> pd.DataFrame:
        """智能获取当日数据"""
        today = datetime.now().strftime('%Y%m%d')
        
        # 无论盘中盘后，都尝试获取实时/最新行情并合并
        # AkShare 的 stock_zh_a_hist 通常包含当天数据（收盘后更新）
        # 盘中则用 stock_zh_a_spot_em
        
        if self.is_trading_hours():
            logger.info(f"[{ts_code}] 盘中模式: 获取实时行情")
            realtime = self._get_single_realtime(ts_code)
            if realtime:
                today_row = self._convert_realtime_to_daily(realtime, ts_code, today)
                return self._merge_with_history(history_df, today_row)
        else:
            # 盘后尝试获取历史数据（可能已更新）
            logger.info(f"[{ts_code}] 盘后模式: 获取最新日线")
            try:
                # 获取最近数据
                start_dt = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
                end_dt = today
                symbol = ts_code.split('.')[0]
                
                df_hist = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start_dt, end_date=end_dt, adjust='qfq')
                
                if df_hist is not None and not df_hist.empty:
                    # 检查是否有今天的
                    # AkShare return date as string '2023-01-01' usually
                    last_date = df_hist.iloc[-1]['日期']
                    if last_date.replace('-', '') == today:
                        # 转换并合并
                        renamed = df_hist.rename(columns={
                            '日期': 'trade_date', '开盘': 'open', '收盘': 'close', 
                            '最高': 'high', '最低': 'low', '成交量': 'vol', 
                            '成交额': 'amount', '涨跌幅': 'pct_chg'
                        })
                        renamed['trade_date'] = renamed['trade_date'].astype(str).str.replace('-', '')
                        return self._merge_with_history(history_df, renamed.iloc[-1].to_dict())
            except Exception as e:
                logger.warning(f"Failed to get daily data: {e}")

            # 如果取不到历史的今天数据（可能没收盘更新），退化为实时
            realtime = self._get_single_realtime(ts_code)
            if realtime:
                today_row = self._convert_realtime_to_daily(realtime, ts_code, today)
                return self._merge_with_history(history_df, today_row)
        
        return history_df
    
    def _get_single_realtime(self, ts_code: str) -> Optional[dict]:
        """获取单只股票实时"""
        try:
            # 效率较低，但简单: 获取全市场过滤
            # 优化: AkShare 没有单只股票的实时接口 (除了sina接口 ak.stock_zh_a_spot_em 是一次性获取)
            # ak.stock_bid_ask_em(symbol="000001") 可以获取买卖盘，包含价格
            
            symbol = ts_code.split('.')[0]
            # 尝试使用个股接口
            df = ak.stock_bid_ask_em(symbol=symbol)
            # columns: item, value
            # item: sell_5, sell_4... latest_price? No, it's order book.
            
            # fallback: stock_zh_a_spot_em 过滤
            # 或者使用 sina 接口 ak.stock_zh_a_spot()
            
            # 使用全市场接口 (注意频次控制)
            df_all = self.get_realtime_quotes([ts_code])
            if not df_all.empty:
                return df_all.iloc[0].to_dict()
            return None
        except Exception:
            return None

    def _convert_realtime_to_daily(self, realtime: dict, ts_code: str, trade_date: str) -> dict:
        """将实时行情转换为日线格式"""
        return {
            'ts_code': ts_code,
            'trade_date': trade_date,
            'open': float(realtime.get('open', 0)),
            'high': float(realtime.get('high', 0)),
            'low': float(realtime.get('low', 0)),
            'close': float(realtime.get('price', 0)), # price is current
            'pre_close': float(realtime.get('pre_close', 0)),
            'pct_chg': float(realtime.get('pct_chg', 0)),
            'vol': float(realtime.get('volume', 0)),
            'amount': float(realtime.get('amount', 0)),
        }

    def _merge_with_history(self, history_df: pd.DataFrame, today_data: dict) -> pd.DataFrame:
        """合并历史数据与当日数据"""
        if history_df is None or history_df.empty:
            return pd.DataFrame([today_data])
        
        history_df = history_df.copy()
        if 'trade_date' in history_df.columns:
            # 统一日期格式 YYYYMMDD
            history_df['trade_date'] = history_df['trade_date'].astype(str).str.replace('-', '').str[:8]
        
        today_date = str(today_data.get('trade_date', ''))
        
        # 移除历史中的今日数据
        history_df = history_df[history_df['trade_date'] != today_date]
        
        # 添加今日数据
        today_df = pd.DataFrame([today_data])
        result = pd.concat([history_df, today_df], ignore_index=True)
        result = result.sort_values('trade_date').reset_index(drop=True)
        
        return result
        
    def get_batch_today_data(self, ts_codes: List[str], on_progress: callable = None) -> Dict[str, pd.DataFrame]:
        results = {}
        # 一次性获取所有实时数据
        logger.info(f"Batch fetching realtime data for {len(ts_codes)} stocks...")
        realtime_df = self.get_realtime_quotes(ts_codes)
        realtime_map = {}
        if not realtime_df.empty:
            realtime_map = realtime_df.set_index('ts_code').to_dict('index')
            
        total = len(ts_codes)
        today = datetime.now().strftime('%Y%m%d')
        
        for i, ts_code in enumerate(ts_codes):
            try:
                # Load history
                filepath = Config.RAW_DATA_DIR / f"{ts_code.replace('.', '_')}.parquet"
                history = pd.DataFrame()
                if filepath.exists():
                    history = pd.read_parquet(filepath).tail(59)
                
                # Merge
                if ts_code in realtime_map:
                    row = self._convert_realtime_to_daily(realtime_map[ts_code], ts_code, today)
                    results[ts_code] = self._merge_with_history(history, row)
                else:
                    results[ts_code] = history
            except Exception:
                pass
                
            if on_progress:
                on_progress(i+1, total, ts_code)
                
        return results

# 单例
_fetcher: Optional[RealtimeFetcher] = None

def get_realtime_fetcher() -> RealtimeFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = RealtimeFetcher()
    return _fetcher
