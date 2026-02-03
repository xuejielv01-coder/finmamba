# -*- coding: utf-8 -*-
"""
特征工程核心 (Preprocessor)
PRD 2.2 实现

特性:
- 资金流因子计算
- 技术指标 (MACD, RSI, BOLL等)
- 截面统计量缓存 (Z-Score)
- 主板严格过滤
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
except ImportError:
    ta = None

import sys
# 确保能导入项目模块
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from utils.tools import is_main_board, robust_zscore

logger = get_logger("Preprocessor")


class Preprocessor:
    """
    数据预处理器
    
    负责:
    1. 主板过滤
    2. 资金流因子计算
    3. 技术指标计算
    4. Z-Score 标准化
    5. 统计量缓存
    """
    
    # 扩展特征定义 (优化后82维)
    FEATURES = [
        # 基本价格特征 (6)
        'open_pct', 'high_pct', 'low_pct', 'close_pct',
        'intraday_ret', 'overnight_ret', 
        # 波动率与流动性 (6)
        'garman_klass_vol', 'parkinson_vol', 'amihud_illiquidity', 
        'vol_cv', 'vol_ma5_ratio', 'turnover_rate',
        # 资金流与量价交互 (12)
        'mf_net_ratio', 'retail_sentiment', 'main_force_ratio',
        'elg_ratio', 'lg_ratio', 'md_ratio', 'sm_ratio',
        'buy_sell_ratio', 'net_mf_vol_ratio', 'net_mf_amount_ratio',
        'large_order_ratio', 'small_order_ratio',
        # 经典技术指标 (20) - 精简有效
        'rsi_6', 'rsi_12', 'rsi_24',
        'macd', 'macd_signal', 'macd_hist',
        'boll_upper_ratio', 'boll_lower_ratio', 'boll_width',
        'ma5_ratio', 'ma20_ratio', 'ma60_ratio', # 移除了ma10
        'atr_ratio', 'cci', 'willr',
        'obv_ratio', 'mtm', 'roc', 'kdj_k', 'kdj_d',
        # 高级Alpha因子 (6)
        'alpha_001', 'alpha_006', 'alpha_plus_di', 'alpha_minus_di', 'alpha_adx', 'alpha_wr',
        # 估值指标 (4)
        'pe_zscore', 'pb_zscore', 'ps_zscore', 'total_mv_rank',
        # 时间特征 (4)
        'weekday', 'month', 'is_month_start', 'is_month_end',
        # 收益与动量标签类特征 (6)
        'ret_5d', 'ret_10d', 'ret_20d', 
        'volatility_20d', 'momentum_20d', 'momentum_60d',
        # 板块情绪因子 (5)
        'sector_up_ratio', 'sector_avg_ret', 'sector_momentum',
        'sector_volatility', 'sector_leader',
        # 市场状态特征 (10)
        'market_index_ret', 'market_index_ma5_ratio', 'market_index_ma20_ratio',
        'market_index_vol_ratio', 'market_index_rsi', 'market_index_macd',
        'industry_rotation_strength', 'market_breadth', 'market_updown_ratio',
        'market_volatility',
        # 冗余位填充 (3) - 凑齐 82 或保持对齐
        'extra_feat_1', 'extra_feat_2', 'extra_feat_3', 
    ]
    
    def __init__(self):
        """初始化预处理器"""
        Config.ensure_dirs()
        self.stats_cache_dir = Config.STATS_CACHE
        logger.info("Preprocessor initialized (Optimized Features)")
    
    def process_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理单日原始数据，计算所有特征
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 确保日期排序
        try:
            # 尝试转换 trade_date 列
            if 'trade_date' not in df.columns:
                logger.error("No 'trade_date' column found in DataFrame")
                return df
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            
            # 处理转换失败的日期
            null_count = df['trade_date'].isnull().sum()
            if null_count > 0:
                logger.warning(f"Found {null_count} null trade_date values, removing those rows")
                df = df.dropna(subset=['trade_date'])
            
            # 确保日期排序
            if not df.empty:
                df = df.sort_values('trade_date').reset_index(drop=True)
                logger.debug(f"Data sorted by trade_date: {df['trade_date'].min()} to {df['trade_date'].max()}")
        except Exception as e:
            logger.error(f"Error processing trade_date: {e}")
            # 尝试恢复，使用默认日期
            from datetime import datetime, timedelta
            if not df.empty:
                dates = [datetime.now() - timedelta(days=i) for i in range(len(df))]
                df['trade_date'] = dates
                df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 1. 计算价格基础特征
        df = self._calculate_price_features(df)
        
        # 2. 计算高级波动率与流动性
        df = self._calculate_volatility_liquidity(df)
        
        # 3. 计算资金流因子 (保持不变)
        df = self._calculate_moneyflow_features(df)
        
        # 4. 计算技术指标 (优化版)
        df = self._calculate_technical_indicators(df)
        
        # 5. 计算估值指标
        df = self._calculate_valuation_features(df)
        
        # 6. 计算时间特征
        df = self._calculate_time_features(df)
        
        # 7. 计算收益特征
        df = self._calculate_return_features(df)
        
        # 8. 计算高级Alpha因子
        df = self._calculate_advanced_alphas(df)
        
        # 9. 填充保留位
        for col in ['extra_feat_1', 'extra_feat_2', 'extra_feat_3']:
            df[col] = 0.0
            
        # 10. 清洗数据
        df = self._clean_data(df)
        
        return df
    
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格相关特征 (增强版)"""
        prev_close = df['close'].shift(1)
        
        # 基础涨跌幅
        df['open_pct'] = (df['open'] / prev_close - 1) * 100
        df['high_pct'] = (df['high'] / prev_close - 1) * 100
        df['low_pct'] = (df['low'] / prev_close - 1) * 100
        df['close_pct'] = (df['close'] / prev_close - 1) * 100
        
        # 日内与隔夜收益
        # Intraday: Close vs Open
        df['intraday_ret'] = (df['close'] / df['open'] - 1) * 100
        # Overnight: Open vs Prev Close
        df['overnight_ret'] = (df['open'] / prev_close - 1) * 100
        
        return df

    def _calculate_volatility_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算高级波动率与流动性特征"""
        # Garman-Klass Volatility: 0.5 * ln(H/L)^2 - (2ln2 - 1) * ln(C/O)^2
        # Use log quantities roughly: (H/L - 1)^2 ...
        # Standard definition:
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        df['garman_klass_vol'] = np.sqrt(np.maximum(0, gk_var)) * 100
        
        # Parkinson Volatility: sqrt(1/(4ln2)) * ln(H/L)
        df['parkinson_vol'] = np.sqrt(1 / (4 * np.log(2))) * log_hl * 100
        
        # Amihud Illiquidity: |Ret| / (Volume * Price)
        # Using daily dollar volume proxy: amount
        if 'amount' not in df.columns:
            df['amount'] = df['vol'] * df['close']
        
        abs_ret = df['close'].pct_change().abs()
        df['amihud_illiquidity'] = (abs_ret / (df['amount'].replace(0, np.nan) + 1e-8)) * 1e8 # Scale up
        
        # Volume Variability (CV: Std/Mean)
        vol_ma20 = df['vol'].rolling(20).mean()
        vol_std20 = df['vol'].rolling(20).std()
        df['vol_cv'] = vol_std20 / (vol_ma20 + 1e-8)
        
        # Basic Volume Ratio
        df['vol_ma5_ratio'] = df['vol'] / (df['vol'].rolling(5).mean() + 1e-8)
        
        return df
    
    def _calculate_moneyflow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算资金流因子 (PRD 2.2)
        
        核心因子:
        1. mf_net_ratio: 净资金流比率
        2. retail_sentiment: 散户情绪
        3. main_force_ratio: 主力净流入占比
        """
        # 检查是否有资金流数据
        has_mf = 'buy_elg_amount' in df.columns
        
        if has_mf:
            # 确保有金额数据
            df['amount'] = df.get('amount', df['vol'] * df['close'])
            
            # 1. 净资金流比率
            buy_total = df.get('buy_elg_amount', 0) + df.get('buy_lg_amount', 0) + \
                       df.get('buy_md_amount', 0) + df.get('buy_sm_amount', 0)
            sell_total = df.get('sell_elg_amount', 0) + df.get('sell_lg_amount', 0) + \
                        df.get('sell_md_amount', 0) + df.get('sell_sm_amount', 0)
            df['mf_net_ratio'] = (buy_total - sell_total) / (df['amount'] + 1e-8)
            
            # 2. 散户情绪 (小单)
            df['retail_sentiment'] = (df.get('buy_sm_amount', 0) - df.get('sell_sm_amount', 0)) / (df['amount'] + 1e-8)
            
            # 3. 主力净流入 (超大单 + 大单)
            main_buy = df.get('buy_elg_amount', 0) + df.get('buy_lg_amount', 0)
            main_sell = df.get('sell_elg_amount', 0) + df.get('sell_lg_amount', 0)
            df['main_force_ratio'] = (main_buy - main_sell) / (df['amount'] + 1e-8)
            
            # 4. 各类资金占比
            total_amount = df['amount'] + 1e-8
            df['elg_ratio'] = (df.get('buy_elg_amount', 0) + df.get('sell_elg_amount', 0)) / total_amount
            df['lg_ratio'] = (df.get('buy_lg_amount', 0) + df.get('sell_lg_amount', 0)) / total_amount
            df['md_ratio'] = (df.get('buy_md_amount', 0) + df.get('sell_md_amount', 0)) / total_amount
            df['sm_ratio'] = (df.get('buy_sm_amount', 0) + df.get('sell_sm_amount', 0)) / total_amount
            
            # 5. 买卖比率
            df['buy_sell_ratio'] = buy_total / (sell_total + 1e-8)
            
            # 6. 净流入量/金额比率
            df['net_mf_vol_ratio'] = df.get('net_mf_vol', 0) / (df['vol'] + 1e-8)
            df['net_mf_amount_ratio'] = df.get('net_mf_amount', 0) / (df['amount'] + 1e-8)
            
            # 7. 大单小单比率
            large_order = df.get('buy_elg_vol', 0) + df.get('sell_elg_vol', 0) + \
                         df.get('buy_lg_vol', 0) + df.get('sell_lg_vol', 0)
            small_order = df.get('buy_sm_vol', 0) + df.get('sell_sm_vol', 0)
            df['large_order_ratio'] = large_order / (df['vol'] + 1e-8)
            df['small_order_ratio'] = small_order / (df['vol'] + 1e-8)
        else:
            # 无资金流数据时填充 0
            mf_cols = ['mf_net_ratio', 'retail_sentiment', 'main_force_ratio',
                      'elg_ratio', 'lg_ratio', 'md_ratio', 'sm_ratio',
                      'buy_sell_ratio', 'net_mf_vol_ratio', 'net_mf_amount_ratio',
                      'large_order_ratio', 'small_order_ratio']
            for col in mf_cols:
                df[col] = 0.0
        
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标 (精简版)"""
        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['vol']
        
        # RSI
        df['rsi_6'] = self._calculate_rsi(close, 6)
        df['rsi_12'] = self._calculate_rsi(close, 12)
        df['rsi_24'] = self._calculate_rsi(close, 24)
        
        # MACD
        macd_data = self._calculate_macd(close)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['hist']
        
        # Bollinger Bands
        boll_data = self._calculate_bollinger(close)
        df['boll_upper_ratio'] = (boll_data['upper'] / close - 1) * 100
        df['boll_lower_ratio'] = (boll_data['lower'] / close - 1) * 100
        df['boll_width'] = (boll_data['upper'] - boll_data['lower']) / close * 100
        
        # MA Ratios (Mean Reversion)
        df['ma5_ratio'] = (close / close.rolling(5).mean() - 1) * 100
        df['ma20_ratio'] = (close / close.rolling(20).mean() - 1) * 100
        df['ma60_ratio'] = (close / close.rolling(60).mean() - 1) * 100
        
        # ATR
        atr = self._calculate_atr(high, low, close)
        df['atr_ratio'] = atr / close * 100
        
        # CCI
        df['cci'] = self._calculate_cci(high, low, close)
        
        # Williams %R
        df['willr'] = self._calculate_willr(high, low, close)
        
        # OBV
        # Using accumulated OBV ratio to MA
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        df['obv_ratio'] = obv / (obv.rolling(20).mean() + 1e-8)
        
        # Momentum & ROC
        df['mtm'] = close - close.shift(10)
        df['roc'] = close.pct_change(10) * 100
        
        # KDJ
        kdj = self._calculate_kdj(high, low, close)
        df['kdj_k'] = kdj['k']
        df['kdj_d'] = kdj['d']
        
        return df

    def _calculate_advanced_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算常见的高效 Alpha 因子"""
        high = df['high']
        low = df['low']
        close = df['close']
        open_ = df['open']
        vol = df['vol']
        
        # Alpha 001 (Simplified): (close - close[1]) signed * vol
        # Rank of Ts_ArgMax(SignedPower(...)) is too complex for simple pandas, use approx.
        # Proxy: Momentum * Volume Interaction
        df['alpha_001'] = (close.pct_change() * np.log(vol + 1)).rolling(6).mean() 
        
        # Alpha 006 (GTJA style): -1 * Correlation(Open, Volume, 10)
        # Sign of open-price changes vs volume changes
        try:
             df['alpha_006'] = open_.rolling(10).corr(vol) * -1
        except:
             df['alpha_006'] = 0.0
             
        # ADX-like components (Directional Movement)
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        tr = self._calculate_atr(high, low, close, 1).fillna(0) # True Range
        
        smooth_pd = pd.Series(plus_dm).rolling(14).mean()
        smooth_md = pd.Series(minus_dm).rolling(14).mean()
        smooth_tr = tr.rolling(14).mean()
        
        df['alpha_plus_di'] = 100 * smooth_pd / (smooth_tr + 1e-8)
        df['alpha_minus_di'] = 100 * smooth_md / (smooth_tr + 1e-8)
        
        dx = 100 * np.abs(df['alpha_plus_di'] - df['alpha_minus_di']) / (df['alpha_plus_di'] + df['alpha_minus_di'] + 1e-8)
        df['alpha_adx'] = dx.rolling(14).mean()
        
        # Alpha WR: Williams R-like but smoothed
        period = 14
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        df['alpha_wr'] = -100 * (hh - close) / (hh - ll + 1e-8)
        
        return df
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """计算 RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, close: pd.Series) -> Dict:
        """计算 MACD"""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return {'macd': macd, 'signal': signal, 'hist': hist}
    
    def _calculate_bollinger(self, close: pd.Series, period: int = 20) -> Dict:
        """计算 Bollinger Bands"""
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return {'ma': ma, 'upper': upper, 'lower': lower}
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算 ATR"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """计算 CCI"""
        tp = (high + low + close) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * md + 1e-8)
        return cci
    
    def _calculate_willr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算 Williams %R"""
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        willr = -100 * (hh - close) / (hh - ll + 1e-8)
        return willr
    
    def _calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> Dict:
        """计算 KDJ 指标"""
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        
        # 计算 RSV
        rsv = (close - ll) / (hh - ll + 1e-8) * 100
        
        # 计算 K、D 值
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        
        return {'k': k, 'd': d}
    
    def _calculate_valuation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算估值相关特征"""
        # PE/PB/PS Z-Score (截面标准化，这里先计算原值，后续做截面标准化)
        df['pe_zscore'] = df.get('pe', 0)
        df['pb_zscore'] = df.get('pb', 0)
        df['ps_zscore'] = df.get('ps', 0)
        
        # 换手率和量比
        df['turnover_rate'] = df.get('turnover_rate', 0)
        df['volume_ratio'] = df.get('volume_ratio', 1)
        
        # 市值排名 (百分位)
        df['total_mv_rank'] = df.get('total_mv', 0)
        
        return df
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算时间特征"""
        try:
            # 确保 trade_date 是 datetime 类型
            if 'trade_date' not in df.columns:
                logger.error("No 'trade_date' column found for time features calculation")
                # 填充默认值
                df['weekday'] = 0.0
                df['month'] = 0.0
                df['is_month_start'] = 0.0
                df['is_month_end'] = 0.0
                return df
            
            # 转换为 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
                
                # 处理转换失败的日期
                null_count = df['trade_date'].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Found {null_count} null trade_date values in time features calculation")
            
            # 计算时间特征
            df['weekday'] = df['trade_date'].dt.dayofweek / 4.0  # 归一化到 [0, 1]
            df['month'] = (df['trade_date'].dt.month - 1) / 11.0  # 归一化到 [0, 1]
            df['is_month_start'] = df['trade_date'].dt.is_month_start.astype(float)
            df['is_month_end'] = df['trade_date'].dt.is_month_end.astype(float)
            
            # 处理可能的 NaN 值
            time_features = ['weekday', 'month', 'is_month_start', 'is_month_end']
            for feature in time_features:
                df[feature] = df[feature].fillna(0.0)
                
        except Exception as e:
            logger.error(f"Error calculating time features: {e}")
            # 填充默认值
            df['weekday'] = 0.0
            df['month'] = 0.0
            df['is_month_start'] = 0.0
            df['is_month_end'] = 0.0
        
        return df
    
    def _calculate_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算收益特征 (用于标签) - 多期限"""
        close = df['close']
        
        # 收益特征
        df['ret_1d'] = close.pct_change(1) * 100
        df['ret_3d'] = close.pct_change(3) * 100   # 新增: 3日收益
        df['ret_5d'] = close.pct_change(5) * 100
        df['ret_10d'] = close.pct_change(10) * 100
        df['ret_20d'] = close.pct_change(20) * 100
        df['ret_60d'] = close.pct_change(60) * 100
        
        # 波动率特征
        df['volatility_5d'] = close.pct_change().rolling(5).std() * 100
        df['volatility_20d'] = close.pct_change().rolling(20).std() * 100
        df['volatility_60d'] = close.pct_change().rolling(60).std() * 100
        
        # 动量特征
        df['momentum_5d'] = close.pct_change(5) * 100
        df['momentum_20d'] = close.pct_change(20) * 100
        df['momentum_60d'] = close.pct_change(60) * 100
        
        return df
    
    def _calculate_market_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场状态特征
        
        由于没有实际的大盘指数数据，这里使用全市场股票的平均值来模拟市场状态特征
        """
        # 检查必要的列
        if 'trade_date' not in df.columns or 'close' not in df.columns or 'vol' not in df.columns:
            logger.warning("Missing required columns for market state features calculation")
            # 填充默认值
            market_state_cols = [
                'market_index_ret', 'market_index_ma5_ratio', 'market_index_ma20_ratio',
                'market_index_vol_ratio', 'market_index_rsi', 'market_index_macd',
                'industry_rotation_strength', 'market_breadth', 'market_updown_ratio',
                'market_volatility'
            ]
            for col in market_state_cols:
                df[col] = 0.0
            return df
        
        try:
            # 模拟大盘指数：使用所有股票的平均收盘价
            market_close = df.groupby('trade_date')['close'].mean()
            market_vol = df.groupby('trade_date')['vol'].mean()
            
            # 1. 市场指数收益
            df['market_index_ret'] = df['trade_date'].map(market_close.pct_change() * 100)
            
            # 2. 市场指数均线比率
            market_ma5 = market_close.rolling(5).mean()
            market_ma20 = market_close.rolling(20).mean()
            df['market_index_ma5_ratio'] = df['trade_date'].map((market_close / market_ma5 - 1) * 100)
            df['market_index_ma20_ratio'] = df['trade_date'].map((market_close / market_ma20 - 1) * 100)
            
            # 3. 市场指数成交量比率
            market_vol_ma20 = market_vol.rolling(20).mean()
            df['market_index_vol_ratio'] = df['trade_date'].map(market_vol / market_vol_ma20)
            
            # 4. 市场指数RSI
            delta = market_close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            market_rsi = 100 - (100 / (1 + rs))
            df['market_index_rsi'] = df['trade_date'].map(market_rsi)
            
            # 5. 市场指数MACD
            ema12 = market_close.ewm(span=12, adjust=False).mean()
            ema26 = market_close.ewm(span=26, adjust=False).mean()
            market_macd = ema12 - ema26
            df['market_index_macd'] = df['trade_date'].map(market_macd)
            
            # 6. 行业轮动强度 (模拟)
            # 使用行业收益的标准差来衡量行业轮动强度
            if 'industry' in df.columns:
                industry_ret = df.groupby(['trade_date', 'industry'])['close'].pct_change().reset_index()
                industry_rotation = industry_ret.groupby('trade_date')['close'].std() * 100
                df['industry_rotation_strength'] = df['trade_date'].map(industry_rotation)
            else:
                # 没有行业数据时，使用市场波动率代替
                df['industry_rotation_strength'] = df['trade_date'].map(market_close.pct_change().rolling(20).std() * 100)
            
            # 7. 市场广度 (模拟)
            # 使用上涨股票数与下跌股票数的比率
            daily_ret = df.groupby('trade_date')['close'].pct_change().reset_index()
            market_breadth = daily_ret.groupby('trade_date')['close'].apply(lambda x: (x > 0).sum() / max((x < 0).sum(), 1))
            df['market_breadth'] = df['trade_date'].map(market_breadth)
            
            # 8. 市场涨跌比率
            up_down_ratio = daily_ret.groupby('trade_date')['close'].apply(lambda x: (x > 0).sum() / len(x))
            df['market_updown_ratio'] = df['trade_date'].map(up_down_ratio)
            
            # 9. 市场波动率
            market_volatility = market_close.pct_change().rolling(20).std() * 100
            df['market_volatility'] = df['trade_date'].map(market_volatility)
            
        except Exception as e:
            logger.error(f"Error calculating market state features: {e}")
            # 填充默认值
            market_state_cols = [
                'market_index_ret', 'market_index_ma5_ratio', 'market_index_ma20_ratio',
                'market_index_vol_ratio', 'market_index_rsi', 'market_index_macd',
                'industry_rotation_strength', 'market_breadth', 'market_updown_ratio',
                'market_volatility'
            ]
            for col in market_state_cols:
                df[col] = 0.0
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        # 删除收盘价为空的行
        df = df.dropna(subset=['close'])
        
        # 确保所有特征列都存在并填充
        for col in self.FEATURES:
            if col not in df.columns:
                # 如果列不存在，创建并填充0
                df[col] = 0.0
            else:
                # 如果列存在，填充NaN值
                df[col] = df[col].fillna(0.0)
        
        # 异常值处理 (winsorize)
        for col in self.FEATURES:
            df[col] = df[col].clip(lower=-10, upper=10)
        
        return df
    
    def calculate_cross_sectional_stats(
        self, 
        all_stocks_df: pd.DataFrame, 
        date: str
    ) -> Dict:
        """
        计算截面统计量并缓存
        
        Args:
            all_stocks_df: 全市场数据 (某日)
            date: 日期 YYYYMMDD
        
        Returns:
            包含 median 和 mad 的字典
        """
        stats = {'date': date, 'median': [], 'mad': []}
        
        for feat in self.FEATURES:
            if feat in all_stocks_df.columns:
                col = all_stocks_df[feat]
                median = col.median()
                mad = (col - median).abs().median()
                stats['median'].append(float(median) if pd.notna(median) else 0.0)
                stats['mad'].append(float(mad) if pd.notna(mad) else 1.0)
            else:
                stats['median'].append(0.0)
                stats['mad'].append(1.0)
        
        # 保存缓存
        cache_file = self.stats_cache_dir / f"{date}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False)
        
        logger.debug(f"Stats cached for {date}")
        return stats
    
    def load_stats_cache(self, date: str) -> Optional[Dict]:
        """
        加载统计量缓存
        
        Args:
            date: 日期 YYYYMMDD
        
        Returns:
            缓存的统计量字典
        """
        cache_file = self.stats_cache_dir / f"{date}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_latest_stats_cache(self) -> Optional[Dict]:
        """
        获取最近一个可用日期的统计缓存
        
        Returns:
            最近一个可用日期的统计缓存字典
        """
        # 获取所有缓存文件
        cache_files = list(self.stats_cache_dir.glob("*.json"))
        if not cache_files:
            return None
        
        # 按文件名排序，获取最新的文件
        cache_files.sort(key=lambda x: x.name, reverse=True)
        
        # 尝试加载最近的几个缓存文件
        for cache_file in cache_files[:10]:  # 尝试前10个最近的文件
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    logger.info(f"Found recent stats cache from {stats.get('date')}")
                    return stats
            except Exception as e:
                logger.error(f"Failed to load cache file {cache_file}: {e}")
                continue
        
        return None
    
    def generate_stats_cache(self, date: str, use_prev_date=True) -> Optional[Dict]:
        """
        生成指定日期的统计缓存
        
        Args:
            date: 日期 YYYYMMDD
            use_prev_date: 如果没有当日数据，是否尝试使用前一天的数据
        
        Returns:
            生成的统计量字典
        """
        from data.downloader import TushareDownloader
        from data.data_manager import TradingCalendar
        from pathlib import Path
        import pandas as pd
        from datetime import datetime, timedelta
        
        # 检查是否是交易日
        is_trading, actual_date = TradingCalendar.should_update_data(date)
        
        if not is_trading:
            logger.info(f"Date {date} is not a trading day (weekend/holiday)")
            # 如果不是交易日，检查最近交易日的缓存
            cache_file = self.stats_cache_dir / f"{actual_date}.json"
            if cache_file.exists():
                logger.info(f"Using cached stats from last trading day: {actual_date}")
                return self.load_stats_cache(actual_date)
            # 否则用最近交易日继续
            date = actual_date
        
        # 检查缓存是否已存在
        cache_file = self.stats_cache_dir / f"{date}.json"
        if cache_file.exists():
            logger.info(f"Stats cache already exists for {date}")
            return self.load_stats_cache(date)
        
        logger.info(f"Generating stats cache for {date}")
        
        try:
            # 初始化下载器
            downloader = TushareDownloader()
            
            # 获取主板股票列表
            stocks_df = downloader.get_main_board_stocks()
            stock_list = stocks_df['ts_code'].tolist()
            logger.info(f"Got {len(stock_list)} main board stocks")
            
            # 收集当日所有股票数据
            all_stocks_data = []
            
            # 优先使用本地已有的数据，处理前20只股票
            for ts_code in stock_list[:20]:
                file_path = Config.RAW_DATA_DIR / f"{ts_code.replace('.', '_')}.parquet"
                df = None
                
                # 首先尝试读取本地数据
                if file_path.exists():
                    # 读取本地数据
                    df = pd.read_parquet(file_path)
                    logger.debug(f"Loaded local data for {ts_code}, shape: {df.shape}")
                    
                    # 检查日期格式
                    if df['trade_date'].dtype == 'object':
                        # 尝试将日期转换为YYYYMMDD格式
                        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
                    
                    # 打印日期范围
                    logger.debug(f"{ts_code}: Date range in local data: {df['trade_date'].min()} to {df['trade_date'].max()}")
                    
                    # 检查是否有指定日期的数据
                    has_date_data = (date in df['trade_date'].values)
                    if not has_date_data:
                        # 不再立即触发下载，先尝试使用最近的可用数据
                        sorted_dates = sorted(df['trade_date'].unique(), reverse=True)
                        if sorted_dates:
                            latest_date = sorted_dates[0]
                            logger.debug(f"No data for {ts_code} on {date}, using latest available: {latest_date}")
                            # 使用最近的数据作为替代
                            df_date = df[df['trade_date'] == latest_date]
                            if not df_date.empty:
                                all_stocks_data.append(df_date)
                        continue  # 不触发下载，继续下一只股票
                else:
                    # 本地没有数据，跳过（不在这里触发下载）
                    logger.debug(f"No local data file for {ts_code}, skipping...")
                    continue
                
                # 筛选指定日期的数据
                df_date = df[df['trade_date'] == date]
                if not df_date.empty:
                    logger.debug(f"Found data for {ts_code} on {date}, shape: {df_date.shape}")
                    all_stocks_data.append(df_date)
                else:
                    logger.warning(f"No data for {ts_code} on {date} after download")
                    # 检查是否有最近的日期数据
                    recent_dates = df['trade_date'].sort_values(ascending=False).head(5)
                    logger.info(f"{ts_code}: Recent dates available: {list(recent_dates)}")
            
            logger.info(f"Collected data for {len(all_stocks_data)} stocks")
            
            if all_stocks_data:
                # 合并所有股票的数据
                all_stocks_df = pd.concat(all_stocks_data, ignore_index=True)
                logger.info(f"Merged data shape: {all_stocks_df.shape}")
                
                # 预处理数据
                processed_data = []
                for _, group in all_stocks_df.groupby('ts_code'):
                    try:
                        processed = self.process_daily_data(group)
                        if not processed.empty:
                            processed_data.append(processed.tail(1))  # 只取最新一行
                            logger.debug(f"Processed data for {group['ts_code'].iloc[0]}, shape: {processed.shape}")
                    except Exception as e:
                        logger.error(f"Failed to process data for {group['ts_code'].iloc[0]}: {e}")
                        continue
                
                logger.info(f"Processed data for {len(processed_data)} stocks")
                
                if processed_data:
                    processed_df = pd.concat(processed_data, ignore_index=True)
                    logger.info(f"Final processed data shape: {processed_df.shape}")
                    
                    # 检查是否包含所有必要的特征列
                    missing_features = [feat for feat in self.FEATURES if feat not in processed_df.columns]
                    if missing_features:
                        logger.warning(f"Missing features: {missing_features}")
                        # 为缺失的特征创建默认值列
                        for feat in missing_features:
                            processed_df[feat] = 0.0
                    
                    # 计算截面统计量
                    stats = self.calculate_cross_sectional_stats(processed_df, date)
                    logger.info(f"Generated stats cache for {date}")
                    return stats
                else:
                    logger.warning(f"No processed data available for {date}")
            else:
                logger.warning(f"No data available for {date}")
                
                # 如果没有当日数据，尝试使用前一天的数据
                if use_prev_date:
                    # 计算前一天的日期
                    prev_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                    logger.info(f"Trying to generate stats cache for previous day: {prev_date}")
                    stats = self.generate_stats_cache(prev_date, use_prev_date=False)
                    if stats is not None:
                        return stats
                
                # 如果还是没有数据，直接返回最近的可用缓存
                logger.warning(f"No data available for {date} and previous day, returning latest available cache")
                return self.get_latest_stats_cache()
        
        except Exception as e:
            logger.error(f"Failed to generate stats cache: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def apply_zscore(
        self, 
        df: pd.DataFrame, 
        stats: Dict = None
    ) -> pd.DataFrame:
        """
        应用 Z-Score 标准化
        
        使用 (X - median) / (1.4826 * mad) 公式
        
        Args:
            df: 输入 DataFrame
            stats: 统计量字典 (含 median 和 mad)
        
        Returns:
            标准化后的 DataFrame
        """
        df = df.copy()
        
        if stats is None:
            # 在线计算
            for feat in self.FEATURES:
                if feat in df.columns:
                    df[feat] = robust_zscore(df[feat])
        else:
            # 使用缓存的统计量
            for i, feat in enumerate(self.FEATURES):
                if feat in df.columns:
                    median = stats['median'][i]
                    mad = stats['mad'][i]
                    mad = max(mad, 1e-8)  # 避免除零
                    df[feat] = (df[feat] - median) / (1.4826 * mad)
        
        return df
    
    def process_all_data(self, save_processed: bool = True) -> pd.DataFrame:
        """
        处理所有原始数据
        
        Returns:
            处理后的合并 DataFrame
        """
        raw_dir = Config.RAW_DATA_DIR
        all_dfs = []
        
        parquet_files = list(raw_dir.glob("*.parquet"))
        total = len(parquet_files)
        
        logger.info(f"Processing {total} files...")
        
        for i, parquet_file in enumerate(parquet_files):
            if 'index_' in parquet_file.name:
                continue
            
            try:
                df = pd.read_parquet(parquet_file)
                
                # Check for trade_date column variations
                if 'trade_date' not in df.columns:
                    # Try to find date column
                    for col in ['Date', 'date', 'time', 'timestamp']:
                        if col in df.columns:
                            df = df.rename(columns={col: 'trade_date'})
                            break
                
                if 'trade_date' not in df.columns:
                    logger.error(f"Missing 'trade_date' column in {parquet_file}. Columns: {list(df.columns)}")
                    continue

                # 主板过滤
                if 'ts_code' in df.columns:
                    ts_code = df['ts_code'].iloc[0]
                    if not is_main_board(ts_code):
                        continue
                
                # 处理特征
                df = self.process_daily_data(df)
                if not df.empty:
                    all_dfs.append(df)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{total} files")
                    
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        if not all_dfs:
            logger.warning("No data processed")
            return pd.DataFrame()
        
        # 合并
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # 确保日期格式统一后再排序
        if 'trade_date' in merged_df.columns:
            merged_df['trade_date'] = pd.to_datetime(merged_df['trade_date'], errors='coerce')
            merged_df = merged_df.dropna(subset=['trade_date'])
        
        merged_df = merged_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # Calculate Market State Features (using aggregated data)
        logger.info("Calculating market state features...")
        
        # 检查必要的列是否存在
        required_cols = ['trade_date', 'close', 'vol']
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for market state features: {missing_cols}")
            logger.warning("Skipping market state features calculation")
        else:
            merged_df = self._calculate_market_state_features(merged_df)
        
        # 按日期计算截面统计量 并 进行标准化
        logger.info("Calculating cross-sectional statistics and normalizing data...")
        
        normalized_dfs = []
        # Group by trade_date (datetime)
        for date, group in merged_df.groupby('trade_date'):
            date_str = date.strftime('%Y%m%d')
            
            # Calculate and cache statistics
            stats = self.calculate_cross_sectional_stats(group, date_str)
            
            # Apply Normalization
            norm_group = self.apply_zscore(group, stats)
            normalized_dfs.append(norm_group)
            
        if normalized_dfs:
            final_df = pd.concat(normalized_dfs, ignore_index=True)
            final_df = final_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        else:
            final_df = merged_df # Fallback
        
        # 保存处理后的数据
        if save_processed:
            save_path = Config.PROCESSED_DATA_DIR / "all_features.parquet"
            # Ensure trade_date is string YYYYMMDD for compatibility
            final_df_save = final_df.copy()
            if 'trade_date' in final_df_save.columns:
                if pd.api.types.is_datetime64_any_dtype(final_df_save['trade_date']):
                    final_df_save['trade_date'] = final_df_save['trade_date'].dt.strftime('%Y%m%d')
                else:
                    # ensure string format consistency just in case
                    pass

            final_df_save.to_parquet(save_path, index=False)
            logger.info(f"Processed (and normalized) data saved to {save_path}")
        
        return final_df


# 便捷函数
def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """便捷预处理函数"""
    preprocessor = Preprocessor()
    return preprocessor.process_daily_data(df)
