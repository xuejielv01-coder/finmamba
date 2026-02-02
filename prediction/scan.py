# -*- coding: utf-8 -*-
"""
批量选股扫描器 (Scanner)
PRD 4.1 实现

特性:
- 批量预测全市场股票
- 大盘风控过滤
- 黑名单过滤
- 输出 CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Set
from datetime import datetime
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from utils.seeder import get_device
from data.preprocessor import Preprocessor
from models.finmamba import AlphaModel
from data.dataset import FastAlphaDataset

logger = get_logger("Scanner")


class MarketRegimeFilter:
    """
    市场环境过滤器
    基于大盘指数判断当前市场状态
    """
    
    def __init__(self, index_code: str = None, ma_period: int = None):
        self.index_code = index_code or Config.INDEX_CODE
        self.ma_period = ma_period or Config.MA_PERIOD
        self.is_risk_mode = False
    
    def check_market_status(self) -> bool:
        """
        检查市场状态
        Returns: True = 风险模式 (空头排列), False = 正常模式
        """
        try:
            index_name = self.index_code.replace('.', '_')
            index_file = Config.RAW_DATA_DIR / f"index_{index_name}.parquet"
            
            if not index_file.exists():
                logger.warning(f"Index data not found: {index_file}")
                return False
            
            df = pd.read_parquet(index_file)
            df = df.sort_values('trade_date').tail(self.ma_period + 5)
            
            if len(df) < self.ma_period:
                return False
            
            # 计算 MA
            df['ma'] = df['close'].rolling(self.ma_period).mean()
            
            latest = df.iloc[-1]
            current_price = latest['close']
            ma_value = latest['ma']
            
            # 空头排列: 价格 < MA
            self.is_risk_mode = current_price < ma_value
            
            if self.is_risk_mode:
                logger.warning(f"Market RISK mode: {current_price:.2f} < MA{self.ma_period}({ma_value:.2f})")
            else:
                logger.info(f"Market NORMAL mode: {current_price:.2f} >= MA{self.ma_period}({ma_value:.2f})")
            
            return self.is_risk_mode
            
        except Exception as e:
            logger.error(f"Market status check failed: {e}")
            return False


class BlacklistFilter:
    """
    黑名单过滤器
    """
    
    def __init__(self, blacklist_file: Path = None):
        self.blacklist_file = blacklist_file or Config.BLACKLIST_FILE
        self.blacklist = self._load_blacklist()
    
    def _load_blacklist(self) -> Set[str]:
        """加载黑名单"""
        if not self.blacklist_file.exists():
            return set()
        
        try:
            df = pd.read_csv(self.blacklist_file, comment='#')
            if 'ts_code' in df.columns:
                return set(df['ts_code'].tolist())
            return set()
        except Exception as e:
            logger.error(f"Failed to load blacklist: {e}")
            return set()
    
    def is_blacklisted(self, ts_code: str) -> bool:
        return ts_code in self.blacklist
    
    def filter(self, stock_list: List[str]) -> List[str]:
        return [s for s in stock_list if s not in self.blacklist]


class StrategyFilter:
    """
    策略过滤器 - '金鹤' 策略
    """
    def __init__(self):
        pass

    def apply(self, results: pd.DataFrame, all_stocks_data: List[pd.DataFrame]) -> pd.DataFrame:
        """应用策略过滤"""
        if results.empty:
            return results
            
        # 创建 data map 用于快速查找
        data_map = {df['ts_code'].iloc[0]: df for df in all_stocks_data}
        
        filtered_results = []
        
        for _, row in results.iterrows():
            ts_code = row['ts_code']
            score = row['score']
            df = data_map.get(ts_code)
            
            if df is None or len(df) < 20:
                continue
                
            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # === '金鹤' 策略逻辑 ===
            reasons = []
            signal_strength = 0
            
            # 1. 模型分数据 (Core)
            if score < 0.55: # 放宽一点初筛
                continue
            signal_strength += (score - 0.5) * 10 
            
            # 2. 趋势共振 (Trend)
            close = latest['close']
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            ma60 = df['close'].rolling(60).mean().iloc[-1]
            
            if close > ma20 and ma20 > ma60:
                reasons.append("多头排列")
                signal_strength += 2
            elif close > ma20:
                reasons.append("站上月线")
                signal_strength += 1
            else:
                # 趋势不好，除非分数极高否则过滤
                if score < 0.7:
                    continue
            
            # 3. 量能确认 (Volume)
            vol = latest['vol']
            vol_ma5 = df['vol'].rolling(5).mean().iloc[-1]
            if vol > vol_ma5:
                reasons.append("量能温和放大")
                signal_strength += 1
            
            # 4. 筹码稳定 (Volatility)
            # 剔除最近暴涨暴跌的
            atr = (df['high'] - df['low']).mean()
            if (latest['high'] - latest['low']) > 3 * atr:
                # 剧烈波动，风险大
                if score < 0.8: continue
            
            # 5. 资金流 (Money Flow) - 既然已预处理，假设模型已学到
            # 这里可以用显式规则加强
            
            row = row.to_dict()
            row['strategy'] = 'GoldenCrane'
            row['reasons'] = ", ".join(reasons)
            row['signal_strength'] = round(signal_strength, 2)
            
            filtered_results.append(row)
            
        return pd.DataFrame(filtered_results)


class Scanner:
    """
    批量选股扫描器
    """
    
    def __init__(
        self,
        model: AlphaModel = None,
        model_path: Path = None,
        use_market_filter: bool = None,
        device: torch.device = None
    ):
        self.device = device or get_device()
        
        # 加载模型
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model(model_path or Config.BEST_MODEL_PATH)
        
        self.model.eval()
        
        # 预处理器
        self.preprocessor = Preprocessor()
        
        # 风控过滤器
        self.use_market_filter = use_market_filter if use_market_filter is not None else Config.USE_MARKET_FILTER
        self.market_filter = MarketRegimeFilter() if self.use_market_filter else None
        
        # 黑名单过滤器
        self.blacklist_filter = BlacklistFilter()
        
        logger.info("Scanner initialized")
    
    def _load_model(self, model_path: Path) -> AlphaModel:
        """加载模型"""
        # 注意: AlphaModel (FinMamba) 初始化参数默认 use_industry=True
        # 如果训练时的配置不同，这里可能需要传参
        model = AlphaModel(
            d_model=Config.D_MODEL,
            n_layers=Config.N_LAYERS,
            n_industries=Config.N_INDUSTRIES
        )
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            try:
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load state dict: {e}")
        else:
            logger.warning(f"Model not found: {model_path}, using untrained model")
        
        return model.to(self.device)
    
    def daily_scan(
        self,
        date: str = None,
        top_k: int = None,
        output_csv: bool = True
    ) -> pd.DataFrame:
        """
        每日批量扫描
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        # 检查日期是否在前半年内
        from datetime import timedelta
        scan_date = datetime.strptime(date, '%Y%m%d')
        six_months_ago = datetime.now() - timedelta(days=180)
        
        if scan_date < six_months_ago:
            logger.warning(f"Scan date {date} is more than 6 months ago, skipping scan")
            return pd.DataFrame()
        
        logger.info(f"Starting daily scan for {date}")
        
        # 1. 加载行业映射 (Key Update)
        self.industry_map = FastAlphaDataset.get_stock_industry_map()
        logger.info(f"Loaded industry map with {len(self.industry_map)} stocks")

        # 检查市场状态
        risk_mode = False
        if self.market_filter:
            risk_mode = self.market_filter.check_market_status()
        
        # 调整 top_k
        if top_k is None:
            top_k = Config.TOP_K_RISK if risk_mode else Config.TOP_K_DEFAULT
        
        # 加载统计量缓存
        stats = self.preprocessor.load_stats_cache(date)
        if stats is None:
            logger.info(f"Stats cache not found for {date}, generating cache...")
            stats = self.preprocessor.generate_stats_cache(date)
            if stats is None:
                stats = self.preprocessor.get_latest_stats_cache()
                if stats:
                    logger.info(f"Using recent stats from {stats.get('date')}")
        
        # 获取所有股票数据
        all_stocks_data = self._load_all_stocks_latest(date)
        
        if not all_stocks_data:
            logger.warning("No stock data available for scanning")
            return pd.DataFrame()
        
        # 批量预测
        results = self._batch_predict(all_stocks_data, stats)
        
        # 过滤黑名单
        results = results[~results['ts_code'].isin(self.blacklist_filter.blacklist)]
        
        # 应用策略过滤 (High Availability)
        strategy_filter = StrategyFilter()
        results = strategy_filter.apply(results, all_stocks_data)
        
        if results.empty:
            logger.info("No stocks passed strategy filter")
            return pd.DataFrame()
        
        # 排序并取 top_k (按信号强度和分数综合排序)
        results = results.sort_values(['signal_strength', 'score'], ascending=[False, False]).head(top_k)
        
        results['risk_status'] = 'RISK' if risk_mode else 'NORMAL'
        results['scan_date'] = date
        
        if output_csv:
            self._save_csv(results, date, risk_mode)
        
        logger.info(f"Scan completed: {len(results)} stocks selected")
        return results
    
    def _load_all_stocks_latest(self, date: str) -> List[pd.DataFrame]:
        """加载所有股票最新数据"""
        raw_dir = Config.RAW_DATA_DIR
        all_data = []
        
        # 这里为了效率，可能需要根据具体情况优化。
        # 目前假设 raw_dir 下是按股票存储的 parquet
        files = list(raw_dir.glob("*.parquet"))
        
        for parquet_file in files:
            if parquet_file.name.startswith('index_'):
                continue
            
            try:
                # 优化: 只读需要的列和最后几行? 
                # 但 Parquet 需要读Footer。这里保持原逻辑但加个 try-catch
                df = pd.read_parquet(parquet_file)
                
                # 过滤主板
                if 'ts_code' not in df.columns: continue
                ts_code = df['ts_code'].iloc[0]
                if not ts_code.startswith(('60', '00', '30', '68')): # 包含科创/创业? 原逻辑 strict on 60/00
                    # USER REQUEST: "配合十分合理高效" -> Should we include 30/68? 
                    # 原代码: if not ts_code.startswith(('60', '00')): continue
                    # 建议保持原逻辑，除非用户要求扩展
                    if not ts_code.startswith(('60', '00')):
                        continue
                
                # 取最近数据
                df = df.sort_values('trade_date').tail(Config.SEQ_LEN + 20)
                
                if len(df) >= Config.SEQ_LEN:
                    all_data.append(df)
                    
            except Exception:
                continue
        
        logger.info(f"Loaded {len(all_data)} stocks for scanning")
        return all_data
    
    def _batch_predict(
        self, 
        all_stocks_data: List[pd.DataFrame],
        stats: Dict = None
    ) -> pd.DataFrame:
        """批量预测"""
        results = []
        
        for stock_df in all_stocks_data:
            try:
                ts_code = stock_df['ts_code'].iloc[0]
                
                # 预处理
                processed = self.preprocessor.process_daily_data(stock_df)
                
                # 标准化
                if stats:
                    processed = self.preprocessor.apply_zscore(processed, stats)
                
                # 提取特征
                features = processed[self.preprocessor.FEATURES].values[-Config.SEQ_LEN:].astype(np.float32)
                
                if len(features) < Config.SEQ_LEN:
                    continue
                
                # 转为张量
                X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 获取行业ID (Key Update)
                ind_id = self.industry_map.get(ts_code, 0)
                ind_ids = torch.tensor([ind_id], dtype=torch.long).to(self.device)
                
                # 预测
                with torch.no_grad():
                    # FinMamba forward signature: (x, industry_ids)
                    score = self.model(X, industry_ids=ind_ids).cpu().numpy().item()
                
                results.append({
                    'ts_code': ts_code,
                    'score': score,
                    'name': stock_df.get('name', [''])[0] if 'name' in stock_df.columns else ''
                })
                
            except Exception:
                continue
        
        return pd.DataFrame(results)
    
    def _save_csv(self, results: pd.DataFrame, date: str, risk_mode: bool):
        """保存 CSV"""
        output_dir = Config.DATA_ROOT / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"scan_{date}.csv"
        
        header_lines = []
        if risk_mode:
            header_lines.append("# WARNING: Market in RISK mode! Please be cautious.")
        header_lines.append(f"# Scan Date: {date}")
        header_lines.append(f"# Total Stocks: {len(results)}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in header_lines:
                f.write(line + '\n')
            results.to_csv(f, index=False)
        
        logger.info(f"Results saved to {output_file}")


def daily_scan(date: str = None, top_k: int = None, **kwargs) -> pd.DataFrame:
    """便捷函数：每日扫描"""
    scanner = Scanner(**kwargs)
    return scanner.daily_scan(date, top_k=top_k)
