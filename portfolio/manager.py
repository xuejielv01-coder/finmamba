# -*- coding: utf-8 -*-
"""
持仓管理器 (Portfolio Manager)

功能:
- 持仓 CRUD 操作
- 盈亏计算
- 持仓统计
- 自动获取最新股票价格
- 自动计算收益
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from portfolio.storage import PortfolioStorage
from data.downloader import TushareDownloader

logger = get_logger("PortfolioManager")


class PortfolioManager:
    """
    持仓管理器
    
    功能:
    1. 添加/更新/删除持仓
    2. 查询持仓信息
    3. 计算盈亏
    4. 持仓统计
    """
    
    # 默认止盈止损比例
    DEFAULT_TAKE_PROFIT = 0.10  # 10%
    DEFAULT_STOP_LOSS = 0.05    # 5%
    
    def __init__(self, storage: PortfolioStorage = None):
        """
        初始化持仓管理器
        
        Args:
            storage: 存储实例
        """
        self.storage = storage or PortfolioStorage()
        self._positions: Dict[str, dict] = {}
        self._downloader = TushareDownloader()  # 初始化Tushare下载器
        self._load_positions()
        self.refresh_prices_and_calculate_returns()  # 加载后自动刷新价格并计算收益
        
        logger.info(f"Portfolio manager initialized with {len(self._positions)} positions")
    
    def _load_positions(self):
        """加载持仓数据"""
        self._positions = self.storage.load()
    
    def _save_positions(self) -> bool:
        """保存持仓数据"""
        return self.storage.save(self._positions)
    
    def add_position(
        self,
        ts_code: str,
        cost_price: float,
        quantity: int,
        buy_date: str = None,
        name: str = "",
        target_profit: float = None,
        stop_loss: float = None,
        notes: str = ""
    ) -> bool:
        """
        添加持仓
        
        Args:
            ts_code: 股票代码 (如 600000.SH)
            cost_price: 成本价
            quantity: 持仓数量
            buy_date: 买入日期 YYYYMMDD
            name: 股票名称
            target_profit: 止盈比例 (如 0.10 表示 10%)
            stop_loss: 止损比例 (如 0.05 表示 5%)
            notes: 备注
            
        Returns:
            是否添加成功
        """
        if ts_code in self._positions:
            logger.warning(f"Position already exists: {ts_code}, use update_position instead")
            return False
        
        position = {
            'ts_code': ts_code,
            'name': name,
            'cost_price': float(cost_price),
            'quantity': int(quantity),
            'buy_date': buy_date or datetime.now().strftime('%Y%m%d'),
            'target_profit': target_profit if target_profit is not None else self.DEFAULT_TAKE_PROFIT,
            'stop_loss': stop_loss if stop_loss is not None else self.DEFAULT_STOP_LOSS,
            'notes': notes,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self._positions[ts_code] = position
        
        if self._save_positions():
            logger.info(f"Added position: {ts_code} @ {cost_price} x {quantity}")
            return True
        return False
    
    def update_position(
        self,
        ts_code: str,
        cost_price: float = None,
        quantity: int = None,
        target_profit: float = None,
        stop_loss: float = None,
        notes: str = None
    ) -> bool:
        """
        更新持仓
        
        Args:
            ts_code: 股票代码
            cost_price: 新成本价 (可选)
            quantity: 新数量 (可选)
            target_profit: 新止盈比例 (可选)
            stop_loss: 新止损比例 (可选)
            notes: 新备注 (可选)
            
        Returns:
            是否更新成功
        """
        if ts_code not in self._positions:
            logger.warning(f"Position not found: {ts_code}")
            return False
        
        position = self._positions[ts_code]
        
        if cost_price is not None:
            position['cost_price'] = float(cost_price)
        if quantity is not None:
            position['quantity'] = int(quantity)
        if target_profit is not None:
            position['target_profit'] = float(target_profit)
        if stop_loss is not None:
            position['stop_loss'] = float(stop_loss)
        if notes is not None:
            position['notes'] = notes
        
        position['updated_at'] = datetime.now().isoformat()
        
        if self._save_positions():
            logger.info(f"Updated position: {ts_code}")
            return True
        return False
    
    def remove_position(self, ts_code: str, backup: bool = True) -> bool:
        """
        删除持仓 (卖出后)
        
        Args:
            ts_code: 股票代码
            backup: 是否先备份
            
        Returns:
            是否删除成功
        """
        if ts_code not in self._positions:
            logger.warning(f"Position not found: {ts_code}")
            return False
        
        if backup:
            self.storage.backup()
        
        del self._positions[ts_code]
        
        if self._save_positions():
            logger.info(f"Removed position: {ts_code}")
            return True
        return False
    
    def get_position(self, ts_code: str) -> Optional[dict]:
        """
        获取单个持仓信息
        
        Args:
            ts_code: 股票代码
            
        Returns:
            持仓信息字典
        """
        return self._positions.get(ts_code)
    
    def get_all_positions(self) -> List[dict]:
        """
        获取所有持仓
        
        Returns:
            持仓列表
        """
        return list(self._positions.values())
    
    def get_position_codes(self) -> List[str]:
        """
        获取所有持仓股票代码
        
        Returns:
            股票代码列表
        """
        return list(self._positions.keys())
    
    def calculate_profit_loss(self, ts_code: str, current_price: float) -> dict:
        """
        计算单个持仓的盈亏
        
        Args:
            ts_code: 股票代码
            current_price: 当前价格
            
        Returns:
            盈亏信息字典
        """
        position = self.get_position(ts_code)
        if not position:
            return {}
        
        cost_price = position['cost_price']
        quantity = position['quantity']
        
        # 盈亏金额
        profit_amount = (current_price - cost_price) * quantity
        # 盈亏比例
        profit_rate = (current_price - cost_price) / cost_price if cost_price > 0 else 0
        # 市值
        market_value = current_price * quantity
        # 成本
        cost_total = cost_price * quantity
        
        return {
            'ts_code': ts_code,
            'name': position.get('name', ''),
            'cost_price': cost_price,
            'current_price': current_price,
            'quantity': quantity,
            'cost_total': cost_total,
            'market_value': market_value,
            'profit_amount': profit_amount,
            'profit_rate': profit_rate,
            'target_profit': position.get('target_profit', self.DEFAULT_TAKE_PROFIT),
            'stop_loss': position.get('stop_loss', self.DEFAULT_STOP_LOSS),
            'is_take_profit': profit_rate >= position.get('target_profit', self.DEFAULT_TAKE_PROFIT),
            'is_stop_loss': profit_rate <= -position.get('stop_loss', self.DEFAULT_STOP_LOSS)
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float] = None) -> dict:
        """
        获取持仓汇总
        
        Args:
            current_prices: 当前价格字典 {ts_code: price}
            
        Returns:
            汇总信息
        """
        positions = self.get_all_positions()
        
        if not positions:
            return {
                'total_positions': 0,
                'total_cost': 0,
                'total_market_value': 0,
                'total_profit': 0,
                'total_profit_rate': 0,
                'positions': []
            }
        
        total_cost = 0
        total_market_value = 0
        total_profit = 0
        position_details = []
        
        for pos in positions:
            ts_code = pos['ts_code']
            quantity = pos['quantity']
            cost_price = pos['cost_price']
            
            cost = cost_price * quantity
            total_cost += cost
            
            # 优先使用已有的实时数据（如果存在）
            if 'current_price' in pos and 'profit_amount' in pos:
                current_price = pos['current_price']
                market_value = pos['market_value']
                profit_amount = pos['profit_amount']
                profit_rate = pos['profit_rate']
                
                detail = {
                    **pos,
                    'cost_total': cost,
                    'is_take_profit': profit_rate >= pos.get('target_profit', self.DEFAULT_TAKE_PROFIT),
                    'is_stop_loss': profit_rate <= -pos.get('stop_loss', self.DEFAULT_STOP_LOSS)
                }
            elif current_prices and ts_code in current_prices:
                current_price = current_prices[ts_code]
                market_value = current_price * quantity
                profit_amount = (current_price - cost_price) * quantity
                profit_rate = (current_price - cost_price) / cost_price if cost_price > 0 else 0
                
                detail = self.calculate_profit_loss(ts_code, current_price)
            else:
                current_price = None
                market_value = cost  # 无当前价格时使用成本
                profit_amount = 0
                profit_rate = 0
                
                detail = {**pos, 'current_price': None, 'profit_amount': 0, 'profit_rate': 0, 'cost_total': cost}
            
            total_market_value += market_value
            total_profit += profit_amount
            position_details.append(detail)
        
        total_profit_rate = total_profit / total_cost if total_cost > 0 else 0
        
        return {
            'total_positions': len(positions),
            'total_cost': total_cost,
            'total_market_value': total_market_value,
            'total_profit': total_profit,
            'total_profit_rate': total_profit_rate,
            'positions': position_details
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为 DataFrame
        
        Returns:
            持仓 DataFrame
        """
        positions = self.get_all_positions()
        if not positions:
            return pd.DataFrame()
        
        return pd.DataFrame(positions)
    
    def get_latest_prices(self, ts_codes: List[str]) -> Dict[str, float]:
        """
        获取指定股票的最新价格
        
        Args:
            ts_codes: 股票代码列表
            
        Returns:
            最新价格字典 {ts_code: price}
        """
        if not ts_codes:
            return {}
        
        latest_prices = {}
        
        try:
            # 使用Tushare API的daily接口获取最新价格
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y%m%d')
            
            for ts_code in ts_codes:
                # 获取最近30天的数据，然后取最新一天的收盘价
                df = self._downloader._api_call('daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    # 按日期降序排序，取第一个值
                    df_sorted = df.sort_values('trade_date', ascending=False)
                    latest_price = df_sorted.iloc[0]['close']
                    latest_prices[ts_code] = latest_price
                    logger.debug(f"Got latest price for {ts_code}: {latest_price}")
        except Exception as e:
            logger.error(f"Failed to get latest prices: {e}")
        
        return latest_prices
    
    def refresh_prices_and_calculate_returns(self):
        """
        自动刷新所有持仓的最新价格并计算收益
        
        1. 获取所有持仓股票代码
        2. 调用Tushare API获取最新价格
        3. 更新持仓数据，添加当前价格和收益信息
        4. 保存更新后的持仓数据
        """
        logger.info("Refreshing latest prices and calculating returns...")
        
        # 获取所有持仓股票代码
        ts_codes = self.get_position_codes()
        if not ts_codes:
            logger.info("No positions to update")
            return
        
        # 获取最新价格
        latest_prices = self.get_latest_prices(ts_codes)
        if not latest_prices:
            logger.warning("Failed to get any latest prices")
            return
        
        # 更新持仓数据，添加当前价格和收益信息
        updated = False
        for ts_code, current_price in latest_prices.items():
            if ts_code in self._positions:
                # 计算收益
                position = self._positions[ts_code]
                cost_price = position['cost_price']
                quantity = position['quantity']
                
                # 计算盈亏金额和比例
                profit_amount = (current_price - cost_price) * quantity
                profit_rate = (current_price - cost_price) / cost_price if cost_price > 0 else 0
                
                # 更新持仓数据
                position['current_price'] = current_price
                position['profit_amount'] = profit_amount
                position['profit_rate'] = profit_rate
                position['market_value'] = current_price * quantity
                position['updated_at'] = datetime.now().isoformat()
                
                updated = True
                logger.debug(f"Updated {ts_code}: current_price={current_price:.2f}, profit_rate={profit_rate:.2%}")
        
        # 保存更新后的持仓数据
        if updated:
            if self._save_positions():
                logger.info(f"Updated prices and returns for {len(latest_prices)} positions")
            else:
                logger.error("Failed to save updated positions")
    
    def refresh(self):
        """刷新持仓数据 (从存储重新加载)"""
        self._load_positions()
        self.refresh_prices_and_calculate_returns()  # 刷新后自动获取最新价格并计算收益
        logger.info(f"Refreshed portfolio: {len(self._positions)} positions")
