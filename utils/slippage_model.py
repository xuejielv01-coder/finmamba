# -*- coding: utf-8 -*-
"""
滑点模型模块
改进16：实现基于成交量的滑点计算

功能:
- 固定滑点模型
- 成交量加权滑点
- 动态滑点计算
- 市场冲击成本模型
- 滑点敏感性分析
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("SlippageModel")


class FixedSlippage:
    """固定滑点模型"""
    
    def __init__(self, slippage_pct: float = 0.001):
        """
        初始化固定滑点
        
        Args:
            slippage_pct: 滑点比例（如0.001表示0.1%）
        """
        self.slippage_pct = slippage_pct
    
    def calculate(self, 
                 price: float, 
                 quantity: float, 
                 side: str = 'buy') -> Dict:
        """
        计算滑点
        
        Args:
            price: 理论成交价格
            quantity: 交易数量（股数）
            side: 交易方向 ('buy' 或 'sell')
            
        Returns:
            滑点信息字典
        """
        slippage = price * self.slippage_pct
        
        if side == 'buy':
            actual_price = price + slippage
        else:
            actual_price = price - slippage
        
        slippage_cost = abs(slippage) * quantity
        
        return {
            'theoretical_price': price,
            'actual_price': actual_price,
            'slippage': slippage,
            'slippage_pct': self.slippage_pct,
            'slippage_cost': slippage_cost,
            'side': side,
        }


class VolumeWeightedSlippage:
    """成交量加权滑点模型"""
    
    def __init__(self, 
                 base_slippage: float = 0.0005,
                 volume_impact_factor: float = 0.1):
        """
        初始化成交量加权滑点
        
        滑点 = base_slippage * (1 + volume_impact_factor * (订单量 / ADV))
        
        Args:
            base_slippage: 基础滑点比例
            volume_impact_factor: 成交量影响因子
        """
        self.base_slippage = base_slippage
        self.volume_impact_factor = volume_impact_factor
    
    def calculate(self,
                 price: float,
                 quantity: float,
                 daily_volume: float,
                 side: str = 'buy') -> Dict:
        """
        计算成交量加权滑点
        
        Args:
            price: 理论成交价格
            quantity: 交易数量
            daily_volume: 当日成交量（或平均日成交量ADV）
            side: 交易方向
            
        Returns:
            滑点信息字典
        """
        if daily_volume <= 0:
            participation_rate = 1.0
        else:
            participation_rate = quantity / daily_volume
        
        # 滑点随参与率非线性增加
        slippage_pct = self.base_slippage * (1 + self.volume_impact_factor * np.sqrt(participation_rate))
        slippage_pct = min(slippage_pct, 0.02)  # 最大2%滑点
        
        slippage = price * slippage_pct
        
        if side == 'buy':
            actual_price = price + slippage
        else:
            actual_price = price - slippage
        
        return {
            'theoretical_price': price,
            'actual_price': actual_price,
            'slippage': slippage,
            'slippage_pct': slippage_pct,
            'slippage_cost': abs(slippage) * quantity,
            'participation_rate': participation_rate,
            'daily_volume': daily_volume,
            'side': side,
        }


class MarketImpactModel:
    """
    市场冲击成本模型
    
    基于Almgren-Chriss模型的简化版本：
    Impact = α * σ * (Q / ADV)^β
    
    其中:
    - σ: 股票波动率
    - Q: 订单量
    - ADV: 平均日成交量
    - α, β: 模型参数
    """
    
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.6,
                 gamma: float = 0.1):
        """
        初始化市场冲击模型
        
        Args:
            alpha: 冲击系数
            beta: 参与率指数（通常在0.5-1之间）
            gamma: 临时冲击系数
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def calculate_permanent_impact(self,
                                   quantity: float,
                                   adv: float,
                                   volatility: float) -> float:
        """
        计算永久冲击（持久性价格变化）
        
        Args:
            quantity: 交易数量
            adv: 平均日成交量
            volatility: 股票波动率
        """
        if adv <= 0:
            return 0.0
        
        participation_rate = quantity / adv
        permanent_impact = self.alpha * volatility * np.power(participation_rate, self.beta)
        
        return float(permanent_impact)
    
    def calculate_temporary_impact(self,
                                   quantity: float,
                                   adv: float,
                                   volatility: float) -> float:
        """
        计算临时冲击（立即恢复的价格变化）
        """
        if adv <= 0:
            return 0.0
        
        participation_rate = quantity / adv
        temporary_impact = self.gamma * volatility * np.sqrt(participation_rate)
        
        return float(temporary_impact)
    
    def calculate_total_cost(self,
                            price: float,
                            quantity: float,
                            adv: float,
                            volatility: float,
                            side: str = 'buy') -> Dict:
        """
        计算总市场冲击成本
        
        Args:
            price: 理论价格
            quantity: 交易数量
            adv: 平均日成交量
            volatility: 股票波动率
            side: 交易方向
            
        Returns:
            市场冲击信息
        """
        permanent = self.calculate_permanent_impact(quantity, adv, volatility)
        temporary = self.calculate_temporary_impact(quantity, adv, volatility)
        
        total_impact_pct = permanent + temporary
        
        if side == 'buy':
            actual_price = price * (1 + total_impact_pct)
        else:
            actual_price = price * (1 - total_impact_pct)
        
        return {
            'theoretical_price': price,
            'actual_price': actual_price,
            'permanent_impact': permanent,
            'temporary_impact': temporary,
            'total_impact_pct': total_impact_pct,
            'impact_cost': abs(price * total_impact_pct) * quantity,
            'participation_rate': quantity / adv if adv > 0 else 1.0,
            'side': side,
        }


class DynamicSlippage:
    """动态滑点模型"""
    
    def __init__(self):
        """初始化动态滑点模型"""
        self.fixed_model = FixedSlippage()
        self.volume_model = VolumeWeightedSlippage()
        self.impact_model = MarketImpactModel()
    
    def calculate(self,
                 price: float,
                 quantity: float,
                 daily_volume: float = None,
                 adv: float = None,
                 volatility: float = None,
                 spread: float = None,
                 side: str = 'buy') -> Dict:
        """
        根据可用信息动态选择最合适的滑点模型
        
        Args:
            price: 理论价格
            quantity: 交易数量
            daily_volume: 当日成交量
            adv: 平均日成交量
            volatility: 波动率
            spread: 买卖价差
            side: 交易方向
            
        Returns:
            滑点信息
        """
        # 计算不同模型的滑点
        results = {}
        
        # 1. 固定滑点（基准）
        fixed_result = self.fixed_model.calculate(price, quantity, side)
        results['fixed'] = fixed_result
        
        # 2. 成交量加权滑点
        if daily_volume is not None and daily_volume > 0:
            volume_result = self.volume_model.calculate(
                price, quantity, daily_volume, side
            )
            results['volume_weighted'] = volume_result
        
        # 3. 市场冲击模型
        if adv is not None and volatility is not None:
            impact_result = self.impact_model.calculate_total_cost(
                price, quantity, adv, volatility, side
            )
            results['market_impact'] = impact_result
        
        # 选择最保守（最高滑点）的模型
        slippages = []
        for model_name, result in results.items():
            if 'slippage_pct' in result:
                slippages.append((model_name, result['slippage_pct']))
            elif 'total_impact_pct' in result:
                slippages.append((model_name, result['total_impact_pct']))
        
        if slippages:
            best_model, best_slippage = max(slippages, key=lambda x: x[1])
            final_result = results[best_model].copy()
            final_result['model_used'] = best_model
        else:
            final_result = fixed_result.copy()
            final_result['model_used'] = 'fixed'
        
        final_result['all_models'] = {k: v.get('slippage_pct', v.get('total_impact_pct', 0)) 
                                      for k, v in results.items()}
        
        return final_result


class SlippageSensitivityAnalyzer:
    """滑点敏感性分析"""
    
    def __init__(self, slippage_model: FixedSlippage = None):
        """初始化分析器"""
        self.slippage_model = slippage_model or FixedSlippage()
    
    def analyze_sensitivity(self,
                           base_return: float,
                           slippage_range: List[float] = None,
                           n_trades: int = 100) -> pd.DataFrame:
        """
        分析滑点对收益的敏感性
        
        Args:
            base_return: 基准收益率（无滑点）
            slippage_range: 滑点范围列表
            n_trades: 交易次数
            
        Returns:
            敏感性分析结果
        """
        if slippage_range is None:
            slippage_range = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
        
        results = []
        
        for slippage in slippage_range:
            # 每次交易的滑点损失
            total_slippage_cost = slippage * 2 * n_trades  # 买入+卖出
            
            # 调整后收益
            adjusted_return = base_return - total_slippage_cost
            
            # 收益损失比例
            loss_pct = total_slippage_cost / base_return if base_return > 0 else 0
            
            results.append({
                'slippage_pct': slippage * 100,
                'slippage_bps': slippage * 10000,
                'total_slippage_cost': total_slippage_cost,
                'base_return': base_return * 100,
                'adjusted_return': adjusted_return * 100,
                'return_loss': total_slippage_cost * 100,
                'return_loss_pct': loss_pct * 100,
            })
        
        return pd.DataFrame(results)
    
    def find_breakeven_slippage(self,
                                expected_return: float,
                                n_trades: int) -> float:
        """
        找到盈亏平衡滑点
        
        Args:
            expected_return: 预期收益率
            n_trades: 交易次数
            
        Returns:
            盈亏平衡滑点比例
        """
        if n_trades <= 0:
            return float('inf')
        
        # 收益 = 预期收益 - 滑点*2*交易次数
        # 盈亏平衡: 0 = expected_return - slippage * 2 * n_trades
        breakeven = expected_return / (2 * n_trades)
        
        return float(breakeven)


class TransactionCostCalculator:
    """交易成本计算器（综合滑点和佣金）"""
    
    def __init__(self,
                 slippage_model: DynamicSlippage = None,
                 commission_rate: float = 0.0003,  # 万3佣金
                 min_commission: float = 5.0,  # 最低5元
                 stamp_tax: float = 0.001,  # 印花税（卖出）
                 transfer_fee: float = 0.00002):  # 过户费
        """
        初始化交易成本计算器
        
        Args:
            slippage_model: 滑点模型
            commission_rate: 佣金费率
            min_commission: 最低佣金
            stamp_tax: 印花税（仅卖出收取）
            transfer_fee: 过户费
        """
        self.slippage_model = slippage_model or DynamicSlippage()
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax = stamp_tax
        self.transfer_fee = transfer_fee
    
    def calculate_buy_cost(self,
                          price: float,
                          quantity: float,
                          daily_volume: float = None,
                          **kwargs) -> Dict:
        """
        计算买入成本
        
        Args:
            price: 理论价格
            quantity: 股数
            daily_volume: 日成交量
            
        Returns:
            成本明细
        """
        trade_value = price * quantity
        
        # 滑点成本
        slippage_result = self.slippage_model.calculate(
            price, quantity, daily_volume=daily_volume, side='buy', **kwargs
        )
        slippage_cost = slippage_result['slippage_cost']
        
        # 佣金
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 过户费
        transfer = trade_value * self.transfer_fee
        
        total_cost = slippage_cost + commission + transfer
        
        return {
            'trade_value': trade_value,
            'slippage_cost': slippage_cost,
            'commission': commission,
            'transfer_fee': transfer,
            'total_cost': total_cost,
            'total_cost_pct': total_cost / trade_value if trade_value > 0 else 0,
            'actual_price': slippage_result['actual_price'],
            'side': 'buy',
        }
    
    def calculate_sell_cost(self,
                           price: float,
                           quantity: float,
                           daily_volume: float = None,
                           **kwargs) -> Dict:
        """
        计算卖出成本
        """
        trade_value = price * quantity
        
        # 滑点成本
        slippage_result = self.slippage_model.calculate(
            price, quantity, daily_volume=daily_volume, side='sell', **kwargs
        )
        slippage_cost = slippage_result['slippage_cost']
        
        # 佣金
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 印花税（仅卖出）
        tax = trade_value * self.stamp_tax
        
        # 过户费
        transfer = trade_value * self.transfer_fee
        
        total_cost = slippage_cost + commission + tax + transfer
        
        return {
            'trade_value': trade_value,
            'slippage_cost': slippage_cost,
            'commission': commission,
            'stamp_tax': tax,
            'transfer_fee': transfer,
            'total_cost': total_cost,
            'total_cost_pct': total_cost / trade_value if trade_value > 0 else 0,
            'actual_price': slippage_result['actual_price'],
            'side': 'sell',
        }
    
    def calculate_round_trip_cost(self,
                                 price: float,
                                 quantity: float,
                                 **kwargs) -> Dict:
        """
        计算完整交易周期（买入+卖出）的成本
        """
        buy_cost = self.calculate_buy_cost(price, quantity, **kwargs)
        sell_cost = self.calculate_sell_cost(price, quantity, **kwargs)
        
        trade_value = price * quantity
        total_cost = buy_cost['total_cost'] + sell_cost['total_cost']
        
        return {
            'trade_value': trade_value,
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'total_cost': total_cost,
            'total_cost_pct': total_cost / trade_value if trade_value > 0 else 0,
            'breakeven_return': total_cost / trade_value if trade_value > 0 else 0,
        }


if __name__ == "__main__":
    print("滑点模型测试")
    print("="*50)
    
    # 测试固定滑点
    print("\n1. 固定滑点模型")
    fixed = FixedSlippage(slippage_pct=0.001)
    result = fixed.calculate(price=10.0, quantity=1000, side='buy')
    print(f"  理论价格: {result['theoretical_price']}")
    print(f"  实际价格: {result['actual_price']:.4f}")
    print(f"  滑点成本: {result['slippage_cost']:.2f}")
    
    # 测试成交量加权滑点
    print("\n2. 成交量加权滑点模型")
    volume_model = VolumeWeightedSlippage()
    result = volume_model.calculate(
        price=10.0, quantity=100000, daily_volume=1000000, side='buy'
    )
    print(f"  参与率: {result['participation_rate']:.2%}")
    print(f"  滑点比例: {result['slippage_pct']:.4%}")
    print(f"  实际价格: {result['actual_price']:.4f}")
    
    # 测试市场冲击模型
    print("\n3. 市场冲击模型")
    impact_model = MarketImpactModel()
    result = impact_model.calculate_total_cost(
        price=10.0, quantity=50000, adv=500000, volatility=0.02, side='buy'
    )
    print(f"  永久冲击: {result['permanent_impact']:.4%}")
    print(f"  临时冲击: {result['temporary_impact']:.4%}")
    print(f"  总冲击: {result['total_impact_pct']:.4%}")
    
    # 测试敏感性分析
    print("\n4. 滑点敏感性分析")
    analyzer = SlippageSensitivityAnalyzer()
    sensitivity = analyzer.analyze_sensitivity(base_return=0.10, n_trades=50)
    print(sensitivity.to_string(index=False))
    
    # 测试综合交易成本
    print("\n5. 综合交易成本")
    calc = TransactionCostCalculator()
    round_trip = calc.calculate_round_trip_cost(price=10.0, quantity=1000)
    print(f"  买入成本: {round_trip['buy_cost']['total_cost']:.2f}")
    print(f"  卖出成本: {round_trip['sell_cost']['total_cost']:.2f}")
    print(f"  总成本: {round_trip['total_cost']:.2f} ({round_trip['total_cost_pct']:.4%})")
    print(f"  盈亏平衡收益: {round_trip['breakeven_return']:.4%}")
    
    print("\n滑点模型测试完成!")
