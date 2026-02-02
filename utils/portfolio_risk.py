# -*- coding: utf-8 -*-
"""
组合风险度量模块
改进19：实现VaR/CVaR计算和组合风险分析

功能:
- VaR (Value at Risk) 计算
- CVaR (Conditional VaR) 计算
- 最大回撤监控
- 波动率分解
- 相关性风险分析
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("PortfolioRisk")


class VaRCalculator:
    """VaR (Value at Risk) 计算器"""
    
    @staticmethod
    def historical_var(returns: np.ndarray, 
                       confidence_level: float = 0.95,
                       holding_period: int = 1) -> float:
        """
        历史模拟法计算VaR
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            holding_period: 持有期（天）
            
        Returns:
            VaR值（负数表示损失）
        """
        returns = np.asarray(returns).flatten()
        
        if len(returns) < 30:
            logger.warning("Sample size too small for reliable VaR estimation")
        
        # 调整为持有期收益
        if holding_period > 1:
            adjusted_returns = []
            for i in range(len(returns) - holding_period + 1):
                period_return = np.prod(1 + returns[i:i+holding_period]) - 1
                adjusted_returns.append(period_return)
            returns = np.array(adjusted_returns)
        
        # 计算VaR（分位数）
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        return var
    
    @staticmethod
    def parametric_var(returns: np.ndarray,
                       confidence_level: float = 0.95,
                       holding_period: int = 1) -> float:
        """
        参数法（正态分布）计算VaR
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            holding_period: 持有期
            
        Returns:
            VaR值
        """
        returns = np.asarray(returns).flatten()
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # 调整为持有期
        mean_adjusted = mean * holding_period
        std_adjusted = std * np.sqrt(holding_period)
        
        # 正态分布分位数
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean_adjusted + z_score * std_adjusted
        
        return var
    
    @staticmethod
    def monte_carlo_var(returns: np.ndarray,
                        confidence_level: float = 0.95,
                        holding_period: int = 1,
                        n_simulations: int = 10000) -> float:
        """
        蒙特卡洛模拟计算VaR
        
        Args:
            returns: 历史收益率
            confidence_level: 置信水平
            holding_period: 持有期
            n_simulations: 模拟次数
            
        Returns:
            VaR值
        """
        returns = np.asarray(returns).flatten()
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # 模拟收益路径
        simulated_returns = np.random.normal(
            mean * holding_period,
            std * np.sqrt(holding_period),
            n_simulations
        )
        
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return var


class CVaRCalculator:
    """CVaR (Conditional VaR / Expected Shortfall) 计算器"""
    
    @staticmethod
    def calculate(returns: np.ndarray, 
                  confidence_level: float = 0.95) -> float:
        """
        计算CVaR（条件VaR/期望损失）
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            CVaR值
        """
        returns = np.asarray(returns).flatten()
        
        # VaR阈值
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        
        # 超过VaR的损失的平均值
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return var_threshold
        
        cvar = np.mean(tail_losses)
        
        return cvar


class DrawdownAnalyzer:
    """回撤分析器"""
    
    @staticmethod
    def calculate_drawdowns(prices: np.ndarray) -> Dict:
        """
        计算回撤序列和统计
        
        Args:
            prices: 价格或净值序列
            
        Returns:
            回撤分析结果
        """
        prices = np.asarray(prices).flatten()
        
        # 计算累计最高点
        running_max = np.maximum.accumulate(prices)
        
        # 计算回撤
        drawdowns = (prices - running_max) / running_max
        
        # 最大回撤
        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        # 最大回撤恢复时间
        recovery_idx = max_dd_idx
        for i in range(max_dd_idx, len(prices)):
            if prices[i] >= running_max[max_dd_idx - 1]:
                recovery_idx = i
                break
        
        recovery_days = recovery_idx - max_dd_idx if recovery_idx > max_dd_idx else None
        
        # 回撤统计
        return {
            'drawdown_series': drawdowns,
            'max_drawdown': float(max_drawdown),
            'max_drawdown_idx': int(max_dd_idx),
            'recovery_days': recovery_days,
            'avg_drawdown': float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0,
            'drawdown_std': float(np.std(drawdowns)),
            'current_drawdown': float(drawdowns[-1]),
        }
    
    @staticmethod
    def get_underwater_chart(prices: np.ndarray) -> np.ndarray:
        """获取水下曲线（持续回撤）"""
        prices = np.asarray(prices).flatten()
        running_max = np.maximum.accumulate(prices)
        underwater = (prices - running_max) / running_max
        return underwater
    
    @staticmethod
    def get_drawdown_periods(prices: np.ndarray, 
                             threshold: float = -0.05) -> List[Dict]:
        """
        获取显著回撤期
        
        Args:
            prices: 价格序列
            threshold: 回撤阈值
            
        Returns:
            回撤期列表
        """
        drawdowns = DrawdownAnalyzer.get_underwater_chart(prices)
        
        periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdowns):
            if not in_drawdown and dd < threshold:
                in_drawdown = True
                start_idx = i
            elif in_drawdown and dd >= 0:
                periods.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'duration': i - start_idx,
                    'max_drawdown': float(np.min(drawdowns[start_idx:i+1])),
                })
                in_drawdown = False
        
        # 如果还在回撤中
        if in_drawdown:
            periods.append({
                'start_idx': start_idx,
                'end_idx': len(drawdowns) - 1,
                'duration': len(drawdowns) - start_idx,
                'max_drawdown': float(np.min(drawdowns[start_idx:])),
                'ongoing': True,
            })
        
        return periods


class VolatilityAnalyzer:
    """波动率分析器"""
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, 
                             annualize: bool = True,
                             trading_days: int = 252) -> float:
        """
        计算波动率
        
        Args:
            returns: 收益率序列
            annualize: 是否年化
            trading_days: 年交易日数
            
        Returns:
            波动率
        """
        returns = np.asarray(returns).flatten()
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(trading_days)
        
        return float(vol)
    
    @staticmethod
    def calculate_rolling_volatility(returns: np.ndarray,
                                     window: int = 20,
                                     annualize: bool = True) -> np.ndarray:
        """计算滚动波动率"""
        returns = pd.Series(returns)
        rolling_vol = returns.rolling(window=window).std()
        
        if annualize:
            rolling_vol *= np.sqrt(252)
        
        return rolling_vol.values
    
    @staticmethod
    def calculate_ewm_volatility(returns: np.ndarray,
                                 span: int = 20,
                                 annualize: bool = True) -> np.ndarray:
        """计算指数加权移动平均波动率"""
        returns = pd.Series(returns)
        ewm_vol = returns.ewm(span=span).std()
        
        if annualize:
            ewm_vol *= np.sqrt(252)
        
        return ewm_vol.values
    
    @staticmethod
    def decompose_volatility(returns: np.ndarray,
                            market_returns: np.ndarray) -> Dict:
        """
        波动率分解（系统性 vs 特异性）
        
        Args:
            returns: 组合收益率
            market_returns: 市场收益率
            
        Returns:
            波动率分解结果
        """
        returns = np.asarray(returns).flatten()
        market_returns = np.asarray(market_returns).flatten()
        
        # 确保长度一致
        min_len = min(len(returns), len(market_returns))
        returns = returns[:min_len]
        market_returns = market_returns[:min_len]
        
        # 计算Beta
        cov = np.cov(returns, market_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
        
        # 总波动率
        total_vol = np.std(returns) * np.sqrt(252)
        
        # 市场波动率
        market_vol = np.std(market_returns) * np.sqrt(252)
        
        # 系统性波动率
        systematic_vol = abs(beta) * market_vol
        
        # 特异性波动率
        residuals = returns - beta * market_returns
        idiosyncratic_vol = np.std(residuals) * np.sqrt(252)
        
        return {
            'total_volatility': float(total_vol),
            'systematic_volatility': float(systematic_vol),
            'idiosyncratic_volatility': float(idiosyncratic_vol),
            'beta': float(beta),
            'systematic_pct': float(systematic_vol**2 / total_vol**2) if total_vol > 0 else 0,
        }


class CorrelationRiskAnalyzer:
    """相关性风险分析器"""
    
    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        return returns_df.corr()
    
    @staticmethod
    def calculate_average_correlation(returns_df: pd.DataFrame) -> float:
        """计算平均相关性"""
        corr_matrix = returns_df.corr()
        
        # 获取上三角（排除对角线）
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.values[mask]
        
        return float(np.mean(upper_triangle))
    
    @staticmethod
    def identify_high_correlation_pairs(returns_df: pd.DataFrame,
                                        threshold: float = 0.7) -> List[Tuple]:
        """
        识别高相关性股票对
        
        Args:
            returns_df: 收益率DataFrame
            threshold: 相关性阈值
            
        Returns:
            高相关性股票对列表
        """
        corr_matrix = returns_df.corr()
        
        pairs = []
        stocks = corr_matrix.columns.tolist()
        
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                if i < j:  # 只取上三角
                    corr = corr_matrix.loc[stock1, stock2]
                    if abs(corr) >= threshold:
                        pairs.append({
                            'stock1': stock1,
                            'stock2': stock2,
                            'correlation': float(corr),
                        })
        
        # 按相关性排序
        pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return pairs
    
    @staticmethod
    def calculate_concentration_risk(weights: np.ndarray) -> Dict:
        """
        计算集中度风险
        
        Args:
            weights: 持仓权重
            
        Returns:
            集中度指标
        """
        weights = np.asarray(weights).flatten()
        weights = weights / np.sum(weights)  # 确保和为1
        
        # HHI指数
        hhi = np.sum(weights ** 2)
        
        # 有效资产数（等效分散度）
        effective_n = 1 / hhi if hhi > 0 else 0
        
        # 最大权重
        max_weight = np.max(weights)
        
        # 前N集中度
        sorted_weights = np.sort(weights)[::-1]
        top3_concentration = np.sum(sorted_weights[:3])
        top5_concentration = np.sum(sorted_weights[:5])
        
        return {
            'hhi': float(hhi),
            'effective_n': float(effective_n),
            'max_weight': float(max_weight),
            'top3_concentration': float(top3_concentration),
            'top5_concentration': float(top5_concentration),
            'n_positions': len(weights),
        }


class IndustryRiskAnalyzer:
    """行业风险分析器"""
    
    def __init__(self, industry_mapping: Dict[str, str] = None):
        """
        初始化
        
        Args:
            industry_mapping: 股票到行业的映射
        """
        self.industry_mapping = industry_mapping or {}
    
    def set_industry_mapping(self, mapping: Dict[str, str]):
        """设置行业映射"""
        self.industry_mapping = mapping
    
    def calculate_industry_exposure(self, 
                                    positions: Dict[str, float]) -> Dict[str, float]:
        """
        计算行业暴露
        
        Args:
            positions: {股票代码: 持仓金额}
            
        Returns:
            {行业: 暴露比例}
        """
        total_value = sum(positions.values())
        if total_value == 0:
            return {}
        
        industry_exposure = defaultdict(float)
        
        for stock, value in positions.items():
            industry = self.industry_mapping.get(stock, 'Unknown')
            industry_exposure[industry] += value / total_value
        
        return dict(industry_exposure)
    
    def check_concentration_limit(self,
                                  positions: Dict[str, float],
                                  max_industry_weight: float = 0.3) -> List[Dict]:
        """
        检查行业集中度限制
        
        Args:
            positions: 持仓
            max_industry_weight: 最大行业权重
            
        Returns:
            超限行业列表
        """
        exposure = self.calculate_industry_exposure(positions)
        
        violations = []
        for industry, weight in exposure.items():
            if weight > max_industry_weight:
                violations.append({
                    'industry': industry,
                    'weight': weight,
                    'limit': max_industry_weight,
                    'excess': weight - max_industry_weight,
                })
        
        return violations


class PortfolioRiskManager:
    """组合风险管理器"""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Config.DATA_DIR / "risk_reports"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.correlation_analyzer = CorrelationRiskAnalyzer()
        self.industry_analyzer = IndustryRiskAnalyzer()
        
        # 风险阈值
        self.risk_limits = {
            'max_var_95': -0.05,  # 5%的VaR限制
            'max_drawdown': -0.15,  # 15%的最大回撤限制
            'max_volatility': 0.30,  # 30%年化波动率限制
            'max_correlation': 0.8,  # 持仓相关性限制
            'max_industry_weight': 0.3,  # 行业权重限制
        }
        
        logger.info("Portfolio risk manager initialized")
    
    def set_risk_limits(self, limits: Dict):
        """设置风险限额"""
        self.risk_limits.update(limits)
    
    def calculate_comprehensive_risk(self,
                                     returns: np.ndarray,
                                     prices: np.ndarray = None,
                                     market_returns: np.ndarray = None,
                                     position_returns: pd.DataFrame = None,
                                     weights: np.ndarray = None) -> Dict:
        """
        计算综合风险指标
        
        Args:
            returns: 组合收益率
            prices: 净值序列
            market_returns: 市场收益率
            position_returns: 各持仓收益率
            weights: 持仓权重
            
        Returns:
            综合风险报告
        """
        returns = np.asarray(returns).flatten()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'var': {},
            'cvar': {},
            'drawdown': {},
            'volatility': {},
            'correlation': {},
            'concentration': {},
            'risk_score': 0,
            'alerts': [],
        }
        
        # VaR计算
        report['var']['historical_95'] = self.var_calculator.historical_var(returns, 0.95)
        report['var']['historical_99'] = self.var_calculator.historical_var(returns, 0.99)
        report['var']['parametric_95'] = self.var_calculator.parametric_var(returns, 0.95)
        
        # CVaR计算
        report['cvar']['95'] = self.cvar_calculator.calculate(returns, 0.95)
        report['cvar']['99'] = self.cvar_calculator.calculate(returns, 0.99)
        
        # 回撤分析
        if prices is not None:
            report['drawdown'] = self.drawdown_analyzer.calculate_drawdowns(prices)
        else:
            # 从收益计算净值
            cumulative = np.cumprod(1 + returns)
            report['drawdown'] = self.drawdown_analyzer.calculate_drawdowns(cumulative)
        
        # 波动率
        report['volatility']['annualized'] = self.volatility_analyzer.calculate_volatility(returns)
        
        if market_returns is not None:
            report['volatility']['decomposition'] = self.volatility_analyzer.decompose_volatility(
                returns, market_returns
            )
        
        # 相关性分析
        if position_returns is not None:
            report['correlation']['average'] = self.correlation_analyzer.calculate_average_correlation(
                position_returns
            )
            report['correlation']['high_pairs'] = self.correlation_analyzer.identify_high_correlation_pairs(
                position_returns
            )[:5]  # 前5对
        
        # 集中度分析
        if weights is not None:
            report['concentration'] = self.correlation_analyzer.calculate_concentration_risk(weights)
        
        # 风险评分和警报
        report['risk_score'], report['alerts'] = self._calculate_risk_score(report)
        
        return report
    
    def _calculate_risk_score(self, report: Dict) -> Tuple[float, List[str]]:
        """计算风险评分和生成警报"""
        score = 100.0
        alerts = []
        
        # 检查VaR
        var_95 = report['var'].get('historical_95', 0)
        if var_95 < self.risk_limits['max_var_95']:
            score -= 20
            alerts.append(f"VaR(95%)超限: {var_95:.2%} < {self.risk_limits['max_var_95']:.2%}")
        
        # 检查最大回撤
        max_dd = report['drawdown'].get('max_drawdown', 0)
        if max_dd < self.risk_limits['max_drawdown']:
            score -= 25
            alerts.append(f"最大回撤超限: {max_dd:.2%} < {self.risk_limits['max_drawdown']:.2%}")
        
        # 检查波动率
        vol = report['volatility'].get('annualized', 0)
        if vol > self.risk_limits['max_volatility']:
            score -= 15
            alerts.append(f"波动率超限: {vol:.2%} > {self.risk_limits['max_volatility']:.2%}")
        
        # 检查相关性
        avg_corr = report['correlation'].get('average', 0)
        if avg_corr > self.risk_limits['max_correlation']:
            score -= 10
            alerts.append(f"平均相关性过高: {avg_corr:.2f} > {self.risk_limits['max_correlation']:.2f}")
        
        # 检查集中度
        if report['concentration']:
            hhi = report['concentration'].get('hhi', 0)
            if hhi > 0.25:  # HHI > 0.25 表示高度集中
                score -= 15
                alerts.append(f"持仓过于集中: HHI={hhi:.3f}")
        
        score = max(0, min(100, score))
        
        return score, alerts
    
    def generate_risk_report(self, 
                            returns: np.ndarray,
                            **kwargs) -> str:
        """生成风险报告文本"""
        report = self.calculate_comprehensive_risk(returns, **kwargs)
        
        text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    组合风险报告                                ║
╠══════════════════════════════════════════════════════════════╣
║ 生成时间: {report['timestamp'][:19]:<40}║
║ 风险评分: {report['risk_score']:.1f}/100{' ' * 43}║
╠══════════════════════════════════════════════════════════════╣
║ VaR分析:                                                     ║
║   历史VaR(95%): {report['var']['historical_95']*100:>8.2f}%{' ' * 35}║
║   历史VaR(99%): {report['var']['historical_99']*100:>8.2f}%{' ' * 35}║
║   CVaR(95%):    {report['cvar']['95']*100:>8.2f}%{' ' * 35}║
╠══════════════════════════════════════════════════════════════╣
║ 回撤分析:                                                    ║
║   最大回撤:     {report['drawdown'].get('max_drawdown', 0)*100:>8.2f}%{' ' * 35}║
║   当前回撤:     {report['drawdown'].get('current_drawdown', 0)*100:>8.2f}%{' ' * 35}║
╠══════════════════════════════════════════════════════════════╣
║ 波动率:                                                      ║
║   年化波动率:   {report['volatility'].get('annualized', 0)*100:>8.2f}%{' ' * 35}║
"""
        
        if report['alerts']:
            text += "╠══════════════════════════════════════════════════════════════╣\n"
            text += "║ ⚠️ 风险警报:                                                  ║\n"
            for alert in report['alerts']:
                text += f"║   • {alert:<55}║\n"
        
        text += "╚══════════════════════════════════════════════════════════════╝"
        
        return text
    
    def save_report(self, report: Dict):
        """保存风险报告"""
        filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Risk report saved to {filepath}")


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成模拟数据
    n_days = 252
    returns = np.random.randn(n_days) * 0.02  # 2%日波动率
    market_returns = np.random.randn(n_days) * 0.015
    
    # 净值
    prices = np.cumprod(1 + returns)
    
    # 测试风险管理器
    risk_manager = PortfolioRiskManager()
    
    # 计算综合风险
    report = risk_manager.calculate_comprehensive_risk(
        returns=returns,
        prices=prices,
        market_returns=market_returns,
        weights=np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    )
    
    # 打印报告
    print(risk_manager.generate_risk_report(returns, prices=prices, market_returns=market_returns))
    
    print("\n详细指标:")
    print(f"VaR(95%): {report['var']['historical_95']:.4f}")
    print(f"CVaR(95%): {report['cvar']['95']:.4f}")
    print(f"最大回撤: {report['drawdown']['max_drawdown']:.4f}")
    print(f"年化波动率: {report['volatility']['annualized']:.4f}")
    print(f"风险评分: {report['risk_score']:.1f}")
    
    print("\n风险度量模块测试完成!")
