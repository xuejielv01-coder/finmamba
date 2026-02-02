# -*- coding: utf-8 -*-
"""
压力测试模块
改进21：实现组合压力测试和场景分析

功能:
- 历史场景压力测试
- 假设场景压力测试
- 蒙特卡洛压力测试
- 极端事件模拟
- 压力测试报告
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from utils.portfolio_risk import VaRCalculator, DrawdownAnalyzer

logger = get_logger("StressTest")


class HistoricalScenario:
    """历史场景定义"""
    
    # 预定义的历史危机场景
    SCENARIOS = {
        '2008_financial_crisis': {
            'name': '2008年金融危机',
            'description': '次贷危机引发的全球金融海啸',
            'start_date': '2008-09-01',
            'end_date': '2009-03-01',
            'market_shock': -0.50,  # 市场约下跌50%
            'volatility_multiplier': 3.0,
            'correlation_increase': 0.3,
        },
        '2015_china_crash': {
            'name': '2015年A股股灾',
            'description': '杠杆牛市崩盘',
            'start_date': '2015-06-12',
            'end_date': '2015-08-26',
            'market_shock': -0.45,
            'volatility_multiplier': 4.0,
            'correlation_increase': 0.4,
        },
        '2020_covid_crash': {
            'name': '2020年新冠疫情',
            'description': 'COVID-19疫情冲击',
            'start_date': '2020-02-20',
            'end_date': '2020-03-23',
            'market_shock': -0.35,
            'volatility_multiplier': 5.0,
            'correlation_increase': 0.5,
        },
        '2022_tech_crash': {
            'name': '2022年科技股调整',
            'description': '加息周期科技股回调',
            'start_date': '2022-01-01',
            'end_date': '2022-10-01',
            'market_shock': -0.30,
            'volatility_multiplier': 2.0,
            'correlation_increase': 0.2,
        },
    }
    
    @classmethod
    def get_scenario(cls, name: str) -> Dict:
        """获取预定义场景"""
        return cls.SCENARIOS.get(name, {})
    
    @classmethod
    def list_scenarios(cls) -> List[str]:
        """列出所有预定义场景"""
        return list(cls.SCENARIOS.keys())


class ScenarioGenerator:
    """压力场景生成器"""
    
    def __init__(self, seed: int = None):
        """初始化场景生成器"""
        if seed is not None:
            np.random.seed(seed)
    
    def generate_market_shock(self,
                              returns: np.ndarray,
                              shock_pct: float,
                              duration_days: int = 20) -> np.ndarray:
        """
        生成市场冲击场景
        
        Args:
            returns: 原始收益率序列
            shock_pct: 冲击幅度（如-0.3表示30%下跌）
            duration_days: 冲击持续天数
            
        Returns:
            冲击后的收益率序列
        """
        stressed_returns = returns.copy()
        
        # 在最后duration_days天应用冲击
        daily_shock = shock_pct / duration_days
        
        if len(stressed_returns) >= duration_days:
            stressed_returns[-duration_days:] += daily_shock
        
        return stressed_returns
    
    def generate_volatility_stress(self,
                                    returns: np.ndarray,
                                    vol_multiplier: float = 2.0) -> np.ndarray:
        """
        生成波动率压力场景
        
        Args:
            returns: 原始收益率序列
            vol_multiplier: 波动率放大倍数
            
        Returns:
            压力后的收益率序列
        """
        mean = np.mean(returns)
        centered = returns - mean
        
        # 放大波动率
        stressed = mean + centered * vol_multiplier
        
        return stressed
    
    def generate_correlation_stress(self,
                                     returns_matrix: np.ndarray,
                                     correlation_increase: float = 0.3) -> np.ndarray:
        """
        生成相关性压力场景
        
        在危机时期，资产相关性通常会增加
        
        Args:
            returns_matrix: 资产收益率矩阵 (n_assets, n_periods)
            correlation_increase: 相关性增加量
            
        Returns:
            压力后的收益率矩阵
        """
        # 计算原始相关性矩阵
        original_corr = np.corrcoef(returns_matrix)
        
        # 增加相关性（向1靠拢）
        stressed_corr = original_corr + correlation_increase * (1 - original_corr)
        stressed_corr = np.clip(stressed_corr, -1, 1)
        np.fill_diagonal(stressed_corr, 1.0)
        
        # 使用Cholesky分解生成符合新相关性的收益率
        try:
            L = np.linalg.cholesky(stressed_corr)
        except np.linalg.LinAlgError:
            # 如果矩阵非正定，使用近似方法
            eigvals, eigvecs = np.linalg.eigh(stressed_corr)
            eigvals = np.maximum(eigvals, 1e-6)
            stressed_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(stressed_corr)
        
        # 标准化收益率
        means = np.mean(returns_matrix, axis=1, keepdims=True)
        stds = np.std(returns_matrix, axis=1, keepdims=True)
        standardized = (returns_matrix - means) / (stds + 1e-10)
        
        # 应用新的相关性结构
        stressed_standardized = L @ np.linalg.solve(
            np.linalg.cholesky(np.corrcoef(returns_matrix)), 
            standardized
        )
        
        # 恢复原始尺度
        stressed_returns = stressed_standardized * stds + means
        
        return stressed_returns
    
    def generate_tail_event(self,
                            returns: np.ndarray,
                            percentile: float = 1.0,
                            n_events: int = 5) -> np.ndarray:
        """
        生成尾部事件场景
        
        Args:
            returns: 原始收益率
            percentile: 尾部百分位（如1表示1%最差情况）
            n_events: 尾部事件数量
            
        Returns:
            包含尾部事件的收益率
        """
        stressed = returns.copy()
        
        # 获取极端负收益
        threshold = np.percentile(returns, percentile)
        
        # 在最后n_events天插入极端事件
        if len(stressed) >= n_events:
            stressed[-n_events:] = threshold
        
        return stressed


class StressTester:
    """压力测试执行器"""
    
    def __init__(self):
        """初始化压力测试器"""
        self.scenario_generator = ScenarioGenerator()
        self.var_calculator = VaRCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        
        self.results: List[Dict] = []
    
    def run_historical_scenario(self,
                                portfolio_returns: np.ndarray,
                                scenario_name: str) -> Dict:
        """
        运行历史场景压力测试
        
        Args:
            portfolio_returns: 组合收益率
            scenario_name: 场景名称
            
        Returns:
            压力测试结果
        """
        scenario = HistoricalScenario.get_scenario(scenario_name)
        
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # 应用场景冲击
        shocked_returns = self.scenario_generator.generate_market_shock(
            portfolio_returns,
            shock_pct=scenario['market_shock'],
            duration_days=20
        )
        
        # 应用波动率压力
        stressed_returns = self.scenario_generator.generate_volatility_stress(
            shocked_returns,
            vol_multiplier=scenario['volatility_multiplier']
        )
        
        # 计算压力下的指标
        cumulative_return = np.prod(1 + stressed_returns) - 1
        stressed_prices = np.cumprod(1 + stressed_returns)
        
        result = {
            'scenario': scenario_name,
            'scenario_info': scenario,
            'original_return': float(np.prod(1 + portfolio_returns) - 1),
            'stressed_return': float(cumulative_return),
            'max_drawdown': float(self.drawdown_analyzer.calculate_drawdowns(stressed_prices)['max_drawdown']),
            'stressed_volatility': float(np.std(stressed_returns) * np.sqrt(252)),
            'var_95': float(self.var_calculator.historical_var(stressed_returns, 0.95)),
            'worst_day': float(np.min(stressed_returns)),
            'n_negative_days': int(np.sum(stressed_returns < 0)),
        }
        
        self.results.append(result)
        
        return result
    
    def run_hypothetical_scenario(self,
                                   portfolio_returns: np.ndarray,
                                   market_shock: float = -0.20,
                                   vol_multiplier: float = 2.0,
                                   scenario_name: str = 'Custom') -> Dict:
        """
        运行假设场景压力测试
        
        Args:
            portfolio_returns: 组合收益率
            market_shock: 市场冲击幅度
            vol_multiplier: 波动率放大倍数
            scenario_name: 场景名称
            
        Returns:
            压力测试结果
        """
        # 应用冲击
        shocked = self.scenario_generator.generate_market_shock(
            portfolio_returns, market_shock, 20
        )
        stressed = self.scenario_generator.generate_volatility_stress(
            shocked, vol_multiplier
        )
        
        stressed_prices = np.cumprod(1 + stressed)
        
        result = {
            'scenario': scenario_name,
            'market_shock': market_shock,
            'vol_multiplier': vol_multiplier,
            'original_return': float(np.prod(1 + portfolio_returns) - 1),
            'stressed_return': float(np.prod(1 + stressed) - 1),
            'max_drawdown': float(self.drawdown_analyzer.calculate_drawdowns(stressed_prices)['max_drawdown']),
            'stressed_volatility': float(np.std(stressed) * np.sqrt(252)),
            'var_95': float(self.var_calculator.historical_var(stressed, 0.95)),
        }
        
        self.results.append(result)
        
        return result
    
    def run_monte_carlo_stress(self,
                                portfolio_returns: np.ndarray,
                                n_simulations: int = 1000,
                                stress_factor: float = 2.0) -> Dict:
        """
        运行蒙特卡洛压力测试
        
        Args:
            portfolio_returns: 组合收益率
            n_simulations: 模拟次数
            stress_factor: 压力因子
            
        Returns:
            压力测试结果
        """
        mean = np.mean(portfolio_returns)
        std = np.std(portfolio_returns) * stress_factor
        
        simulated_returns = []
        simulated_drawdowns = []
        
        n_days = len(portfolio_returns)
        
        for _ in range(n_simulations):
            # 生成压力收益率路径
            sim_returns = np.random.normal(mean, std, n_days)
            sim_prices = np.cumprod(1 + sim_returns)
            
            total_return = sim_prices[-1] - 1
            dd = self.drawdown_analyzer.calculate_drawdowns(sim_prices)['max_drawdown']
            
            simulated_returns.append(total_return)
            simulated_drawdowns.append(dd)
        
        simulated_returns = np.array(simulated_returns)
        simulated_drawdowns = np.array(simulated_drawdowns)
        
        result = {
            'scenario': 'Monte Carlo',
            'n_simulations': n_simulations,
            'stress_factor': stress_factor,
            'return_mean': float(np.mean(simulated_returns)),
            'return_std': float(np.std(simulated_returns)),
            'return_5th_percentile': float(np.percentile(simulated_returns, 5)),
            'return_1st_percentile': float(np.percentile(simulated_returns, 1)),
            'max_drawdown_mean': float(np.mean(simulated_drawdowns)),
            'max_drawdown_95th': float(np.percentile(simulated_drawdowns, 95)),
            'worst_case_return': float(np.min(simulated_returns)),
            'worst_case_drawdown': float(np.min(simulated_drawdowns)),
            'prob_loss_20pct': float(np.mean(simulated_returns < -0.20)),
            'prob_loss_50pct': float(np.mean(simulated_returns < -0.50)),
        }
        
        self.results.append(result)
        
        return result
    
    def run_sensitivity_analysis(self,
                                  portfolio_returns: np.ndarray,
                                  shock_levels: List[float] = None) -> pd.DataFrame:
        """
        运行敏感性分析
        
        分析组合对不同冲击程度的敏感性
        
        Args:
            portfolio_returns: 组合收益率
            shock_levels: 冲击水平列表
            
        Returns:
            敏感性分析结果
        """
        if shock_levels is None:
            shock_levels = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50]
        
        results = []
        
        for shock in shock_levels:
            stressed = self.scenario_generator.generate_market_shock(
                portfolio_returns, shock, 20
            )
            stressed_prices = np.cumprod(1 + stressed)
            
            results.append({
                'market_shock': shock * 100,  # 百分比
                'portfolio_return': (np.prod(1 + stressed) - 1) * 100,
                'max_drawdown': self.drawdown_analyzer.calculate_drawdowns(stressed_prices)['max_drawdown'] * 100,
                'volatility': np.std(stressed) * np.sqrt(252) * 100,
            })
        
        return pd.DataFrame(results)
    
    def run_all_historical_scenarios(self,
                                      portfolio_returns: np.ndarray) -> pd.DataFrame:
        """
        运行所有历史场景
        
        Args:
            portfolio_returns: 组合收益率
            
        Returns:
            所有场景结果
        """
        results = []
        
        for scenario_name in HistoricalScenario.list_scenarios():
            result = self.run_historical_scenario(portfolio_returns, scenario_name)
            results.append(result)
        
        return pd.DataFrame(results)


class StressTestReport:
    """压力测试报告生成器"""
    
    def __init__(self, tester: StressTester = None):
        """初始化报告生成器"""
        self.tester = tester or StressTester()
    
    def generate_comprehensive_report(self,
                                       portfolio_returns: np.ndarray,
                                       portfolio_name: str = 'Portfolio') -> Dict:
        """
        生成综合压力测试报告
        
        Args:
            portfolio_returns: 组合收益率
            portfolio_name: 组合名称
            
        Returns:
            综合报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_name': portfolio_name,
            'baseline': {},
            'historical_scenarios': [],
            'monte_carlo': {},
            'sensitivity': {},
            'summary': {},
        }
        
        # 基准指标
        prices = np.cumprod(1 + portfolio_returns)
        report['baseline'] = {
            'total_return': float(prices[-1] - 1),
            'annualized_return': float(np.mean(portfolio_returns) * 252),
            'volatility': float(np.std(portfolio_returns) * np.sqrt(252)),
            'max_drawdown': float(DrawdownAnalyzer.calculate_drawdowns(prices)['max_drawdown']),
            'var_95': float(VaRCalculator.historical_var(portfolio_returns, 0.95)),
        }
        
        # 历史场景
        for scenario_name in HistoricalScenario.list_scenarios():
            try:
                result = self.tester.run_historical_scenario(portfolio_returns, scenario_name)
                report['historical_scenarios'].append(result)
            except Exception as e:
                logger.warning(f"Failed to run scenario {scenario_name}: {e}")
        
        # 蒙特卡洛压力测试
        report['monte_carlo'] = self.tester.run_monte_carlo_stress(
            portfolio_returns, n_simulations=1000
        )
        
        # 敏感性分析
        sensitivity_df = self.tester.run_sensitivity_analysis(portfolio_returns)
        report['sensitivity'] = sensitivity_df.to_dict('records')
        
        # 汇总
        if report['historical_scenarios']:
            worst_scenario = min(report['historical_scenarios'], 
                                 key=lambda x: x['stressed_return'])
            report['summary'] = {
                'worst_historical_scenario': worst_scenario['scenario'],
                'worst_historical_return': worst_scenario['stressed_return'],
                'worst_historical_drawdown': worst_scenario['max_drawdown'],
                'mc_1pct_var': report['monte_carlo']['return_1st_percentile'],
                'mc_worst_case': report['monte_carlo']['worst_case_return'],
                'prob_severe_loss': report['monte_carlo']['prob_loss_20pct'],
            }
        
        return report
    
    def format_text_report(self, report: Dict) -> str:
        """格式化文本报告"""
        text = []
        text.append("=" * 60)
        text.append("            压力测试综合报告")
        text.append("=" * 60)
        text.append(f"组合名称: {report['portfolio_name']}")
        text.append(f"报告时间: {report['timestamp'][:19]}")
        text.append("")
        
        # 基准
        text.append("📊 基准指标:")
        text.append("-" * 40)
        baseline = report['baseline']
        text.append(f"  总收益率: {baseline['total_return']*100:.2f}%")
        text.append(f"  年化收益: {baseline['annualized_return']*100:.2f}%")
        text.append(f"  波动率: {baseline['volatility']*100:.2f}%")
        text.append(f"  最大回撤: {baseline['max_drawdown']*100:.2f}%")
        text.append(f"  VaR(95%): {baseline['var_95']*100:.2f}%")
        text.append("")
        
        # 历史场景
        text.append("📈 历史场景压力测试:")
        text.append("-" * 40)
        for scenario in report['historical_scenarios']:
            info = scenario.get('scenario_info', {})
            text.append(f"  【{info.get('name', scenario['scenario'])}】")
            text.append(f"    压力收益: {scenario['stressed_return']*100:.2f}%")
            text.append(f"    最大回撤: {scenario['max_drawdown']*100:.2f}%")
        text.append("")
        
        # 蒙特卡洛
        text.append("🎲 蒙特卡洛压力测试:")
        text.append("-" * 40)
        mc = report['monte_carlo']
        text.append(f"  模拟次数: {mc['n_simulations']}")
        text.append(f"  平均收益: {mc['return_mean']*100:.2f}%")
        text.append(f"  5%分位收益: {mc['return_5th_percentile']*100:.2f}%")
        text.append(f"  1%分位收益: {mc['return_1st_percentile']*100:.2f}%")
        text.append(f"  最坏情况: {mc['worst_case_return']*100:.2f}%")
        text.append(f"  亏损>20%概率: {mc['prob_loss_20pct']*100:.2f}%")
        text.append("")
        
        # 汇总
        if report['summary']:
            text.append("⚠️ 风险汇总:")
            text.append("-" * 40)
            summary = report['summary']
            text.append(f"  最差历史场景: {summary['worst_historical_scenario']}")
            text.append(f"  最差历史收益: {summary['worst_historical_return']*100:.2f}%")
            text.append(f"  MC 1%VaR: {summary['mc_1pct_var']*100:.2f}%")
        
        text.append("")
        text.append("=" * 60)
        
        return "\n".join(text)


if __name__ == "__main__":
    print("压力测试模块测试")
    print("="*50)
    
    # 生成模拟组合收益率
    np.random.seed(42)
    n_days = 252  # 一年
    portfolio_returns = np.random.randn(n_days) * 0.02  # 2%日波动
    
    # 创建压力测试器
    tester = StressTester()
    
    # 测试历史场景
    print("\n1. 历史场景压力测试")
    for scenario in HistoricalScenario.list_scenarios()[:2]:
        result = tester.run_historical_scenario(portfolio_returns, scenario)
        print(f"  {result['scenario_info']['name']}:")
        print(f"    原始收益: {result['original_return']*100:.2f}%")
        print(f"    压力收益: {result['stressed_return']*100:.2f}%")
    
    # 测试蒙特卡洛
    print("\n2. 蒙特卡洛压力测试")
    mc_result = tester.run_monte_carlo_stress(portfolio_returns, n_simulations=500)
    print(f"  5%分位收益: {mc_result['return_5th_percentile']*100:.2f}%")
    print(f"  最坏情况: {mc_result['worst_case_return']*100:.2f}%")
    
    # 测试敏感性分析
    print("\n3. 敏感性分析")
    sensitivity = tester.run_sensitivity_analysis(portfolio_returns)
    print(sensitivity.to_string(index=False))
    
    # 综合报告
    print("\n4. 综合压力测试报告")
    reporter = StressTestReport()
    report = reporter.generate_comprehensive_report(portfolio_returns, "测试组合")
    print(reporter.format_text_report(report))
    
    print("\n压力测试模块测试完成!")
