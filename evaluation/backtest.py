# -*- coding: utf-8 -*-
"""
向量化回测引擎
PRD 5.1 实现

特性:
- 向量化计算提升速度
- T日预测 -> T+1开盘买入 -> T+2开盘卖出
- 费用扣除
- 净值曲线
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("Backtest")


class Backtester:
    """
    向量化回测引擎
    
    策略逻辑:
    - T日: 模型预测
    - T+1日开盘: 买入
    - T+2日开盘: 卖出
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.003,
        top_k: int = 10,
        benchmark_code: str = None
    ):
        """
        初始化回测器
        
        Args:
            transaction_cost: 单次交易成本 (买卖各一半)
            top_k: 每日持仓数量
            benchmark_code: 基准指数代码
        """
        self.transaction_cost = transaction_cost
        self.top_k = top_k
        self.benchmark_code = benchmark_code or Config.INDEX_CODE
        
        # 回测结果
        self.results: Dict = {}
        self.trades: List[Dict] = []
        
        logger.info("Backtester initialized")
    
    def run(self, predictions_df: pd.DataFrame = None, predictions_dir: Path = None, start_date: str = None, end_date: str = None) -> Dict:
        """
        运行回测
        
        Args:
            predictions_df: 预测结果 DataFrame (columns: date, ts_code, score)
            predictions_dir: 预测 CSV 文件目录
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            回测结果字典
        """
        # 加载预测数据
        if predictions_df is None:
            predictions_df = self._load_predictions(predictions_dir or Config.DATA_ROOT / "predictions")
        
        if predictions_df.empty:
            logger.warning("No prediction data for backtesting")
            return {}
        
        # 确保必要的列存在
        required_columns = ['date', 'ts_code', 'score']
        for col in required_columns:
            if col not in predictions_df.columns:
                # 尝试从其他列映射
                if col == 'date':
                    if 'scan_date' in predictions_df.columns:
                        predictions_df['date'] = predictions_df['scan_date']
                    elif 'scan_date' in predictions_df.columns:
                        predictions_df['date'] = predictions_df['scan_date']
                    elif 'signal_date' in predictions_df.columns:
                        predictions_df['date'] = predictions_df['signal_date']
                    elif 'date' not in predictions_df.columns:
                        # 如果没有任何日期列，使用当前日期
                        from datetime import datetime
                        predictions_df['date'] = datetime.now().strftime('%Y%m%d')
                else:
                    logger.warning(f"Missing required column: {col}")
                    return {}
        
        # 日期过滤
        from datetime import datetime, timedelta
        
        # 计算当前日期
        current_date = datetime.now()
        
        # 计算回测数据开始日期：当前日期往前推6个月
        backtest_start_date = current_date - timedelta(days=Config.BACKTEST_START_MONTHS * 30)
        backtest_start_str = backtest_start_date.strftime('%Y%m%d')
        
        # 如果没有指定开始日期，使用前半年作为默认开始日期
        if not start_date:
            start_date = backtest_start_str
            logger.info(f"Using default backtest start date: {start_date}")
        
        # 如果没有指定结束日期，使用当前日期
        if not end_date:
            end_date = current_date.strftime('%Y%m%d')
            logger.info(f"Using default backtest end date: {end_date}")
        
        # 确保回测开始日期不早于前半年
        if start_date < backtest_start_str:
            logger.warning(f"Backtest start date {start_date} is too early, using {backtest_start_str} instead")
            start_date = backtest_start_str
        
        # 过滤预测数据
        predictions_df = predictions_df[predictions_df['date'] >= start_date]
        predictions_df = predictions_df[predictions_df['date'] <= end_date]  
        
        logger.info(f"Backtest date range: {start_date} to {end_date}")
        
        if predictions_df.empty:
            logger.warning("No prediction data after date filtering")
            return {}
        
        # 加载价格数据
        price_data = self._load_price_data(predictions_df['ts_code'].unique())
        
        if price_data.empty:
            logger.warning("No price data for backtesting")
            return {}
        
        # 向量化回测
        self._vectorized_backtest(predictions_df, price_data)
        
        # 计算统计指标
        self._calculate_metrics()
        
        # 加载基准
        self._add_benchmark()
        
        # 计算高级金融指标 (Alpha, Beta, IR 等)
        if 'daily_returns' in self.results and not self.results['daily_returns'].empty:
            from evaluation.metrics import SOTAMetrics
            
            strategy_ret = self.results['daily_returns'].set_index('date')['return']
            bench_ret = None
            if 'benchmark' in self.results and not self.results['benchmark'].empty:
                bench_ret = self.results['benchmark'].set_index('trade_date')['benchmark_return']
            
            adv_metrics = SOTAMetrics.calculate_advanced_metrics(strategy_ret, bench_ret)
            self.results['metrics'].update(adv_metrics)
        
        # 确保返回的结果字典包含必要的键，即使没有交易发生
        if 'daily_returns' not in self.results:
            # 创建空的 daily_returns DataFrame
            self.results['daily_returns'] = pd.DataFrame(columns=['date', 'return', 'n_stocks', 'cum_return'])
            
        if 'metrics' not in self.results:
            # 创建空的 metrics 字典
            self.results['metrics'] = {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'n_trades': 0,
                'n_days': 0
            }
            
        if 'benchmark' not in self.results:
            # 创建空的 benchmark DataFrame
            self.results['benchmark'] = pd.DataFrame(columns=['trade_date', 'benchmark_return', 'benchmark_cum'])
        
        return self.results
    
    def _load_predictions(self, predictions_dir: Path) -> pd.DataFrame:
        """加载预测文件"""
        all_preds = []
        
        for csv_file in predictions_dir.glob("scan_*.csv"):
            try:
                df = pd.read_csv(csv_file, comment='#')
                
                # 从文件名提取日期
                date_str = csv_file.stem.replace('scan_', '')
                
                # 确保date列存在且正确
                df['date'] = date_str
                
                # 也设置signal_date，保持一致性
                df['signal_date'] = date_str
                
                all_preds.append(df)
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
        
        if not all_preds:
            return pd.DataFrame()
        
        return pd.concat(all_preds, ignore_index=True)
    
    def _load_price_data(self, ts_codes) -> pd.DataFrame:
        """加载价格数据"""
        all_prices = []
        
        for ts_code in ts_codes:
            filename = ts_code.replace('.', '_') + '.parquet'
            filepath = Config.RAW_DATA_DIR / filename
            
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    df = df[['ts_code', 'trade_date', 'open', 'close']].copy()
                    all_prices.append(df)
                except Exception:
                    continue
        
        if not all_prices:
            return pd.DataFrame()
        
        prices = pd.concat(all_prices, ignore_index=True)
        prices['trade_date'] = pd.to_datetime(prices['trade_date']).dt.strftime('%Y%m%d')
        return prices
    
    def _vectorized_backtest(self, predictions: pd.DataFrame, prices: pd.DataFrame):
        """向量化回测核心"""
        predictions = predictions.copy()
        
        # 确保日期格式统一
        predictions['date'] = pd.to_datetime(predictions['date']).dt.strftime('%Y%m%d')
        
        # 创建日期映射 (交易日列表)
        all_trade_dates = sorted(prices['trade_date'].unique())
        
        # 获取所有日期
        dates = sorted(predictions['date'].unique())
        
        date_to_next = {}
        date_to_next2 = {}
        for i, d in enumerate(all_trade_dates):
            if i + 1 < len(all_trade_dates):
                date_to_next[d] = all_trade_dates[i + 1]
            if i + 2 < len(all_trade_dates):
                date_to_next2[d] = all_trade_dates[i + 2]
        
        # 记录每日收益
        daily_returns = []
        self.trades = []
        
        for date in dates:
            # 获取当日预测的 top_k 股票
            day_preds = predictions[predictions['date'] == date].nlargest(self.top_k, 'score')
            
            if day_preds.empty:
                continue
            
            # 获取 T+1 和 T+2 日期
            buy_date = date_to_next.get(date)
            sell_date = date_to_next2.get(date)
            
            # 如果找不到直接的 T+1/T+2 日期，尝试使用最近的可用日期
            if not buy_date:
                # 找到大于等于当前日期的第一个交易日期
                next_dates = [d for d in all_trade_dates if d > date]
                if next_dates:
                    buy_date = next_dates[0]
                    # 尝试为新的 buy_date 找到 T+1 作为 sell_date
                    sell_date = date_to_next.get(buy_date)
                    if not sell_date:
                        next_sell_dates = [d for d in all_trade_dates if d > buy_date]
                        sell_date = next_sell_dates[0] if next_sell_dates else None
            
            if not buy_date or not sell_date:
                continue
            
            # 计算每只股票的收益
            stock_returns = []
            
            for _, row in day_preds.iterrows():
                ts_code = row['ts_code']
                
                # 获取买入价 (T+1 开盘价)
                buy_price_row = prices[(prices['ts_code'] == ts_code) & 
                                       (prices['trade_date'] == buy_date)]
                # 获取卖出价 (T+2 开盘价)
                sell_price_row = prices[(prices['ts_code'] == ts_code) & 
                                        (prices['trade_date'] == sell_date)]
                
                if buy_price_row.empty or sell_price_row.empty:
                    continue
                
                buy_price = buy_price_row['open'].iloc[0]
                sell_price = sell_price_row['open'].iloc[0]
                
                # 计算收益 (扣除交易成本)
                ret = (sell_price / buy_price - 1) - self.transaction_cost
                stock_returns.append(ret)
                
                # 记录交易
                self.trades.append({
                    'signal_date': date,
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'ts_code': ts_code,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': ret
                })
            
            if stock_returns:
                # 等权组合收益
                portfolio_return = np.mean(stock_returns)
                daily_returns.append({
                    'date': sell_date,
                    'return': portfolio_return,
                    'n_stocks': len(stock_returns)
                })
        
        # 汇总结果
        if daily_returns:
            self.results['daily_returns'] = pd.DataFrame(daily_returns)
            self.results['daily_returns'] = self.results['daily_returns'].sort_values('date')
            
            # 计算累计净值
            self.results['daily_returns']['cum_return'] = (
                1 + self.results['daily_returns']['return']
            ).cumprod()
            
            self.results['trades'] = pd.DataFrame(self.trades)
    
    def _calculate_metrics(self):
        """计算绩效指标"""
        if 'daily_returns' not in self.results or self.results['daily_returns'].empty:
            # 如果没有 daily_returns，创建空的 metrics 字典
            self.results['metrics'] = {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'n_trades': 0,
                'n_days': 0
            }
            return
        
        daily_returns = self.results['daily_returns']['return']
        
        # 总收益
        total_return = (1 + daily_returns).prod() - 1
        
        # 年化收益
        n_days = len(daily_returns)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        
        # 夏普比率
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
        
        # 最大回撤
        cum_returns = (1 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
        
        # 平均收益/亏损
        avg_win = daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0
        avg_loss = daily_returns[daily_returns < 0].mean() if (daily_returns < 0).any() else 0
        
        # 盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        self.results['metrics'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'n_trades': len(self.trades),
            'n_days': n_days
        }
        
        logger.info(f"Total Return: {total_return*100:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
    
    def _add_benchmark(self):
        """添加基准收益"""
        try:
            index_file = Config.RAW_DATA_DIR / f"index_{self.benchmark_code.replace('.', '_')}.parquet"
            
            if not index_file.exists():
                return
            
            df = pd.read_parquet(index_file)
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            df = df.sort_values('trade_date')
            
            # 计算基准日收益
            df['benchmark_return'] = df['close'].pct_change()
            
            # 与策略日期对齐
            if 'daily_returns' in self.results:
                strategy_dates = set(self.results['daily_returns']['date'].tolist())
                df = df[df['trade_date'].isin(strategy_dates)]
                
                # 累计收益
                df['benchmark_cum'] = (1 + df['benchmark_return']).cumprod()
                
                self.results['benchmark'] = df[['trade_date', 'benchmark_return', 'benchmark_cum']]
                
        except Exception as e:
            logger.error(f"Failed to load benchmark: {e}")
    
    def plot(self, save_path: Path = None, show: bool = True):
        """
        绘制回测结果
        
        Args:
            save_path: 图片保存路径
            show: 是否显示
        """
        if 'daily_returns' not in self.results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 净值曲线
        ax1 = axes[0]
        dates = pd.to_datetime(self.results['daily_returns']['date'])
        strategy_cum = self.results['daily_returns']['cum_return']
        
        ax1.plot(dates, strategy_cum, label='Strategy', color='blue', linewidth=2)
        
        if 'benchmark' in self.results:
            bench_dates = pd.to_datetime(self.results['benchmark']['trade_date'])
            ax1.plot(bench_dates, self.results['benchmark']['benchmark_cum'], 
                    label='Benchmark', color='gray', linestyle='--')
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Net Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 日收益分布
        ax2 = axes[1]
        daily_ret = self.results['daily_returns']['return'] * 100
        ax2.bar(range(len(daily_ret)), daily_ret, 
               color=['green' if r > 0 else 'red' for r in daily_ret],
               alpha=0.7)
        ax2.set_title('Daily Returns (%)')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Return %')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self) -> str:
        """生成回测报告"""
        if 'metrics' not in self.results:
            return "No backtest results available."
        
        m = self.results['metrics']
        
        report = f"""
═══════════════════════════════════════════════
        DeepAlpha Backtest Report
═══════════════════════════════════════════════

📊 Performance Summary
─────────────────────
Total Return:     {m['total_return']*100:>10.2f}%
Annual Return:    {m['annual_return']*100:>10.2f}%
Sharpe Ratio:     {m['sharpe_ratio']:>10.2f}
Max Drawdown:     {m['max_drawdown']*100:>10.2f}%

📈 Trading Statistics
─────────────────────
Win Rate:         {m['win_rate']*100:>10.2f}%
Avg Win:          {m['avg_win']*100:>10.2f}%
Avg Loss:         {m['avg_loss']*100:>10.2f}%
Profit Factor:    {m['profit_factor']:>10.2f}
Total Trades:     {m['n_trades']:>10d}
Trading Days:     {m['n_days']:>10d}

═══════════════════════════════════════════════
"""
        return report


def run_backtest(**kwargs) -> Dict:
    """便捷函数：运行回测"""
    backtester = Backtester(**kwargs)
    return backtester.run()
