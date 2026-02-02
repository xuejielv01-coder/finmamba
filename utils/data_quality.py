# -*- coding: utf-8 -*-
"""
数据质量监控模块
改进4：实现数据质量Dashboard

功能:
- 监控缺失值、异常值、分布变化
- 实现数据漂移检测
- 生成数据质量报告
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("DataQuality")


class DataQualityMetrics:
    """数据质量指标计算"""
    
    @staticmethod
    def calculate_missing_ratio(df: pd.DataFrame) -> Dict[str, float]:
        """计算每列的缺失率"""
        return (df.isnull().sum() / len(df)).to_dict()
    
    @staticmethod
    def calculate_outlier_ratio(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, float]:
        """
        计算每列的异常值比例（基于Z-Score）
        
        Args:
            df: 数据框
            threshold: Z-Score阈值
        """
        outlier_ratios = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                z_scores = np.abs(stats.zscore(values))
                outlier_ratios[col] = (z_scores > threshold).mean()
            else:
                outlier_ratios[col] = 0.0
        
        return outlier_ratios
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """计算每列的基础统计量"""
        stats_dict = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                stats_dict[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'skewness': float(values.skew()) if len(values) > 2 else 0.0,
                    'kurtosis': float(values.kurtosis()) if len(values) > 3 else 0.0,
                    'q1': float(values.quantile(0.25)),
                    'q3': float(values.quantile(0.75)),
                }
            else:
                stats_dict[col] = {}
        
        return stats_dict


class DataDriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, reference_window: int = 30, threshold: float = 0.05):
        """
        初始化漂移检测器
        
        Args:
            reference_window: 参考窗口大小（天数）
            threshold: 检测阈值
        """
        self.reference_window = reference_window
        self.threshold = threshold
        self.reference_stats: Dict[str, Dict] = {}
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        拟合参考分布
        
        Args:
            df: 参考数据
            feature_cols: 特征列名
        """
        for col in feature_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    self.reference_stats[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                    }
        
        logger.info(f"Data drift detector fitted with {len(self.reference_stats)} features")
    
    def detect(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        检测数据漂移
        
        Args:
            df: 当前数据
            
        Returns:
            漂移检测结果
        """
        drift_results = {}
        
        for col, ref_stats in self.reference_stats.items():
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            current_mean = float(values.mean())
            current_std = float(values.std())
            
            # 计算均值漂移（使用标准化差异）
            if ref_stats['std'] > 1e-10:
                mean_drift = abs(current_mean - ref_stats['mean']) / ref_stats['std']
            else:
                mean_drift = 0.0
            
            # 计算方差比
            if ref_stats['std'] > 1e-10:
                std_ratio = current_std / ref_stats['std']
            else:
                std_ratio = 1.0
            
            # KS检验（需要参考数据）
            # 这里使用简化的方法：基于均值和标准差
            is_drifted = mean_drift > 2.0 or std_ratio > 2.0 or std_ratio < 0.5
            
            drift_results[col] = {
                'mean_drift': mean_drift,
                'std_ratio': std_ratio,
                'is_drifted': is_drifted,
                'current_mean': current_mean,
                'current_std': current_std,
                'reference_mean': ref_stats['mean'],
                'reference_std': ref_stats['std'],
            }
        
        # 统计漂移特征数量
        n_drifted = sum(1 for r in drift_results.values() if r['is_drifted'])
        logger.info(f"Data drift detection: {n_drifted}/{len(drift_results)} features drifted")
        
        return drift_results


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, storage_dir: Path = None):
        """
        初始化监控器
        
        Args:
            storage_dir: 质量报告存储目录
        """
        self.storage_dir = storage_dir or Config.DATA_DIR / "quality_reports"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = DataQualityMetrics()
        self.drift_detector = DataDriftDetector()
        
        # 历史报告
        self.history: List[Dict] = []
        self._load_history()
        
        logger.info(f"Data quality monitor initialized, storage: {self.storage_dir}")
    
    def _load_history(self):
        """加载历史报告"""
        history_file = self.storage_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} historical reports")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self.history = []
    
    def _save_history(self):
        """保存历史报告"""
        history_file = self.storage_dir / "history.json"
        try:
            # 只保留最近30天的报告
            cutoff = datetime.now() - timedelta(days=30)
            self.history = [
                h for h in self.history 
                if datetime.fromisoformat(h['timestamp']) > cutoff
            ]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def analyze_stock_data(self, ts_code: str, df: pd.DataFrame) -> Dict:
        """
        分析单只股票的数据质量
        
        Args:
            ts_code: 股票代码
            df: 股票数据
            
        Returns:
            质量报告
        """
        report = {
            'ts_code': ts_code,
            'timestamp': datetime.now().isoformat(),
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'date_range': None,
            'missing_ratios': {},
            'outlier_ratios': {},
            'statistics': {},
            'issues': [],
            'quality_score': 100.0,
        }
        
        if df.empty:
            report['issues'].append("数据为空")
            report['quality_score'] = 0.0
            return report
        
        # 日期范围
        if 'trade_date' in df.columns:
            report['date_range'] = {
                'start': str(df['trade_date'].min()),
                'end': str(df['trade_date'].max()),
            }
        
        # 缺失率
        report['missing_ratios'] = self.metrics.calculate_missing_ratio(df)
        
        # 检查高缺失率列
        high_missing = [
            col for col, ratio in report['missing_ratios'].items() 
            if ratio > 0.1
        ]
        if high_missing:
            report['issues'].append(f"高缺失率列: {', '.join(high_missing)}")
            report['quality_score'] -= len(high_missing) * 5
        
        # 异常值
        report['outlier_ratios'] = self.metrics.calculate_outlier_ratio(df)
        
        # 检查高异常值列
        high_outlier = [
            col for col, ratio in report['outlier_ratios'].items() 
            if ratio > 0.05
        ]
        if high_outlier:
            report['issues'].append(f"高异常值列: {', '.join(high_outlier)}")
            report['quality_score'] -= len(high_outlier) * 3
        
        # 基础统计
        report['statistics'] = self.metrics.calculate_statistics(df)
        
        # 检查数据完整性
        if len(df) < 20:
            report['issues'].append(f"数据量不足: 仅{len(df)}条")
            report['quality_score'] -= 20
        
        # 确保质量分数在0-100之间
        report['quality_score'] = max(0.0, min(100.0, report['quality_score']))
        
        return report
    
    def analyze_batch(self, data_dir: Path = None, sample_size: int = 100) -> Dict:
        """
        批量分析数据质量
        
        Args:
            data_dir: 数据目录
            sample_size: 采样数量
            
        Returns:
            汇总报告
        """
        data_dir = data_dir or Config.RAW_DATA_DIR
        
        # 获取所有数据文件
        files = list(data_dir.glob("*.parquet"))
        
        if len(files) > sample_size:
            import random
            files = random.sample(files, sample_size)
        
        logger.info(f"Analyzing {len(files)} stock data files...")
        
        all_reports = []
        issues_summary = defaultdict(int)
        quality_scores = []
        
        for file_path in files:
            try:
                ts_code = file_path.stem.replace('_', '.')
                df = pd.read_parquet(file_path)
                
                report = self.analyze_stock_data(ts_code, df)
                all_reports.append(report)
                quality_scores.append(report['quality_score'])
                
                for issue in report['issues']:
                    issues_summary[issue] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # 汇总报告
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_stocks_analyzed': len(all_reports),
            'avg_quality_score': float(np.mean(quality_scores)) if quality_scores else 0.0,
            'min_quality_score': float(np.min(quality_scores)) if quality_scores else 0.0,
            'max_quality_score': float(np.max(quality_scores)) if quality_scores else 0.0,
            'issues_summary': dict(issues_summary),
            'n_low_quality': sum(1 for s in quality_scores if s < 70),
            'n_high_quality': sum(1 for s in quality_scores if s >= 90),
        }
        
        # 保存报告
        self.history.append(summary)
        self._save_history()
        
        # 保存详细报告
        report_file = self.storage_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'details': all_reports
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch analysis complete. Avg quality score: {summary['avg_quality_score']:.1f}")
        
        return summary
    
    def get_quality_trend(self) -> pd.DataFrame:
        """获取质量趋势数据"""
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    def generate_html_report(self, summary: Dict = None) -> str:
        """
        生成HTML格式的质量报告
        
        Args:
            summary: 汇总报告
            
        Returns:
            HTML字符串
        """
        if summary is None:
            summary = self.history[-1] if self.history else {}
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>数据质量报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', sans-serif;
                    background: #0f0f23;
                    color: #ffffff;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1 {{
                    color: #e94560;
                    border-bottom: 2px solid #e94560;
                    padding-bottom: 10px;
                }}
                .card {{
                    background: #1a1a3e;
                    border-radius: 12px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                }}
                .metric {{
                    display: inline-block;
                    width: 200px;
                    text-align: center;
                    padding: 15px;
                    margin: 10px;
                    background: #16213e;
                    border-radius: 8px;
                }}
                .metric-value {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #00c853;
                }}
                .metric-label {{
                    color: #888;
                    font-size: 12px;
                    margin-top: 5px;
                }}
                .issue-item {{
                    background: #2a2a4e;
                    padding: 10px 15px;
                    margin: 5px 0;
                    border-radius: 6px;
                    border-left: 3px solid #ff5252;
                }}
                .good {{ color: #00c853; }}
                .warning {{ color: #ffeb3b; }}
                .bad {{ color: #ff5252; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 数据质量监控报告</h1>
                <p>生成时间: {summary.get('timestamp', 'N/A')}</p>
                
                <div class="card">
                    <h2>📈 总体指标</h2>
                    <div class="metric">
                        <div class="metric-value">{summary.get('n_stocks_analyzed', 0)}</div>
                        <div class="metric-label">分析股票数</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {'good' if summary.get('avg_quality_score', 0) >= 80 else 'warning' if summary.get('avg_quality_score', 0) >= 60 else 'bad'}">{summary.get('avg_quality_score', 0):.1f}</div>
                        <div class="metric-label">平均质量分数</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value good">{summary.get('n_high_quality', 0)}</div>
                        <div class="metric-label">高质量数据</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value bad">{summary.get('n_low_quality', 0)}</div>
                        <div class="metric-label">低质量数据</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>⚠️ 问题汇总</h2>
        """
        
        for issue, count in summary.get('issues_summary', {}).items():
            html += f'<div class="issue-item">{issue}: <strong>{count}</strong> 个股票</div>\n'
        
        if not summary.get('issues_summary'):
            html += '<p class="good">暂无发现问题 ✓</p>'
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


class DataQualityDashboard:
    """数据质量仪表盘（GUI组件）"""
    
    def __init__(self, monitor: DataQualityMonitor = None):
        self.monitor = monitor or DataQualityMonitor()
    
    def get_summary_data(self) -> Dict:
        """获取仪表盘数据"""
        summary = self.monitor.history[-1] if self.monitor.history else {}
        trend = self.monitor.get_quality_trend()
        
        return {
            'summary': summary,
            'trend': trend.to_dict() if not trend.empty else {},
            'last_updated': summary.get('timestamp', 'Never'),
        }


# 命令行工具
def run_quality_check():
    """运行数据质量检查"""
    monitor = DataQualityMonitor()
    summary = monitor.analyze_batch(sample_size=50)
    
    print("\n" + "="*60)
    print("📊 数据质量检查报告")
    print("="*60)
    print(f"分析股票数: {summary['n_stocks_analyzed']}")
    print(f"平均质量分数: {summary['avg_quality_score']:.1f}")
    print(f"高质量数据: {summary['n_high_quality']}")
    print(f"低质量数据: {summary['n_low_quality']}")
    print("\n问题汇总:")
    for issue, count in summary['issues_summary'].items():
        print(f"  - {issue}: {count}")
    print("="*60)
    
    # 生成HTML报告
    html = monitor.generate_html_report(summary)
    report_path = monitor.storage_dir / "latest_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    run_quality_check()
