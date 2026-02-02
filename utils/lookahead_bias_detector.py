# -*- coding: utf-8 -*-
"""
前瞻偏差检测模块
改进25：实现数据泄漏和前瞻性偏差检测

功能:
- 时间序列验证完整性检查
- 特征-标签相关性检测
- 前瞻性特征检测
- 数据泄漏风险评估
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("LookaheadBiasDetector")


class TemporalLeakageDetector:
    """时间泄漏检测器"""
    
    def __init__(self, date_column: str = 'trade_date'):
        """
        初始化时间泄漏检测器
        
        Args:
            date_column: 日期列名
        """
        self.date_column = date_column
        self.issues: List[Dict] = []
    
    def check_train_test_overlap(self,
                                 train_df: pd.DataFrame,
                                 test_df: pd.DataFrame) -> Dict:
        """
        检查训练集和测试集的时间重叠
        
        Args:
            train_df: 训练集
            test_df: 测试集
            
        Returns:
            检查结果
        """
        if self.date_column not in train_df.columns or self.date_column not in test_df.columns:
            return {
                'passed': False,
                'error': f'Date column "{self.date_column}" not found',
                'overlap_dates': [],
            }
        
        train_dates = set(train_df[self.date_column].unique())
        test_dates = set(test_df[self.date_column].unique())
        
        overlap = train_dates & test_dates
        
        result = {
            'passed': len(overlap) == 0,
            'train_date_range': (min(train_dates), max(train_dates)),
            'test_date_range': (min(test_dates), max(test_dates)),
            'overlap_count': len(overlap),
            'overlap_dates': sorted(list(overlap))[:10],  # 前10个重叠日期
        }
        
        if not result['passed']:
            self.issues.append({
                'type': 'train_test_overlap',
                'severity': 'critical',
                'message': f"Found {len(overlap)} overlapping dates between train and test sets",
            })
        
        return result
    
    def check_future_data_usage(self,
                                df: pd.DataFrame,
                                feature_columns: List[str],
                                label_column: str,
                                lookback: int = 20) -> Dict:
        """
        检查是否使用了未来数据
        
        通过检查特征和未来标签的相关性来检测
        """
        if self.date_column not in df.columns:
            return {'passed': False, 'error': 'Date column not found'}
        
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        suspicious_features = []
        
        for col in feature_columns:
            if col not in df.columns or label_column not in df.columns:
                continue
            
            # 计算特征与未来各期标签的相关性
            for shift in range(1, lookback + 1):
                # 未来标签
                future_label = df[label_column].shift(-shift)
                
                # 计算相关性
                valid_mask = ~(df[col].isna() | future_label.isna())
                if valid_mask.sum() < 30:
                    continue
                
                corr = df[col][valid_mask].corr(future_label[valid_mask])
                
                # 与当期标签的相关性
                current_corr = df[col].corr(df[label_column])
                
                # 如果与未来标签的相关性显著高于当期，可能存在泄漏
                if abs(corr) > abs(current_corr) * 1.5 and abs(corr) > 0.1:
                    suspicious_features.append({
                        'feature': col,
                        'future_shift': shift,
                        'future_corr': float(corr),
                        'current_corr': float(current_corr),
                        'corr_ratio': float(abs(corr) / (abs(current_corr) + 1e-10)),
                    })
        
        passed = len(suspicious_features) == 0
        
        if not passed:
            self.issues.append({
                'type': 'future_data_usage',
                'severity': 'critical',
                'message': f"Found {len(suspicious_features)} features with suspicious future correlations",
                'details': suspicious_features[:5],
            })
        
        return {
            'passed': passed,
            'suspicious_features': suspicious_features,
            'n_suspicious': len(suspicious_features),
        }
    
    def check_data_ordering(self, df: pd.DataFrame) -> Dict:
        """
        检查数据是否按正确的时间顺序排列
        """
        if self.date_column not in df.columns:
            return {'passed': False, 'error': 'Date column not found'}
        
        dates = df[self.date_column].values
        is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
        
        if not is_sorted:
            # 找出乱序位置
            disorder_positions = [i for i in range(len(dates)-1) if dates[i] > dates[i+1]]
            
            self.issues.append({
                'type': 'data_ordering',
                'severity': 'warning',
                'message': f"Data not properly sorted by date at {len(disorder_positions)} positions",
            })
        
        return {
            'passed': is_sorted,
            'is_sorted': is_sorted,
            'n_disorder_positions': 0 if is_sorted else len([i for i in range(len(dates)-1) if dates[i] > dates[i+1]]),
        }


class FeatureLeakageDetector:
    """特征泄漏检测器"""
    
    def __init__(self):
        self.issues: List[Dict] = []
        
        # 已知的危险特征模式
        self.dangerous_patterns = [
            'future', 'next', 'forward', 'target', 'label', 'return_t1',
            'price_t+', 'close_t+', 'high_t+', 'low_t+',
        ]
    
    def check_feature_names(self, feature_columns: List[str]) -> Dict:
        """
        检查特征名称是否包含危险模式
        """
        suspicious_names = []
        
        for col in feature_columns:
            col_lower = col.lower()
            for pattern in self.dangerous_patterns:
                if pattern in col_lower:
                    suspicious_names.append({
                        'feature': col,
                        'matched_pattern': pattern,
                    })
                    break
        
        passed = len(suspicious_names) == 0
        
        if not passed:
            self.issues.append({
                'type': 'suspicious_feature_names',
                'severity': 'warning',
                'message': f"Found {len(suspicious_names)} features with suspicious names",
                'details': suspicious_names,
            })
        
        return {
            'passed': passed,
            'suspicious_names': suspicious_names,
        }
    
    def check_perfect_correlation(self,
                                  df: pd.DataFrame,
                                  feature_columns: List[str],
                                  label_column: str,
                                  threshold: float = 0.99) -> Dict:
        """
        检查与标签完美相关的特征（可能是泄漏的标志）
        """
        if label_column not in df.columns:
            return {'passed': False, 'error': 'Label column not found'}
        
        perfect_correlations = []
        
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            corr = df[col].corr(df[label_column])
            
            if abs(corr) >= threshold:
                perfect_correlations.append({
                    'feature': col,
                    'correlation': float(corr),
                })
        
        passed = len(perfect_correlations) == 0
        
        if not passed:
            self.issues.append({
                'type': 'perfect_correlation',
                'severity': 'critical',
                'message': f"Found {len(perfect_correlations)} features with near-perfect correlation to label",
                'details': perfect_correlations,
            })
        
        return {
            'passed': passed,
            'perfect_correlations': perfect_correlations,
        }
    
    def check_information_leakage_via_mean(self,
                                           train_df: pd.DataFrame,
                                           test_df: pd.DataFrame,
                                           feature_columns: List[str]) -> Dict:
        """
        检查是否使用整体数据计算的统计量（而非仅训练集）
        
        如果测试集特征的标准化使用了包含测试集的统计量，会造成泄漏
        """
        suspicious_features = []
        
        for col in feature_columns:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            
            train_mean = train_df[col].mean()
            train_std = train_df[col].std()
            
            test_mean = test_df[col].mean()
            test_std = test_df[col].std()
            
            # 如果测试集的均值/标准差与训练集过于接近，可能存在泄漏
            # （实际中测试集通常应该有一定漂移）
            mean_ratio = abs(test_mean - train_mean) / (train_std + 1e-10)
            
            # 这里只是启发式检查
            if mean_ratio < 0.01 and len(train_df) > 100 and len(test_df) > 20:
                suspicious_features.append({
                    'feature': col,
                    'train_mean': float(train_mean),
                    'test_mean': float(test_mean),
                    'mean_ratio': float(mean_ratio),
                })
        
        # 这个检查通常会产生假阳性，设为低优先级
        return {
            'passed': True,  # 总是通过，仅作为信息
            'potentially_suspicious': suspicious_features[:5],
            'note': 'This check is informational and may produce false positives',
        }


class CrossValidationChecker:
    """交叉验证完整性检查器"""
    
    def __init__(self, date_column: str = 'trade_date'):
        self.date_column = date_column
        self.issues: List[Dict] = []
    
    def check_time_series_cv(self,
                             df: pd.DataFrame,
                             n_splits: int,
                             train_ratio: float = 0.7) -> Dict:
        """
        检查时间序列交叉验证的正确性
        
        正确的时间序列CV应该：
        1. 训练集总是在验证集之前
        2. 没有时间重叠
        """
        if self.date_column not in df.columns:
            return {'passed': False, 'error': 'Date column not found'}
        
        dates = sorted(df[self.date_column].unique())
        n_dates = len(dates)
        
        splits_info = []
        all_valid = True
        
        for i in range(n_splits):
            # 扩展窗口方法
            train_end_idx = int(n_dates * (train_ratio + (1 - train_ratio) * i / n_splits))
            val_start_idx = train_end_idx
            val_end_idx = int(n_dates * (train_ratio + (1 - train_ratio) * (i + 1) / n_splits))
            
            train_dates = dates[:train_end_idx]
            val_dates = dates[val_start_idx:val_end_idx]
            
            # 检查无重叠
            overlap = set(train_dates) & set(val_dates)
            is_valid = len(overlap) == 0 and (not train_dates or not val_dates or max(train_dates) < min(val_dates))
            
            if not is_valid:
                all_valid = False
            
            splits_info.append({
                'split': i,
                'train_start': train_dates[0] if train_dates else None,
                'train_end': train_dates[-1] if train_dates else None,
                'val_start': val_dates[0] if val_dates else None,
                'val_end': val_dates[-1] if val_dates else None,
                'is_valid': is_valid,
            })
        
        if not all_valid:
            self.issues.append({
                'type': 'cv_temporal_violation',
                'severity': 'critical',
                'message': 'Time series cross-validation has temporal violations',
            })
        
        return {
            'passed': all_valid,
            'n_splits': n_splits,
            'splits': splits_info,
        }
    
    def suggest_purge_embargo(self,
                              df: pd.DataFrame,
                              lookback: int,
                              horizon: int) -> Dict:
        """
        建议Purge和Embargo参数
        
        Purge: 训练集和验证集之间的间隔，防止标签泄漏
        Embargo: 验证集后的额外间隔
        """
        suggested_purge = max(lookback, horizon)  # 至少等于lookback
        suggested_embargo = horizon  # 至少等于预测horizon
        
        return {
            'suggested_purge_days': suggested_purge,
            'suggested_embargo_days': suggested_embargo,
            'lookback': lookback,
            'horizon': horizon,
            'explanation': f"With lookback={lookback} and horizon={horizon}, "
                          f"use purge={suggested_purge} and embargo={suggested_embargo} "
                          f"to prevent data leakage in time series CV.",
        }


class LookaheadBiasReport:
    """前瞻偏差综合报告"""
    
    def __init__(self, date_column: str = 'trade_date'):
        self.temporal_detector = TemporalLeakageDetector(date_column)
        self.feature_detector = FeatureLeakageDetector()
        self.cv_checker = CrossValidationChecker(date_column)
    
    def comprehensive_check(self,
                           train_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           feature_columns: List[str],
                           label_column: str,
                           lookback: int = 20,
                           horizon: int = 1) -> Dict:
        """
        执行全面的前瞻偏差检查
        
        Args:
            train_df: 训练数据
            test_df: 测试数据
            feature_columns: 特征列名
            label_column: 标签列名
            lookback: 模型回溯期
            horizon: 预测horizon
            
        Returns:
            综合检查报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_passed': True,
            'risk_level': 'low',
            'issues': [],
        }
        
        # 1. 训练/测试集时间重叠检查
        logger.info("Checking train/test temporal overlap...")
        overlap_check = self.temporal_detector.check_train_test_overlap(train_df, test_df)
        report['checks']['train_test_overlap'] = overlap_check
        if not overlap_check['passed']:
            report['overall_passed'] = False
            report['risk_level'] = 'critical'
        
        # 2. 数据排序检查
        logger.info("Checking data ordering...")
        order_check = self.temporal_detector.check_data_ordering(
            pd.concat([train_df, test_df], ignore_index=True)
        )
        report['checks']['data_ordering'] = order_check
        
        # 3. 特征名称检查
        logger.info("Checking feature names...")
        name_check = self.feature_detector.check_feature_names(feature_columns)
        report['checks']['feature_names'] = name_check
        if not name_check['passed']:
            if report['risk_level'] != 'critical':
                report['risk_level'] = 'warning'
        
        # 4. 完美相关性检查
        logger.info("Checking perfect correlations...")
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        corr_check = self.feature_detector.check_perfect_correlation(
            combined_df, feature_columns, label_column
        )
        report['checks']['perfect_correlation'] = corr_check
        if not corr_check['passed']:
            report['overall_passed'] = False
            report['risk_level'] = 'critical'
        
        # 5. 未来数据使用检查
        logger.info("Checking future data usage...")
        future_check = self.temporal_detector.check_future_data_usage(
            combined_df, feature_columns, label_column, lookback
        )
        report['checks']['future_data_usage'] = future_check
        if not future_check['passed']:
            report['overall_passed'] = False
            report['risk_level'] = 'critical'
        
        # 6. CV参数建议
        cv_suggestion = self.cv_checker.suggest_purge_embargo(
            combined_df, lookback, horizon
        )
        report['cv_recommendation'] = cv_suggestion
        
        # 收集所有问题
        report['issues'] = (
            self.temporal_detector.issues + 
            self.feature_detector.issues + 
            self.cv_checker.issues
        )
        
        # 生成摘要
        n_critical = sum(1 for i in report['issues'] if i.get('severity') == 'critical')
        n_warning = sum(1 for i in report['issues'] if i.get('severity') == 'warning')
        
        report['summary'] = {
            'n_checks': len(report['checks']),
            'n_passed': sum(1 for c in report['checks'].values() if c.get('passed', True)),
            'n_failed': sum(1 for c in report['checks'].values() if not c.get('passed', True)),
            'n_critical_issues': n_critical,
            'n_warning_issues': n_warning,
        }
        
        logger.info(f"Lookahead bias check complete. Risk level: {report['risk_level']}")
        
        return report
    
    def generate_text_report(self, report: Dict) -> str:
        """生成文本报告"""
        text = f"""
╔══════════════════════════════════════════════════════════════╗
║               前瞻偏差检测报告                                 ║
╠══════════════════════════════════════════════════════════════╣
║ 检查时间: {report['timestamp'][:19]:<41}║
║ 风险等级: {report['risk_level'].upper():<41}║
║ 总体通过: {'✓ 是' if report['overall_passed'] else '✗ 否':<41}║
╠══════════════════════════════════════════════════════════════╣
║ 检查摘要:                                                    ║
║   通过: {report['summary']['n_passed']:<3} / {report['summary']['n_checks']:<3}{' ' * 44}║
║   失败: {report['summary']['n_failed']:<53}║
║   严重问题: {report['summary']['n_critical_issues']:<49}║
║   警告: {report['summary']['n_warning_issues']:<53}║
"""
        
        if report['issues']:
            text += "╠══════════════════════════════════════════════════════════════╣\n"
            text += "║ 发现的问题:                                                  ║\n"
            for issue in report['issues'][:5]:
                severity = issue.get('severity', 'info').upper()
                msg = issue.get('message', '')[:50]
                text += f"║   [{severity}] {msg:<50}║\n"
        
        text += "╠══════════════════════════════════════════════════════════════╣\n"
        text += "║ CV建议:                                                      ║\n"
        cv = report.get('cv_recommendation', {})
        text += f"║   Purge天数: {cv.get('suggested_purge_days', 'N/A'):<48}║\n"
        text += f"║   Embargo天数: {cv.get('suggested_embargo_days', 'N/A'):<46}║\n"
        
        text += "╚══════════════════════════════════════════════════════════════╝"
        
        return text


def quick_check(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                feature_columns: List[str],
                label_column: str) -> bool:
    """
    快速前瞻偏差检查
    
    Returns:
        True if no critical issues found
    """
    reporter = LookaheadBiasReport()
    report = reporter.comprehensive_check(
        train_df, test_df, feature_columns, label_column
    )
    
    print(reporter.generate_text_report(report))
    
    return report['overall_passed']


if __name__ == "__main__":
    print("前瞻偏差检测模块测试")
    print("="*50)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    
    # 模拟特征
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame({
        'trade_date': dates,
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'future_price': np.random.randn(n_samples),  # 可疑特征名
        'returns': np.random.randn(n_samples) * 0.02,
    })
    
    # 分割训练/测试集
    train_df = df.iloc[:400].copy()
    test_df = df.iloc[400:].copy()
    
    feature_cols = ['feature_1', 'feature_2', 'future_price']
    label_col = 'returns'
    
    # 运行检查
    result = quick_check(train_df, test_df, feature_cols, label_col)
    
    print(f"\n检测结果: {'通过' if result else '存在问题'}")
    print("\n前瞻偏差检测模块测试完成!")
