# -*- coding: utf-8 -*-
"""
数据质量评估脚本
对下载的原始股票数据进行全面质量检查
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import get_logger

logger = get_logger("DataQualityAudit")

class DataQualityAudit:
    """
    数据质量评估类
    """
    
    def __init__(self, data_dir):
        """
        初始化
        
        Args:
            data_dir: 原始数据目录
        """
        self.data_dir = Path(data_dir)
        self.audit_results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'date_ranges': {},
            'missing_values': {},
            'duplicate_dates': {},
            'data_consistency': {},
            'file_sizes': [],
            'overall_quality': 'Unknown'
        }
    
    def run_audit(self):
        """
        运行完整的数据质量评估
        """
        logger.info(f"开始数据质量评估，检查目录: {self.data_dir}")
        
        # 获取所有数据文件
        parquet_files = list(self.data_dir.glob("*.parquet"))
        self.audit_results['total_files'] = len(parquet_files)
        
        logger.info(f"找到 {len(parquet_files)} 个数据文件")
        
        # 逐个文件检查
        for i, file_path in enumerate(parquet_files):
            if (i + 1) % 100 == 0:
                logger.info(f"已检查 {i + 1}/{len(parquet_files)} 个文件")
            
            self._audit_single_file(file_path)
        
        # 生成评估报告
        self._generate_report()
        
        return self.audit_results
    
    def _audit_single_file(self, file_path):
        """
        评估单个数据文件
        
        Args:
            file_path: 文件路径
        """
        ts_code = file_path.stem.replace('_', '.')
        
        try:
            # 读取文件
            df = pd.read_parquet(file_path)
            self.audit_results['valid_files'] += 1
            
            # 记录文件大小
            self.audit_results['file_sizes'].append(file_path.stat().st_size)
            
            # 检查日期范围
            if 'trade_date' in df.columns:
                min_date = df['trade_date'].min()
                max_date = df['trade_date'].max()
                date_range = f"{min_date} to {max_date}"
                self.audit_results['date_ranges'][ts_code] = date_range
                
                # 检查重复日期
                if df['trade_date'].duplicated().any():
                    duplicate_count = df['trade_date'].duplicated().sum()
                    self.audit_results['duplicate_dates'][ts_code] = duplicate_count
            
            # 检查缺失值
            missing_stats = df.isnull().sum()
            total_missing = missing_stats.sum()
            if total_missing > 0:
                self.audit_results['missing_values'][ts_code] = {
                    'total': total_missing,
                    'columns': {col: int(count) for col, count in missing_stats.items() if count > 0}
                }
            
            # 检查数据一致性
            self._check_data_consistency(df, ts_code)
            
        except Exception as e:
            logger.error(f"文件 {file_path.name} 读取失败: {e}")
            self.audit_results['invalid_files'] += 1
    
    def _check_data_consistency(self, df, ts_code):
        """
        检查数据一致性
        
        Args:
            df: 数据框
            ts_code: 股票代码
        """
        consistency_issues = []
        
        # 检查必要列是否存在
        required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            consistency_issues.append(f"缺失必要列: {missing_columns}")
        
        # 检查价格合理性
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val <= 0:
                    consistency_issues.append(f"{col} 存在非正值: {min_val}")
                if max_val > 10000:
                    consistency_issues.append(f"{col} 值异常大: {max_val}")
        
        # 检查成交量合理性
        if 'vol' in df.columns:
            min_vol = df['vol'].min()
            if min_vol < 0:
                consistency_issues.append(f"成交量存在负值: {min_vol}")
        
        if consistency_issues:
            self.audit_results['data_consistency'][ts_code] = consistency_issues
    
    def _generate_report(self):
        """
        生成评估报告
        """
        logger.info("\n=== 数据质量评估报告 ===")
        logger.info(f"总文件数: {self.audit_results['total_files']}")
        logger.info(f"有效文件数: {self.audit_results['valid_files']}")
        logger.info(f"无效文件数: {self.audit_results['invalid_files']}")
        
        # 计算文件大小统计
        if self.audit_results['file_sizes']:
            avg_size = np.mean(self.audit_results['file_sizes']) / 1024  # KB
            max_size = np.max(self.audit_results['file_sizes']) / 1024  # KB
            min_size = np.min(self.audit_results['file_sizes']) / 1024  # KB
            logger.info(f"平均文件大小: {avg_size:.2f} KB")
            logger.info(f"最大文件大小: {max_size:.2f} KB")
            logger.info(f"最小文件大小: {min_size:.2f} KB")
        
        # 检查日期覆盖情况
        if self.audit_results['date_ranges']:
            date_ranges = list(self.audit_results['date_ranges'].values())
            logger.info(f"日期覆盖检查: {len(date_ranges)} 个文件有有效日期范围")
        
        # 检查缺失值情况
        if self.audit_results['missing_values']:
            logger.warning(f"存在缺失值的文件数: {len(self.audit_results['missing_values'])}")
        
        # 检查重复日期情况
        if self.audit_results['duplicate_dates']:
            logger.warning(f"存在重复日期的文件数: {len(self.audit_results['duplicate_dates'])}")
        
        # 检查数据一致性问题
        if self.audit_results['data_consistency']:
            logger.warning(f"存在数据一致性问题的文件数: {len(self.audit_results['data_consistency'])}")
        
        # 计算整体质量评分
        valid_ratio = self.audit_results['valid_files'] / self.audit_results['total_files'] * 100
        logger.info(f"文件有效性: {valid_ratio:.2f}%")
        
        # 确定整体质量等级
        if valid_ratio >= 95:
            quality = "Excellent"
        elif valid_ratio >= 90:
            quality = "Good"
        elif valid_ratio >= 80:
            quality = "Fair"
        else:
            quality = "Poor"
        
        self.audit_results['overall_quality'] = quality
        logger.info(f"整体数据质量: {quality}")
        
        # 生成详细报告文件
        self._write_detailed_report()
    
    def _write_detailed_report(self):
        """
        写入详细报告到文件
        """
        report_dir = Path("data/quality_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 数据质量评估详细报告 ===\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估目录: {self.data_dir}\n\n")
            
            f.write("1. 基本统计\n")
            f.write(f"   总文件数: {self.audit_results['total_files']}\n")
            f.write(f"   有效文件数: {self.audit_results['valid_files']}\n")
            f.write(f"   无效文件数: {self.audit_results['invalid_files']}\n")
            
            if self.audit_results['file_sizes']:
                avg_size = np.mean(self.audit_results['file_sizes']) / 1024
                f.write(f"   平均文件大小: {avg_size:.2f} KB\n")
            
            f.write(f"   整体数据质量: {self.audit_results['overall_quality']}\n\n")
            
            f.write("2. 日期范围检查\n")
            if self.audit_results['date_ranges']:
                f.write(f"   有效日期范围文件数: {len(self.audit_results['date_ranges'])}\n")
                # 写入前10个文件的日期范围作为示例
                sample_ranges = list(self.audit_results['date_ranges'].items())[:10]
                f.write("   示例日期范围:\n")
                for ts_code, date_range in sample_ranges:
                    f.write(f"     {ts_code}: {date_range}\n")
            else:
                f.write("   没有有效日期范围\n")
            
            f.write("\n3. 缺失值检查\n")
            if self.audit_results['missing_values']:
                f.write(f"   存在缺失值的文件数: {len(self.audit_results['missing_values'])}\n")
            else:
                f.write("   所有文件无缺失值\n")
            
            f.write("\n4. 重复日期检查\n")
            if self.audit_results['duplicate_dates']:
                f.write(f"   存在重复日期的文件数: {len(self.audit_results['duplicate_dates'])}\n")
            else:
                f.write("   所有文件无重复日期\n")
            
            f.write("\n5. 数据一致性检查\n")
            if self.audit_results['data_consistency']:
                f.write(f"   存在一致性问题的文件数: {len(self.audit_results['data_consistency'])}\n")
            else:
                f.write("   所有文件数据一致\n")
        
        logger.info(f"详细报告已保存到: {report_file}")


if __name__ == "__main__":
    """
    运行数据质量评估
    """
    data_dir = "data/raw"
    
    audit = DataQualityAudit(data_dir)
    results = audit.run_audit()
    
    print("\n=== 数据质量评估摘要 ===")
    print(f"总文件数: {results['total_files']}")
    print(f"有效文件数: {results['valid_files']}")
    print(f"无效文件数: {results['invalid_files']}")
    print(f"整体数据质量: {results['overall_quality']}")
    print(f"详细报告已生成到 data/quality_reports 目录")
