# -*- coding: utf-8 -*-
"""
数据缺失值详细检查脚本
对每个股票文件的数据缺失情况进行详细评估
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

logger = get_logger("DataMissingValuesAudit")

class DataMissingValuesAudit:
    """
    数据缺失值详细检查类
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
            'files_with_missing_values': 0,
            'files_without_missing_values': 0,
            'missing_values_summary': {},
            'column_missing_stats': {},
            'detailed_missing_info': {}
        }
    
    def run_audit(self):
        """
        运行详细的缺失值检查
        """
        logger.info(f"开始详细缺失值检查，检查目录: {self.data_dir}")
        
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
        检查单个文件的缺失值情况
        
        Args:
            file_path: 文件路径
        """
        ts_code = file_path.stem.replace('_', '.')
        
        try:
            # 读取文件
            df = pd.read_parquet(file_path)
            
            # 检查缺失值
            missing_stats = df.isnull().sum()
            total_missing = missing_stats.sum()
            
            if total_missing > 0:
                self.audit_results['files_with_missing_values'] += 1
                
                # 记录详细的缺失值信息
                missing_columns = {}
                for col, count in missing_stats.items():
                    if count > 0:
                        missing_columns[col] = int(count)
                
                self.audit_results['detailed_missing_info'][ts_code] = {
                    'total_missing': int(total_missing),
                    'missing_columns': missing_columns,
                    'total_rows': len(df),
                    'missing_percentage': (total_missing / (len(df) * len(df.columns))) * 100
                }
                
                # 更新列缺失统计
                for col, count in missing_columns.items():
                    if col not in self.audit_results['column_missing_stats']:
                        self.audit_results['column_missing_stats'][col] = {
                            'total_missing': 0,
                            'affected_files': 0
                        }
                    self.audit_results['column_missing_stats'][col]['total_missing'] += count
                    self.audit_results['column_missing_stats'][col]['affected_files'] += 1
            else:
                self.audit_results['files_without_missing_values'] += 1
                
        except Exception as e:
            logger.error(f"文件 {file_path.name} 读取失败: {e}")
    
    def _generate_report(self):
        """
        生成详细的缺失值报告
        """
        logger.info("\n=== 数据缺失值详细检查报告 ===")
        logger.info(f"总文件数: {self.audit_results['total_files']}")
        logger.info(f"存在缺失值的文件数: {self.audit_results['files_with_missing_values']}")
        logger.info(f"无缺失值的文件数: {self.audit_results['files_without_missing_values']}")
        
        # 计算缺失值文件百分比
        if self.audit_results['total_files'] > 0:
            missing_percentage = (self.audit_results['files_with_missing_values'] / self.audit_results['total_files']) * 100
            logger.info(f"存在缺失值的文件百分比: {missing_percentage:.2f}%")
        
        # 检查列缺失统计
        if self.audit_results['column_missing_stats']:
            logger.info("\n列缺失值统计:")
            # 按受影响文件数排序
            sorted_cols = sorted(
                self.audit_results['column_missing_stats'].items(),
                key=lambda x: x[1]['affected_files'],
                reverse=True
            )
            
            for col, stats in sorted_cols[:10]:  # 显示前10个受影响最严重的列
                logger.info(f"  {col}: 受影响文件数={stats['affected_files']}, 总缺失值={stats['total_missing']}")
        else:
            logger.info("\n所有列均无缺失值")
        
        # 检查详细缺失信息
        if self.audit_results['detailed_missing_info']:
            logger.info(f"\n详细缺失值信息 (前10个文件):")
            # 按缺失值数量排序
            sorted_files = sorted(
                self.audit_results['detailed_missing_info'].items(),
                key=lambda x: x[1]['total_missing'],
                reverse=True
            )
            
            for ts_code, info in sorted_files[:10]:  # 显示前10个缺失值最多的文件
                logger.info(f"  {ts_code}: 缺失值={info['total_missing']}, 缺失百分比={info['missing_percentage']:.2f}%")
                logger.info(f"    缺失列: {info['missing_columns']}")
        else:
            logger.info("\n所有文件均无缺失值")
        
        # 生成详细报告文件
        self._write_detailed_report()
    
    def _write_detailed_report(self):
        """
        写入详细报告到文件
        """
        report_dir = Path("data/quality_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"missing_values_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 数据缺失值详细检查报告 ===\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估目录: {self.data_dir}\n\n")
            
            f.write("1. 基本统计\n")
            f.write(f"   总文件数: {self.audit_results['total_files']}\n")
            f.write(f"   存在缺失值的文件数: {self.audit_results['files_with_missing_values']}\n")
            f.write(f"   无缺失值的文件数: {self.audit_results['files_without_missing_values']}\n")
            
            if self.audit_results['total_files'] > 0:
                missing_percentage = (self.audit_results['files_with_missing_values'] / self.audit_results['total_files']) * 100
                f.write(f"   存在缺失值的文件百分比: {missing_percentage:.2f}%\n\n")
            
            f.write("2. 列缺失值统计\n")
            if self.audit_results['column_missing_stats']:
                # 按受影响文件数排序
                sorted_cols = sorted(
                    self.audit_results['column_missing_stats'].items(),
                    key=lambda x: x[1]['affected_files'],
                    reverse=True
                )
                
                for col, stats in sorted_cols:
                    f.write(f"   {col}: 受影响文件数={stats['affected_files']}, 总缺失值={stats['total_missing']}\n")
            else:
                f.write("   所有列均无缺失值\n")
            
            f.write("\n3. 详细缺失值信息\n")
            if self.audit_results['detailed_missing_info']:
                # 按缺失值数量排序
                sorted_files = sorted(
                    self.audit_results['detailed_missing_info'].items(),
                    key=lambda x: x[1]['total_missing'],
                    reverse=True
                )
                
                for ts_code, info in sorted_files:
                    f.write(f"   {ts_code}:\n")
                    f.write(f"     缺失值总数: {info['total_missing']}\n")
                    f.write(f"     总行数: {info['total_rows']}\n")
                    f.write(f"     缺失百分比: {info['missing_percentage']:.2f}%\n")
                    f.write(f"     缺失列: {info['missing_columns']}\n")
            else:
                f.write("   所有文件均无缺失值\n")
        
        logger.info(f"详细缺失值报告已保存到: {report_file}")


if __name__ == "__main__":
    """
    运行缺失值详细检查
    """
    data_dir = "data/raw"
    
    audit = DataMissingValuesAudit(data_dir)
    results = audit.run_audit()
    
    print("\n=== 数据缺失值检查摘要 ===")
    print(f"总文件数: {results['total_files']}")
    print(f"存在缺失值的文件数: {results['files_with_missing_values']}")
    print(f"无缺失值的文件数: {results['files_without_missing_values']}")
    
    if results['total_files'] > 0:
        missing_percentage = (results['files_with_missing_values'] / results['total_files']) * 100
        print(f"存在缺失值的文件百分比: {missing_percentage:.2f}%")
    
    if results['detailed_missing_info']:
        print(f"\n存在缺失值的文件详情已保存到报告中")
    else:
        print("\n所有文件均无缺失值")
    
    print(f"详细报告已生成到 data/quality_reports 目录")
