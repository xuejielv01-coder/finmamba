# -*- coding: utf-8 -*-
"""
日期处理测试脚本
测试雅虎数据源的日期字段检查和格式处理改进
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.downloader import YahooDownloader
from data.preprocessor import Preprocessor
import pandas as pd


def test_date_validation():
    """测试日期验证功能"""
    print("\n=== 测试日期验证功能 ===")
    
    downloader = YahooDownloader()
    
    # 测试各种日期格式
    test_dates = [
        "20240101",  # YYYYMMDD 格式
        "2024-01-01",  # YYYY-MM-DD 格式
        "2024/01/01",  # YYYY/MM/DD 格式
        "2024.01.01",  # YYYY.MM.DD 格式
        "19900101",  # 边界日期（最小）
        datetime.now().strftime("%Y%m%d"),  # 当前日期
        "19891231",  # 早于最小日期
        "20300101",  # 未来日期
    ]
    
    for date_str in test_dates:
        try:
            result = downloader._validate_and_convert_date(date_str)
            print(f"输入: {date_str} -> 输出: {result}")
        except Exception as e:
            print(f"输入: {date_str} -> 错误: {e}")


def test_downloader_date_validation():
    """测试下载器的日期验证功能"""
    print("\n=== 测试下载器日期验证功能 ===")
    
    downloader = YahooDownloader()
    
    # 测试下载单只股票，验证日期处理
    test_cases = [
        ("600000.SH", "20240101", "20240105"),  # 正常日期范围
        ("600000.SH", "19891231", "20240105"),  # 早于最小日期
        ("600000.SH", "20240101", "20300101"),  # 未来日期
        ("600000.SH", "20240105", "20240101"),  # 开始日期晚于结束日期
    ]
    
    for ts_code, start_date, end_date in test_cases:
        try:
            print(f"\n测试 {ts_code}: 开始日期={start_date}, 结束日期={end_date}")
            df = downloader.download_stock_data(ts_code, start_date, end_date)
            if df is not None and not df.empty:
                print(f"✓ 下载成功，数据行数: {len(df)}")
                print(f"  日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
                # 验证日期格式
                date_format_valid = all(len(date) == 8 and date.isdigit() for date in df['trade_date'])
                print(f"  日期格式验证: {'✓ 有效' if date_format_valid else '✗ 无效'}")
            else:
                print(f"✗ 下载失败或返回空数据")
        except Exception as e:
            print(f"✗ 错误: {e}")


def test_preprocessor_date_handling():
    """测试预处理器的日期处理功能"""
    print("\n=== 测试预处理器日期处理功能 ===")
    
    preprocessor = Preprocessor()
    
    # 创建测试数据
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    date_strings = [date.strftime("%Y%m%d") for date in dates]
    
    test_data = pd.DataFrame({
        'trade_date': date_strings,
        'open': [100 + i for i in range(30)],
        'high': [101 + i for i in range(30)],
        'low': [99 + i for i in range(30)],
        'close': [100.5 + i for i in range(30)],
        'vol': [1000000 + i * 10000 for i in range(30)],
        'ts_code': ['600000.SH'] * 30
    })
    
    print(f"测试数据行数: {len(test_data)}")
    print(f"原始日期格式: {test_data['trade_date'].dtype}")
    print(f"原始日期示例: {test_data['trade_date'].head(3).tolist()}")
    
    # 测试预处理
    try:
        processed_data = preprocessor.process_daily_data(test_data)
        print(f"\n预处理后数据行数: {len(processed_data)}")
        print(f"预处理后日期格式: {processed_data['trade_date'].dtype}")
        print(f"预处理后日期示例: {processed_data['trade_date'].head(3).tolist()}")
        
        # 验证时间特征
        time_features = ['weekday', 'month', 'is_month_start', 'is_month_end']
        for feature in time_features:
            if feature in processed_data.columns:
                print(f"{feature}: 均值={processed_data[feature].mean():.4f}, 非空值={processed_data[feature].notnull().sum()}")
        
        print("\n✓ 预处理器日期处理测试通过")
    except Exception as e:
        print(f"\n✗ 预处理器日期处理测试失败: {e}")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试无效日期
    downloader = YahooDownloader()
    
    invalid_dates = ["20240230", "20241301", "invalid"]
    for date_str in invalid_dates:
        try:
            result = downloader._validate_and_convert_date(date_str)
            print(f"无效日期 {date_str} -> 输出: {result}")
        except Exception as e:
            print(f"无效日期 {date_str} -> 错误: {e}")
    
    # 测试空数据框
    preprocessor = Preprocessor()
    empty_df = pd.DataFrame()
    try:
        result = preprocessor.process_daily_data(empty_df)
        print(f"\n空数据框处理: {'✓ 成功' if result.empty else '✗ 失败'}")
    except Exception as e:
        print(f"\n空数据框处理失败: {e}")


if __name__ == "__main__":
    print("开始测试日期处理功能...")
    
    test_date_validation()
    test_downloader_date_validation()
    test_preprocessor_date_handling()
    test_edge_cases()
    
    print("\n=== 测试完成 ===")
