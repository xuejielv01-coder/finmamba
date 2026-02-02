#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试雅虎财经数据源的A股数据下载功能
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.downloader import AkShareDownloader


def test_yahoo_download():
    """测试雅虎财经数据下载"""
    print("=== 测试雅虎财经数据源的A股数据下载功能 ===")
    
    # 初始化下载器
    downloader = AkShareDownloader()
    
    # 测试股票列表
    test_stocks = [
        "600519.SH",  # 贵州茅台
        "000001.SZ"   # 平安银行
    ]
    
    # 计算日期范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    
    print(f"\n下载日期范围: {start_date} 到 {end_date}")
    
    for ts_code in test_stocks:
        print(f"\n=== 测试下载 {ts_code} ===")
        try:
            # 下载数据
            df = downloader.download_stock_data(ts_code, start_date, end_date)
            
            if df is not None and not df.empty:
                print(f"✅ 成功下载 {ts_code} 数据")
                print(f"数据行数: {len(df)}")
                print(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
                print("前5行数据:")
                print(df.head())
                print("\n列名:")
                print(list(df.columns))
            else:
                print(f"❌ 下载 {ts_code} 失败，返回空数据")
                
        except Exception as e:
            print(f"❌ 下载 {ts_code} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_yahoo_download()
