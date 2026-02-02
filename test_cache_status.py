#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试缓存状态和内置股票列表
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.downloader import AkShareDownloader


def test_cache_status():
    """测试缓存状态"""
    print("=== 测试缓存状态和内置股票列表 ===")
    
    # 初始化下载器
    downloader = AkShareDownloader()
    
    # 获取股票列表
    stocks = downloader.get_main_board_stocks()
    print(f"内置股票列表长度: {len(stocks)}")
    print("\n股票列表:")
    print(stocks[['ts_code', 'name']])
    
    # 检查每个股票的缓存状态
    print("\n=== 检查每个股票的缓存状态 ===")
    for _, row in stocks.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        # 构建缓存文件路径
        save_path = Path('data/raw') / f"{ts_code.replace('.', '_')}.parquet"
        exists = save_path.exists()
        
        print(f"{ts_code} ({name}): {'✅ 已缓存' if exists else '❌ 未缓存'}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_cache_status()
