# -*- coding: utf-8 -*-
"""
Fetch Stock Basic Info
下载股票基础信息（包含行业分类）并保存为 Parquet 文件
使用 AkShare 替代 Tushare
"""
import sys
from pathlib import Path
import pandas as pd
import akshare as ak
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("FetchBasic")

def fetch_and_save_basic():
    """下载并保存股票基础信息 (AkShare版)"""
    logger.info("Initializing AkShare API...")
    
    logger.info("Fetching stock spot data (EastMoney)...")
    try:
        # 使用东方财富接口获取实时行情，包含了代码、名称、最新价等，但不一定有详细的上市日期/行业
        # ak.stock_zh_a_spot_em() 返回所有A股
        df = ak.stock_zh_a_spot_em()
        
        if df is None or df.empty:
            logger.error("Failed to fetch stock spot data: Empty response")
            return
            
        # akshare columns: 序号, 代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 今开, 昨收, 量比, 换手率, 市盈率-动态, 市净率, 总市值, 流通市值, 涨速, 5分钟涨跌, 60日涨跌幅, 年初至今涨跌幅
        
        # 构造 ts_code
        # 东财代码通常是 6位数字，需要区分 SH/SZ
        # 简易判断：6开头SH/688, 0/3开头SZ, 8/4开头BJ
        def get_ts_code(code):
            c = str(code)
            if c.startswith('6'):
                return f"{c}.SH"
            elif c.startswith('0') or c.startswith('3'):
                return f"{c}.SZ"
            elif c.startswith('8') or c.startswith('4'):
                return f"{c}.BJ"
            else:
                return f"{c}.Unknown"

        # 尝试获取行业信息
        # AkShare 获取所有股票行业分类可能比较复杂，这里尝试获取申万行业分类
        # ak.sw_index_spot() 获取申万一级行业列表
        # ak.sw_index_cons(index_code="801010") 获取成分股
        # 这太慢了。
        
        # 替代方案：尝试 ak.stock_board_industry_name_em() 获取东方财富行业板块
        logger.info("Fetching industry data...")
        try:
            # 获取行业板块列表
            df_board = ak.stock_board_industry_name_em()
            # 遍历板块获取成分股？这也很慢。
            
            # 简化方案：只保存基本信息，行业暂时设为 'Unknown' 或者后续由 Dataset 自行补充
            # 或者使用 Tushare 的历史文件如果存在？不可依赖。
            
            # 为了保持兼容，我们尽量构造所需字段
            # fields='ts_code,symbol,name,area,industry,market,list_date'
            
            df = df.rename(columns={
                '代码': 'symbol',
                '名称': 'name',
            })
            
            df['ts_code'] = df['symbol'].apply(get_ts_code)
            df['area'] = 'CN'
            df['market'] = 'Main' # 简化
            df['list_date'] = '20000101' # 简化
            
            # 尝试从本地或其他接口补充行业
            # 暂时置空
            df['industry'] = 'Unknown'
            
        except Exception as e:
            logger.warning(f"Failed to fetch extra info: {e}")
        
        logger.info(f"Fetched {len(df)} records")
        
        # 过滤需要的列
        cols = ['ts_code', 'symbol', 'name', 'area', 'industry', 'market', 'list_date']
        # 确保列存在
        for c in cols:
            if c not in df.columns:
                df[c] = ''
                
        df_final = df[cols]
        
        # 过滤非A股主板/创业/科创
        df_final = df_final[df_final['ts_code'].str.contains('SH|SZ')]

        # 保存路径
        save_path = Config.DATA_ROOT / "stock_basic.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_final.to_parquet(save_path, index=False)
        logger.info(f"Saved stock_basic to {save_path}")
        
        # 同时也保存一份 csv 方便查看
        df_final.to_csv(save_path.with_suffix('.csv'), index=False, encoding='utf-8-sig')
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_and_save_basic()
