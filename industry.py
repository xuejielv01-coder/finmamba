import tushare as ts
import pandas as pd
from pathlib import Path

# 初始化 tushare（需要注册获取 token）
ts.set_token('30d634a271dcc867e5a14046e9b570af049a0326df7dd1cf64d28021')
pro = ts.pro_api()

# 获取股票基本信息
stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')

# 保存为 parquet 文件
data_dir = Path('/mnt/workspace/0114finmamba/data')
data_dir.mkdir(parents=True, exist_ok=True)
stock_basic.to_parquet(data_dir / 'stock_basic.parquet', index=False)
