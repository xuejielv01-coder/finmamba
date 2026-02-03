# -*- coding: utf-8 -*-
"""
DeepAlpha-Stock 全局配置中心
PRD v3.4 配置清单实现
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """全局配置单例类"""
    
    # ============ System ============
    SEED: int = 42
    LOG_FILE: str = "logs/deepalpha.log"
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # ============ Data ============
    # 从环境变量读取或使用默认值
    TUSHARE_TOKEN: str = os.getenv("TUSHARE_TOKEN", "30d634a271dcc867e5a14046e9b570af049a0326df7dd1cf64d28021")
    DATA_ROOT: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_ROOT / "raw"
    PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"
    STATS_CACHE: Path = DATA_ROOT / "stats"
    MANIFEST_FILE: Path = DATA_ROOT / "manifest.json"
    
    # ============ Unified Cache Paths ============
    CACHE_DIR: Path = DATA_ROOT / "cache"
    SCORE_CACHE_DIR: Path = CACHE_DIR / "scores"
    KLINE_CACHE_DIR: Path = CACHE_DIR / "kline"
    INDICATOR_CACHE_DIR: Path = CACHE_DIR / "indicators"
    MODEL_CACHE_DIR: Path = CACHE_DIR / "models"
    QUALITY_REPORT_DIR: Path = DATA_ROOT / "quality_reports"
    
    # Cache expiry settings (hours)
    CACHE_EXPIRY_STOCK_DATA: int = 24    # 股票数据24小时过期
    CACHE_EXPIRY_KLINE: int = 12          # K线数据12小时过期
    CACHE_EXPIRY_STATS: int = 48          # 统计量48小时过期
    CACHE_EXPIRY_STOCK_LIST: int = 168    # 股票列表一周过期

    
    # ============ Constraints ============    
    ONLY_MAIN_BOARD: bool = True  # 仅主板股票 (60, 00 开头)
    DROP_ST: bool = False  # 剔除 ST 股票
    MAX_HOLDING_DAYS: int = 1  # 默认日频调仓
    DOWNLOAD_YEARS: int = 3  # 下载历史数据年数 (前3年)
    
    # ============ Data Split ============
    # 训练数据：前2年到前半年
    # 回测数据：前半年到现在
    TRAIN_END_MONTHS: int = 6  # 训练数据结束时间：当前日期往前推6个月
    TRAIN_START_YEARS: int = 2  # 训练数据开始时间：当前日期往前推2年（改为2年）
    BACKTEST_START_MONTHS: int = 6  # 回测数据开始时间：当前日期往前推6个月
    
    # ============ FinMamba Architecture ============
    SEQ_LEN: int = 60  # 回看天数
    FEATURE_DIM: int = 82  # 扩展特征维度 (原始特征 + 新增技术指标 + 波动率特征 + 动量特征 + 市场状态特征)
    D_MODEL: int = 128  # 模型隐藏维度（从96→128）
    N_LAYERS: int = 4  # Mamba 层数（从3→4）
    N_TRANSFORMER_LAYERS: int = 3  # Transformer 层数（从2→3）
    N_HEADS: int = 8  # 多头注意力头数 (从6→8，确保128能被8整除)
    D_STATE: int = 64  # SSM 状态维度（从32→64）
    D_CONV: int = 4  # SSM 卷积核大小
    EXPAND: int = 2  # SSM 扩展因子
    MAMBA_LEVELS: tuple = (1, 5)  # 多尺度: 日级和周级
    N_INDUSTRIES: int = 111  # 实际约111个行业，预留空间 (申万一级行业)
    USE_GRAPH: bool = True  # 使用行业嵌入
    N_GCN_LAYERS: int = 2  # GCN层数
    PATCH_LEN: int = 12  # 保留兼容性
    STRIDE: int = 8  # 保留兼容性

    # ============ Training (Optimized for Strong Generalization) ============
    TRAIN_YEARS: int = 2  # 使用2年数据训练（与 TRAIN_START_YEARS 保持一致，从3→2）
    BATCH_SIZE: int = 32  # 进一步减小batch size至32，适应6GB GPU显存
    GRAD_ACCUM_STEPS: int = 4  # 梯度累积步长，模拟更大的批次大小 (384 * 4 = 1536)
    LR_INIT: float = 5e-4  # 学习率（从1e-3→5e-4，更深的模型需要更小的学习率）
    WEIGHT_DECAY: float = 3e-3  # 权重衰减（从5e-3→3e-3，平衡正则化和模型表达能力）
    MAX_EPOCHS: int = 100  # 最大epoch数（从50→100，增加训练轮数）
    PATIENCE: int = 15  # 早停耐心值（从8→15，给予模型更多训练时间）
    DROPOUT: float = 0.25  # 增加dropout比例（从0.2→0.25，防止过拟合）
    SLIDE_STEP: int = 3  # 滑动窗口步长，可根据需要调整，默认为5，平衡速度和性能

    # ============ Downloader ============
    CONCURRENT_WORKERS = 16  # 并发下载线程数（增加以提高速度）
    API_RATE_LIMIT = 0.3  # API调用间隔（秒）（适当减少以提高速度）
    MAX_RETRY_TIMES = 3  # 最大重试次数（减少以避免长时间等待）
    RETRY_DELAY = 3.0  # 重试延迟（秒）（减少以提高速度）
    COMPILE_WARMUP = True
    LOG_FIRST_BATCH_TIMING = True

    # ============ SOTA Thresholds (Targeting IC > 0.2) ============
    SOTA_TARGET_IC = 0.20  # 提高目标 IC (0.055 -> 0.20)
    SOTA_MIN_IC = 0.05  # 提高最低 IC 阈值 (0.02 -> 0.05)
    SOTA_TARGET_ICIR = 1.0  # 提高目标 ICIR (0.55 -> 1.0)
    SOTA_MIN_IC_SAVE = 0.10  # 提高保存模型的IC阈值 (0.04 -> 0.10)
    BAD_EPOCH_LIMIT = 15  # 增加bad epoch限制 (10 -> 15)
    
    # ============ Risk Control ============
    INDEX_CODE = '000905.SH'  # 中证500作为大盘风向标
    USE_MARKET_FILTER = True  # 是否开启大盘风控
    MA_PERIOD = 20  # 均线周期
    TOP_K_DEFAULT = 10  # 默认选股数量
    TOP_K_RISK = 5  # 风险模式选股数量
    
    # ============ Prediction ============
    RADAR_TIMEOUT_MS = 100  # 个股雷达目标响应时间
    CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值
    
    # ============ Blacklist ============
    BLACKLIST_FILE = PROJECT_ROOT / "config" / "blacklist.csv"
    
    # ============ Model Paths ============
    MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
    BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
    
    # ============ Downloader (Optimized for A100) ============
    CONCURRENT_WORKERS = 64  # 并发下载线程数（A100服务器通常有更多CPU核心）
    API_RATE_LIMIT = 0.05  # API调用间隔（秒）（A100可处理更快的数据下载）
    MAX_RETRY_TIMES = 10  # 最大重试次数（确保数据完整性）
    RETRY_DELAY = 3.0  # 重试延迟（秒）（平衡速度和稳定性）
    
    # ============ Memory Optimization (for 96GB RAM) ============
    MEMORY_LIMIT = 0.8  # 使用80%的可用内存
    PARALLEL_WORKERS = min(os.cpu_count(), 32)  # 并行处理核心数
    BATCH_PROCESS_SIZE = 500  # 批处理大小
    DATA_CHUNK_SIZE = 1000000  # 数据分块大小
    
    # ============ Portfolio ============
    PORTFOLIO_DIR = DATA_ROOT / "portfolio"
    PORTFOLIO_FILE = PORTFOLIO_DIR / "holdings.json"
    DEFAULT_TAKE_PROFIT = 0.10  # 默认止盈 10%
    DEFAULT_STOP_LOSS = 0.05    # 默认止损 5%
    
    # ============ Automation ============
    DAILY_SCAN_TIME = "08:30"           # 每日选股时间
    MONITOR_INTERVAL_MINUTES = 5         # 监控间隔(分钟)
    DATA_UPDATE_TIME = "15:30"          # 数据更新时间
    MIN_EXPECTED_RETURN = 0.10          # 最低预期收益率 10%
    
    # ============ Email Notification ============
    ENABLE_EMAIL_NOTIFICATION = True
    EMAIL_SMTP_SERVER = ""              # SMTP服务器地址
    EMAIL_SMTP_PORT = 465               # SMTP端口 (SSL)
    EMAIL_USERNAME = ""                 # 发件人邮箱
    EMAIL_PASSWORD = ""                 # 邮箱密码/授权码
    EMAIL_TO = ""                       # 收件人邮箱
    
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """确保所有必要目录存在"""
        dirs = [
            cls.DATA_ROOT,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.STATS_CACHE,
            cls.MODEL_DIR,
            cls.PROJECT_ROOT / "logs",
            cls.PROJECT_ROOT / "config",
            # 新增统一缓存目录
            cls.CACHE_DIR,
            cls.SCORE_CACHE_DIR,
            cls.KLINE_CACHE_DIR,
            cls.INDICATOR_CACHE_DIR,
            cls.MODEL_CACHE_DIR,
            cls.QUALITY_REPORT_DIR,
            cls.PORTFOLIO_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> bool:
        """验证配置有效性"""
        assert cls.SEED >= 0, "SEED must be non-negative"
        assert cls.SEQ_LEN > 0, "SEQ_LEN must be positive"
        assert cls.FEATURE_DIM > 0, "FEATURE_DIM must be positive"
        assert cls.TUSHARE_TOKEN, "TUSHARE_TOKEN is required"
        return True
    
    @classmethod
    def update_model_params(cls, **kwargs) -> None:
        """
        动态更新模型参数
        
        Args:
            **kwargs: 模型参数 (d_model, n_layers, d_state, expand, levels, use_graph, n_gcn_layers)
        """
        if 'd_model' in kwargs:
            cls.D_MODEL = kwargs['d_model']
        if 'n_layers' in kwargs:
            cls.N_LAYERS = kwargs['n_layers']
        if 'd_state' in kwargs:
            cls.D_STATE = kwargs['d_state']
        if 'expand' in kwargs:
            cls.EXPAND = kwargs['expand']
        if 'levels' in kwargs:
            cls.MAMBA_LEVELS = kwargs['levels']
        if 'use_graph' in kwargs:
            cls.USE_GRAPH = kwargs['use_graph']
        if 'n_gcn_layers' in kwargs:
            cls.N_GCN_LAYERS = kwargs['n_gcn_layers']
        if 'dropout' in kwargs:
            cls.DROPOUT = kwargs['dropout']
    
    @classmethod
    def update_training_params(cls, **kwargs) -> None:
        """
        动态更新训练参数
        
        Args:
            **kwargs: 训练参数 (max_epochs, batch_size, lr_init, patience, weight_decay)
        """
        if 'max_epochs' in kwargs:
            cls.MAX_EPOCHS = kwargs['max_epochs']
        if 'batch_size' in kwargs:
            cls.BATCH_SIZE = kwargs['batch_size']
        if 'lr_init' in kwargs:
            cls.LR_INIT = kwargs['lr_init']
        if 'patience' in kwargs:
            cls.PATIENCE = kwargs['patience']
        if 'weight_decay' in kwargs:
            cls.WEIGHT_DECAY = kwargs['weight_decay']
    
    @classmethod
    def get_model_config(cls) -> dict:
        """获取当前模型配置"""
        return {
            'seq_len': cls.SEQ_LEN,
            'feature_dim': cls.FEATURE_DIM,
            'd_model': cls.D_MODEL,
            'n_layers': cls.N_LAYERS,
            'd_state': cls.D_STATE,
            'd_conv': cls.D_CONV,
            'expand': cls.EXPAND,
            'levels': cls.MAMBA_LEVELS,
            'n_industries': cls.N_INDUSTRIES,
            'use_graph': cls.USE_GRAPH,
            'n_gcn_layers': cls.N_GCN_LAYERS,
            'dropout': cls.DROPOUT
        }


# 初始化目录
Config.ensure_dirs()
