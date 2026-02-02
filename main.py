# -*- coding: utf-8 -*-
"""
DeepAlpha-Stock 量化预测系统
主入口文件

使用方法:
    python main.py          # 启动 GUI
    python main.py --cli    # 命令行模式
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_gui():
    """运行 GUI 界面"""
    from gui.main_window import main
    main()


def run_cli(args):
    """运行命令行模式"""
    from config.config import Config
    from utils.logger import get_logger
    from utils.seeder import seed_everything
    
    logger = get_logger("CLI")
    seed_everything(Config.SEED)
    
    if args.download:
        logger.info("Starting data download...")
        from data.downloader import TushareDownloader
        downloader = TushareDownloader()
        downloader.download_all(force_update=args.force)
        downloader.download_index_data()
    
    if args.preprocess:
        logger.info("Starting data preprocessing...")
        from data.preprocessor import Preprocessor
        preprocessor = Preprocessor()
        preprocessor.process_all_data()
    
    if args.train:
        logger.info("Starting model training...")
        from data.dataset import AlphaDataModule
        from training.trainer import Trainer
        
        data_module = AlphaDataModule()
        trainer = Trainer(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        trainer.train()
    
    if args.scan:
        logger.info("Starting stock scanning...")
        from prediction.scan import Scanner
        scanner = Scanner()
        results = scanner.daily_scan(top_k=args.top_k)
        print(results.to_string())
    
    if args.diagnose:
        logger.info(f"Diagnosing stock: {args.diagnose}")
        from prediction.radar import Radar
        radar = Radar()
        result = radar.diagnose_single(args.diagnose)
        
        print("\n" + "="*50)
        print(f"股票代码: {result['ts_code']}")
        print(f"预测方向: {result['direction']}")
        print(f"预期涨幅: {result['magnitude']}")
        print(f"置信度: {result['confidence']*100:.1f}%")
        print(f"响应时间: {result['latency_ms']:.0f}ms")
        print("="*50 + "\n")
    
    if args.backtest:
        logger.info("Starting backtesting...")
        from evaluation.backtest import Backtester
        backtester = Backtester(
            transaction_cost=args.cost,
            top_k=args.top_k
        )
        results = backtester.run()
        print(backtester.generate_report())


def main():
    parser = argparse.ArgumentParser(
        description='DeepAlpha-Stock Quantitative Prediction System',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--cli', action='store_true',
        help='Run in CLI mode instead of GUI'
    )
    
    # 数据操作
    parser.add_argument(
        '--download', action='store_true',
        help='Download market data'
    )
    parser.add_argument(
        '--preprocess', action='store_true',
        help='Preprocess downloaded data'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force full data update'
    )
    
    # 训练
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model'
    )
    
    # 预测
    parser.add_argument(
        '--scan', action='store_true',
        help='Run daily stock scan'
    )
    parser.add_argument(
        '--diagnose', type=str, metavar='CODE',
        help='Diagnose a single stock (e.g., 600000.SH)'
    )
    parser.add_argument(
        '--top-k', type=int, default=10,
        help='Number of top stocks to select'
    )
    
    # 回测
    parser.add_argument(
        '--backtest', action='store_true',
        help='Run backtesting'
    )
    parser.add_argument(
        '--cost', type=float, default=0.003,
        help='Transaction cost'
    )
    
    args = parser.parse_args()
    
    if args.cli or any([args.download, args.preprocess, args.train, 
                        args.scan, args.diagnose, args.backtest]):
        run_cli(args)
    else:
        run_gui()


if __name__ == '__main__':
    main()
