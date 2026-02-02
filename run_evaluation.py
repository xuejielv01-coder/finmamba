
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.config import Config
from utils.logger import get_logger
from evaluation.metrics import SOTAMetrics

logger = get_logger("Evaluator")

def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load all prediction CSVs"""
    all_preds = []
    if not predictions_dir.exists():
        logger.error(f"Prediction directory not found: {predictions_dir}")
        return pd.DataFrame()
        
    for csv_file in predictions_dir.glob("scan_*.csv"):
        try:
            df = pd.read_csv(csv_file, comment='#')
            # Extract date from filename scan_YYYYMMDD.csv
            date_str = csv_file.stem.replace('scan_', '')
            df['date'] = date_str
            all_preds.append(df)
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
            
    if not all_preds:
        return pd.DataFrame()
        
    return pd.concat(all_preds, ignore_index=True)

def load_actual_returns(ts_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load actual forward returns for validation
    We need T+1, T+5 etc returns.
    Here we calculate T+1 return as (Open_T+2 / Open_T+1 - 1) consistent with Backtester
    Or just Close_T+1 / Close_T - 1 for simple labeling?
    
    Standard IC is usually correl(Score_T, Ret_T+1).
    """
    all_returns = []
    
    # Batch loading could be slow, optimizing for demonstration
    # In production, we might read from a unified returns table
    
    # We'll use a simplified approach: read daily data and calculate returns
    raw_dir = Config.RAW_DATA_DIR
    
    count = 0
    total = len(ts_codes)
    
    for ts_code in ts_codes:
        filename = ts_code.replace('.', '_') + '.parquet'
        filepath = raw_dir / filename
        
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                df = df.sort_values('trade_date')
                
                # Calculate Forward Return (Next Day Return)
                # Ret_T = (Close_T+1 / Close_T) - 1
                # Or based on Open prices for actionable trading
                
                # Let's use Close-to-Close for standard Alpha evaluation
                df['return'] = df['close'].shift(-1) / df['close'] - 1
                
                # Align date format
                if 'trade_date' in df.columns:
                     # ensure string YYYYMMDD
                     df['date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
                
                # Filter date
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df_filtered = df.loc[mask, ['ts_code', 'date', 'return']].dropna()
                
                if not df_filtered.empty:
                    all_returns.append(df_filtered)
                    
            except Exception:
                pass
        
        count += 1
        if count % 500 == 0:
            print(f"Loaded returns for {count}/{total} stocks...")
            
    if not all_returns:
        return pd.DataFrame()
        
    return pd.concat(all_returns, ignore_index=True)

def main():
    print("Starting Comprehensive Model Evaluation...")
    
    # 1. Load Predictions
    pred_dir = Config.DATA_ROOT / "predictions"
    print(f"Loading predictions from {pred_dir}...")
    preds = load_predictions(pred_dir)
    
    if preds.empty:
        print("No prediction data found! Please run 'scan' first.")
        return
        
    print(f"Loaded {len(preds)} prediction records.")
    
    # 2. Determine Date Range from Preds
    start_date = preds['date'].min()
    end_date = preds['date'].max()
    print(f"Date Range: {start_date} to {end_date}")
    
    # 3. Load Actual Returns
    ts_codes = preds['ts_code'].unique().tolist()
    print(f"Loading actual returns for {len(ts_codes)} stocks...")
    actuals = load_actual_returns(ts_codes, start_date, end_date)
    
    if actuals.empty:
        print("Failed to load actual returns.")
        return
        
    print(f"Loaded {len(actuals)} return records.")
    
    # 4. Run Evaluation
    evaluator = SOTAMetrics()
    results = evaluator.evaluate(
        predictions=preds,
        actuals=actuals,
        date_col='date',
        pred_col='score',
        target_col='return'
    )
    
    # 5. Print Report
    print("\n" + "="*50)
    print("EVALUATION GENERATED")
    print("="*50 + "\n")
    print(evaluator.generate_report())
    
    # Save report
    report_file = Config.DATA_ROOT / f"evaluation_report_{end_date}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(evaluator.generate_report())
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
