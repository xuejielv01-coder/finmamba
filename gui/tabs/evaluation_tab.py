from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox
)
from PyQt6.QtCore import QThread, pyqtSignal

class BacktestWorker(QThread):
    finished_signal = pyqtSignal(object)

    def __init__(self, cost, top_k):
        super().__init__()
        self.cost = cost
        self.top_k = top_k

    def run(self):
        try:
            from evaluation.backtest import Backtester
            backtester = Backtester(
                transaction_cost=self.cost,
                top_k=self.top_k
            )
            backtester.run()
            report = backtester.generate_report()
            self.finished_signal.emit(report)
        except Exception as e:
            self.finished_signal.emit(f"Error: {str(e)}")

class EvaluationWorker(QThread):
    finished_signal = pyqtSignal(str)

    def run(self):
        try:
            from evaluation.metrics import SOTAMetrics
            from config.config import Config
            import pandas as pd
            from pathlib import Path
            
            # 1. Load Predictions
            pred_dir = Config.DATA_ROOT / "predictions"
            all_preds = []
            if pred_dir.exists():
                for csv_file in pred_dir.glob("scan_*.csv"):
                    try:
                        df = pd.read_csv(csv_file, comment='#')
                        date_str = csv_file.stem.replace('scan_', '')
                        df['date'] = date_str
                        all_preds.append(df)
                    except: pass
            
            if not all_preds:
                self.finished_signal.emit("Error: No prediction files found in data/predictions. Run daily scan first.")
                return

            preds = pd.concat(all_preds, ignore_index=True)
            
            # 2. Load Actuals (Simplified for GUI speed)
            raw_dir = Config.RAW_DATA_DIR
            ts_codes = preds['ts_code'].unique()
            all_returns = []
            
            # Limit stocks to avoid GUI freeze if too many
            # QThread runs in background but excessive IO is bad
            count = 0 
            for ts_code in ts_codes:
                if count > 500: break # Sample for speed if too many
                p_file = raw_dir / f"{ts_code.replace('.', '_')}.parquet"
                if p_file.exists():
                    try:
                        df = pd.read_parquet(p_file)
                        df['return'] = df['close'].shift(-1) / df['close'] - 1
                        if 'trade_date' in df.columns:
                            df['date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
                        all_returns.append(df[['ts_code', 'date', 'return']].dropna())
                        count += 1
                    except: pass
            
            if not all_returns:
                self.finished_signal.emit("Error: Could not load actual return data.")
                return
                
            actuals = pd.concat(all_returns, ignore_index=True)
            
            # 3. Evaluate
            evaluator = SOTAMetrics()
            evaluator.evaluate(preds, actuals, date_col='date', pred_col='score', target_col='return')
            
            report = evaluator.generate_report()
            self.finished_signal.emit(report)
            
        except Exception as e:
            import traceback
            self.finished_signal.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30,30,30,30)
        
        header = QLabel("Model Evaluation & Backtesting")
        header.setObjectName("Header")
        layout.addWidget(header)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        
        # Backtest Group
        group = QGroupBox("Backtest Settings")
        form = QFormLayout()
        
        self.spin_cost = QDoubleSpinBox()
        self.spin_cost.setRange(0.0001, 0.05)
        self.spin_cost.setValue(0.003)
        self.spin_cost.setDecimals(4)
        form.addRow("Transaction Cost:", self.spin_cost)
        
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 100)
        self.spin_topk.setValue(10)
        form.addRow("Top K Stocks:", self.spin_topk)
        
        group.setLayout(form)
        ctrl_layout.addWidget(group)
        
        # Actions Group
        action_group = QGroupBox("Actions")
        vbox = QVBoxLayout()
        
        self.btn_run = QPushButton("Run Backtest (Strategy)")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.run_backtest)
        vbox.addWidget(self.btn_run)
        
        self.btn_eval = QPushButton("Run Model Evaluation (Metrics)")
        self.btn_eval.setMinimumHeight(40)
        self.btn_eval.clicked.connect(self.run_evaluation)
        vbox.addWidget(self.btn_eval)
        
        action_group.setLayout(vbox)
        ctrl_layout.addWidget(action_group)
        
        layout.addLayout(ctrl_layout)
        
        # Report Area
        self.report_area = QTextEdit()
        self.report_area.setReadOnly(True)
        self.report_area.setFontFamily("Consolas")
        self.report_area.setPlaceholderText("Report will appear here...")
        layout.addWidget(self.report_area)
        
        self.setLayout(layout)

    def run_backtest(self):
        self.report_area.setText("Running backtest... please wait.")
        self.btn_run.setEnabled(False)
        self.btn_eval.setEnabled(False)
        
        cost = self.spin_cost.value()
        top_k = self.spin_topk.value()
        
        self.worker = BacktestWorker(cost, top_k)
        self.worker.finished_signal.connect(self.display_report)
        self.worker.start()

    def run_evaluation(self):
        self.report_area.setText("Running comprehensive model evaluation... please wait.")
        self.btn_run.setEnabled(False)
        self.btn_eval.setEnabled(False)
        
        self.eval_worker = EvaluationWorker()
        self.eval_worker.finished_signal.connect(self.display_report)
        self.eval_worker.start()

    def display_report(self, report):
        self.btn_run.setEnabled(True)
        self.btn_eval.setEnabled(True)
        self.report_area.setText(str(report))
