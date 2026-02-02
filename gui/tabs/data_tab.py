from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QProgressBar, QMessageBox, QDateEdit, QFormLayout, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
import sys
import os

class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.log_signal.emit("Starting task...")
            # Inject signals into kwargs if the task accepts them, or just rely on task setups
            # Here we expect the task_func to handle the logic, possibly using self as context if bound
            self.task_func(*self.args, **self.kwargs)
            self.finished_signal.emit(True, "Task completed successfully.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("Market Data Management")
        header.setObjectName("Header")
        layout.addWidget(header)

        # Settings
        settings_group = QGroupBox("Download Settings")
        form_layout = QFormLayout()
        
        # Default dates: 3 years ago to today
        today = QDate.currentDate()
        start = today.addYears(-3)
        
        self.date_start = QDateEdit()
        self.date_start.setDate(start)
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        
        self.date_end = QDateEdit()
        self.date_end.setDate(today)
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        
        form_layout.addRow("Start Date:", self.date_start)
        form_layout.addRow("End Date:", self.date_end)
        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)
        
        # Data Statistics
        stats_group = QGroupBox("Data Statistics")
        stats_layout = QFormLayout()
        
        self.lbl_stock_count = QLabel("-")
        self.lbl_date_range = QLabel("-")
        
        stats_layout.addRow("Downloaded Stocks:", self.lbl_stock_count)
        stats_layout.addRow("Overall Date Range:", self.lbl_date_range)
        
        self.btn_refresh = QPushButton("Refresh Stats")
        self.btn_refresh.setFixedWidth(120)
        self.btn_refresh.clicked.connect(self.refresh_stats)
        
        stats_group.setLayout(stats_layout)
        
        # Horizontal layout for stats group and refresh button
        stats_container = QVBoxLayout()
        stats_container.addLayout(stats_layout)
        stats_container.addWidget(self.btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)
        stats_group.setLayout(stats_container)
        
        layout.addWidget(stats_group)

        # Buttons Container
        btn_layout = QHBoxLayout()
        
        self.btn_download = QPushButton("Download Data")
        self.btn_download.setMinimumHeight(50)
        self.btn_download.clicked.connect(self.start_download)
        
        self.btn_process = QPushButton("Preprocess Data")
        self.btn_process.setObjectName("SecondaryButton") # Use ghost style
        self.btn_process.setMinimumHeight(50)
        self.btn_process.clicked.connect(self.start_preprocess)

        btn_layout.addWidget(self.btn_download)
        btn_layout.addWidget(self.btn_process)
        layout.addLayout(btn_layout)

        # Progress/Status
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0) # Indeterminate
        self.progress.hide()
        layout.addWidget(self.progress)

        # Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Logs will appear here...")
        layout.addWidget(self.log_output)
        
        # Initial stats load
        self.refresh_stats()

        self.setLayout(layout)

    def log(self, message):
        self.log_output.append(message)

    def set_loading(self, loading):
        if loading:
            self.progress.show()
            self.btn_download.setEnabled(False)
            self.btn_process.setEnabled(False)
            self.btn_refresh.setEnabled(False)
        else:
            self.progress.hide()
            self.btn_download.setEnabled(True)
            self.btn_process.setEnabled(True)
            self.btn_refresh.setEnabled(True)

    def show_result(self, success, message):
        self.set_loading(False)
        self.log(f"[{'SUCCESS' if success else 'ERROR'}] {message}")
        if not success:
            QMessageBox.critical(self, "Error", message)
            
    def refresh_stats(self):
        self.lbl_stock_count.setText("Scanning...")
        self.lbl_date_range.setText("Scanning...")
        
        def stats_task():
            from config.config import Config
            import pandas as pd
            from pathlib import Path
            
            raw_dir = Config.RAW_DATA_DIR
            if not raw_dir.exists():
                return 0, "No Data"
                
            files = list(raw_dir.glob("*.parquet"))
            # Filter out index/basic files
            stock_files = [f for f in files if not f.name.startswith("index_") and f.name != "stock_basic.parquet"]
            count = len(stock_files)
            
            if count == 0:
                return 0, "No Data"
            
            # Check range of a few files to estimate (checking all is too slow)
            # Or just check global min/max if we want accuracy
            # For speed, check 5 random files
            min_dates = []
            max_dates = []
            
            import random
            sample_files = random.sample(stock_files, min(len(stock_files), 10))
            
            for f in sample_files:
                try:
                    df = pd.read_parquet(f, columns=['trade_date'])
                    if not df.empty:
                        # trade_date is string YYYYMMDD usually
                        min_dates.append(df['trade_date'].min())
                        max_dates.append(df['trade_date'].max())
                except:
                    pass
            
            date_range_str = "Unknown"
            if min_dates and max_dates:
                overall_min = min(min_dates)
                overall_max = max(max_dates)
                date_range_str = f"{overall_min} ~ {overall_max}"
                
            return count, date_range_str

        # Re-using WorkerThread and passing result via a shared variable or callback is messy.
        # Let's define a StatsWorker class inside refresh_stats or globally.
        self.stats_worker = StatsWorker(stats_task)
        self.stats_worker.result_signal.connect(self.update_stats_ui)
        self.stats_worker.start()

    def on_stats_finished(self, success, msg):
        pass # Not used for this custom flow

    def update_stats_ui(self, count, date_range):
        self.lbl_stock_count.setText(str(count))
        self.lbl_date_range.setText(date_range)
        self.log(f"Stats updated: {count} stocks, range {date_range}")

    def update_progress(self, percentage, message):
        self.progress.setValue(percentage)
        # Optional: Update status label or log infrequently
        # self.log(message) # Can be too verbose
        
    def start_download(self):
        s_date = self.date_start.date().toString("yyyyMMdd")
        e_date = self.date_end.date().toString("yyyyMMdd")
        self.log(f"Initializing download from {s_date} to {e_date}...")
        self.set_loading(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        
        # Define the task wrapper that sets up the callback
        def download_task(worker_instance, start, end):
             from data.downloader import AkShareDownloader
             downloader = AkShareDownloader()
             
             # Setup callback
             def progress_cb(pct, msg):
                 worker_instance.progress_signal.emit(pct, msg)
                 
             downloader.progress_callback = progress_cb
             
             # Start download (getting full list first if needed, usually handles internally or we call get_main_board)
             # AkShareDownloader normally downloads for all stocks if we call a wrapper, 
             # but download_with_thread_pool needs a list.
             # Let's get the list first.
             worker_instance.log_signal.emit("Fetching stock list...")
             df_stocks = downloader.get_main_board_stocks()
             stock_list = df_stocks['ts_code'].tolist()
             
             worker_instance.log_signal.emit(f"Found {len(stock_list)} stocks. Starting download...")
             downloader.download_with_thread_pool(stock_list, force_update=False, start_date=start, end_date=end)
             
             # Index data
             worker_instance.log_signal.emit("Downloading index data...")
             downloader.download_index_data()

        self.worker = WorkerThread(download_task, start=s_date, end=e_date)
        # We need to pass the worker instance to the task so it can emit signals
        self.worker.kwargs['worker_instance'] = self.worker
        
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.show_result)
        self.worker.start()

    def start_preprocess(self):
        self.log("Initializing preprocessing...")
        self.set_loading(True)

        def process_task():
            from data.preprocessor import Preprocessor
            preprocessor = Preprocessor()
            preprocessor.process_all_data()

        self.worker = WorkerThread(process_task)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.show_result)
        self.worker.start()

class StatsWorker(QThread):
    result_signal = pyqtSignal(int, str)
    
    def __init__(self, task_func):
        super().__init__()
        self.task_func = task_func

    def run(self):
        try:
            count, dates = self.task_func()
            self.result_signal.emit(count, dates)
        except Exception:
            self.result_signal.emit(0, "Error")
