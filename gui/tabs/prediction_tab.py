from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class ScanWorker(QThread):
    finished_signal = pyqtSignal(object)
    
    def run(self):
        try:
            from prediction.scan import Scanner
            scanner = Scanner()
            results = scanner.daily_scan(top_k=50)
            self.finished_signal.emit(results)
        except Exception as e:
            self.finished_signal.emit(str(e))

class DiagnoseWorker(QThread):
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, code):
        super().__init__()
        self.code = code

    def run(self):
        try:
            from prediction.radar import Radar
            radar = Radar()
            res = radar.diagnose_single(self.code)
            self.finished_signal.emit(res)
        except Exception as e:
            self.finished_signal.emit({"error": str(e)})

class PredictionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Splitter to divide Scan (Left) and Radar (Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Side: Scanner ---
        scan_widget = QWidget()
        scan_layout = QVBoxLayout()
        scan_layout.setContentsMargins(0, 0, 10, 0)
        
        scan_header = QLabel("Daily Stock Scanner")
        scan_header.setObjectName("Header")
        scan_layout.addWidget(scan_header)
        
        self.btn_scan = QPushButton("Run Daily Scan")
        self.btn_scan.clicked.connect(self.run_scan)
        scan_layout.addWidget(self.btn_scan)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Code", "Date", "Score", "Mag.", "Dir."])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        scan_layout.addWidget(self.table)
        
        scan_widget.setLayout(scan_layout)
        splitter.addWidget(scan_widget)

        # --- Right Side: Radar / Diagnostics ---
        radar_widget = QWidget()
        radar_layout = QVBoxLayout()
        radar_layout.setContentsMargins(10, 0, 0, 0)
        
        radar_header = QLabel("Stock Radar Diagnosis")
        radar_header.setObjectName("Header")
        radar_layout.addWidget(radar_header)
        
        input_layout = QHBoxLayout()
        self.input_code = QLineEdit()
        self.input_code.setPlaceholderText("e.g. 600000.SH")
        self.btn_radar = QPushButton("Diagnose")
        self.btn_radar.clicked.connect(self.run_diagnosis)
        input_layout.addWidget(self.input_code)
        input_layout.addWidget(self.btn_radar)
        radar_layout.addLayout(input_layout)
        
        # Results Display
        self.res_group = QGroupBox("Analysis Result")
        res_layout = QVBoxLayout()
        
        self.lbl_code = QLabel("-")
        self.lbl_dir = QLabel("-")
        self.lbl_mag = QLabel("-")
        self.lbl_conf = QLabel("-")
        
        res_layout.addWidget(QLabel("Stock Code:"))
        res_layout.addWidget(self.lbl_code)
        res_layout.addWidget(QLabel("Direction:"))
        res_layout.addWidget(self.lbl_dir)
        res_layout.addWidget(QLabel("Magnitude:"))
        res_layout.addWidget(self.lbl_mag)
        res_layout.addWidget(QLabel("Confidence:"))
        res_layout.addWidget(self.lbl_conf)
        res_layout.addStretch()
        
        self.res_group.setLayout(res_layout)
        radar_layout.addWidget(self.res_group)
        
        radar_widget.setLayout(radar_layout)
        splitter.addWidget(radar_widget)
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        self.setLayout(layout)

    def run_scan(self):
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Scanning...")
        self.worker_scan = ScanWorker()
        self.worker_scan.finished_signal.connect(self.on_scan_finished)
        self.worker_scan.start()

    def on_scan_finished(self, results):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("Run Daily Scan")
        
        if isinstance(results, str): # Error
            # Handle error
            return

        # Assume results is a DataFrame
        self.table.setRowCount(len(results))
        for i, (idx, row) in enumerate(results.iterrows()):
             self.table.setItem(i, 0, QTableWidgetItem(str(row.get('ts_code', row.get('code', '')))))
             self.table.setItem(i, 1, QTableWidgetItem(str(row.get('trade_date', ''))))
             self.table.setItem(i, 2, QTableWidgetItem(f"{row.get('score', 0):.4f}"))
             self.table.setItem(i, 3, QTableWidgetItem(str(row.get('magnitude', ''))))
             self.table.setItem(i, 4, QTableWidgetItem(str(row.get('direction', ''))))

    def run_diagnosis(self):
        code = self.input_code.text().strip()
        if not code: return
        
        self.btn_radar.setEnabled(False)
        self.btn_radar.setText("...")
        self.worker_radar = DiagnoseWorker(code)
        self.worker_radar.finished_signal.connect(self.on_radar_finished)
        self.worker_radar.start()

    def on_radar_finished(self, res):
        self.btn_radar.setEnabled(True)
        self.btn_radar.setText("Diagnose")
        
        if "error" in res:
            self.lbl_code.setText("Error")
            self.lbl_dir.setText(res['error'])
            return
            
        self.lbl_code.setText(res.get('ts_code', '-'))
        
        direction = res.get('direction', '-')
        self.lbl_dir.setText(direction)
        if direction == 'UP':
            self.lbl_dir.setStyleSheet("color: #10b981; font-weight: bold; font-size: 20px;")
        elif direction == 'DOWN':
            self.lbl_dir.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 20px;")
        else:
            self.lbl_dir.setStyleSheet("color: #f8fafc;")

        self.lbl_mag.setText(str(res.get('magnitude', '-')))
        conf = res.get('confidence', 0)
        self.lbl_conf.setText(f"{conf*100:.1f}%")
