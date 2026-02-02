import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QStatusBar, QLabel, QWidget
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gui.tabs.data_tab import DataTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.prediction_tab import PredictionTab
from gui.tabs.evaluation_tab import EvaluationTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FinMamba AI Terminal")
        self.resize(1200, 800)
        
        # Load Stylesheet
        self.load_styles()

        # Central Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize Tabs
        self.tab_data = DataTab()
        self.tab_train = TrainingTab()
        self.tab_pred = PredictionTab()
        self.tab_eval = EvaluationTab()

        self.tabs.addTab(self.tab_data, "Data Management")
        self.tabs.addTab(self.tab_train, "Training & Config")
        self.tabs.addTab(self.tab_pred, "Stock Selection & Radar")
        self.tabs.addTab(self.tab_eval, "Evaluation")

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("System Ready")
        
    def load_styles(self):
        style_path = Path(__file__).parent / "styles.qss"
        if style_path.exists():
            with open(style_path, "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
