from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QTextEdit,
    QProgressBar, QDateEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate

class TrainWorker(QThread):
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, train_range, val_range, **kwargs):
        super().__init__()
        self.train_range = train_range
        self.val_range = val_range
        self.kwargs = kwargs

    def run(self):
        try:
            # Import here to avoid heavy load at start
            from data.dataset import AlphaDataModule
            from training.trainer import Trainer
            
            # Real implementation would take params from UI
            data_module = AlphaDataModule(
                train_range=self.train_range, 
                val_range=self.val_range
            )
            trainer = Trainer(
                train_loader=data_module.train_dataloader(),
                val_loader=data_module.val_dataloader(),
                **self.kwargs
            )
            trainer.train()
            self.finished_signal.emit(True, "Training finished.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        header = QLabel("Model Training")
        header.setObjectName("Header")
        layout.addWidget(header)

        main_split = QHBoxLayout()
        
        # Left Panel: Settings
        settings_group = QGroupBox("Hyperparameters")
        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        form_layout.addRow("Epochs:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(16, 1024)
        self.spin_batch.setValue(64)
        form_layout.addRow("Batch Size:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(4)
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setSingleStep(0.0001)
        form_layout.addRow("Learning Rate:", self.spin_lr)

        settings_group.setLayout(form_layout)
        main_split.addWidget(settings_group, 1)

        # Right Panel: Date Ranges
        date_group = QGroupBox("Date Ranges")
        date_form = QFormLayout()
        date_form.setSpacing(15)
        
        today = QDate.currentDate()
        # Defaults
        t_start = today.addYears(-2)
        t_end = today.addMonths(-6)
        v_start = t_end.addDays(1)
        v_end = today
        
        self.train_start = self._create_date_edit(t_start)
        self.train_end = self._create_date_edit(t_end)
        self.val_start = self._create_date_edit(v_start)
        self.val_end = self._create_date_edit(v_end)
        
        date_form.addRow("Train Start:", self.train_start)
        date_form.addRow("Train End:", self.train_end)
        date_form.addRow("Val Start:", self.val_start)
        date_form.addRow("Val End:", self.val_end)
        
        date_group.setLayout(date_form)
        main_split.addWidget(date_group, 1)
        
        layout.addLayout(main_split)

        # Training Controls
        self.btn_train = QPushButton("Start Training")
        self.btn_train.setMinimumHeight(45)
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        # Logs
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Training logs...")
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def _create_date_edit(self, date_val):
        de = QDateEdit()
        de.setDate(date_val)
        de.setCalendarPopup(True)
        de.setDisplayFormat("yyyy-MM-dd")
        return de

    def start_training(self):
        self.log_output.append("Starting training...")
        self.btn_train.setEnabled(False)
        self.progress.show()
        
        # Get params
        t_start = self.train_start.date().toString("yyyy-MM-dd")
        t_end = self.train_end.date().toString("yyyy-MM-dd")
        v_start = self.val_start.date().toString("yyyy-MM-dd")
        v_end = self.val_end.date().toString("yyyy-MM-dd")
        
        # We assume trainer might accept args, but mostly config updates
        # For simplicity, we just pass what we can or update config if needed (not implemented here safely)
        # But we do pass ranges to DataModule
        
        self.worker = TrainWorker(
            train_range=(t_start, t_end),
            val_range=(v_start, v_end),
            # pass other params to update config if supported or just for logging
        ) 
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, msg):
        self.btn_train.setEnabled(True)
        self.progress.hide()
        self.log_output.append(f"[{'DONE' if success else 'FAIL'}] {msg}")
