import sys
import os
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import json

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import project modules
# We import them lazily or inside functions if they are heavy, 
# but usually config is needed early.
try:
    from config.config import Config
    from utils.logger import get_logger
except ImportError:
    # If project structure is different, we might fail here.
    # We will assume the structure is correct as verified.
    pass

app = FastAPI(title="FinMamba GUI")

# Global State
task_status = {
    "status": "idle", # idle, running, error
    "message": "Ready",
    "last_update": ""
}

class DiagnoseRequest(BaseModel):
    code: str

def update_status(status, message):
    task_status["status"] = status
    task_status["message"] = message
    # In a real app we might want a timestamp or log

# Async Background Tasks
def run_download_task(force: bool):
    try:
        update_status("running", "Downloading data...")
        from data.downloader import TushareDownloader
        downloader = TushareDownloader()
        downloader.download_all(force_update=force)
        downloader.download_index_data()
        update_status("idle", "Data download completed successfully.")
    except Exception as e:
        update_status("error", f"Download failed: {str(e)}")

def run_preprocess_task():
    try:
        update_status("running", "Preprocessing data...")
        from data.preprocessor import Preprocessor
        preprocessor = Preprocessor()
        preprocessor.process_all_data()
        update_status("idle", "Data preprocessing completed successfully.")
    except Exception as e:
        update_status("error", f"Preprocessing failed: {str(e)}")

def run_train_task():
    try:
        update_status("running", "Training model... (check console for logs)")
        from data.dataset import AlphaDataModule
        from training.trainer import Trainer
        
        data_module = AlphaDataModule()
        trainer = Trainer(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        trainer.train()
        update_status("idle", "Model training completed successfully.")
    except Exception as e:
        update_status("error", f"Training failed: {str(e)}")

def run_backtest_task(cost, top_k):
    try:
        update_status("running", "Running backtest...")
        from evaluation.backtest import Backtester
        backtester = Backtester(transaction_cost=cost, top_k=top_k)
        results = backtester.run()
        report = backtester.generate_report()
        # We could save report to a file or store in memory to serve
        update_status("idle", "Backtest completed. Check logs/output.")
    except Exception as e:
        update_status("error", f"Backtest failed: {str(e)}")

# API Endpoints

@app.get("/api/status")
async def get_status():
    return task_status

@app.post("/api/action/download")
async def api_download(background_tasks: BackgroundTasks, force: bool = False):
    if task_status["status"] == "running":
        return JSONResponse(status_code=400, content={"message": "A task is already running"})
    background_tasks.add_task(run_download_task, force)
    return {"message": "Download started"}

@app.post("/api/action/preprocess")
async def api_preprocess(background_tasks: BackgroundTasks):
    if task_status["status"] == "running":
        return JSONResponse(status_code=400, content={"message": "A task is already running"})
    background_tasks.add_task(run_preprocess_task)
    return {"message": "Preprocessing started"}

@app.post("/api/action/train")
async def api_train(background_tasks: BackgroundTasks):
    if task_status["status"] == "running":
        return JSONResponse(status_code=400, content={"message": "A task is already running"})
    background_tasks.add_task(run_train_task)
    return {"message": "Training started"}

@app.get("/api/scan")
async def api_scan(top_k: int = 10):
    try:
        from prediction.scan import Scanner
        scanner = Scanner()
        results = scanner.daily_scan(top_k=top_k)
        if hasattr(results, 'to_dict'):
            return results.to_dict(orient="records")
        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/diagnose")
async def api_diagnose(req: DiagnoseRequest):
    try:
        from prediction.radar import Radar
        radar = Radar()
        result = radar.diagnose_single(req.code)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Mount static files
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True)

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
