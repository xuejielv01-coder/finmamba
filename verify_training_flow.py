
import sys
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.config import Config
from data.dataset import AlphaDataModule, FastAlphaDataset
from training.trainer import Trainer

class TestTrainingFlow(unittest.TestCase):
    def setUp(self):
        # Setup paths
        self.original_processed_dir = Config.PROCESSED_DATA_DIR
        self.original_data_root = Config.DATA_ROOT
        
        self.test_dir = Path("test_training_flow_dir")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        Config.PROCESSED_DATA_DIR = self.test_dir
        Config.DATA_ROOT = self.test_dir
        (self.test_dir / "fast_cache").mkdir(parents=True, exist_ok=True)
        
        # Create dummy stock_basic.parquet
        self.stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'industry': ['Bank', 'RealEstate']
        })
        self.stock_basic.to_parquet(self.test_dir / 'stock_basic.parquet')

    def tearDown(self):
        # Cleanup
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup test dir: {e}")
        
        Config.PROCESSED_DATA_DIR = self.original_processed_dir
        Config.DATA_ROOT = self.original_data_root

    @patch('data.dataset.Preprocessor')
    def test_full_training_loop(self, MockPreprocessor):
        print("\n=== Testing Full Training Flow ===")
        
        # 1. Setup Mock Preprocessor to generate dummy data
        mock_instance = MockPreprocessor.return_value
        
        def side_effect_process(save_processed=True):
            print("[Mock] Generating dummy all_features.parquet...")
            # Generate 100 days of data for 2 stocks
            dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
            data = []
            for date in dates:
                for ts_code in ['000001.SZ', '000002.SZ']:
                    row = {
                        'trade_date': date,
                        'ts_code': ts_code,
                        'close': 10.0 + np.random.randn(),
                        'ret_rank': 0.5, # Label
                        'industry': 'Bank' if ts_code == '000001.SZ' else 'RealEstate'
                    }
                    # Add feature cols
                    for i in range(Config.FEATURE_DIM): # Assuming default feature dim
                        row[f'feat_{i}'] = np.random.randn()
                    # Add features expected by Preprocessor.FEATURES if any specific ones are needed
                    # But dataset.py uses Preprocessor.FEATURES. 
                    # We need to mock Preprocessor.FEATURES too or ensure FastAlphaDataset reads it from somewhere.
                    # FastAlphaDataset reads Preprocessor.FEATURES.
                    # Let's mock that class attribute if possible, or just add enough columns.
                    
                    data.append(row)
            
            df = pd.DataFrame(data)
            # Ensure features match what FastAlphaDataset expects
            # We'll just assume FeatureEmbedding expects Config.FEATURE_DIM input
            # But FastAlphaDataset.feature_cols comes from Preprocessor.FEATURES
            
            df.to_parquet(Config.PROCESSED_DATA_DIR / "all_features.parquet")
            return df
            
        mock_instance.process_all_data.side_effect = side_effect_process
        
        # Mock Preprocessor.FEATURES
        # We need to patch the class attribute
        with patch('data.preprocessor.Preprocessor.FEATURES', [f'feat_{i}' for i in range(Config.FEATURE_DIM)]):
            
            # 2. Initialize DataModule (should trigger preprocessing)
            print("Initializing AlphaDataModule...")
            # We need to ensure Config.FEATURE_DIM matches model expectation
            Config.FEATURE_DIM = 10 # Small dim for test
            Config.SEQ_LEN = 5
            
            # Create feature cols list matching Config.FEATURE_DIM
            feature_cols = [f'feat_{i}' for i in range(Config.FEATURE_DIM)]
            
            # Re-patch FEATURES with correct length
            with patch('data.dataset.Preprocessor.FEATURES', feature_cols):
                
                dm = AlphaDataModule(batch_size=4, num_workers=0, auto_clear_cache=True)
                
                # Check if train dataset has samples
                print(f"Train samples: {len(dm.train_dataset)}")
                self.assertGreater(len(dm.train_dataset), 0, "Train dataset should not be empty")
                
                # 3. Initialize Trainer
                print("Initializing Trainer...")
                # Mock model to avoid huge download/init
                # Actually, let's use the real model but small config
                Config.D_MODEL = 16
                Config.N_LAYERS = 1
                Config.N_TRANSFORMER_LAYERS = 1
                Config.N_HEADS = 2
                Config.D_STATE = 4
                Config.N_INDUSTRIES = 5
                
                trainer = Trainer(
                    train_loader=dm.train_dataloader(),
                    val_loader=dm.val_dataloader(),
                    max_epochs=1,
                    use_amp=False, # Disable AMP for CPU test stability
                    device=torch.device('cpu')
                )
                
                # 4. Run Training
                print("Running dummy training epoch...")
                metrics = trainer.train_epoch()
                
                print("Training metrics:", metrics)
                self.assertIn('loss', metrics)
                
                print("SUCCESS: Full training flow verified!")

if __name__ == '__main__':
    unittest.main()
