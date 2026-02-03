import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import Config
from training.trainer import Trainer


class DummyDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        X = torch.randn(Config.SEQ_LEN, Config.FEATURE_DIM)
        y = torch.randn(())
        ind = torch.tensor(0, dtype=torch.long)
        info = {"ts_code": "DUMMY", "date": "2020-01-01"}
        return X, y, ind, info


def main():
    Config.ENABLE_COMPILE = False
    Config.USE_AMP = False
    Config.BATCH_SIZE = 8

    dl = DataLoader(DummyDataset(), batch_size=8, shuffle=False, num_workers=0)
    trainer = Trainer(train_loader=dl, val_loader=dl, max_epochs=1, use_amp=False, device=torch.device("cpu"))
    trainer.current_epoch = 0
    metrics = trainer.train_epoch()
    print("OK", metrics)


if __name__ == "__main__":
    main()
