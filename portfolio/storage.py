# -*- coding: utf-8 -*-
"""
持仓存储模块 (Portfolio Storage)

功能:
- JSON 文件持久化
- 自动备份
- 数据加载/保存
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger

logger = get_logger("PortfolioStorage")


class PortfolioStorage:
    """
    持仓数据存储类
    
    使用 JSON 文件存储持仓数据，支持自动备份
    """
    
    def __init__(self, storage_path: Path = None):
        """
        初始化存储
        
        Args:
            storage_path: 存储文件路径
        """
        self.storage_path = storage_path or (Config.DATA_ROOT / "portfolio" / "holdings.json")
        self.backup_dir = self.storage_path.parent / "backups"
        self._lock = threading.Lock()
        
        # 确保目录存在
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Portfolio storage initialized: {self.storage_path}")
    
    def load(self) -> Dict[str, dict]:
        """
        加载持仓数据
        
        Returns:
            持仓字典 {ts_code: position_data}
        """
        with self._lock:
            if not self.storage_path.exists():
                return {}
            
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} positions from storage")
                return data
            except Exception as e:
                logger.error(f"Failed to load portfolio: {e}")
                return {}
    
    def save(self, positions: Dict[str, dict]) -> bool:
        """
        保存持仓数据
        
        Args:
            positions: 持仓字典
            
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                # 原子化写入
                temp_path = self.storage_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(positions, f, ensure_ascii=False, indent=2)
                
                # 重命名
                temp_path.replace(self.storage_path)
                logger.debug(f"Saved {len(positions)} positions to storage")
                return True
            except Exception as e:
                logger.error(f"Failed to save portfolio: {e}")
                return False
    
    def backup(self) -> Optional[Path]:
        """
        创建备份
        
        Returns:
            备份文件路径
        """
        if not self.storage_path.exists():
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"holdings_{timestamp}.json"
            shutil.copy2(self.storage_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            
            # 清理旧备份 (保留最近 10 个)
            self._cleanup_old_backups(keep=10)
            
            return backup_path
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def _cleanup_old_backups(self, keep: int = 10):
        """清理旧备份"""
        backups = sorted(self.backup_dir.glob("holdings_*.json"), reverse=True)
        for backup in backups[keep:]:
            backup.unlink()
            logger.debug(f"Removed old backup: {backup}")
    
    def restore_from_backup(self, backup_path: Path) -> bool:
        """
        从备份恢复
        
        Args:
            backup_path: 备份文件路径
            
        Returns:
            是否恢复成功
        """
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, self.storage_path)
            logger.info(f"Restored from backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Path]:
        """列出所有备份"""
        return sorted(self.backup_dir.glob("holdings_*.json"), reverse=True)
