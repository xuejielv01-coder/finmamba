# -*- coding: utf-8 -*-
"""
个股极速雷达 (Radar)
PRD 4.2 实现

特性:
- 单股快速诊断 (< 100ms)
- 使用缓存统计量
- 涨跌方向判断
- 相似股推荐
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from utils.logger import get_logger
from utils.seeder import get_device
from utils.tools import is_main_board
from data.preprocessor import Preprocessor
from models.finmamba import AlphaModel
from data.dataset import FastAlphaDataset

logger = get_logger("Radar")


class Radar:
    """
    个股极速雷达
    
    功能:
    1. 快速单股诊断
    2. 涨跌预测
    3. 置信度评估
    4. 相似股推荐
    """
    
    # 分数到预期收益的映射系数
    SCORE_TO_RETURN_FACTOR = 0.02
    
    def __init__(
        self,
        model: AlphaModel = None,
        model_path: Path = None,
        device: torch.device = None,
        preload: bool = True
    ):
        """
        初始化雷达
        """
        self.device = device or get_device()
        
        # 加载模型
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model(model_path or Config.BEST_MODEL_PATH)
        
        self.model.eval()
        
        # 预处理器
        self.preprocessor = Preprocessor()
        
        # 加载行业映射 (Key Update)
        self.industry_map = FastAlphaDataset.get_stock_industry_map()
        logger.info(f"Loaded industry map with {len(self.industry_map)} stocks")
        
        # 缓存最近加载的统计量
        self._stats_cache: Dict[str, Dict] = {}
        
        # 缓存股票 embedding (用于相似股推荐)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # 预热 (减少首次推理延迟)
        if preload:
            self._warmup()
        
        logger.info("Radar initialized")
    
    def _load_model(self, model_path: Path) -> AlphaModel:
        """加载模型"""
        model = AlphaModel(
            d_model=Config.D_MODEL,
            n_layers=Config.N_LAYERS,
            n_industries=Config.N_INDUSTRIES
        )
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            try:
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load state dict: {e}")
        else:
            logger.warning(f"Model not found: {model_path}")
        
        return model.to(self.device)
    
    def _warmup(self):
        """模型预热"""
        # 构造 dummy 输入 (x + industry_ids)
        dummy_x = torch.randn(1, Config.SEQ_LEN, Config.FEATURE_DIM).to(self.device)
        dummy_ind = torch.zeros(1, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_x, dummy_ind)
        logger.debug("Model warmup complete")
    
    def diagnose_single(
        self,
        ts_code: str,
        date: str = None,
        return_embedding: bool = False
    ) -> Dict:
        """
        单股诊断
        """
        start_time = time.perf_counter()
        
        result = {
            'ts_code': ts_code,
            'date': date or 'latest',
            'score': 0.0,
            'direction': '中性',
            'magnitude': '0.0%',
            'confidence': 0.0,
            'risk_level': 'medium',
            'latency_ms': 0.0,
            'status': 'success',
            'message': ''
        }
        
        try:
            # 1. 合法性校验
            if not is_main_board(ts_code):
                result['status'] = 'error'
                result['message'] = '非主板股票'
                return result
            
            # 2. 加载数据 (仅最近60行)
            stock_data = self._load_stock_data(ts_code)
            if stock_data is None or len(stock_data) < Config.SEQ_LEN:
                result['status'] = 'error'
                result['message'] = '数据不足'
                return result
            
            # 3. 加载统计量缓存
            if date is None:
                date = stock_data['trade_date'].iloc[-1]
                if hasattr(date, 'strftime'):
                    date = date.strftime('%Y%m%d')
                else:
                    date = str(date).replace('-', '')
            
            result['date'] = date
            stats = self._get_stats(date)
            
            # 尝试获取最近的缓存
            if stats is None:
                stats = self.preprocessor.get_latest_stats_cache()
            
            # 4. 预处理和标准化
            processed = self.preprocessor.process_daily_data(stock_data)
            if stats:
                processed = self.preprocessor.apply_zscore(processed, stats)
            
            # 5. 提取特征
            features = processed[self.preprocessor.FEATURES].values[-Config.SEQ_LEN:].astype(np.float32)
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 获取行业ID
            ind_id = self.industry_map.get(ts_code, 0)
            ind_ids = torch.tensor([ind_id], dtype=torch.long).to(self.device)
            
            # 6. 推理
            with torch.no_grad():
                # FinMamba input: (x, industry_ids)
                model_output = self.model(X, ind_ids)
                
                if return_embedding:
                    self._embedding_cache[ts_code] = model_output.cpu().numpy().flatten()
                    result['embedding'] = self._embedding_cache[ts_code].tolist()
                
                score = model_output.cpu().numpy().item()
            
            # 7. 结果解析
            result['score'] = float(score)
            
            # 高级技术分析
            close = stock_data['close'].iloc[-1]
            ma20 = stock_data['close'].rolling(20).mean().iloc[-1]
            ma60 = stock_data['close'].rolling(60).mean().iloc[-1]
            vol_ma5 = stock_data['vol'].rolling(5).mean().iloc[-1]
            
            technical_signals = []
            if close > ma20: technical_signals.append("站上月线")
            if ma20 > ma60: technical_signals.append("均线多头")
            if stock_data['vol'].iloc[-1] > vol_ma5 * 1.5: technical_signals.append("放量")
            
            result['analysis'] = " | ".join(technical_signals) if technical_signals else "技术面弱势"
            
            # 支撑/压力位 (简单估计)
            recent_high = stock_data['high'].tail(20).max()
            recent_low = stock_data['low'].tail(20).min()
            result['pressure'] = f"{recent_high:.2f}"
            result['support'] = f"{recent_low:.2f}"

            # 综合决策
            if score > 0.65 and close > ma20:
                result['direction'] = '强烈看涨'
                result['action'] = 'BUY'
                result['risk_level'] = 'low'
            elif score > 0.55:
                result['direction'] = '看涨'
                result['action'] = 'ACCUMULATE'
                result['risk_level'] = 'medium'
            elif score < 0.4:
                result['direction'] = '看跌'
                result['action'] = 'SELL'
                result['risk_level'] = 'high'
            else:
                result['direction'] = '震荡'
                result['action'] = 'HOLD'
                result['risk_level'] = 'medium'
            
            # 预期涨幅
            expected_return = (score - 0.5) * self.SCORE_TO_RETURN_FACTOR * 100
            sign = '+' if expected_return > 0 else ''
            result['magnitude'] = f"{sign}{expected_return:.2f}%"
            
            # 置信度
            confidence = min(abs(score - 0.5) * 2, 1.0)
            result['confidence'] = float(confidence)
            
        except Exception as e:
            result['status'] = 'error'
            result['message'] = str(e)
            logger.error(f"Diagnosis failed for {ts_code}: {e}")
        
        # 计算延迟
        result['latency_ms'] = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def _load_stock_data(self, ts_code: str) -> Optional[pd.DataFrame]:
        """
        加载股票数据
        """
        filename = ts_code.replace('.', '_') + '.parquet'
        filepath = Config.RAW_DATA_DIR / filename
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_parquet(filepath)
            df = df.sort_values('trade_date').tail(Config.SEQ_LEN + 10)
            return df
        except Exception as e:
            logger.error(f"Failed to load {ts_code}: {e}")
            return None
    
    def _get_stats(self, date: str) -> Optional[Dict]:
        """获取统计量缓存 (带内存缓存)"""
        if date in self._stats_cache:
            return self._stats_cache[date]
        
        stats = self.preprocessor.load_stats_cache(date)
        if stats:
            if len(self._stats_cache) > 5:
                oldest_key = next(iter(self._stats_cache))
                del self._stats_cache[oldest_key]
            self._stats_cache[date] = stats
        
        return stats
    
    def find_similar_stocks(
        self,
        ts_code: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        查找相似股票 (基于 embedding 余弦相似度)
        """
        if ts_code not in self._embedding_cache:
            self.diagnose_single(ts_code, return_embedding=True)
        
        if ts_code not in self._embedding_cache:
            return []
        
        target_emb = self._embedding_cache[ts_code]
        
        similarities = []
        for code, emb in self._embedding_cache.items():
            if code == ts_code:
                continue
            
            sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb) + 1e-8)
            similarities.append({'ts_code': code, 'similarity': float(sim)})
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def batch_diagnose(
        self,
        ts_codes: List[str],
        date: str = None
    ) -> List[Dict]:
        """批量诊断"""
        results = []
        for ts_code in ts_codes:
            result = self.diagnose_single(ts_code, date)
            results.append(result)
        return results


def diagnose_single(ts_code: str, **kwargs) -> Dict:
    """便捷函数：单股诊断"""
    radar = Radar(preload=True)
    return radar.diagnose_single(ts_code, **kwargs)
