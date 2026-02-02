# -*- coding: utf-8 -*-
"""
日志系统
配置 FileHandler 和 StreamHandler
格式: [Time] [Level] Message
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class Logger:
    """统一日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if Logger._initialized:
            return
        Logger._initialized = True
        self._loggers = {}
    
    def get_logger(self, name: str = "DeepAlpha", log_file: str = None) -> logging.Logger:
        """
        获取或创建 Logger
        
        所有子 logger 使用 DeepAlpha 作为命名空间前缀
        例如 get_logger("DataTab") 实际返回 "DeepAlpha.DataTab"
        
        Args:
            name: Logger 名称
            log_file: 日志文件路径，为空则使用默认路径
        
        Returns:
            配置好的 Logger 实例
        """
        # 构建完整的 logger 名称
        if name == "DeepAlpha":
            full_name = "DeepAlpha"
        else:
            full_name = f"DeepAlpha.{name}"
            # 确保根 logger 已创建
            if "DeepAlpha" not in self._loggers:
                self.get_logger("DeepAlpha", log_file)
        
        if full_name in self._loggers:
            return self._loggers[full_name]
        
        logger = logging.getLogger(full_name)
        logger.setLevel(logging.DEBUG)
        
        # 只为根 logger (DeepAlpha) 添加 handlers
        # 子 logger 会自动传播到父 logger
        if name == "DeepAlpha":
            logger.handlers.clear()  # 清除已有的 handlers
            logger.propagate = False  # 根 logger 不传播
            
            # 格式化器: [Time] [Level] [Name] Message
            formatter = logging.Formatter(
                fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 控制台 Handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 文件 Handler (追加模式)
            if log_file is None:
                # 使用默认路径
                log_dir = Path(__file__).parent.parent / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "deepalpha.log"
            else:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            # 子 logger 清除自己的 handlers，让它传播到父 logger
            logger.handlers.clear()
            logger.propagate = True
        
        self._loggers[full_name] = logger
        return logger


def get_logger(name: str = "DeepAlpha") -> logging.Logger:
    """便捷函数，获取 Logger"""
    return Logger().get_logger(name)


# 创建根 logger (确保先创建)
_root_logger = get_logger("DeepAlpha")
# 向后兼容别名
logger = _root_logger


# GUI 日志 Handler - 用于在 GUI 中显示日志
class QTextEditHandler(logging.Handler):
    """
    Qt 文本框日志 Handler
    可以将日志输出到 QTextEdit 控件
    使用 QMetaObject.invokeMethod 确保跨线程安全
    """
    
    def __init__(self, text_edit=None):
        super().__init__()
        self.text_edit = text_edit
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter(
            fmt='[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
    
    def set_text_edit(self, text_edit):
        """设置 QTextEdit 控件"""
        self.text_edit = text_edit
    
    def emit(self, record):
        if self.text_edit is None:
            return
        try:
            msg = self.format(record)
            # 根据日志级别设置颜色
            color = self._get_color(record.levelno)
            # 转义 HTML 特殊字符
            import html as html_module
            msg_escaped = html_module.escape(msg)
            html_text = f'<span style="color: {color};">{msg_escaped}</span>'
            
            # 使用 QMetaObject.invokeMethod 确保在主线程更新
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(
                self.text_edit, 
                "append",
                Qt.QueuedConnection,
                Q_ARG(str, html_text)
            )
        except Exception:
            self.handleError(record)
    
    def _get_color(self, level: int) -> str:
        """根据日志级别返回颜色"""
        if level >= logging.ERROR:
            return '#FF5555'  # 红色
        elif level >= logging.WARNING:
            return '#FFAA00'  # 橙色
        elif level >= logging.INFO:
            return '#55FF55'  # 绿色
        else:
            return '#AAAAAA'  # 灰色
