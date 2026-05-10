"""
工具函数模块
重试装饰器、日志配置、通用工具
"""
import time
import logging
import functools
import re
import numpy as np


# ============ 日志配置 ============

def setup_logging(level=logging.INFO, log_file=None):
    """配置全局日志"""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


logger = logging.getLogger("quant")


# ============ 重试装饰器 ============

def retry(max_retries=3, base_delay=1.0, exceptions=(Exception,)):
    """
    指数退避重试装饰器
    用法: @retry(max_retries=3, base_delay=1.0)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} 第{attempt+1}次失败: {e}, {delay}s后重试")
                    time.sleep(delay)
            logger.error(f"{func.__name__} 重试{max_retries}次后仍失败")
            raise last_exception
        return wrapper
    return decorator


# ============ 数值解析 ============

def parse_cn_number(value):
    """
    解析中文数值（如 '1.23亿' -> 123000000）
    支持：亿、万、% 等后缀
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return np.nan

    value = value.strip()
    if not value or value == "--" or value == "-":
        return np.nan

    multiplier = 1.0
    if value.endswith("亿"):
        multiplier = 1e8
        value = value[:-1]
    elif value.endswith("万"):
        multiplier = 1e4
        value = value[:-1]
    elif value.endswith("%"):
        multiplier = 0.01
        value = value[:-1]

    try:
        return float(value) * multiplier
    except (ValueError, TypeError):
        return np.nan


def safe_numeric(series):
    """安全地将 Series 转为数值类型"""
    return series.apply(parse_cn_number) if hasattr(series, 'apply') else np.nan
