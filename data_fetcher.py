"""
数据获取模块
使用akshare获取A股财务数据和行情数据
修复：缓存机制、重试、数据校验
"""
import os
import json
import logging
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import numpy as np

from utils import retry, parse_cn_number

logger = logging.getLogger("quant.data")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _is_cache_valid(filepath, max_age_days):
    """检查缓存文件是否在有效期内"""
    if not os.path.exists(filepath):
        return False
    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now() - mod_time < timedelta(days=max_age_days)


# ============ 股票池 ============

def get_stock_pool(pool_name="hs300"):
    """获取股票池"""
    cache_file = os.path.join(CACHE_DIR, f"stock_pool_{pool_name}.csv")

    if _is_cache_valid(cache_file, max_age_days=7):
        logger.info(f"从缓存加载股票池: {pool_name}")
        return pd.read_csv(cache_file, dtype={"code": str})

    logger.info(f"从网络获取股票池: {pool_name}")
    if pool_name == "hs300":
        df = ak.index_stock_cons_df(symbol="000300")
        df = df.rename(columns={"品种代码": "code", "品种名称": "name"})
    elif pool_name == "zz500":
        df = ak.index_stock_cons_df(symbol="000905")
        df = df.rename(columns={"品种代码": "code", "品种名称": "name"})
    else:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={"代码": "code", "名称": "name"})
        df = df[["code", "name"]]

    df.to_csv(cache_file, index=False)
    return df


# ============ 财务数据（原始接口，带重试）============

@retry(max_retries=3, base_delay=1.0)
def get_financial_indicators(stock_code):
    """获取财务分析指标（多年）"""
    df = ak.stock_financial_analysis_indicator(symbol=stock_code)
    if df is None or df.empty:
        return None
    return df


@retry(max_retries=3, base_delay=1.0)
def get_cashflow_raw(stock_code):
    """获取现金流量表原始数据"""
    df = ak.stock_cash_flow_sheet_by_report_em(symbol=stock_code)
    if df is None or df.empty:
        return None
    return df


@retry(max_retries=3, base_delay=1.0)
def get_profit_raw(stock_code):
    """获取利润表原始数据"""
    df = ak.stock_profit_sheet_by_report_em(symbol=stock_code)
    if df is None or df.empty:
        return None
    return df


@retry(max_retries=3, base_delay=1.0)
def get_balance_sheet_raw(stock_code):
    """获取资产负债表原始数据"""
    df = ak.stock_balance_sheet_by_report_em(symbol=stock_code)
    if df is None or df.empty:
        return None
    return df


# ============ 行情数据 ============

@retry(max_retries=3, base_delay=1.0)
def get_price_data(stock_code, start_date="20210101", end_date=None):
    """
    获取行情数据（日K线）
    缓存策略：按 stock_code + start_date 缓存，不含 end_date
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_file = os.path.join(CACHE_DIR, f"price_{stock_code}_{start_date}.csv")

    if _is_cache_valid(cache_file, max_age_days=1):
        logger.debug(f"从缓存加载行情: {stock_code}")
        df = pd.read_csv(cache_file)
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
        return df

    logger.info(f"从网络获取行情: {stock_code}")
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    if df is not None and not df.empty:
        required_cols = ["日期", "收盘"]
        if not all(c in df.columns for c in required_cols):
            logger.warning(f"{stock_code} 行情数据缺少必要列: {df.columns.tolist()}")
            return None
        df.to_csv(cache_file, index=False)
    else:
        logger.warning(f"{stock_code} 无行情数据")

    return df


@retry(max_retries=3, base_delay=1.0)
def get_index_daily(symbol="000300", start_date="20210101"):
    """获取指数日线数据"""
    cache_file = os.path.join(CACHE_DIR, f"index_{symbol}_{start_date}.csv")

    if _is_cache_valid(cache_file, max_age_days=1):
        return pd.read_csv(cache_file)

    df = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date)
    if df is not None and not df.empty:
        df.to_csv(cache_file, index=False)
    return df


# ============ 实时行情（一次获取）============

@retry(max_retries=3, base_delay=1.0)
def get_realtime_quotes():
    """获取全市场实时行情"""
    cache_file = os.path.join(CACHE_DIR, "realtime_quotes.csv")

    if _is_cache_valid(cache_file, max_age_days=0.5):
        return pd.read_csv(cache_file, dtype={"代码": str})

    df = ak.stock_zh_a_spot_em()
    if df is not None and not df.empty:
        df.to_csv(cache_file, index=False)
    return df


# ============ 分红数据 ============

@retry(max_retries=3, base_delay=1.0)
def get_dividend_data(stock_code):
    """获取分红数据"""
    cache_file = os.path.join(CACHE_DIR, f"dividend_{stock_code}.json")

    if _is_cache_valid(cache_file, max_age_days=90):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        df = ak.stock_dividend_cninfo(symbol=stock_code)
        if df is not None and not df.empty:
            result = df.iloc[0].to_dict()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, default=str)
            return result
    except Exception as e:
        logger.warning(f"{stock_code} 获取分红数据失败: {e}")

    return None


# ============ 行业分类 ============

def get_industry_classification():
    """获取行业分类"""
    cache_file = os.path.join(CACHE_DIR, "industry.csv")

    if _is_cache_valid(cache_file, max_age_days=30):
        return pd.read_csv(cache_file)

    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        logger.warning(f"获取行业分类失败: {e}")
        return None
