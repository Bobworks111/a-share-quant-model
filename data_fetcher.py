"""
数据获取模块
使用akshare获取A股财务数据和行情数据
"""
import akshare as ak
import pandas as pd
import os
import json
from datetime import datetime, timedelta

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_stock_pool(pool_name="hs300"):
    """获取股票池"""
    cache_file = f"{CACHE_DIR}/stock_pool_{pool_name}.csv"

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(days=7):
            return pd.read_csv(cache_file, dtype={"code": str})

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


def get_financial_data(stock_code):
    """获取单只股票的财务数据"""
    cache_file = f"{CACHE_DIR}/financial_{stock_code}.json"

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(days=1):
            with open(cache_file, "r") as f:
                return json.load(f)

    try:
        # 获取财务指标
        df = ak.stock_financial_analysis_indicator(symbol=stock_code)
        if df is None or df.empty:
            return None

        # 获取最新数据
        latest = df.iloc[0].to_dict()

        # 获取股息率
        try:
            div_df = ak.stock_dividend_cninfo(symbol=stock_code)
            if div_df is not None and not div_df.empty:
                latest["dividend_yield"] = float(div_df.iloc[0].get("股息率", 0))
            else:
                latest["dividend_yield"] = 0
        except:
            latest["dividend_yield"] = 0

        with open(cache_file, "w") as f:
            json.dump(latest, f, ensure_ascii=False, default=str)

        return latest
    except Exception as e:
        print(f"获取{stock_code}财务数据失败: {e}")
        return None


def get_price_data(stock_code, start_date="20210101", end_date=None):
    """获取行情数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_file = f"{CACHE_DIR}/price_{stock_code}_{start_date}_{end_date}.csv"

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(days=1):
            return pd.read_csv(cache_file, parse_dates=["日期"])

    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        print(f"获取{stock_code}行情数据失败: {e}")
        return None


def get_realtime_quote(stock_code):
    """获取实时行情"""
    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == stock_code]
        if not row.empty:
            return row.iloc[0].to_dict()
    except:
        pass
    return None


def batch_fetch_financials(stock_codes, progress=True):
    """批量获取财务数据"""
    results = {}
    total = len(stock_codes)
    for i, code in enumerate(stock_codes):
        if progress and (i + 1) % 10 == 0:
            print(f"进度: {i+1}/{total}")
        data = get_financial_data(code)
        if data:
            results[code] = data
    return results


def get_industry_classification():
    """获取行业分类"""
    cache_file = f"{CACHE_DIR}/industry.csv"

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(days=30):
            return pd.read_csv(cache_file, dtype={"code": str})

    try:
        df = ak.stock_board_industry_name_em()
        return df
    except:
        return None
