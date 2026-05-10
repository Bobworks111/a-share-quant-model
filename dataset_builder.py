"""
数据集构建器
按时间线构建数据集，消除前瞻偏差
核心接口：get_snapshot(stock_codes, as_of_date) -> DataFrame
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from data_fetcher import (
    get_financial_indicators, get_cashflow_raw, get_profit_raw,
    get_balance_sheet_raw, get_price_data, get_realtime_quotes,
    get_dividend_data, get_index_daily
)
from utils import parse_cn_number

logger = logging.getLogger("quant.builder")


def _extract_multi_year(df, years=5, as_of_date=None):
    """
    从财务指标表中提取多年数据
    as_of_date: 截止日期，只使用该日期之前的报表
    """
    if df is None or df.empty:
        return None

    # 如果有报告日期列，过滤掉未来数据
    date_col = None
    for col in df.columns:
        if "日期" in col or "报告" in col:
            date_col = col
            break

    if date_col and as_of_date:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            mask = dates <= pd.Timestamp(as_of_date)
            df = df[mask]
        except Exception:
            pass

    df = df.head(years)
    if df.empty:
        return None

    result = {}

    # ROE
    roe_col = [c for c in df.columns if "净资产收益率" in c and "摊薄" in c]
    if roe_col:
        roe_values = pd.to_numeric(df[roe_col[0]], errors="coerce").dropna()
        if len(roe_values) >= 2:
            result["roe_5y_mean"] = roe_values.mean()
            result["roe_5y_std"] = roe_values.std()
            if roe_values.iloc[-1] > 0:
                result["roe_growth_3y"] = (roe_values.iloc[0] / roe_values.iloc[-1] - 1) * 100
        elif len(roe_values) == 1:
            result["roe_5y_mean"] = roe_values.iloc[0]
            result["roe_5y_std"] = 0

    # 毛利率
    gm_col = [c for c in df.columns if "销售毛利率" in c or "毛利率" in c]
    if gm_col:
        gm_values = pd.to_numeric(df[gm_col[0]], errors="coerce").dropna()
        if len(gm_values) >= 2:
            result["gross_margin_5y_mean"] = gm_values.mean()
            result["gross_margin_5y_std"] = gm_values.std()
        elif len(gm_values) == 1:
            result["gross_margin_5y_mean"] = gm_values.iloc[0]
            result["gross_margin_5y_std"] = 0

    # 资产负债率
    debt_col = [c for c in df.columns if "资产负债率" in c]
    if debt_col:
        debt_values = pd.to_numeric(df[debt_col[0]], errors="coerce").dropna()
        if len(debt_values) > 0:
            result["debt_ratio"] = debt_values.iloc[0]

    # 流动比率
    cr_col = [c for c in df.columns if "流动比率" in c]
    if cr_col:
        cr_values = pd.to_numeric(df[cr_col[0]], errors="coerce").dropna()
        if len(cr_values) > 0:
            result["current_ratio"] = cr_values.iloc[0]

    # 净利润增速
    pg_col = [c for c in df.columns if "净利润增长率" in c or "净利润同比" in c]
    if pg_col:
        pg_values = pd.to_numeric(df[pg_col[0]], errors="coerce").dropna()
        if len(pg_values) >= 3:
            result["profit_growth_3y"] = pg_values.head(3).mean()
            mean_val = pg_values.head(3).mean()
            result["growth_stability"] = pg_values.head(3).std() / abs(mean_val) if mean_val != 0 else 999
        elif len(pg_values) > 0:
            result["profit_growth_3y"] = pg_values.iloc[0]
            result["growth_stability"] = 0

    # 营收增速
    rg_col = [c for c in df.columns if "营业收入增长率" in c or "营收同比" in c]
    if rg_col:
        rg_values = pd.to_numeric(df[rg_col[0]], errors="coerce").dropna()
        if len(rg_values) >= 3:
            result["revenue_growth_3y"] = rg_values.head(3).mean()
            if "profit_growth_3y" in result and result.get("revenue_growth_3y", 0) != 0:
                result["profit_revenue_ratio"] = result["profit_growth_3y"] / result["revenue_growth_3y"]
        elif len(rg_values) > 0:
            result["revenue_growth_3y"] = rg_values.iloc[0]

    # 连续正利润年数
    if pg_col:
        pg_values = pd.to_numeric(df[pg_col[0]], errors="coerce").dropna()
        consecutive = 0
        for v in pg_values:
            if v > 0:
                consecutive += 1
            else:
                break
        result["consecutive_profit_years"] = consecutive

    # 利息覆盖倍数
    ic_col = [c for c in df.columns if "利息保障倍数" in c or "利息覆盖" in c]
    if ic_col:
        ic_values = pd.to_numeric(df[ic_col[0]], errors="coerce").dropna()
        if len(ic_values) > 0:
            result["interest_coverage"] = ic_values.iloc[0]

    return result


def _extract_cashflow(df, as_of_date=None):
    """从现金流量表提取数据"""
    if df is None or df.empty:
        return {}

    # 按日期过滤
    if as_of_date:
        date_col = None
        for col in df.columns:
            if "日期" in col or "报告" in col:
                date_col = col
                break
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                df = df[dates <= pd.Timestamp(as_of_date)]
            except Exception:
                pass

    if df.empty:
        return {}

    result = {}
    latest = df.iloc[0]

    ocf_col = [c for c in df.columns if "经营活动产生的现金流量净额" in c]
    if ocf_col:
        result["operating_cashflow"] = pd.to_numeric(latest[ocf_col[0]], errors="coerce")

    capex_col = [c for c in df.columns if "购建固定资产" in c]
    if capex_col:
        result["capex"] = pd.to_numeric(latest[capex_col[0]], errors="coerce")

    # 近3年现金流连续性
    if ocf_col and len(df) >= 3:
        ocf_values = pd.to_numeric(df[ocf_col[0]].head(3), errors="coerce").dropna()
        result["cashflow_continuity"] = 1 if all(v > 0 for v in ocf_values) else 0

    return result


def _extract_profit(df, as_of_date=None):
    """从利润表提取数据"""
    if df is None or df.empty:
        return {}

    if as_of_date:
        date_col = None
        for col in df.columns:
            if "日期" in col or "报告" in col:
                date_col = col
                break
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                df = df[dates <= pd.Timestamp(as_of_date)]
            except Exception:
                pass

    if df.empty:
        return {}

    result = {}
    latest = df.iloc[0]

    np_col = [c for c in df.columns if "净利润" in c and "扣非" not in c]
    if np_col:
        result["net_profit"] = pd.to_numeric(latest[np_col[0]], errors="coerce")

    rev_col = [c for c in df.columns if "营业总收入" in c or "营业收入" in c]
    if rev_col:
        result["revenue"] = pd.to_numeric(latest[rev_col[0]], errors="coerce")

    return result


def _extract_balance_sheet(df, as_of_date=None):
    """从资产负债表提取数据"""
    if df is None or df.empty:
        return {}

    if as_of_date:
        date_col = None
        for col in df.columns:
            if "日期" in col or "报告" in col:
                date_col = col
                break
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                df = df[dates <= pd.Timestamp(as_of_date)]
            except Exception:
                pass

    if df.empty:
        return {}

    result = {}
    latest = df.iloc[0]

    ta_col = [c for c in df.columns if "资产总计" in c or "总资产" in c]
    if ta_col:
        result["total_assets"] = pd.to_numeric(latest[ta_col[0]], errors="coerce")

    return result


def calc_beta_and_volatility(stock_code, as_of_date, market_code="000300", window_days=252):
    """
    计算Beta和特异性波动率
    使用 as_of_date 之前 window_days 个交易日的数据
    """
    try:
        # 计算起始日期（往前多取一些以确保有足够交易日）
        start_dt = pd.Timestamp(as_of_date) - timedelta(days=int(window_days * 1.5))
        start_str = start_dt.strftime("%Y%m%d")
        end_str = pd.Timestamp(as_of_date).strftime("%Y%m%d")

        stock_df = get_price_data(stock_code, start_date=start_str, end_date=end_str)
        if stock_df is None or stock_df.empty or len(stock_df) < 60:
            return {}

        index_df = get_index_daily(market_code, start_date=start_str)
        if index_df is None or index_df.empty:
            return {}

        # 日期对齐
        stock_df["日期"] = pd.to_datetime(stock_df["日期"])
        index_df["date"] = pd.to_datetime(index_df["date"])

        # 只使用 as_of_date 之前的数据
        as_of_ts = pd.Timestamp(as_of_date)
        stock_df = stock_df[stock_df["日期"] <= as_of_ts].set_index("日期").sort_index()
        index_df = index_df[index_df["date"] <= as_of_ts].set_index("date").sort_index()

        # 取最近 window_days 个交易日
        if len(stock_df) > window_days:
            stock_df = stock_df.tail(window_days)
        if len(index_df) > window_days:
            index_df = index_df.tail(window_days)

        common_dates = stock_df.index.intersection(index_df.index)
        if len(common_dates) < 60:
            return {}

        stock_ret = stock_df.loc[common_dates, "收盘"].pct_change().dropna()
        bench_ret = index_df.loc[common_dates, "close"].pct_change().dropna()

        common = stock_ret.index.intersection(bench_ret.index)
        if len(common) < 60:
            return {}

        stock_ret = stock_ret.loc[common]
        bench_ret = bench_ret.loc[common]

        cov = np.cov(stock_ret, bench_ret)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

        predicted = beta * bench_ret
        residuals = stock_ret - predicted
        idio_vol = residuals.std() * np.sqrt(252)

        return {"beta": beta, "idio_volatility": idio_vol}

    except Exception as e:
        logger.warning(f"{stock_code} 计算Beta失败: {e}")
        return {}


def calc_pe_percentile(stock_code, current_pe, as_of_date, years=5):
    """
    计算PE的历史分位数
    使用 as_of_date 之前 years 年的PE数据
    """
    try:
        start_dt = pd.Timestamp(as_of_date) - timedelta(days=years * 365)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = pd.Timestamp(as_of_date).strftime("%Y%m%d")

        price_df = get_price_data(stock_code, start_date=start_str, end_date=end_str)
        if price_df is None or price_df.empty or len(price_df) < 60:
            return 0.5

        # 简化：用价格分位数近似PE分位数
        # （严格来说需要历史EPS数据来计算历史PE序列）
        price_df["日期"] = pd.to_datetime(price_df["日期"])
        price_df = price_df[price_df["日期"] <= pd.Timestamp(as_of_date)]

        if price_df.empty:
            return 0.5

        prices = price_df["收盘"].dropna()
        if len(prices) < 20:
            return 0.5

        # PE与价格负相关：价格越高PE越高，分位数越大
        current_price = prices.iloc[-1]
        percentile = (prices < current_price).sum() / len(prices)
        return percentile

    except Exception as e:
        logger.warning(f"{stock_code} 计算PE分位数失败: {e}")
        return 0.5


def get_snapshot(stock_codes, as_of_date, realtime_quotes_df=None):
    """
    获取指定日期的数据快照
    核心接口：只使用 as_of_date 之前已发布的数据，消除前瞻偏差

    参数:
        stock_codes: 股票代码列表
        as_of_date: 截止日期（str, 如 "2023-06-30"）
        realtime_quotes_df: 预加载的实时行情DataFrame（避免重复请求）

    返回:
        DataFrame，每行一只股票，包含所有因子所需的指标
    """
    logger.info(f"构建数据快照: {len(stock_codes)}只股票, 截止{as_of_date}")

    # 一次获取实时行情（用于PE/PB/股息率等）
    if realtime_quotes_df is None:
        try:
            realtime_quotes_df = get_realtime_quotes()
        except Exception:
            realtime_quotes_df = pd.DataFrame()

    rows = []
    total = len(stock_codes)

    for i, code in enumerate(stock_codes):
        if (i + 1) % 10 == 0:
            logger.info(f"进度: {i+1}/{total}")

        # 1. 多年财务数据（带日期过滤）
        try:
            fin_df = get_financial_indicators(code)
            multi_year = _extract_multi_year(fin_df, years=5, as_of_date=as_of_date)
            if multi_year is None:
                continue
        except Exception as e:
            logger.warning(f"{code} 获取财务指标失败: {e}")
            continue

        multi_year["code"] = code

        # 2. 现金流数据
        try:
            cf_df = get_cashflow_raw(code)
            cf_data = _extract_cashflow(cf_df, as_of_date=as_of_date)
            multi_year.update(cf_data)
        except Exception as e:
            logger.debug(f"{code} 获取现金流失败: {e}")

        # 3. 利润表数据
        try:
            profit_df = get_profit_raw(code)
            profit_data = _extract_profit(profit_df, as_of_date=as_of_date)
            multi_year.update(profit_data)
        except Exception as e:
            logger.debug(f"{code} 获取利润表失败: {e}")

        # 4. 资产负债表数据
        try:
            bs_df = get_balance_sheet_raw(code)
            bs_data = _extract_balance_sheet(bs_df, as_of_date=as_of_date)
            multi_year.update(bs_data)
        except Exception as e:
            logger.debug(f"{code} 获取资产负债表失败: {e}")

        # 5. 计算衍生指标
        # 自由现金流/净利润
        if "operating_cashflow" in multi_year and "capex" in multi_year and "net_profit" in multi_year:
            fcf = multi_year["operating_cashflow"] - multi_year.get("capex", 0)
            np_val = multi_year["net_profit"]
            if np_val and np_val != 0:
                multi_year["fcf_to_profit"] = fcf / np_val
            ta = multi_year.get("total_assets")
            if ta and ta != 0:
                multi_year["fcf_to_assets"] = fcf / ta

        # 资本开支/营收
        if "capex" in multi_year and "revenue" in multi_year:
            rev = multi_year["revenue"]
            if rev and rev != 0:
                multi_year["capex_to_revenue"] = abs(multi_year["capex"]) / rev

        # 应计比率
        if all(k in multi_year for k in ["net_profit", "operating_cashflow", "total_assets"]):
            ta = multi_year["total_assets"]
            if ta and ta != 0:
                multi_year["accruals_ratio"] = (multi_year["net_profit"] - multi_year["operating_cashflow"]) / ta

        # 6. Beta 和特异性波动率（使用截止日期前的数据）
        beta_vol = calc_beta_and_volatility(code, as_of_date=as_of_date)
        multi_year.update(beta_vol)

        # 7. PE/PB/名称（从预加载的实时行情中获取）
        if not realtime_quotes_df.empty:
            spot_row = realtime_quotes_df[realtime_quotes_df["代码"] == code]
            if not spot_row.empty:
                row = spot_row.iloc[0]
                multi_year["pe"] = parse_cn_number(row.get("市盈率-动态", 0))
                multi_year["pb"] = parse_cn_number(row.get("市净率", 0))
                multi_year["name"] = row.get("名称", code)

        # 8. PE历史分位数（用真实数据计算）
        pe_val = multi_year.get("pe")
        if pe_val and pe_val > 0:
            multi_year["pe_percentile"] = calc_pe_percentile(code, pe_val, as_of_date=as_of_date)

        # 9. 股息率
        div_data = get_dividend_data(code)
        if div_data:
            multi_year["dividend_yield"] = parse_cn_number(div_data.get("股息率", 0))
        else:
            multi_year["dividend_yield"] = 0

        # 10. 派息比例
        if div_data and "net_profit" in multi_year and multi_year["net_profit"] > 0:
            div_amount = parse_cn_number(div_data.get("每股股利", 0))
            if div_amount:
                multi_year["dividend_payout"] = min(div_amount * 1e8 / multi_year["net_profit"], 1.0)

        rows.append(multi_year)

    if not rows:
        logger.warning("无有效数据")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"快照构建完成: {len(df)}只股票")
    return df
