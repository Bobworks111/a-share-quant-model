"""
因子研究工具
IC分析、分层回测、因子相关性、因子衰减
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

from config import FACTORS, HARD_CONSTRAINTS
from factor_engine import (
    calc_quality_factors, calc_earnings_quality_factors,
    calc_growth_factors, calc_safety_factors, calc_valuation_factors,
    preprocess_indicator, winsorize_mad, zscore
)

logger = logging.getLogger("quant.research")


# ============ IC 分析 ============

def calc_single_factor_ic(factor_values, forward_returns):
    """
    计算单期因子 IC（Information Coefficient）
    IC = Spearman秩相关(因子值, 下期收益)

    返回: IC值（float）
    """
    # 去除NaN
    mask = factor_values.notna() & forward_returns.notna()
    if mask.sum() < 5:
        return np.nan

    f = factor_values[mask]
    r = forward_returns[mask]

    ic, _ = stats.spearmanr(f, r)
    return ic


def calc_factor_ic_series(factor_name, factor_values_fn, price_data, stock_codes,
                          start_date, end_date, freq="Q"):
    """
    计算因子的 IC 时间序列

    参数:
        factor_name: 因子名称
        factor_values_fn: 函数(snapshot_df) -> Series，计算因子值
        price_data: {code: DataFrame} 行情数据
        stock_codes: 股票代码列表
        start_date, end_date: 时间范围
        freq: 频率（Q=季度）

    返回:
        DataFrame: 日期, IC值
    """
    from dataset_builder import get_snapshot
    from data_fetcher import get_price_data

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    ic_records = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        try:
            # 获取该日期的数据快照
            snapshot = get_snapshot(stock_codes, as_of_date=date_str)
            if snapshot.empty or len(snapshot) < 5:
                continue

            # 计算因子值
            factor_values = factor_values_fn(snapshot)
            if factor_values is None or factor_values.isna().all():
                continue

            # 计算下期收益（下一季度）
            next_date = (date + pd.DateOffset(months=3)).strftime("%Y-%m-%d")
            forward_returns = pd.Series(index=snapshot.index, dtype=float)

            for idx, row in snapshot.iterrows():
                code = row["code"]
                if code not in price_data:
                    continue
                df = price_data[code]
                df["日期"] = pd.to_datetime(df["日期"])

                start_prices = df[df["日期"] >= date]
                end_prices = df[df["日期"] <= pd.Timestamp(next_date)]

                if not start_prices.empty and not end_prices.empty:
                    p_start = start_prices.iloc[0]["收盘"]
                    p_end = end_prices.iloc[-1]["收盘"]
                    if p_start > 0:
                        forward_returns.loc[idx] = (p_end - p_start) / p_start

            # 计算IC
            ic = calc_single_factor_ic(factor_values, forward_returns)
            if not np.isnan(ic):
                ic_records.append({"date": date, "ic": ic})

        except Exception as e:
            logger.debug(f"{date_str} IC计算失败: {e}")

    if not ic_records:
        return pd.DataFrame(columns=["date", "ic"])

    return pd.DataFrame(ic_records)


def analyze_factor_ic(ic_series_df, factor_name=""):
    """
    分析因子IC统计量

    返回:
        dict: {
            "factor": str,
            "ic_mean": float,
            "ic_std": float,
            "ic_ir": float,       # IC均值/IC标准差
            "ic_positive_rate": float,  # IC>0的比例
            "ic_abs_mean": float,
            "n_periods": int,
            "is_effective": bool,  # 是否有效（|IC均值|>0.03 且 IC_IR>0.5）
        }
    """
    if ic_series_df.empty:
        return {"factor": factor_name, "is_effective": False, "n_periods": 0}

    ic = ic_series_df["ic"]
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0

    result = {
        "factor": factor_name,
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "ic_ir": round(ic_ir, 4),
        "ic_positive_rate": round((ic > 0).mean(), 4),
        "ic_abs_mean": round(ic.abs().mean(), 4),
        "n_periods": len(ic),
        "is_effective": abs(ic_mean) > 0.03 and abs(ic_ir) > 0.5,
    }

    return result


# ============ 分层回测 ============

def quantile_backtest(factor_values, forward_returns, n_quantiles=5):
    """
    分层回测：按因子值分为N组，计算每组的平均收益

    参数:
        factor_values: Series，因子值
        forward_returns: Series，下期收益
        n_quantiles: 分组数

    返回:
        DataFrame: 分组, 平均收益, 股票数
    """
    mask = factor_values.notna() & forward_returns.notna()
    f = factor_values[mask]
    r = forward_returns[mask]

    if len(f) < n_quantiles * 2:
        return pd.DataFrame()

    # 分组
    try:
        groups = pd.qcut(f, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        groups = pd.cut(f, n_quantiles, labels=False)

    result = []
    for g in sorted(groups.unique()):
        group_mask = groups == g
        group_returns = r[group_mask]
        result.append({
            "quantile": int(g) + 1,
            "mean_return": group_returns.mean(),
            "median_return": group_returns.median(),
            "std_return": group_returns.std(),
            "n_stocks": int(group_mask.sum()),
        })

    df = pd.DataFrame(result)

    # 检验单调性
    if len(df) >= 3:
        monotonic_up = all(df["mean_return"].diff().dropna() > 0)
        monotonic_down = all(df["mean_return"].diff().dropna() < 0)
        df.attrs["monotonic"] = monotonic_up or monotonic_down
    else:
        df.attrs["monotonic"] = False

    return df


def run_quantile_analysis(factor_name, factor_values_fn, price_data, stock_codes,
                          start_date, end_date, n_quantiles=5):
    """
    运行完整的分层回测分析

    返回:
        dict: {
            "factor": str,
            "quantile_returns": DataFrame,  # 各组累计收益
            "long_short_return": float,      # 多空收益（Q5-Q1）
            "monotonic": bool,               # 是否单调
        }
    """
    from dataset_builder import get_snapshot

    dates = pd.date_range(start=start_date, end=end_date, freq="Q")
    all_quantile_returns = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        next_date = (date + pd.DateOffset(months=3)).strftime("%Y-%m-%d")

        try:
            snapshot = get_snapshot(stock_codes, as_of_date=date_str)
            if snapshot.empty:
                continue

            factor_values = factor_values_fn(snapshot)
            forward_returns = pd.Series(index=snapshot.index, dtype=float)

            for idx, row in snapshot.iterrows():
                code = row["code"]
                if code not in price_data:
                    continue
                df = price_data[code]
                df["日期"] = pd.to_datetime(df["日期"])
                start_prices = df[df["日期"] >= date]
                end_prices = df[df["日期"] <= pd.Timestamp(next_date)]
                if not start_prices.empty and not end_prices.empty:
                    p_start = start_prices.iloc[0]["收盘"]
                    p_end = end_prices.iloc[-1]["收盘"]
                    if p_start > 0:
                        forward_returns.loc[idx] = (p_end - p_start) / p_start

            qr = quantile_backtest(factor_values, forward_returns, n_quantiles)
            if not qr.empty:
                qr["date"] = date
                all_quantile_returns.append(qr)

        except Exception as e:
            logger.debug(f"{date_str} 分层回测失败: {e}")

    if not all_quantile_returns:
        return {"factor": factor_name, "quantile_returns": pd.DataFrame(),
                "long_short_return": 0, "monotonic": False}

    combined = pd.concat(all_quantile_returns)
    # 各组累计收益
    cumulative = combined.groupby("quantile")["mean_return"].mean()
    long_short = cumulative.iloc[-1] - cumulative.iloc[0] if len(cumulative) >= 2 else 0

    # 单调性检验
    monotonic = False
    if len(cumulative) >= 3:
        diffs = cumulative.diff().dropna()
        monotonic = all(diffs > 0) or all(diffs < 0)

    return {
        "factor": factor_name,
        "quantile_returns": combined,
        "cumulative_returns": cumulative.to_dict(),
        "long_short_return": round(long_short, 4),
        "monotonic": monotonic,
    }


# ============ 因子相关性分析 ============

def calc_factor_correlation(snapshot_df, factors_config=None):
    """
    计算因子间的相关性矩阵

    返回:
        DataFrame: 因子相关矩阵（Spearman）
    """
    if factors_config is None:
        factors_config = FACTORS

    factor_calculators = {
        "quality": calc_quality_factors,
        "earnings_quality": calc_earnings_quality_factors,
        "growth": calc_growth_factors,
        "safety": calc_safety_factors,
        "valuation": calc_valuation_factors,
    }

    factor_scores = pd.DataFrame(index=snapshot_df.index)

    for name, calc_fn in factor_calculators.items():
        if name in factors_config:
            try:
                raw = calc_fn(snapshot_df)
                score = pd.Series(0.0, index=snapshot_df.index)
                config = factors_config[name]
                available_weight = 0

                for ind_name, params in config["indicators"].items():
                    if ind_name in raw.columns:
                        col = pd.to_numeric(raw[ind_name], errors="coerce")
                        if col.isna().all():
                            continue
                        processed = preprocess_indicator(col)
                        if params["ascending"]:
                            processed = -processed
                        score += processed * params["weight"]
                        available_weight += params["weight"]

                if available_weight > 0:
                    score = score / available_weight
                factor_scores[name] = score
            except Exception as e:
                logger.debug(f"{name} 因子计算失败: {e}")

    if factor_scores.empty:
        return pd.DataFrame()

    # Spearman相关矩阵
    corr_matrix = factor_scores.corr(method="spearman")
    return corr_matrix


def find_correlated_pairs(corr_matrix, threshold=0.7):
    """
    找出高度相关的因子对

    返回:
        list of dict: [{"factor_1", "factor_2", "correlation"}]
    """
    pairs = []
    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                pairs.append({
                    "factor_1": cols[i],
                    "factor_2": cols[j],
                    "correlation": round(corr_val, 4),
                })

    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


# ============ 因子衰减分析 ============

def calc_factor_decay(factor_name, factor_values_fn, price_data, stock_codes,
                      as_of_date, holding_periods=[5, 20, 60, 120]):
    """
    计算因子在不同持有期的IC（衰减分析）

    参数:
        holding_periods: 持有期列表（交易日数）

    返回:
        dict: {holding_period: ic_value}
    """
    from dataset_builder import get_snapshot

    try:
        snapshot = get_snapshot(stock_codes, as_of_date=as_of_date)
        if snapshot.empty:
            return {}

        factor_values = factor_values_fn(snapshot)
        if factor_values is None or factor_values.isna().all():
            return {}

        results = {}
        for period in holding_periods:
            end_date = (pd.Timestamp(as_of_date) + timedelta(days=int(period * 1.5))).strftime("%Y-%m-%d")
            forward_returns = pd.Series(index=snapshot.index, dtype=float)

            for idx, row in snapshot.iterrows():
                code = row["code"]
                if code not in price_data:
                    continue
                df = price_data[code]
                df["日期"] = pd.to_datetime(df["日期"])
                start_prices = df[df["日期"] >= pd.Timestamp(as_of_date)]
                end_prices = df[df["日期"] <= pd.Timestamp(end_date)]

                if not start_prices.empty and not end_prices.empty:
                    p_start = start_prices.iloc[0]["收盘"]
                    p_end = end_prices.iloc[-1]["收盘"]
                    if p_start > 0:
                        forward_returns.loc[idx] = (p_end - p_start) / p_start

            ic = calc_single_factor_ic(factor_values, forward_returns)
            results[period] = round(ic, 4) if not np.isnan(ic) else None

        return results

    except Exception as e:
        logger.warning(f"因子衰减分析失败: {e}")
        return {}


# ============ 综合因子研究报告 ============

def generate_research_report(ic_results=None, correlation_results=None,
                             quantile_results=None, decay_results=None):
    """生成因子研究综合报告"""
    report = []
    report.append("=" * 80)
    report.append("因子研究报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 80)

    # IC分析
    if ic_results:
        report.append("\n【一、IC分析】")
        report.append("-" * 80)
        header = f"  {'因子':<20} {'IC均值':<10} {'IC_IR':<10} {'IC>0比例':<10} {'期数':<8} {'有效':<6}"
        report.append(header)
        for r in ic_results:
            effective = "是" if r.get("is_effective") else "否"
            report.append(
                f"  {r['factor']:<20} {r['ic_mean']:<10} {r['ic_ir']:<10} "
                f"{r['ic_positive_rate']:<10} {r['n_periods']:<8} {effective:<6}"
            )

    # 分层回测
    if quantile_results:
        report.append("\n【二、分层回测】")
        report.append("-" * 80)
        for r in quantile_results:
            report.append(f"\n  {r['factor']}:")
            report.append(f"    多空收益(Q5-Q1): {r['long_short_return']:.4f}")
            report.append(f"    单调性: {'是' if r['monotonic'] else '否'}")
            if "cumulative_returns" in r:
                for q, ret in r["cumulative_returns"].items():
                    report.append(f"    Q{q}: {ret:.4f}")

    # 因子相关性
    if correlation_results:
        report.append("\n【三、因子相关性】")
        report.append("-" * 80)
        if "correlation_matrix" in correlation_results:
            report.append("  相关矩阵:")
            corr = correlation_results["correlation_matrix"]
            report.append("  " + " ".join(f"{c:>12}" for c in corr.columns))
            for idx, row in corr.iterrows():
                report.append(f"  {idx:<12}" + " ".join(f"{v:>12.3f}" for v in row))

        if "high_correlation_pairs" in correlation_results:
            pairs = correlation_results["high_correlation_pairs"]
            if pairs:
                report.append(f"\n  高度相关因子对 (|r| > 0.7):")
                for p in pairs:
                    report.append(f"    {p['factor_1']} <-> {p['factor_2']}: {p['correlation']:.3f}")

    # 因子衰减
    if decay_results:
        report.append("\n【四、因子衰减】")
        report.append("-" * 80)
        for factor_name, decay in decay_results.items():
            report.append(f"\n  {factor_name}:")
            for period, ic in decay.items():
                if ic is not None:
                    report.append(f"    {period}日: IC = {ic:.4f}")

    report.append("\n" + "=" * 80)
    return "\n".join(report)
