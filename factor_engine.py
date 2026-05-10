"""
因子计算引擎
消费股价值投资量化模型
学术依据：Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)
"""
import pandas as pd
import numpy as np


# ============ 去极值 ============

def winsorize_mad(series, n=5):
    """MAD法去极值：中位数 ± n倍MAD"""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + n * 1.4826 * mad
    lower = median - n * 1.4826 * mad
    return series.clip(lower, upper)


# ============ 标准化 ============

def zscore(series):
    """z-score标准化"""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


# ============ 因子预处理 ============

def preprocess_indicator(series):
    """单个指标预处理：去极值 → z-score标准化"""
    s = winsorize_mad(series)
    s = s.fillna(s.median())
    return zscore(s)


# ============ 因子得分计算 ============

def calc_factor_score(df, factor_name, factor_config):
    """
    计算单个因子的综合得分
    df: 包含所有指标列的DataFrame
    factor_name: 因子名称
    factor_config: 因子配置（含indicators和weight）
    """
    indicators = factor_config["indicators"]
    scores = pd.DataFrame(index=df.index)

    for ind_name, params in indicators.items():
        if ind_name in df.columns:
            col = pd.to_numeric(df[ind_name], errors="coerce")
            processed = preprocess_indicator(col)
            # ascending=True表示越小越好，取负号使方向统一（越大越好）
            if params["ascending"]:
                processed = -processed
            scores[ind_name] = processed * params["weight"]

    return scores.sum(axis=1)


# ============ 公司质量因子（QMJ盈利能力）============

def calc_quality_factors(df):
    """
    计算公司质量因子
    学术依据：QMJ盈利能力维度（ROE、毛利率、现金流/资产）
    """
    result = pd.DataFrame(index=df.index)

    # ROE均值和变异系数
    if "roe_5y_mean" in df.columns:
        result["roe_mean"] = pd.to_numeric(df["roe_5y_mean"], errors="coerce")
    if "roe_5y_std" in df.columns and "roe_5y_mean" in df.columns:
        roe_mean = pd.to_numeric(df["roe_5y_mean"], errors="coerce")
        roe_std = pd.to_numeric(df["roe_5y_std"], errors="coerce")
        result["roe_cv"] = roe_std / roe_mean.replace(0, np.nan)

    # 毛利率均值和变异系数
    if "gross_margin_5y_mean" in df.columns:
        result["gross_margin_mean"] = pd.to_numeric(df["gross_margin_5y_mean"], errors="coerce")
    if "gross_margin_5y_std" in df.columns and "gross_margin_5y_mean" in df.columns:
        gm_mean = pd.to_numeric(df["gross_margin_5y_mean"], errors="coerce")
        gm_std = pd.to_numeric(df["gross_margin_5y_std"], errors="coerce")
        result["gross_margin_cv"] = gm_std / gm_mean.replace(0, np.nan)

    # 自由现金流/净利润
    if "fcf_to_profit" in df.columns:
        result["fcf_to_profit"] = pd.to_numeric(df["fcf_to_profit"], errors="coerce")

    # 自由现金流/总资产（QMJ核心指标）
    if "fcf_to_assets" in df.columns:
        result["fcf_to_assets"] = pd.to_numeric(df["fcf_to_assets"], errors="coerce")

    # 资本开支/营收
    if "capex_to_revenue" in df.columns:
        result["capex_to_revenue"] = pd.to_numeric(df["capex_to_revenue"], errors="coerce")

    return result


# ============ 盈利质量因子（QMJ）============

def calc_earnings_quality_factors(df):
    """
    计算盈利质量因子
    学术依据：QMJ盈利质量维度（应计比率、派息、现金流连续性）
    """
    result = pd.DataFrame(index=df.index)

    # 应计比率 = (净利润 - 经营现金流) / 总资产
    # 越低说明盈利质量越高（利润更多是现金而非应计）
    if "accruals_ratio" in df.columns:
        result["accruals_ratio"] = pd.to_numeric(df["accruals_ratio"], errors="coerce")

    # 派息比例
    if "dividend_payout" in df.columns:
        result["dividend_payout"] = pd.to_numeric(df["dividend_payout"], errors="coerce")

    # 现金流连续性
    if "cashflow_continuity" in df.columns:
        result["cashflow_continuity"] = pd.to_numeric(df["cashflow_continuity"], errors="coerce")

    return result


# ============ 成长性因子（QMJ成长 + Lynch PEG）============

def calc_growth_factors(df):
    """
    计算成长性因子
    学术依据：QMJ成长维度 + Lynch PEG策略
    """
    result = pd.DataFrame(index=df.index)

    # 净利润3年复合增速
    if "profit_growth_3y" in df.columns:
        result["profit_growth_3y"] = pd.to_numeric(df["profit_growth_3y"], errors="coerce")

    # 营收3年复合增速
    if "revenue_growth_3y" in df.columns:
        result["revenue_growth_3y"] = pd.to_numeric(df["revenue_growth_3y"], errors="coerce")

    # ROE增长率（QMJ成长指标）
    if "roe_growth_3y" in df.columns:
        result["roe_growth_3y"] = pd.to_numeric(df["roe_growth_3y"], errors="coerce")

    # 调整PEG = PEG × (1 + 增速稳定性)
    # 论文发现：单纯PEG可能选到价值陷阱，需结合增长质量
    if "pe" in df.columns and "profit_growth_3y" in df.columns:
        pe = pd.to_numeric(df["pe"], errors="coerce")
        growth = pd.to_numeric(df["profit_growth_3y"], errors="coerce")
        stability = pd.to_numeric(df.get("growth_stability", pd.Series(0, index=df.index)), errors="coerce")

        # PEG只在增速>0时有意义
        peg = pe / growth.replace(0, np.nan)
        peg = peg.where(growth > 0, np.nan)

        # 调整PEG：增长越稳定，调整后PEG越低（越好）
        peg_adjusted = peg * (1 + stability.clip(0, 2))  # 限制稳定性影响
        result["peg_adjusted"] = peg_adjusted

    # 增速稳定性
    if "growth_stability" in df.columns:
        result["growth_stability"] = pd.to_numeric(df["growth_stability"], errors="coerce")

    # 利润增速/营收增速（经营杠杆）
    if "profit_revenue_ratio" in df.columns:
        result["profit_revenue_ratio"] = pd.to_numeric(df["profit_revenue_ratio"], errors="coerce")

    return result


# ============ 安全性因子（QMJ安全性）============

def calc_safety_factors(df):
    """
    计算安全性因子
    学术依据：QMJ安全性维度（低Beta、低波动、低杠杆）
    """
    result = pd.DataFrame(index=df.index)

    # Beta（市场敏感度）
    if "beta" in df.columns:
        result["beta"] = pd.to_numeric(df["beta"], errors="coerce")

    # 特异性波动率（残差波动）
    if "idio_volatility" in df.columns:
        result["idio_volatility"] = pd.to_numeric(df["idio_volatility"], errors="coerce")

    # 资产负债率
    if "debt_ratio" in df.columns:
        result["debt_ratio"] = pd.to_numeric(df["debt_ratio"], errors="coerce")

    # 利息覆盖倍数
    if "interest_coverage" in df.columns:
        result["interest_coverage"] = pd.to_numeric(df["interest_coverage"], errors="coerce")

    # 流动比率
    if "current_ratio" in df.columns:
        result["current_ratio"] = pd.to_numeric(df["current_ratio"], errors="coerce")

    return result


# ============ 估值合理性因子 ============

def calc_valuation_factors(df):
    """
    计算估值合理性因子
    学术依据：Buffett合理价格 + Lynch PEG
    """
    result = pd.DataFrame(index=df.index)

    if "pe_percentile" in df.columns:
        result["pe_percentile"] = pd.to_numeric(df["pe_percentile"], errors="coerce")

    if "dividend_yield" in df.columns:
        result["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce")

    # PEG复用成长因子的PEG
    if "peg" in df.columns:
        result["peg_valuation"] = pd.to_numeric(df["peg"], errors="coerce")
    elif "pe" in df.columns and "profit_growth_3y" in df.columns:
        pe = pd.to_numeric(df["pe"], errors="coerce")
        growth = pd.to_numeric(df["profit_growth_3y"], errors="coerce")
        peg = pe / growth.replace(0, np.nan)
        peg = peg.where(growth > 0, np.nan)
        result["peg_valuation"] = peg

    return result


# ============ 综合打分 ============

def calculate_all_factors(df, factors_config):
    """
    计算所有因子并合成综合得分
    df: 包含原始指标的DataFrame
    factors_config: config.FACTORS
    返回: 带有各因子得分和总分的DataFrame
    """
    result = df.copy()

    # 计算各因子的原始指标
    quality_df = calc_quality_factors(df)
    earnings_df = calc_earnings_quality_factors(df)
    growth_df = calc_growth_factors(df)
    safety_df = calc_safety_factors(df)
    valuation_df = calc_valuation_factors(df)

    # 计算各因子得分
    total_score = pd.Series(0.0, index=df.index)

    # 质量因子
    if "quality" in factors_config:
        score = calc_factor_score(quality_df, "quality", factors_config["quality"])
        result["score_quality"] = score
        total_score += score * factors_config["quality"]["weight"]

    # 盈利质量因子
    if "earnings_quality" in factors_config:
        score = calc_factor_score(earnings_df, "earnings_quality", factors_config["earnings_quality"])
        result["score_earnings_quality"] = score
        total_score += score * factors_config["earnings_quality"]["weight"]

    # 成长因子
    if "growth" in factors_config:
        score = calc_factor_score(growth_df, "growth", factors_config["growth"])
        result["score_growth"] = score
        total_score += score * factors_config["growth"]["weight"]

    # 安全性因子
    if "safety" in factors_config:
        score = calc_factor_score(safety_df, "safety", factors_config["safety"])
        result["score_safety"] = score
        total_score += score * factors_config["safety"]["weight"]

    # 估值因子
    if "valuation" in factors_config:
        score = calc_factor_score(valuation_df, "valuation", factors_config["valuation"])
        result["score_valuation"] = score
        total_score += score * factors_config["valuation"]["weight"]

    result["total_score"] = total_score
    result = result.sort_values("total_score", ascending=False)
    result["rank"] = range(1, len(result) + 1)

    return result
