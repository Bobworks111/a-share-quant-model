"""
持仓分析模块
行业集中度、个股集中度、因子暴露归因
"""
import logging
import pandas as pd
import numpy as np

from config import FACTORS
from factor_engine import (
    calc_quality_factors, calc_earnings_quality_factors,
    calc_growth_factors, calc_safety_factors, calc_valuation_factors
)

logger = logging.getLogger("quant.portfolio")


# ============ 行业集中度 ============

def analyze_industry_concentration(snapshot_df, holdings):
    """
    分析持仓的行业集中度

    参数:
        snapshot_df: 数据快照（含行业信息）
        holdings: 持仓股票代码列表

    返回:
        dict: {
            "industry_weights": {行业: 权重},
            "hhi": float,  # HHI指数
            "top_industries": [(行业, 权重)],
        }
    """
    if "industry" not in snapshot_df.columns:
        logger.warning("数据中无行业分类信息")
        return {"industry_weights": {}, "hhi": 0, "top_industries": []}

    portfolio = snapshot_df[snapshot_df["code"].isin(holdings)]
    if portfolio.empty:
        return {"industry_weights": {}, "hhi": 0, "top_industries": []}

    # 等权
    n = len(portfolio)
    industry_weights = portfolio.groupby("industry").size() / n

    # HHI指数（赫芬达尔指数，越接近1越集中）
    hhi = (industry_weights ** 2).sum()

    # 按权重排序
    top = industry_weights.sort_values(ascending=False)

    return {
        "industry_weights": top.to_dict(),
        "hhi": round(hhi, 4),
        "top_industries": list(zip(top.index, top.values)),
    }


# ============ 个股集中度 ============

def analyze_stock_concentration(holdings, weights=None):
    """
    分析个股集中度

    参数:
        holdings: 持仓股票代码列表
        weights: 权重字典（默认等权）

    返回:
        dict: {
            "n_stocks": int,
            "top5_weight": float,
            "top3_weight": float,
            "max_weight": float,
            "equal_weight": float,
        }
    """
    n = len(holdings)
    if n == 0:
        return {}

    if weights is None:
        weights = {code: 1.0 / n for code in holdings}

    sorted_weights = sorted(weights.values(), reverse=True)

    return {
        "n_stocks": n,
        "top5_weight": round(sum(sorted_weights[:5]), 4),
        "top3_weight": round(sum(sorted_weights[:3]), 4),
        "max_weight": round(sorted_weights[0], 4) if sorted_weights else 0,
        "equal_weight": round(1.0 / n, 4),
    }


# ============ 因子暴露归因 ============

def analyze_factor_exposure(snapshot_df, holdings):
    """
    分析持仓的因子暴露

    参数:
        snapshot_df: 数据快照
        holdings: 持仓股票代码列表

    返回:
        dict: {
            "factor_exposures": {因子名: 平均暴露度},
            "factor_contributions": {因子名: 得分贡献},
        }
    """
    portfolio = snapshot_df[snapshot_df["code"].isin(holdings)]
    if portfolio.empty:
        return {}

    factor_calculators = {
        "quality": calc_quality_factors,
        "earnings_quality": calc_earnings_quality_factors,
        "growth": calc_growth_factors,
        "safety": calc_safety_factors,
        "valuation": calc_valuation_factors,
    }

    exposures = {}
    contributions = {}

    for name, calc_fn in factor_calculators.items():
        if name not in FACTORS:
            continue
        try:
            raw = calc_fn(portfolio)
            config = FACTORS[name]

            # 计算各指标的平均值
            factor_score = pd.Series(0.0, index=portfolio.index)
            available_weight = 0

            for ind_name, params in config["indicators"].items():
                if ind_name in raw.columns:
                    col = pd.to_numeric(raw[ind_name], errors="coerce")
                    if col.isna().all():
                        continue
                    mean_val = col.mean()
                    exposures[f"{name}.{ind_name}"] = round(mean_val, 4)
                    # 贡献 = 指标均值 * 权重
                    contributions[f"{name}.{ind_name}"] = round(mean_val * params["weight"], 4)
                    available_weight += params["weight"]

        except Exception as e:
            logger.debug(f"{name} 因子暴露计算失败: {e}")

    return {
        "factor_exposures": exposures,
        "factor_contributions": contributions,
    }


# ============ 收益归因 ============

def attribute_returns(snapshot_df, holdings, period_returns):
    """
    简单收益归因：按因子维度分解收益贡献

    参数:
        snapshot_df: 数据快照
        holdings: 持仓代码列表
        period_returns: {code: return} 收益字典

    返回:
        dict: {因子名: 收益贡献}
    """
    portfolio = snapshot_df[snapshot_df["code"].isin(holdings)].copy()
    if portfolio.empty or not period_returns:
        return {}

    # 将收益映射到股票
    portfolio["return"] = portfolio["code"].map(period_returns)
    portfolio = portfolio.dropna(subset=["return"])

    if portfolio.empty:
        return {}

    factor_calculators = {
        "quality": calc_quality_factors,
        "earnings_quality": calc_earnings_quality_factors,
        "growth": calc_growth_factors,
        "safety": calc_safety_factors,
        "valuation": calc_valuation_factors,
    }

    attribution = {}
    for name, calc_fn in factor_calculators.items():
        if name not in FACTORS:
            continue
        try:
            raw = calc_fn(portfolio)
            config = FACTORS[name]

            # 因子得分与收益的相关性 = 因子对收益的解释力
            factor_score = pd.Series(0.0, index=portfolio.index)
            available_weight = 0

            for ind_name, params in config["indicators"].items():
                if ind_name in raw.columns:
                    col = pd.to_numeric(raw[ind_name], errors="coerce")
                    if col.isna().all():
                        continue
                    from scipy import stats
                    corr, _ = stats.spearmanr(col.fillna(0), portfolio["return"])
                    factor_score += col.fillna(0) * params["weight"]
                    available_weight += params["weight"]

            if available_weight > 0:
                factor_score = factor_score / available_weight
                from scipy import stats
                corr, _ = stats.spearmanr(factor_score, portfolio["return"])
                attribution[name] = round(corr, 4) if not np.isnan(corr) else 0

        except Exception as e:
            logger.debug(f"{name} 收益归因失败: {e}")

    return attribution


# ============ 综合报告 ============

def generate_portfolio_report(snapshot_df, holdings, weights=None, period_returns=None):
    """生成持仓分析报告"""
    report = []
    report.append("=" * 80)
    report.append("持仓分析报告")
    report.append("=" * 80)

    # 1. 个股集中度
    conc = analyze_stock_concentration(holdings, weights)
    report.append("\n【个股集中度】")
    report.append(f"  持仓数: {conc.get('n_stocks', 0)}")
    report.append(f"  最大权重: {conc.get('max_weight', 0):.1%}")
    report.append(f"  前3大权重: {conc.get('top3_weight', 0):.1%}")
    report.append(f"  前5大权重: {conc.get('top5_weight', 0):.1%}")

    # 2. 行业集中度
    industry = analyze_industry_concentration(snapshot_df, holdings)
    report.append(f"\n【行业集中度】")
    report.append(f"  HHI指数: {industry['hhi']:.4f} (1=完全集中)")
    if industry["top_industries"]:
        report.append("  行业分布:")
        for ind, weight in industry["top_industries"][:10]:
            report.append(f"    {ind}: {weight:.1%}")

    # 3. 因子暴露
    exposure = analyze_factor_exposure(snapshot_df, holdings)
    if exposure.get("factor_contributions"):
        report.append(f"\n【因子暴露】")
        for factor_name in ["quality", "earnings_quality", "growth", "safety", "valuation"]:
            related = {k: v for k, v in exposure["factor_contributions"].items()
                       if k.startswith(factor_name)}
            if related:
                report.append(f"  {factor_name}:")
                for k, v in related.items():
                    ind = k.split(".")[1]
                    report.append(f"    {ind}: {v:.4f}")

    # 4. 收益归因
    if period_returns:
        attribution = attribute_returns(snapshot_df, holdings, period_returns)
        if attribution:
            report.append(f"\n【收益归因（因子与收益的Spearman相关）】")
            for factor, corr in sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True):
                direction = "正向" if corr > 0 else "负向"
                report.append(f"  {factor}: {corr:.4f} ({direction})")

    report.append("\n" + "=" * 80)
    return "\n".join(report)
