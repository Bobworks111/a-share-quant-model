"""
策略层
接收数据快照，输出持仓信号
解耦：不依赖回测层，回测层不依赖策略内部实现
"""
import logging
import pandas as pd
import numpy as np

from factor_engine import calculate_all_factors
from config import FACTORS, HARD_CONSTRAINTS

logger = logging.getLogger("quant.strategy")


def apply_hard_constraints(df, constraints=None):
    """
    硬约束筛选（第一阶段）
    排除不符合基本条件的股票
    """
    if constraints is None:
        constraints = HARD_CONSTRAINTS

    mask = pd.Series(True, index=df.index)
    n_total = len(df)

    # 连续正利润
    if "consecutive_profit_years" in df.columns:
        years = pd.to_numeric(df["consecutive_profit_years"], errors="coerce")
        mask &= years >= constraints["consecutive_profit_years"]

    # 资产负债率
    if "debt_ratio" in df.columns:
        debt = pd.to_numeric(df["debt_ratio"], errors="coerce")
        mask &= debt <= constraints["max_debt_ratio"]

    # 连续分红
    if "consecutive_dividend_years" in df.columns:
        div_years = pd.to_numeric(df["consecutive_dividend_years"], errors="coerce")
        mask &= div_years >= constraints["consecutive_dividend_years"]

    # PE > 0 且 PB > 0
    if "pe" in df.columns:
        pe = pd.to_numeric(df["pe"], errors="coerce")
        mask &= pe > 0
    if "pb" in df.columns:
        pb = pd.to_numeric(df["pb"], errors="coerce")
        mask &= pb > 0

    filtered = df[mask].copy()
    logger.info(f"硬约束筛选: {len(filtered)}/{n_total} 只通过")
    return filtered


def score_stocks(df, factors_config=None):
    """
    对筛选后的股票进行多因子打分
    返回带得分的DataFrame
    """
    if factors_config is None:
        factors_config = FACTORS
    return calculate_all_factors(df, factors_config)


def generate_signal(snapshot_df, top_n=10, factors_config=None, constraints=None):
    """
    策略核心接口：接收数据快照，输出持仓信号

    参数:
        snapshot_df: dataset_builder.get_snapshot() 返回的数据
        top_n: 持仓股票数
        factors_config: 因子配置（默认用 config.FACTORS）
        constraints: 硬约束配置（默认用 config.HARD_CONSTRAINTS）

    返回:
        dict: {
            "stocks": [code1, code2, ...],  # 持仓股票代码
            "weights": {code: weight, ...},  # 等权权重
            "scored_df": DataFrame,           # 完整打分结果
        }
    """
    if snapshot_df.empty:
        logger.warning("数据快照为空，无法生成信号")
        return {"stocks": [], "weights": {}, "scored_df": pd.DataFrame()}

    # 第一阶段：硬约束筛选
    filtered = apply_hard_constraints(snapshot_df, constraints)
    if filtered.empty:
        logger.warning("无股票通过硬约束筛选")
        return {"stocks": [], "weights": {}, "scored_df": pd.DataFrame()}

    # 第二阶段：多因子打分
    scored = score_stocks(filtered, factors_config)
    top = scored.head(top_n)

    stocks = top["code"].tolist()
    weights = {code: 1.0 / len(stocks) for code in stocks}

    logger.info(f"信号生成: {len(stocks)}只股票")
    for _, row in top.iterrows():
        logger.info(f"  {row.get('name', row['code'])} "
                    f"| 总分: {row['total_score']:.2f} "
                    f"| 质量: {row.get('score_quality', 0):.2f} "
                    f"| 成长: {row.get('score_growth', 0):.2f} "
                    f"| 安全: {row.get('score_safety', 0):.2f} "
                    f"| 估值: {row.get('score_valuation', 0):.2f}")

    return {
        "stocks": stocks,
        "weights": weights,
        "scored_df": scored,
    }


def generate_report(scored_df, constraints=None, factors_config=None):
    """生成筛选报告"""
    if constraints is None:
        constraints = HARD_CONSTRAINTS
    if factors_config is None:
        factors_config = FACTORS

    report = []
    report.append("=" * 80)
    report.append("A股消费股价值投资量化筛选报告")
    report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("投资理念: 巴菲特(质量) + 林奇(成长) + 马克斯(风险)")
    report.append("学术依据: Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)")
    report.append("=" * 80)
    report.append("")

    # 硬约束
    report.append("硬约束条件:")
    for k, v in constraints.items():
        report.append(f"  {k}: {v}")
    report.append("")

    # 因子权重
    report.append("五因子体系:")
    for k, v in factors_config.items():
        report.append(f"  {k}: {v['weight']*100:.0f}%")
    report.append("")

    # 股票列表
    report.append(f"入选股票 ({len(scored_df)} 只):")
    report.append("-" * 80)
    header = f"{'排名':<4} {'代码':<8} {'名称':<10} {'总分':<8} {'质量':<8} {'盈利':<8} {'成长':<8} {'安全':<8} {'估值':<8}"
    report.append(header)
    report.append("-" * 80)

    for _, row in scored_df.iterrows():
        line = (
            f"{row.get('rank', '-'):<4} "
            f"{row['code']:<8} "
            f"{row.get('name', '-'):<10} "
            f"{row['total_score']:<8.2f} "
            f"{row.get('score_quality', 0):<8.2f} "
            f"{row.get('score_earnings_quality', 0):<8.2f} "
            f"{row.get('score_growth', 0):<8.2f} "
            f"{row.get('score_safety', 0):<8.2f} "
            f"{row.get('score_valuation', 0):<8.2f}"
        )
        report.append(line)

    return "\n".join(report)
