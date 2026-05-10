"""
主入口
消费股价值投资量化模型
学术依据：Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, logger
from config import CONSUMER_STOCKS, STOCK_POOL, OUTPUT_DIR, BACKTEST, FACTORS
from data_fetcher import get_stock_pool, get_price_data
from dataset_builder import get_snapshot
from strategy import generate_signal, generate_report
from backtester import run_backtest, run_rolling_backtest
from fraud_detector import check_pool, generate_fraud_report
from factor_research import (
    calc_factor_ic_series, analyze_factor_ic,
    run_quantile_analysis, calc_factor_correlation, find_correlated_pairs,
    calc_factor_decay, generate_research_report
)
from factor_engine import (
    calc_quality_factors, calc_earnings_quality_factors,
    calc_growth_factors, calc_safety_factors, calc_valuation_factors
)


def get_stock_codes():
    """获取股票池"""
    if STOCK_POOL == "consumer" and CONSUMER_STOCKS:
        return CONSUMER_STOCKS
    pool_df = get_stock_pool(STOCK_POOL)
    return pool_df["code"].tolist()


def run_screener(stock_codes, top_n=10):
    """运行筛选模式：获取当前数据，输出推荐股票"""
    logger.info("获取当前数据快照...")
    snapshot = get_snapshot(stock_codes, as_of_date=datetime.now().strftime("%Y-%m-%d"))

    if snapshot.empty:
        logger.error("无有效数据")
        return

    logger.info(f"获取到 {len(snapshot)} 只股票的有效数据")

    signal = generate_signal(snapshot, top_n=top_n)
    if not signal["stocks"]:
        logger.error("无符合条件的股票")
        return

    # 生成报告
    report = generate_report(signal["scored_df"])
    logger.info("\n" + report)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    report_file = os.path.join(OUTPUT_DIR, f"report_{date_str}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"报告已保存: {report_file}")

    excel_file = os.path.join(OUTPUT_DIR, f"scored_{date_str}.xlsx")
    signal["scored_df"].to_excel(excel_file, index=False)
    logger.info(f"数据已保存: {excel_file}")


def run_backtest_mode(stock_codes, top_n=10, rolling=False):
    """运行回测模式"""
    if rolling:
        results = run_rolling_backtest(stock_codes, top_n=top_n)
    else:
        results = run_backtest(stock_codes, top_n=top_n)
    return results


def run_fraud_check(stock_codes):
    """运行造假风险检测"""
    logger.info("开始造假风险检测...")

    # 获取股票名称
    stock_names = {}
    try:
        quotes_df = get_snapshot(stock_codes, as_of_date=datetime.now().strftime("%Y-%m-%d"))
        if not quotes_df.empty and "name" in quotes_df.columns:
            stock_names = dict(zip(quotes_df["code"], quotes_df["name"]))
    except Exception:
        pass

    results = check_pool(stock_codes, stock_names=stock_names)
    report = generate_fraud_report(results)
    logger.info("\n" + report)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    report_file = os.path.join(OUTPUT_DIR, f"fraud_check_{date_str}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"报告已保存: {report_file}")

    excel_file = os.path.join(OUTPUT_DIR, f"fraud_check_{date_str}.xlsx")
    results.to_excel(excel_file, index=False)
    logger.info(f"数据已保存: {excel_file}")


def run_factor_research(stock_codes):
    """运行因子研究：IC分析、分层回测、相关性、衰减分析"""
    logger.info("开始因子研究...")

    start_date = BACKTEST["start_date"]
    end_date = BACKTEST["end_date"]

    # 预加载行情数据
    logger.info("预加载行情数据...")
    price_data = {}
    for code in stock_codes:
        try:
            df = get_price_data(code, start_date=start_date.replace("-", ""))
            if df is not None and not df.empty:
                price_data[code] = df
        except Exception:
            pass
    logger.info(f"行情加载完成: {len(price_data)}/{len(stock_codes)} 只")

    # 因子计算函数映射
    factor_fns = {
        "quality": lambda df: calc_quality_factors(df).mean(axis=1),
        "earnings_quality": lambda df: calc_earnings_quality_factors(df).mean(axis=1),
        "growth": lambda df: calc_growth_factors(df).mean(axis=1),
        "safety": lambda df: calc_safety_factors(df).mean(axis=1),
        "valuation": lambda df: calc_valuation_factors(df).mean(axis=1),
    }

    # 1. IC分析
    logger.info("\n===== IC分析 =====")
    ic_results = []
    for factor_name, fn in factor_fns.items():
        logger.info(f"计算 {factor_name} IC...")
        ic_df = calc_factor_ic_series(factor_name, fn, price_data, stock_codes,
                                      start_date, end_date)
        result = analyze_factor_ic(ic_df, factor_name)
        ic_results.append(result)
        logger.info(f"  {factor_name}: IC均值={result['ic_mean']}, IC_IR={result['ic_ir']}, "
                     f"有效={'是' if result['is_effective'] else '否'}")

    # 2. 分层回测
    logger.info("\n===== 分层回测 =====")
    quantile_results = []
    for factor_name, fn in factor_fns.items():
        logger.info(f"运行 {factor_name} 分层回测...")
        qr = run_quantile_analysis(factor_name, fn, price_data, stock_codes,
                                   start_date, end_date)
        quantile_results.append(qr)
        logger.info(f"  {factor_name}: 多空收益={qr['long_short_return']:.4f}, "
                     f"单调性={'是' if qr['monotonic'] else '否'}")

    # 3. 因子相关性
    logger.info("\n===== 因子相关性 =====")
    snapshot = get_snapshot(stock_codes, as_of_date=end_date)
    corr_matrix = calc_factor_correlation(snapshot, FACTORS)
    high_corr = find_correlated_pairs(corr_matrix, threshold=0.7)
    correlation_results = {
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_corr,
    }
    if high_corr:
        for p in high_corr:
            logger.info(f"  高相关: {p['factor_1']} <-> {p['factor_2']}: {p['correlation']:.3f}")

    # 4. 因子衰减（用最新日期）
    logger.info("\n===== 因子衰减 =====")
    decay_results = {}
    for factor_name, fn in factor_fns.items():
        decay = calc_factor_decay(factor_name, fn, price_data, stock_codes,
                                  as_of_date=end_date)
        decay_results[factor_name] = decay
        if decay:
            logger.info(f"  {factor_name}: {decay}")

    # 生成报告
    report = generate_research_report(
        ic_results=ic_results,
        correlation_results=correlation_results,
        quantile_results=quantile_results,
        decay_results=decay_results
    )
    logger.info("\n" + report)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    report_file = os.path.join(OUTPUT_DIR, f"factor_research_{date_str}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"报告已保存: {report_file}")


def main():
    """主函数"""
    setup_logging()

    logger.info("=" * 70)
    logger.info("A股消费股价值投资量化模型")
    logger.info("投资理念: 巴菲特(质量) + 林奇(成长) + 马克斯(风险)")
    logger.info("学术依据: Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)")
    logger.info("=" * 70)

    stock_codes = get_stock_codes()
    logger.info(f"股票池: {len(stock_codes)} 只")

    # 模式选择
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "screener"

    if mode == "backtest":
        run_backtest_mode(stock_codes)
    elif mode == "rolling_backtest":
        run_backtest_mode(stock_codes, rolling=True)
    elif mode == "fraud_check":
        run_fraud_check(stock_codes)
    elif mode == "factor_research":
        run_factor_research(stock_codes)
    else:
        run_screener(stock_codes)


if __name__ == "__main__":
    main()
