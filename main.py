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
from config import CONSUMER_STOCKS, STOCK_POOL, OUTPUT_DIR, BACKTEST
from data_fetcher import get_stock_pool
from dataset_builder import get_snapshot
from strategy import generate_signal, generate_report
from backtester import run_backtest


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


def run_backtest_mode(stock_codes, top_n=10):
    """运行回测模式"""
    results = run_backtest(stock_codes, top_n=top_n)
    return results


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
    else:
        run_screener(stock_codes)


if __name__ == "__main__":
    main()
