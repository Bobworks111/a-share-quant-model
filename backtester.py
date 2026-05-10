"""
回测引擎
消除前瞻偏差：每个调仓日只使用该日之前的数据
解耦：通过 strategy.generate_signal 接口获取持仓信号
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from config import BACKTEST
from strategy import generate_signal
from dataset_builder import get_snapshot
from data_fetcher import get_price_data

logger = logging.getLogger("quant.backtest")


def get_rebalance_dates(start_date, end_date, freq="quarterly"):
    """生成调仓日期列表"""
    dates = pd.date_range(start=start_date, end=end_date, freq="QE")
    return dates


def calc_period_return(price_data, stock_codes, start_date, end_date):
    """计算持仓期间收益率"""
    returns = {}
    for code in stock_codes:
        if code not in price_data:
            continue
        df = price_data[code]
        if df is None or df.empty:
            continue

        df["日期"] = pd.to_datetime(df["日期"])
        start_rows = df[df["日期"] >= pd.Timestamp(start_date)]
        end_rows = df[df["日期"] <= pd.Timestamp(end_date)]

        if start_rows.empty or end_rows.empty:
            continue

        p_start = start_rows.iloc[0]["收盘"]
        p_end = end_rows.iloc[-1]["收盘"]

        if p_start > 0:
            returns[code] = (p_end - p_start) / p_start

    return returns


def calc_turnover(old_stocks, new_stocks):
    """计算换手率"""
    old_set = set(old_stocks)
    new_set = set(new_stocks)
    total = max(len(old_set), len(new_set))
    if total == 0:
        return 0.0
    added = len(new_set - old_set)
    removed = len(old_set - new_set)
    return (added + removed) / (2 * total)


def calc_transaction_cost(turnover):
    """
    计算交易成本（只对换仓部分收取）
    佣金 + 印花税(卖出) + 过户费 + 滑点
    """
    commission = BACKTEST.get("commission", 0.001)
    stamp_tax = 0.0005      # A股印花税（卖出）
    transfer_fee = 0.00001  # 过户费
    slippage = BACKTEST.get("slippage", 0.001)

    sell_cost = commission + stamp_tax + transfer_fee + slippage
    buy_cost = commission + transfer_fee + slippage
    return turnover * (sell_cost + buy_cost)


def run_backtest(stock_codes, start_date=None, end_date=None, top_n=None):
    """
    运行回测
    消除前瞻偏差：每个调仓日调用 get_snapshot 获取该时刻的数据
    """
    if start_date is None:
        start_date = BACKTEST["start_date"]
    if end_date is None:
        end_date = BACKTEST["end_date"]
    if top_n is None:
        top_n = BACKTEST.get("top_n", 10)

    initial_cash = BACKTEST["initial_cash"]

    # 获取调仓日期
    rebalance_dates = get_rebalance_dates(start_date, end_date, BACKTEST["rebalance_freq"])
    logger.info(f"回测区间: {start_date} ~ {end_date}")
    logger.info(f"调仓次数: {len(rebalance_dates)}")
    logger.info(f"初始资金: {initial_cash:,.0f}")
    logger.info("-" * 60)

    # 预加载行情数据
    logger.info("预加载行情数据...")
    price_data = {}
    for code in stock_codes:
        try:
            df = get_price_data(code, start_date=start_date.replace("-", ""))
            if df is not None and not df.empty:
                price_data[code] = df
        except Exception as e:
            logger.debug(f"{code} 行情加载失败: {e}")

    logger.info(f"行情加载完成: {len(price_data)}/{len(stock_codes)} 只")

    # 回测结果记录
    portfolio_values = [initial_cash]
    portfolio_dates = [pd.Timestamp(start_date)]
    period_returns = []
    turnover_list = []
    old_stocks = []

    for i in range(len(rebalance_dates)):
        rebal_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else pd.Timestamp(end_date)

        rebal_str = rebal_date.strftime("%Y-%m-%d")
        logger.info(f"调仓日: {rebal_str}")

        # 1. 获取截至调仓日的数据快照（消除前瞻偏差的关键）
        try:
            snapshot = get_snapshot(stock_codes, as_of_date=rebal_str)
            if snapshot.empty:
                logger.warning(f"  {rebal_str}: 无有效数据，跳过")
                continue
        except Exception as e:
            logger.error(f"  {rebal_str}: 数据快照构建失败: {e}")
            continue

        # 2. 通过策略层获取持仓信号
        signal = generate_signal(snapshot, top_n=top_n)
        new_stocks = signal["stocks"]

        if not new_stocks:
            logger.warning(f"  {rebal_str}: 无持仓信号，跳过")
            continue

        # 3. 计算换手率和交易成本
        turnover = calc_turnover(old_stocks, new_stocks)
        tx_cost = calc_transaction_cost(turnover)
        turnover_list.append(turnover)

        # 4. 计算持仓期收益
        returns = calc_period_return(price_data, new_stocks, rebal_date, next_date)

        if not returns:
            logger.warning(f"  {rebal_str}: 无有效收益数据，跳过")
            continue

        # 5. 等权持仓收益 - 扣除交易成本
        avg_return = np.mean(list(returns.values()))
        net_return = avg_return - tx_cost

        # 6. 更新组合价值
        current_value = portfolio_values[-1] * (1 + net_return)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)
        period_returns.append(net_return)

        # 7. 输出
        n_positive = sum(1 for r in returns.values() if r > 0)
        logger.info(f"  持仓{len(returns)}只, 正收益{n_positive}只, "
                    f"换手{turnover:.1%}, 成本{tx_cost:.4f}, "
                    f"组合收益{net_return*100:+.2f}%, 累计{current_value:,.0f}")

        old_stocks = new_stocks

    # 计算绩效
    if len(portfolio_values) < 2:
        logger.error("回测无有效数据")
        return None

    results = calc_performance(portfolio_values, portfolio_dates, period_returns,
                               turnover_list, initial_cash)
    return results


def calc_performance(values, dates, period_returns, turnover_list, initial_cash):
    """计算回测绩效指标"""
    values = np.array(values)
    total_return = (values[-1] - initial_cash) / initial_cash

    # 年化收益率
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)

    # 最大回撤持续时间
    dd_start = 0
    max_dd_duration = 0
    for i in range(1, len(values)):
        if values[i] < peak[i]:
            if dd_start == 0:
                dd_start = i
        else:
            if dd_start > 0:
                max_dd_duration = max(max_dd_duration, i - dd_start)
                dd_start = 0

    # 夏普比率
    sharpe = 0
    sortino = 0
    if period_returns:
        returns_arr = np.array(period_returns)
        rf = 0.025 / 4
        excess = returns_arr - rf
        std = np.std(excess)
        if std > 0:
            sharpe = np.mean(excess) / std * np.sqrt(4)

        downside = excess[excess < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0.001
        sortino = np.mean(excess) / downside_std * np.sqrt(4)

    # Calmar 比率
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 胜率
    win_rate = sum(1 for r in period_returns if r > 0) / len(period_returns) if period_returns else 0

    # 平均换手率
    avg_turnover = np.mean(turnover_list) if turnover_list else 0

    # 输出
    logger.info("\n" + "=" * 60)
    logger.info("回测绩效报告")
    logger.info("=" * 60)
    logger.info(f"回测区间: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"初始资金: {initial_cash:,.0f}")
    logger.info(f"最终资金: {values[-1]:,.0f}")
    logger.info(f"总收益率: {total_return*100:+.2f}%")
    logger.info(f"年化收益率: {annual_return*100:+.2f}%")
    logger.info(f"最大回撤: {max_drawdown*100:.2f}%")
    logger.info(f"最大回撤持续: {max_dd_duration}期")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"Sortino比率: {sortino:.2f}")
    logger.info(f"Calmar比率: {calmar:.2f}")
    logger.info(f"胜率: {win_rate*100:.1f}%")
    logger.info(f"平均换手率: {avg_turnover:.1%}")
    logger.info("=" * 60)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "max_dd_duration": max_dd_duration,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_turnover": avg_turnover,
        "values": values,
        "dates": dates,
    }
