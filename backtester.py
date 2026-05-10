"""
回测引擎
消费股价值投资量化模型
"""
import pandas as pd
import numpy as np
from datetime import datetime
from config import BACKTEST, FACTORS
from factor_model import apply_hard_constraints, score_stocks


def get_rebalance_dates(start_date, end_date, freq="quarterly"):
    """生成调仓日期列表"""
    dates = pd.date_range(start=start_date, end=end_date, freq="QE")
    return dates


def calc_period_return(price_data, stock_codes, start_date, end_date):
    """
    计算持仓期间收益率
    price_data: {stock_code: DataFrame with '日期' and '收盘'}
    """
    returns = {}
    for code in stock_codes:
        if code not in price_data:
            continue
        df = price_data[code]
        if df is None or df.empty:
            continue

        # 找到start_date和end_date对应的收盘价
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


def calc_benchmark_return(benchmark_data, start_date, end_date):
    """计算基准收益率"""
    if benchmark_data is None or benchmark_data.empty:
        return 0.0

    df = benchmark_data.copy()
    df["日期"] = pd.to_datetime(df["日期"])

    start_rows = df[df["日期"] >= pd.Timestamp(start_date)]
    end_rows = df[df["日期"] <= pd.Timestamp(end_date)]

    if start_rows.empty or end_rows.empty:
        return 0.0

    p_start = start_rows.iloc[0]["收盘"]
    p_end = end_rows.iloc[-1]["收盘"]

    return (p_end - p_start) / p_start if p_start > 0 else 0.0


def run_backtest(stock_data, price_data, benchmark_data=None):
    """
    运行回测
    stock_data: DataFrame，包含所有股票的财务指标
    price_data: {stock_code: DataFrame} 行情数据
    benchmark_data: DataFrame 基准行情
    """
    start_date = BACKTEST["start_date"]
    end_date = BACKTEST["end_date"]
    top_n = BACKTEST["top_n"]
    commission = BACKTEST["commission"]
    slippage = BACKTEST["slippage"]
    initial_cash = BACKTEST["initial_cash"]

    # 获取调仓日期
    rebalance_dates = get_rebalance_dates(start_date, end_date, BACKTEST["rebalance_freq"])
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"调仓次数: {len(rebalance_dates)}")
    print(f"初始资金: {initial_cash:,.0f}")
    print("-" * 60)

    # 回测结果记录
    portfolio_values = [initial_cash]
    portfolio_dates = [pd.Timestamp(start_date)]
    holdings = {}
    period_returns = []
    benchmark_returns = []

    for i in range(len(rebalance_dates)):
        rebal_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else pd.Timestamp(end_date)

        # 1. 筛选+打分
        filtered = apply_hard_constraints(stock_data)
        if len(filtered) == 0:
            print(f"  {rebal_date.strftime('%Y-%m-%d')}: 无符合条件的股票，跳过")
            continue

        scored = score_stocks(filtered)
        top_stocks = scored.head(top_n)
        selected_codes = top_stocks["code"].tolist()

        # 2. 计算持仓期收益
        returns = calc_period_return(price_data, selected_codes, rebal_date, next_date)

        if not returns:
            print(f"  {rebal_date.strftime('%Y-%m-%d')}: 无有效收益数据，跳过")
            continue

        # 3. 等权持仓收益
        avg_return = np.mean(list(returns.values()))

        # 4. 扣除交易成本（双边）
        cost = (commission + slippage) * 2
        net_return = avg_return - cost

        # 5. 更新组合价值
        current_value = portfolio_values[-1] * (1 + net_return)
        portfolio_values.append(current_value)
        portfolio_dates.append(next_date)

        # 6. 记录
        period_returns.append(net_return)

        # 7. 基准收益
        if benchmark_data is not None:
            bench_ret = calc_benchmark_return(benchmark_data, rebal_date, next_date)
            benchmark_returns.append(bench_ret)
        else:
            benchmark_returns.append(0)

        # 输出
        n_positive = sum(1 for r in returns.values() if r > 0)
        print(f"  {rebal_date.strftime('%Y-%m-%d')}: "
              f"持仓{len(returns)}只, "
              f"正收益{n_positive}只, "
              f"组合收益{net_return*100:+.2f}%, "
              f"累计{current_value:,.0f}")

        # 更新持仓
        holdings = {code: 1.0 / len(selected_codes) for code in selected_codes}

    # 计算绩效指标
    results = calc_performance(portfolio_values, portfolio_dates, period_returns, benchmark_returns, initial_cash)
    return results


def calc_performance(values, dates, period_returns, benchmark_returns, initial_cash):
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

    # 夏普比率（假设无风险利率2.5%）
    if period_returns:
        returns_arr = np.array(period_returns)
        rf = 0.025 / 4  # 季度无风险利率
        excess = returns_arr - rf
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(4) if np.std(excess) > 0 else 0
    else:
        sharpe = 0

    # 胜率
    win_rate = sum(1 for r in period_returns if r > 0) / len(period_returns) if period_returns else 0

    # 超额收益
    if benchmark_returns:
        bench_total = 1
        for r in benchmark_returns:
            bench_total *= (1 + r)
        bench_total_return = bench_total - 1
        excess_return = total_return - bench_total_return
    else:
        bench_total_return = 0
        excess_return = total_return

    # 输出报告
    print("\n" + "=" * 60)
    print("回测绩效报告")
    print("=" * 60)
    print(f"回测区间: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"初始资金: {initial_cash:,.0f}")
    print(f"最终资金: {values[-1]:,.0f}")
    print(f"总收益率: {total_return*100:+.2f}%")
    print(f"年化收益率: {annual_return*100:+.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"胜率: {win_rate*100:.1f}%")
    if benchmark_returns:
        print(f"基准收益: {bench_total_return*100:+.2f}%")
        print(f"超额收益: {excess_return*100:+.2f}%")
    print("=" * 60)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "excess_return": excess_return,
        "values": values,
        "dates": dates,
    }
