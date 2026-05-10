"""
主入口
消费股价值投资量化模型
学术依据：Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONSUMER_STOCKS, STOCK_POOL, OUTPUT_DIR, BACKTEST
from data_fetcher import get_stock_pool, get_financial_data, get_price_data, batch_fetch_financials
from factor_model import get_top_stocks, generate_report
from backtester import run_backtest


def parse_cn_number(val):
    """解析中文数字（如 1.47亿, 6.28万）"""
    if pd.isna(val) or val == False or val == 'False':
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    try:
        if '亿' in val:
            return float(val.replace('亿', '')) * 1e8
        elif '万' in val:
            return float(val.replace('万', '')) * 1e4
        elif '%' in val:
            return float(val.replace('%', ''))
        else:
            return float(val)
    except:
        return np.nan


def fetch_multi_year_data(stock_code, years=5):
    """
    获取单只股票的多年财务数据，计算因子所需的衍生指标
    使用同花顺财务摘要接口
    """
    import akshare as ak

    try:
        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator='按报告期')
        if df is None or df.empty:
            return None

        df = df.head(years)
        result = {"code": stock_code}

        # --- ROE（净资产收益率-摊薄）---
        roe_col = [c for c in df.columns if "净资产收益率" in c and "摊薄" in c]
        if roe_col:
            roe_values = pd.to_numeric(df[roe_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(roe_values) >= 2:
                result["roe_5y_mean"] = roe_values.mean()
                result["roe_5y_std"] = roe_values.std()
                if roe_values.iloc[-1] > 0:
                    result["roe_growth_3y"] = (roe_values.iloc[0] / roe_values.iloc[-1] - 1) * 100
            elif len(roe_values) == 1:
                result["roe_5y_mean"] = roe_values.iloc[0]
                result["roe_5y_std"] = 0

        # --- 毛利率 ---
        gm_col = [c for c in df.columns if "销售毛利率" in c or "毛利率" in c]
        if gm_col:
            gm_values = pd.to_numeric(df[gm_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(gm_values) >= 2:
                result["gross_margin_5y_mean"] = gm_values.mean()
                result["gross_margin_5y_std"] = gm_values.std()
            elif len(gm_values) == 1:
                result["gross_margin_5y_mean"] = gm_values.iloc[0]
                result["gross_margin_5y_std"] = 0

        # --- 资产负债率 ---
        debt_col = [c for c in df.columns if "资产负债率" in c]
        if debt_col:
            debt_values = pd.to_numeric(df[debt_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(debt_values) > 0:
                result["debt_ratio"] = debt_values.iloc[0]

        # --- 流动比率 ---
        cr_col = [c for c in df.columns if "流动比率" in c]
        if cr_col:
            cr_values = pd.to_numeric(df[cr_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(cr_values) > 0:
                result["current_ratio"] = cr_values.iloc[0]

        # --- 净利润增速 ---
        pg_col = [c for c in df.columns if "净利润同比增长" in c or "净利润同比" in c]
        if not pg_col:
            pg_col = [c for c in df.columns if "净利润增长" in c]
        if pg_col:
            pg_values = pd.to_numeric(df[pg_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(pg_values) >= 3:
                result["profit_growth_3y"] = pg_values.head(3).mean()
                result["growth_stability"] = pg_values.head(3).std() / abs(pg_values.head(3).mean()) if pg_values.head(3).mean() != 0 else 999
            elif len(pg_values) > 0:
                result["profit_growth_3y"] = pg_values.iloc[0]
                result["growth_stability"] = 0

        # --- 营收增速 ---
        rg_col = [c for c in df.columns if "营业总收入同比增长" in c or "营收同比" in c]
        if not rg_col:
            rg_col = [c for c in df.columns if "营业总收入" in c and "同比" in c]
        if rg_col:
            rg_values = pd.to_numeric(df[rg_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(rg_values) >= 3:
                result["revenue_growth_3y"] = rg_values.head(3).mean()
                if "profit_growth_3y" in result and result.get("revenue_growth_3y", 0) != 0:
                    result["profit_revenue_ratio"] = result["profit_growth_3y"] / result["revenue_growth_3y"]
            elif len(rg_values) > 0:
                result["revenue_growth_3y"] = rg_values.iloc[0]

        # --- 连续正利润年数 ---
        np_col = [c for c in df.columns if "净利润" in c and "同比" not in c and "扣非" not in c]
        if np_col:
            np_values = pd.to_numeric(df[np_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            consecutive = 0
            for v in np_values:
                if v > 0:
                    consecutive += 1
                else:
                    break
            result["consecutive_profit_years"] = consecutive

        # --- 速动比率（替代利息覆盖倍数）---
        qr_col = [c for c in df.columns if "速动比率" in c]
        if qr_col:
            qr_values = pd.to_numeric(df[qr_col[0]].apply(parse_cn_number), errors="coerce").dropna()
            if len(qr_values) > 0:
                result["interest_coverage"] = qr_values.iloc[0]

        return result

    except Exception as e:
        print(f"  获取{stock_code}多年数据失败: {e}")
        return None


def fetch_cashflow_data(stock_code):
    """获取现金流相关数据"""
    import akshare as ak

    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=stock_code)
        if df is None or df.empty:
            return {}

        result = {}
        latest = df.iloc[0]

        # 经营现金流
        ocf_col = [c for c in df.columns if "经营活动产生的现金流量净额" in c]
        if ocf_col:
            result["operating_cashflow"] = pd.to_numeric(latest[ocf_col[0]], errors="coerce")

        # 资本开支
        capex_col = [c for c in df.columns if "购建固定资产" in c]
        if capex_col:
            result["capex"] = pd.to_numeric(latest[capex_col[0]], errors="coerce")

        # 近3年经营现金流是否连续为正
        if ocf_col and len(df) >= 3:
            ocf_values = pd.to_numeric(df[ocf_col[0]].head(3), errors="coerce").dropna()
            result["cashflow_continuity"] = 1 if all(v > 0 for v in ocf_values) else 0

        return result

    except Exception as e:
        return {}


def fetch_profit_data(stock_code):
    """获取利润表数据"""
    import akshare as ak

    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=stock_code)
        if df is None or df.empty:
            return {}

        result = {}
        latest = df.iloc[0]

        # 净利润
        np_col = [c for c in df.columns if "净利润" in c and "扣非" not in c]
        if np_col:
            result["net_profit"] = pd.to_numeric(latest[np_col[0]], errors="coerce")

        # 营收
        rev_col = [c for c in df.columns if "营业总收入" in c or "营业收入" in c]
        if rev_col:
            result["revenue"] = pd.to_numeric(latest[rev_col[0]], errors="coerce")

        return result

    except Exception as e:
        return {}


def fetch_balance_sheet_data(stock_code):
    """获取资产负债表数据（用于计算应计比率、总资产）"""
    import akshare as ak

    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=stock_code)
        if df is None or df.empty:
            return {}

        result = {}
        latest = df.iloc[0]

        # 总资产
        ta_col = [c for c in df.columns if "资产总计" in c or "总资产" in c]
        if ta_col:
            result["total_assets"] = pd.to_numeric(latest[ta_col[0]], errors="coerce")

        return result

    except Exception as e:
        return {}


def calc_beta_and_volatility(stock_code, market_code="000300"):
    """
    计算Beta和特异性波动率
    使用近1年日收益率数据
    """
    try:
        # 获取个股行情
        stock_df = get_price_data(stock_code, start_date="20240101")
        if stock_df is None or stock_df.empty or len(stock_df) < 60:
            return {}

        # 获取基准行情
        import akshare as ak
        bench_df = ak.stock_zh_index_daily_em(symbol=market_code, start_date="20240101")
        if bench_df is None or bench_df.empty:
            return {}

        # 计算日收益率
        stock_df["日期"] = pd.to_datetime(stock_df["日期"])
        bench_df["date"] = pd.to_datetime(bench_df["date"])

        stock_df = stock_df.set_index("日期").sort_index()
        bench_df = bench_df.set_index("date").sort_index()

        # 对齐日期
        common_dates = stock_df.index.intersection(bench_df.index)
        if len(common_dates) < 60:
            return {}

        stock_ret = stock_df.loc[common_dates, "收盘"].pct_change().dropna()
        bench_ret = bench_df.loc[common_dates, "close"].pct_change().dropna()

        common = stock_ret.index.intersection(bench_ret.index)
        if len(common) < 60:
            return {}

        stock_ret = stock_ret.loc[common]
        bench_ret = bench_ret.loc[common]

        # 计算Beta = Cov(stock, bench) / Var(bench)
        cov = np.cov(stock_ret, bench_ret)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

        # 计算特异性波动率（回归残差的标准差）
        predicted = beta * bench_ret
        residuals = stock_ret - predicted
        idio_vol = residuals.std() * np.sqrt(252)  # 年化

        return {"beta": beta, "idio_volatility": idio_vol}

    except Exception as e:
        return {}


def calc_accruals_ratio(net_profit, operating_cashflow, total_assets):
    """计算应计比率 = (净利润 - 经营现金流) / 总资产"""
    if total_assets and total_assets != 0:
        return (net_profit - operating_cashflow) / total_assets
    return None


def calc_dividend_payout(stock_code, net_profit):
    """计算派息比例"""
    import akshare as ak

    try:
        div_df = ak.stock_dividend_cninfo(symbol=stock_code)
        if div_df is not None and not div_df.empty:
            # 获取最近一次分红
            div_amount = pd.to_numeric(div_df.iloc[0].get("每股股利", 0), errors="coerce")
            if div_amount and net_profit and net_profit > 0:
                # 简化：用每股股利 / 每股收益估算
                # 实际需要总股本数据，这里用简化方式
                return min(div_amount * 100 / net_profit, 1.0) if net_profit > 0 else 0
    except:
        pass
    return 0


def build_stock_dataset(stock_codes):
    """
    构建完整的股票数据集
    合并多年财务数据、现金流数据、行情数据
    """
    import akshare as ak

    rows = []
    total = len(stock_codes)

    for i, code in enumerate(stock_codes):
        if (i + 1) % 5 == 0:
            print(f"进度: {i+1}/{total}")

        # 1. 多年财务数据
        multi_year = fetch_multi_year_data(code, years=5)
        if multi_year is None:
            continue

        # 2. 现金流数据
        cf_data = fetch_cashflow_data(code)
        multi_year.update(cf_data)

        # 3. 利润表数据
        profit_data = fetch_profit_data(code)
        multi_year.update(profit_data)

        # 4. 资产负债表数据
        bs_data = fetch_balance_sheet_data(code)
        multi_year.update(bs_data)

        # 5. 计算衍生指标
        # 自由现金流/净利润
        if "operating_cashflow" in multi_year and "capex" in multi_year and "net_profit" in multi_year:
            fcf = multi_year["operating_cashflow"] - multi_year.get("capex", 0)
            np_val = multi_year["net_profit"]
            if np_val and np_val != 0:
                multi_year["fcf_to_profit"] = fcf / np_val
            # 自由现金流/总资产（QMJ指标）
            ta = multi_year.get("total_assets")
            if ta and ta != 0:
                multi_year["fcf_to_assets"] = fcf / ta

        # 资本开支/营收
        if "capex" in multi_year and "revenue" in multi_year:
            rev = multi_year["revenue"]
            if rev and rev != 0:
                multi_year["capex_to_revenue"] = abs(multi_year["capex"]) / rev

        # 应计比率（QMJ盈利质量指标）
        if "net_profit" in multi_year and "operating_cashflow" in multi_year and "total_assets" in multi_year:
            accruals = calc_accruals_ratio(
                multi_year["net_profit"],
                multi_year["operating_cashflow"],
                multi_year["total_assets"]
            )
            if accruals is not None:
                multi_year["accruals_ratio"] = accruals

        # 6. Beta和特异性波动率
        beta_vol = calc_beta_and_volatility(code)
        multi_year.update(beta_vol)

        # 7. 获取当前PE/PB/股息率
        try:
            spot_df = ak.stock_zh_a_spot_em()
            spot_row = spot_df[spot_df["代码"] == code]
            if not spot_row.empty:
                row = spot_row.iloc[0]
                multi_year["pe"] = pd.to_numeric(row.get("市盈率-动态", 0), errors="coerce")
                multi_year["pb"] = pd.to_numeric(row.get("市净率", 0), errors="coerce")
                multi_year["name"] = row.get("名称", code)
                multi_year["pe_percentile"] = 0.5  # 占位
        except:
            pass

        # 8. 股息率和派息比例
        try:
            div_df = ak.stock_dividend_cninfo(symbol=code)
            if div_df is not None and not div_df.empty:
                multi_year["dividend_yield"] = float(div_df.iloc[0].get("股息率", 0))
            else:
                multi_year["dividend_yield"] = 0
        except:
            multi_year["dividend_yield"] = 0

        rows.append(multi_year)

    return pd.DataFrame(rows)


def main():
    """主函数"""
    print("=" * 70)
    print("A股消费股价值投资量化模型")
    print("投资理念: 巴菲特(质量) + 林奇(成长) + 马克斯(风险)")
    print("学术依据: Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)")
    print("=" * 70)
    print()

    # 1. 获取股票池
    if STOCK_POOL == "consumer" and CONSUMER_STOCKS:
        stock_codes = CONSUMER_STOCKS
        print(f"使用自定义消费股池: {len(stock_codes)} 只")
    else:
        pool_df = get_stock_pool(STOCK_POOL)
        stock_codes = pool_df["code"].tolist()
        print(f"使用{STOCK_POOL}股票池: {len(stock_codes)} 只")

    # 2. 构建数据集
    print("\n正在获取财务数据...")
    dataset = build_stock_dataset(stock_codes)

    if dataset.empty:
        print("无有效数据，退出")
        return

    print(f"\n获取到 {len(dataset)} 只股票的有效数据")

    # 3. 筛选+打分
    print("\n正在筛选和打分...")
    top_stocks = get_top_stocks(dataset, top_n=10)

    if top_stocks.empty:
        print("无符合条件的股票")
        return

    # 4. 生成报告
    report = generate_report(top_stocks)
    print("\n" + report)

    # 5. 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_file = os.path.join(OUTPUT_DIR, f"report_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {report_file}")

    # 6. 保存Excel
    excel_file = os.path.join(OUTPUT_DIR, f"scored_{datetime.now().strftime('%Y%m%d')}.xlsx")
    top_stocks.to_excel(excel_file, index=False)
    print(f"数据已保存: {excel_file}")


if __name__ == "__main__":
    main()
