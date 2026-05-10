"""
数据造假检测模块
检测方法：Beneish M-Score + 监管处罚 + 财务数据异常
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from data_fetcher import get_financial_indicators, get_penalty_info, get_audit_opinion
from config import FRAUD_CHECK

logger = logging.getLogger("quant.fraud")


# ============ Beneish M-Score ============

def calc_m_score(financial_df):
    """
    计算 Beneish M-Score
    需要至少2年的财务数据

    M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    M > -1.78 为造假嫌疑
    """
    if financial_df is None or len(financial_df) < 2:
        return None, "数据不足"

    df = financial_df.head(2)  # 最近2年

    try:
        # 获取各指标列
        def find_col(keywords):
            for kw in keywords:
                cols = [c for c in df.columns if kw in c]
                if cols:
                    return cols[0]
            return None

        # 应收账款
        ar_col = find_col(["应收账款"])
        # 营收
        rev_col = find_col(["营业总收入", "营业收入"])
        # 毛利率
        gm_col = find_col(["销售毛利率", "毛利率"])
        # 非流动资产
        nca_col = find_col(["非流动资产合计"])
        # 总资产
        ta_col = find_col(["资产总计", "总资产"])
        # 折旧
        dep_col = find_col(["折旧"])
        # 销管费用
        sga_col = find_col(["销售费用", "管理费用"])
        # 负债合计
        debt_col = find_col(["负债合计", "总负债"])
        # 净利润
        np_col = find_col(["净利润"])
        # 经营现金流
        ocf_col = find_col(["经营活动产生的现金流量净额"])

        def safe_get(col_name, idx):
            if col_name is None:
                return np.nan
            val = pd.to_numeric(df.iloc[idx].get(col_name, np.nan), errors="coerce")
            return val

        # 当期(t=0)和上期(t=1)
        # DSRI = (应收t0/营收t0) / (应收t1/营收t1)
        ar0, ar1 = safe_get(ar_col, 0), safe_get(ar_col, 1)
        rev0, rev1 = safe_get(rev_col, 0), safe_get(rev_col, 1)
        if rev0 and rev1 and ar0 is not np.nan and ar1 is not np.nan:
            dsri = (ar0 / rev0) / (ar1 / rev1) if (rev0 and rev1 and ar1) else 1.0
        else:
            dsri = 1.0

        # GMI = 毛利率t1 / 毛利率t0
        gm0 = safe_get(gm_col, 0)
        gm1 = safe_get(gm_col, 1)
        if gm0 and gm1:
            gmi = gm1 / gm0
        else:
            gmi = 1.0

        # AQI = (1 - 非流动资产t0/总资产t0) / (1 - 非流动资产t1/总资产t1)
        nca0, nca1 = safe_get(nca_col, 0), safe_get(nca_col, 1)
        ta0, ta1 = safe_get(ta_col, 0), safe_get(ta_col, 1)
        if all(v is not np.nan and v for v in [nca0, nca1, ta0, ta1]):
            aqi = (1 - nca0 / ta0) / (1 - nca1 / ta1)
        else:
            aqi = 1.0

        # SGI = 营收t0 / 营收t1
        if rev0 and rev1:
            sgi = rev0 / rev1
        else:
            sgi = 1.0

        # DEPI = 折旧率t1 / 折旧率t0
        dep0, dep1 = safe_get(dep_col, 0), safe_get(dep_col, 1)
        if dep0 and dep1 and ta0 and ta1:
            depi = (dep1 / ta1) / (dep0 / ta0)
        else:
            depi = 1.0

        # SGAI = 销管费用率t0 / 销管费用率t1
        sga0, sga1 = safe_get(sga_col, 0), safe_get(sga_col, 1)
        if sga0 and sga1 and rev0 and rev1:
            sgai = (sga0 / rev0) / (sga1 / rev1)
        else:
            sgai = 1.0

        # LVGI = 负债率t0 / 负债率t1
        debt0, debt1 = safe_get(debt_col, 0), safe_get(debt_col, 1)
        if debt0 and debt1 and ta0 and ta1:
            lvgi = (debt0 / ta0) / (debt1 / ta1)
        else:
            lvgi = 1.0

        # TATA = (净利润 - 经营现金流) / 总资产
        np0 = safe_get(np_col, 0)
        ocf0 = safe_get(ocf_col, 0)
        if np0 is not np.nan and ocf0 is not np.nan and ta0:
            tata = (np0 - ocf0) / ta0
        else:
            tata = 0.0

        # M-Score 公式
        m = (-4.84
             + 0.920 * dsri
             + 0.528 * gmi
             + 0.404 * aqi
             + 0.892 * sgi
             + 0.115 * depi
             - 0.172 * sgai
             + 4.679 * tata
             - 0.327 * lvgi)

        # 风险等级
        threshold = FRAUD_CHECK["m_score_threshold"]
        if m > threshold:
            risk = "高风险"
        elif m > threshold - 0.5:
            risk = "中风险"
        else:
            risk = "低风险"

        details = {
            "DSRI": round(dsri, 3),
            "GMI": round(gmi, 3),
            "AQI": round(aqi, 3),
            "SGI": round(sgi, 3),
            "DEPI": round(depi, 3),
            "SGAI": round(sgai, 3),
            "LVGI": round(lvgi, 3),
            "TATA": round(tata, 3),
        }

        return round(m, 3), risk, details

    except Exception as e:
        logger.warning(f"M-Score计算失败: {e}")
        return None, "计算失败", {}


# ============ 财务数据异常检测 ============

def detect_data_anomalies(stock_code, financial_df, cashflow_df=None):
    """
    检测财务数据异常信号
    """
    anomalies = []

    if financial_df is None or financial_df.empty:
        return anomalies

    df = financial_df.head(4)  # 最近4年

    def find_col(keywords):
        for kw in keywords:
            cols = [c for c in df.columns if kw in c]
            if cols:
                return cols[0]
        return None

    # 1. 应收账款增速远超营收增速
    ar_col = find_col(["应收账款"])
    rev_col = find_col(["营业总收入", "营业收入"])
    if ar_col and rev_col and len(df) >= 2:
        ar0 = pd.to_numeric(df.iloc[0].get(ar_col, 0), errors="coerce")
        ar1 = pd.to_numeric(df.iloc[1].get(ar_col, 0), errors="coerce")
        rev0 = pd.to_numeric(df.iloc[0].get(rev_col, 0), errors="coerce")
        rev1 = pd.to_numeric(df.iloc[1].get(rev_col, 0), errors="coerce")
        if all(pd.notna(v) and v for v in [ar0, ar1, rev0, rev1]):
            ar_growth = (ar0 - ar1) / abs(ar1) if ar1 else 0
            rev_growth = (rev0 - rev1) / abs(rev1) if rev1 else 0
            if rev_growth > 0 and ar_growth / rev_growth > FRAUD_CHECK["receivable_growth_limit"]:
                anomalies.append(f"应收账款增速({ar_growth:.1%})远超营收增速({rev_growth:.1%})")

    # 2. 连续多年营收增长但经营现金流为负
    ocf_col = find_col(["经营活动产生的现金流量净额"])
    if rev_col and ocf_col and len(df) >= 2:
        rev_growing = True
        ocf_negative = True
        for i in range(min(3, len(df))):
            rev_val = pd.to_numeric(df.iloc[i].get(rev_col, 0), errors="coerce")
            ocf_val = pd.to_numeric(df.iloc[i].get(ocf_col, 0), errors="coerce")
            if pd.isna(rev_val) or (i > 0 and rev_val <= pd.to_numeric(df.iloc[i-1].get(rev_col, 0), errors="coerce")):
                rev_growing = False
            if pd.isna(ocf_val) or ocf_val > 0:
                ocf_negative = False
        if rev_growing and ocf_negative:
            anomalies.append("连续多年营收增长但经营现金流为负（纸面利润）")

    # 3. 毛利率异常高（>80%且非特殊行业）
    gm_col = find_col(["销售毛利率", "毛利率"])
    if gm_col:
        gm_val = pd.to_numeric(df.iloc[0].get(gm_col, 0), errors="coerce")
        if pd.notna(gm_val) and gm_val > 80:
            anomalies.append(f"毛利率异常高({gm_val:.1f}%)，需关注行业合理性")

    # 4. 非经常性损益占比过高
    nr_col = find_col(["非经常性损益"])
    np_col = find_col(["净利润"])
    if nr_col and np_col:
        nr_val = abs(pd.to_numeric(df.iloc[0].get(nr_col, 0), errors="coerce"))
        np_val = abs(pd.to_numeric(df.iloc[0].get(np_col, 0), errors="coerce"))
        if pd.notna(nr_val) and pd.notna(np_val) and np_val > 0:
            if nr_val / np_val > 0.5:
                anomalies.append(f"非经常性损益占比过高({nr_val/np_val:.1%})，盈利质量存疑")

    return anomalies


# ============ 综合检查 ============

def check_stock(stock_code, stock_name=None):
    """
    综合检查单只股票的造假风险
    """
    result = {
        "code": stock_code,
        "name": stock_name or stock_code,
        "m_score": None,
        "m_score_risk": "未知",
        "m_score_details": {},
        "regulatory_flags": [],
        "data_anomalies": [],
        "audit_opinion": "未知",
        "risk_level": "未知",
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # 1. 获取财务数据，计算 M-Score
    try:
        fin_df = get_financial_indicators(stock_code)
        if fin_df is not None and not fin_df.empty:
            m_score, m_risk, m_details = calc_m_score(fin_df)
            result["m_score"] = m_score
            result["m_score_risk"] = m_risk
            result["m_score_details"] = m_details

            # 2. 数据异常检测
            result["data_anomalies"] = detect_data_anomalies(stock_code, fin_df)
    except Exception as e:
        logger.warning(f"{stock_code} 财务数据获取失败: {e}")

    # 3. 监管处罚信息
    try:
        penalties = get_penalty_info(stock_code)
        if penalties:
            result["regulatory_flags"] = penalties
    except Exception as e:
        logger.debug(f"{stock_code} 监管信息获取失败: {e}")

    # 4. 审计意见
    try:
        audit = get_audit_opinion(stock_code)
        if audit:
            result["audit_opinion"] = audit
            if "非标准" in audit or "保留" in audit or "否定" in audit or "无法表示" in audit:
                result["regulatory_flags"].append(f"审计意见: {audit}")
    except Exception as e:
        logger.debug(f"{stock_code} 审计意见获取失败: {e}")

    # 5. 综合风险等级
    result["risk_level"] = _calc_overall_risk(result)

    return result


def _calc_overall_risk(result):
    """计算综合风险等级"""
    score = 0

    # M-Score
    if result["m_score_risk"] == "高风险":
        score += 3
    elif result["m_score_risk"] == "中风险":
        score += 1

    # 监管处罚
    score += len(result["regulatory_flags"]) * 2

    # 数据异常
    score += len(result["data_anomalies"])

    # 审计意见
    if result["audit_opinion"] != "标准无保留意见" and result["audit_opinion"] != "未知":
        score += 2

    if score >= 5:
        return "高风险"
    elif score >= 2:
        return "中风险"
    else:
        return "低风险"


def check_pool(stock_codes, stock_names=None):
    """批量检查股票池"""
    results = []
    total = len(stock_codes)

    for i, code in enumerate(stock_codes):
        if (i + 1) % 10 == 0:
            logger.info(f"检查进度: {i+1}/{total}")

        name = stock_names.get(code, code) if stock_names else code
        result = check_stock(code, stock_name=name)
        results.append(result)

    return pd.DataFrame(results)


def generate_fraud_report(results_df):
    """生成造假风险报告"""
    report = []
    report.append("=" * 80)
    report.append("数据造假风险检测报告")
    report.append(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"检测方法: Beneish M-Score + 监管处罚 + 财务异常")
    report.append("=" * 80)
    report.append("")

    # 统计
    high = len(results_df[results_df["risk_level"] == "高风险"])
    medium = len(results_df[results_df["risk_level"] == "中风险"])
    low = len(results_df[results_df["risk_level"] == "低风险"])
    report.append(f"检测总数: {len(results_df)} 只")
    report.append(f"  高风险: {high} 只")
    report.append(f"  中风险: {medium} 只")
    report.append(f"  低风险: {low} 只")
    report.append("")

    # 高风险详情
    high_df = results_df[results_df["risk_level"] == "高风险"]
    if not high_df.empty:
        report.append("【高风险股票】")
        report.append("-" * 80)
        for _, row in high_df.iterrows():
            report.append(f"\n  {row['name']} ({row['code']})")
            if row["m_score"] is not None:
                report.append(f"    M-Score: {row['m_score']} (阈值: {FRAUD_CHECK['m_score_threshold']})")
            if row["regulatory_flags"]:
                for flag in row["regulatory_flags"]:
                    report.append(f"    监管警告: {flag}")
            if row["data_anomalies"]:
                for anomaly in row["data_anomalies"]:
                    report.append(f"    数据异常: {anomaly}")
            if row["audit_opinion"] not in ("标准无保留意见", "未知"):
                report.append(f"    审计意见: {row['audit_opinion']}")

    # 中风险详情
    medium_df = results_df[results_df["risk_level"] == "中风险"]
    if not medium_df.empty:
        report.append("\n【中风险股票】")
        report.append("-" * 80)
        for _, row in medium_df.iterrows():
            report.append(f"\n  {row['name']} ({row['code']})")
            if row["m_score"] is not None:
                report.append(f"    M-Score: {row['m_score']}")
            if row["data_anomalies"]:
                for anomaly in row["data_anomalies"]:
                    report.append(f"    数据异常: {anomaly}")

    # M-Score 排行
    report.append("\n【M-Score 排行（从高到低）】")
    report.append("-" * 80)
    m_df = results_df[results_df["m_score"].notna()].sort_values("m_score", ascending=False)
    header = f"  {'代码':<8} {'名称':<10} {'M-Score':<10} {'风险':<8}"
    report.append(header)
    for _, row in m_df.iterrows():
        report.append(f"  {row['code']:<8} {row['name']:<10} {row['m_score']:<10} {row['m_score_risk']:<8}")

    report.append("\n" + "=" * 80)
    report.append("注: M-Score > -1.78 为造假嫌疑（Beneish, 1999）")
    report.append("    本报告仅供参考，不构成投资建议")
    report.append("=" * 80)

    return "\n".join(report)
