"""
单元测试
核心函数测试：去极值、标准化、因子计算、M-Score
"""
import sys
import os
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import parse_cn_number
from factor_engine import winsorize_mad, zscore, rank_score, is_binary, preprocess_indicator
from factor_engine import calc_quality_factors, calc_factor_score
from fraud_detector import calc_m_score


class TestUtils(unittest.TestCase):
    """工具函数测试"""

    def test_parse_cn_number_basic(self):
        self.assertEqual(parse_cn_number(1.5), 1.5)
        self.assertEqual(parse_cn_number(100), 100.0)

    def test_parse_cn_number_wan(self):
        self.assertAlmostEqual(parse_cn_number("1.5万"), 15000.0)

    def test_parse_cn_number_yi(self):
        self.assertAlmostEqual(parse_cn_number("2.5亿"), 250000000.0)

    def test_parse_cn_number_percent(self):
        self.assertAlmostEqual(parse_cn_number("15.5%"), 0.155)

    def test_parse_cn_number_nan(self):
        self.assertTrue(np.isnan(parse_cn_number(None)))
        self.assertTrue(np.isnan(parse_cn_number("--")))
        self.assertTrue(np.isnan(parse_cn_number("")))


class TestFactorEngine(unittest.TestCase):
    """因子引擎测试"""

    def setUp(self):
        np.random.seed(42)
        self.normal_data = pd.Series(np.random.randn(100))

    def test_winsorize_mad_normal(self):
        result = winsorize_mad(self.normal_data)
        # 去极值后范围应该更小
        self.assertLess(result.max(), self.normal_data.max() + 1)
        self.assertGreater(result.min(), self.normal_data.min() - 1)

    def test_winsorize_mad_constant(self):
        """常数序列不应报错"""
        const = pd.Series([5.0] * 20)
        result = winsorize_mad(const)
        self.assertEqual(len(result), 20)

    def test_zscore_normal(self):
        result = zscore(self.normal_data)
        self.assertAlmostEqual(result.mean(), 0, places=5)
        self.assertAlmostEqual(result.std(), 1, places=1)

    def test_zscore_constant(self):
        """常数序列应返回全零"""
        const = pd.Series([5.0] * 20)
        result = zscore(const)
        self.assertTrue((result == 0).all())

    def test_rank_score(self):
        data = pd.Series([10, 20, 30, 40, 50])
        result = rank_score(data)
        self.assertAlmostEqual(result.iloc[0], 0.2)
        self.assertAlmostEqual(result.iloc[-1], 1.0)

    def test_is_binary_true(self):
        data = pd.Series([0, 1, 1, 0, 1])
        self.assertTrue(is_binary(data))

    def test_is_binary_false(self):
        data = pd.Series([0, 1, 2, 3, 4])
        self.assertFalse(is_binary(data))

    def test_preprocess_indicator_binary(self):
        """二元变量应使用rank_score而非zscore"""
        data = pd.Series([0, 1, 1, 0, 1, 0, 0, 1])
        result = preprocess_indicator(data, use_rank=True)
        # rank_score结果应在0-1之间
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

    def test_calc_quality_factors(self):
        """质量因子计算测试"""
        df = pd.DataFrame({
            "roe_5y_mean": [15.0, 20.0, 10.0],
            "roe_5y_std": [2.0, 3.0, 1.0],
            "gross_margin_5y_mean": [40.0, 50.0, 30.0],
            "gross_margin_5y_std": [5.0, 3.0, 8.0],
            "fcf_to_profit": [0.8, 0.9, 0.5],
            "fcf_to_assets": [0.1, 0.15, 0.05],
            "capex_to_revenue": [0.1, 0.05, 0.2],
        })
        result = calc_quality_factors(df)
        self.assertEqual(len(result), 3)
        # 应该有roe_mean列
        self.assertIn("roe_mean", result.columns)

    def test_calc_factor_score_with_missing(self):
        """缺失指标时权重应归一化"""
        df = pd.DataFrame({
            "roe_mean": [1.0, 2.0, 3.0],
            "roe_cv": [0.1, 0.2, 0.3],
            # 故意缺少其他指标
        })
        config = {
            "indicators": {
                "roe_mean": {"weight": 0.20, "ascending": False},
                "roe_cv": {"weight": 0.15, "ascending": True},
                "fcf_to_profit": {"weight": 0.20, "ascending": False},  # 缺失
            }
        }
        score = calc_factor_score(df, "test", config)
        # 不应报错，且得分不为NaN
        self.assertFalse(score.isna().all())
        self.assertEqual(len(score), 3)


class TestFraudDetector(unittest.TestCase):
    """造假检测测试"""

    def test_m_score_insufficient_data(self):
        """数据不足时应返回None"""
        result = calc_m_score(pd.DataFrame())
        self.assertIsNone(result[0])

    def test_m_score_calculation(self):
        """M-Score基本计算测试"""
        # 构造假数据（2年）
        df = pd.DataFrame([
            {
                "应收账款": 100, "营业总收入": 1000, "销售毛利率": 40,
                "非流动资产合计": 500, "资产总计": 2000,
                "折旧": 50, "销售费用": 100, "管理费用": 80,
                "负债合计": 800, "净利润": 200, "经营活动产生的现金流量净额": 150,
            },
            {
                "应收账款": 80, "营业总收入": 900, "销售毛利率": 42,
                "非流动资产合计": 480, "资产总计": 1800,
                "折旧": 45, "销售费用": 90, "管理费用": 70,
                "负债合计": 700, "净利润": 180, "经营活动产生的现金流量净额": 160,
            }
        ])
        result, risk, details = calc_m_score(df)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)
        self.assertIn(risk, ["低风险", "中风险", "高风险"])
        self.assertIn("DSRI", details)


class TestBacktesterIntegration(unittest.TestCase):
    """回测相关集成测试"""

    def test_calc_turnover(self):
        """换手率计算测试"""
        from backtester import calc_turnover
        # 完全不同的持仓
        self.assertAlmostEqual(calc_turnover(["A", "B"], ["C", "D"]), 1.0)
        # 完全相同的持仓
        self.assertAlmostEqual(calc_turnover(["A", "B"], ["A", "B"]), 0.0)
        # 部分重叠
        self.assertAlmostEqual(calc_turnover(["A", "B"], ["A", "C"]), 0.5)

    def test_calc_transaction_cost(self):
        """交易成本计算测试"""
        from backtester import calc_transaction_cost
        # 0换手=0成本
        self.assertAlmostEqual(calc_transaction_cost(0), 0)
        # 正换手应有正成本
        self.assertGreater(calc_transaction_cost(0.5), 0)


class TestStrategyIntegration(unittest.TestCase):
    """策略集成测试"""

    def test_apply_hard_constraints(self):
        """硬约束筛选测试"""
        from strategy import apply_hard_constraints
        df = pd.DataFrame({
            "code": ["A", "B", "C"],
            "consecutive_profit_years": [5, 1, 3],
            "debt_ratio": [50, 80, 60],
            "pe": [15, -5, 20],
            "pb": [2, 1, 3],
        })
        constraints = {
            "consecutive_profit_years": 3,
            "max_debt_ratio": 70,
            "consecutive_dividend_years": 0,
        }
        result = apply_hard_constraints(df, constraints)
        # A应通过（连续5年正利润，负债50%，PE>0）
        # B应被排除（连续1年，负债80%，PE<0）
        # C应通过
        self.assertEqual(len(result), 2)
        self.assertIn("A", result["code"].values)
        self.assertIn("C", result["code"].values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
