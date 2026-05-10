"""
量化模型配置文件
消费股价值投资量化模型（巴菲特+林奇+马克斯）
学术依据：Buffett's Alpha (AQR 2018), Quality Minus Junk (AQR 2019)
"""
import os

# 项目根目录（基于 config.py 的位置）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============ 硬约束（第一阶段筛选）============
HARD_CONSTRAINTS = {
    "consecutive_profit_years": 3,    # 连续正利润年数
    "max_debt_ratio": 70.0,           # 资产负债率上限(%)
    "consecutive_dividend_years": 3,  # 连续分红年数
}

# ============ 因子权重 ============
# 五因子体系（论文改进版）
FACTORS = {
    "quality": {              # 公司质量因子 - 巴菲特 + QMJ盈利能力
        "weight": 0.30,
        "indicators": {
            "roe_mean": {"weight": 0.20, "ascending": False},           # ROE均值
            "roe_cv": {"weight": 0.15, "ascending": True},              # ROE稳定性
            "gross_margin_mean": {"weight": 0.15, "ascending": False},  # 毛利率均值
            "gross_margin_cv": {"weight": 0.10, "ascending": True},     # 毛利率稳定性
            "fcf_to_profit": {"weight": 0.20, "ascending": False},      # 自由现金流/净利润
            "fcf_to_assets": {"weight": 0.10, "ascending": False},      # 自由现金流/总资产（QMJ）
            "capex_to_revenue": {"weight": 0.10, "ascending": True},    # 资本开支率
        }
    },
    "earnings_quality": {     # 盈利质量因子 - QMJ
        "weight": 0.15,
        "indicators": {
            "accruals_ratio": {"weight": 0.35, "ascending": True},      # 应计比率，越低越好
            "dividend_payout": {"weight": 0.35, "ascending": False},    # 派息比例，越高越好
            "cashflow_continuity": {"weight": 0.30, "ascending": False},  # 现金流连续性
        }
    },
    "growth": {               # 成长性因子 - 林奇 + QMJ成长
        "weight": 0.20,
        "indicators": {
            "profit_growth_3y": {"weight": 0.20, "ascending": False},   # 净利润增速
            "revenue_growth_3y": {"weight": 0.15, "ascending": False},  # 营收增速
            "roe_growth_3y": {"weight": 0.20, "ascending": False},      # ROE增长（QMJ）
            "peg_adjusted": {"weight": 0.25, "ascending": True},        # 调整PEG（结合稳定性）
            "profit_revenue_ratio": {"weight": 0.10, "ascending": False},  # 经营杠杆
            "growth_stability": {"weight": 0.10, "ascending": True},    # 增速稳定性
        }
    },
    "safety": {               # 安全性因子 - QMJ安全性
        "weight": 0.15,
        "indicators": {
            "beta": {"weight": 0.20, "ascending": True},                # Beta，越低越好
            "idio_volatility": {"weight": 0.20, "ascending": True},     # 特异性波动，越低越好
            "debt_ratio": {"weight": 0.20, "ascending": True},          # 负债率
            "interest_coverage": {"weight": 0.20, "ascending": False},  # 利息覆盖
            "current_ratio": {"weight": 0.20, "ascending": False},      # 流动比率
        }
    },
    "valuation": {            # 估值合理性因子
        "weight": 0.20,
        "indicators": {
            "pe_percentile": {"weight": 0.35, "ascending": True},       # PE历史分位
            "dividend_yield": {"weight": 0.35, "ascending": False},     # 股息率
            "peg_valuation": {"weight": 0.30, "ascending": True},       # PEG
        }
    },
}

# ============ 回测参数 ============
BACKTEST = {
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
    "initial_cash": 1_000_000,    # 初始资金100万
    "top_n": 10,                   # 持仓股票数
    "rebalance_freq": "quarterly", # 调仓频率
    "commission": 0.001,           # 手续费率
    "slippage": 0.001,             # 滑点
}

# ============ 股票池 ============
STOCK_POOL = "consumer"  # consumer / hs300 / zz500 / custom
CUSTOM_STOCKS = []

CONSUMER_STOCKS = [
    # 白酒
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "000568",  # 泸州老窖
    "002304",  # 洋河股份
    "600809",  # 山西汾酒
    "000596",  # 古井贡酒
    "603369",  # 今世缘
    # 乳制品
    "600887",  # 伊利股份
    "600597",  # 光明乳业
    # 调味品
    "603288",  # 海天味业
    "600872",  # 中炬高新
    # 食品饮料
    "603517",  # 绝味食品
    "600882",  # 妙可蓝多
    "002557",  # 洽洽食品
    # 家电
    "000651",  # 格力电器
    "000333",  # 美的集团
    "002032",  # 苏泊尔
    # 医药消费
    "600276",  # 恒瑞医药
    "000538",  # 云南白药
    "600436",  # 片仔癀
]

# ============ 输出路径 ============
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ============ 造假检测配置 ============
FRAUD_CHECK = {
    "m_score_threshold": -1.78,       # Beneish M-Score 阈值（>此值为造假嫌疑）
    "receivable_growth_limit": 2.0,   # 应收增速/营收增速上限
    "check_interval_days": 30,        # 检查间隔（天）
}
