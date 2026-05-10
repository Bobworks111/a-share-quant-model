# A股消费股价值投资量化模型

基于学术论文的多因子选股模型，融合巴菲特（质量）、彼得·林奇（成长）、霍华德·马克斯（风险）的投资理念。

## 学术依据

| 论文 | 核心贡献 |
|------|---------|
| Buffett's Alpha (AQR, 2018) | 巴菲特超额收益可被质量、价值、低波动因子解释 |
| Quality Minus Junk (AQR, 2019) | 质量因子（盈利能力、成长性、安全性、派息）在全球市场有效 |
| Fama-French Five Factor (2015) | RMW(盈利溢价) + CMA(投资因子) |
| Beneish M-Score (1999) | 财报操纵检测模型（8变量） |

## 运行模式

```bash
# 筛选模式：获取当前推荐股票
python main.py screener

# 回测模式：历史回测
python main.py backtest

# 滚动回测：滚动窗口训练+测试（防过拟合）
python main.py rolling_backtest

# 因子研究：IC分析、分层回测、因子相关性、因子衰减
python main.py factor_research

# 造假检测：Beneish M-Score + 监管处罚 + 财务异常
python main.py fraud_check
```

## 模型架构

```
main.py（5种运行模式）
    │
    ├── strategy.py ──── 硬约束筛选 → 多因子打分 → 持仓信号
    │
    ├── backtester.py ── 季度调仓 / 滚动回测 / 绩效评估
    │
    ├── factor_research.py ── IC分析 / 分层回测 / 因子相关性 / 衰减
    │
    ├── fraud_detector.py ── M-Score / 监管处罚 / 数据异常
    │
    ├── dataset_builder.py ── 按时间线构建数据快照（消除前瞻偏差）
    │
    ├── factor_engine.py ── 去极值 / 标准化 / 因子计算
    │
    ├── data_fetcher.py ── akshare接口 + 缓存 + 重试
    │
    └── config.py + utils.py
```

## 五因子体系

| 因子 | 权重 | 核心指标 | 投资理念来源 |
|------|------|---------|-------------|
| **公司质量** | 30% | ROE均值+稳定性、毛利率、自由现金流/净利润、资本开支率 | 巴菲特 + QMJ |
| **盈利质量** | 15% | 应计比率、派息比例、现金流连续性 | QMJ |
| **成长性** | 20% | 净利润增速、营收增速、ROE增长、调整PEG | 林奇 + QMJ |
| **安全性** | 15% | Beta、特异性波动率、负债率、利息覆盖、流动比率 | QMJ |
| **估值** | 20% | PE历史分位、股息率、PEG | 巴菲特 + 林奇 |

## 项目结构

```
a-share-quant-model/
├── main.py              # 入口（5种模式）
├── config.py            # 配置
├── utils.py             # 工具函数
├── data_fetcher.py      # 数据获取
├── dataset_builder.py   # 数据集构建（消除前瞻偏差）
├── factor_engine.py     # 因子引擎
├── factor_research.py   # 因子研究工具
├── fraud_detector.py    # 造假检测
├── strategy.py          # 策略层
├── backtester.py        # 回测引擎（含滚动回测）
├── requirements.txt
├── docs/
│   └── research_notes.md
└── output/
```

## 配置说明

在 `config.py` 中可以调整：

- `HARD_CONSTRAINTS`：硬约束阈值
- `FACTORS`：因子权重和指标配置
- `BACKTEST`：回测参数
- `FRAUD_CHECK`：造假检测参数
- `CONSUMER_STOCKS`：自定义股票池

## 参考框架

| 框架 | 借鉴的设计 |
|------|-----------|
| [Qlib](https://github.com/microsoft/qlib) | 数据管线设计 |
| [Zipline](https://github.com/stefan-jansen/zipline-reloaded) | Pipeline API |
| [RQAlpha](https://github.com/ricequant/rqalpha) | A股特殊规则处理 |
| [Alphalens](https://github.com/stefan-jansen/alphalens-reloaded) | 因子分析方法 |

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
