# A股消费股价值投资量化模型

基于学术论文的多因子选股模型，融合巴菲特（质量）、彼得·林奇（成长）、霍华德·马克斯（风险）的投资理念。

## 学术依据

| 论文 | 核心贡献 |
|------|---------|
| Buffett's Alpha (AQR, 2018) | 巴菲特超额收益可被质量、价值、低波动因子解释 |
| Quality Minus Junk (AQR, 2019) | 质量因子（盈利能力、成长性、安全性、派息）在全球市场有效 |
| Fama-French Five Factor (2015) | RMW(盈利溢价) + CMA(投资因子) |
| Testing Peter Lynch's Criteria | PEG策略需结合增长质量，避免价值陷阱 |

## 模型架构

```
┌─────────────────────────────────────────────────────────┐
│                    main.py（入口）                        │
│  screener模式: 获取快照 → 筛选 → 报告                    │
│  backtest模式: 季度调仓 → 消除前瞻偏差 → 绩效评估         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    strategy.py（策略层）                   │
│  硬约束筛选 → 多因子打分 → 输出持仓信号                   │
│  接口: generate_signal(snapshot_df) → {stocks, weights}   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 dataset_builder.py（数据集构建）           │
│  get_snapshot(codes, as_of_date) → 按时间线构建数据       │
│  消除前瞻偏差：只使用 as_of_date 之前已发布的数据         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 data_fetcher.py（数据获取）                │
│  akshare接口封装 + 缓存 + 重试 + 数据校验                │
└─────────────────────────────────────────────────────────┘
```

## 五因子体系

| 因子 | 权重 | 核心指标 | 投资理念来源 |
|------|------|---------|-------------|
| **公司质量** | 30% | ROE均值+稳定性、毛利率、自由现金流/净利润、资本开支率 | 巴菲特 + QMJ |
| **盈利质量** | 15% | 应计比率、派息比例、现金流连续性 | QMJ |
| **成长性** | 20% | 净利润增速、营收增速、ROE增长、调整PEG | 林奇 + QMJ |
| **安全性** | 15% | Beta、特异性波动率、负债率、利息覆盖、流动比率 | QMJ |
| **估值** | 20% | PE历史分位、股息率、PEG | 巴菲特 + 林奇 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 筛选模式：获取当前推荐股票
python main.py screener

# 回测模式：运行历史回测
python main.py backtest
```

## 项目结构

```
a-share-quant-model/
├── main.py              # 入口：screener / backtest 两种模式
├── config.py            # 配置：股票池、因子权重、回测参数
├── utils.py             # 工具函数：日志、重试、数值解析
├── data_fetcher.py      # 数据获取：akshare封装、缓存、重试
├── dataset_builder.py   # 数据集构建：按时间线构建快照（消除前瞻偏差）
├── factor_engine.py     # 因子引擎：去极值、标准化、因子计算
├── strategy.py          # 策略层：硬约束筛选 + 多因子打分 → 持仓信号
├── backtester.py        # 回测引擎：季度调仓、交易成本、绩效评估
├── requirements.txt
├── docs/
│   └── research_notes.md  # 学术论文研究总结与模型改进
└── output/              # 输出目录（报告、Excel）
```

## 消除前瞻偏差

本模型通过 `dataset_builder.py` 的 `get_snapshot(codes, as_of_date)` 接口消除前瞻偏差：

- 每个调仓日只使用该日**之前已发布**的财务数据
- Beta 和特异性波动率使用截至调仓日的滚动窗口计算
- PE 历史分位数基于调仓日之前的历史数据

## 配置说明

在 `config.py` 中可以调整：

- `HARD_CONSTRAINTS`：硬约束阈值
- `FACTORS`：因子权重和指标配置
- `BACKTEST`：回测参数（区间、调仓频率、初始资金等）
- `CONSUMER_STOCKS`：自定义股票池

## 输出

运行后会在 `output/` 目录生成：

- `report_YYYYMMDD.txt`：文本格式筛选报告
- `scored_YYYYMMDD.xlsx`：Excel格式打分数据

## 参考框架

本项目的设计参考了以下开源量化框架：

| 框架 | 借鉴的设计 |
|------|-----------|
| [Qlib](https://github.com/microsoft/qlib) | 数据管线设计 |
| [Zipline](https://github.com/stefan-jansen/zipline-reloaded) | Pipeline API |
| [RQAlpha](https://github.com/ricequant/rqalpha) | A股特殊规则处理 |
| [Alphalens](https://github.com/stefan-jansen/alphalens-reloaded) | 因子分析方法 |

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
