# nas100-quant

NAS100 短中线动量策略（含 regime 过滤与波动率目标仓位）

一个基于 NAS100 成分股的横截面打分日频选股回测系统。从你的本金（默认 $1,000,000）开始，每天根据动量与反转信号给 100 只股票打分，挑前 8 名持有 1–20 天，配大盘 regime 过滤与波动率目标仓位，自动止损。

---

## 快速开始

```bash
# 安装依赖（首次）
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 在 .env 中填入 Longport 凭证
# LONGPORT_APP_KEY=...
# LONGPORT_APP_SECRET=...
# LONGPORT_ACCESS_TOKEN=...

# 运行回测（所有结果直接打印到控制台，不写文件）
python backtest.py
```

打开 [`backtest.py`](backtest.py) 顶部的 `CONFIG` 区块即可调参；运行后会打印：
1. 策略配置回显
2. 数据加载进度
3. **每一笔交易的开/平仓**（股票、日期、价位、止损、仓位金额、持仓数、盈亏）
4. 回测汇总（CAGR/Sharpe/MDD/终值/多空各自盈亏/**总交易成本**）
5. 盈利 Top 10 / 亏损 Top 10 交易明细
6. 与 QQQ 基准对比

---

## 策略原理

### 1. 股票池

NAS100 当期成分股共 100 只（[`nas100_universe.py`](nas100_universe.py) 里的静态清单）。
每天动态过滤：20 日平均成交额 ≥ $50M，且当日有完整 OHLC 数据。
未上市 / 退市的股票自动跳过，从其有数据日起参与。

### 2. 信号

每只股票每天计算 6 个信号，分为两类：

| 类别 | 信号 | 含义 |
|---|---|---|
| 动量类 | `mom_20 = close/close.shift(20) - 1` | 20 日涨幅 |
| 动量类 | `mom_60 = close/close.shift(60) - 1` | 60 日涨幅 |
| 反转类 | `IBS = (close-low)/(high-low)` | 日内位置；越接近 1 越超买 |
| 反转类 | `Williams%R(14)` | 14 日相对位置 |
| 反转类 | `rev_5 = -(close/close.shift(5) - 1)` | 5 日反向涨幅 |
| Bias | `trend_up = EMA9 > EMA21` | 短均线趋势方向 |

### 3. 横截面打分（每日）

每个信号在 100 只股票上做 rank，归一到 [-1, +1]：

```
momentum_block = mean(rank(mom_20), rank(mom_60))
reversal_block = mean(-rank(IBS), -rank(-Williams%R), rank(rev_5))
bias           = (trend_up - 0.5) × 2          # ∈ {-1, +1}
composite      = 0.7 × momentum_block + 0.3 × reversal_block + 0.2 × bias
```

`composite` 越大越想做多。NAS100 是趋势市场，实验证明动量权重 0.7 显著优于 0.5。

### 4. 选股

每日按 composite 排序：

- **Top K_LONG (默认 8) 做多**
- **Bottom K_SHORT 做空**（默认 0，即纯多头；多空版可设为 4 等）

### 5. Hysteresis（滞后带）— 降低换手的核心

如果每日"top 8 才进、不在 top 8 就出"，换手非常高。改成：

- **新开仓**：composite 进入 top K_LONG 才开
- **维持持仓**：composite 仍在 top (HYSTERESIS_MULT × K_LONG) 内就保留（默认 4×K = top 32）
- 跌出 32 名才平仓

这能把日均换手从约 50% 降到 8% 左右。

### 6. 大盘 Regime 过滤（熊市保护）

`REGIME_FILTER=True` 时，每日计算 SPY 200 日均线：

- SPY > 200DMA：允许开新多头
- SPY < 200DMA：暂停开新多头（已持仓继续按信号/止损管理）

效果：2022 熊市跌幅从 -34.7% 压到 -12.8%（QQQ 同期 -33.2%）。

### 7. 波动率目标仓位

`VOL_TARGET_ANNUAL=0.20` 时，根据近 20 日组合实际波动调节仓位规模：

```
realized_vol = std(daily_returns_last_20d) × sqrt(252)
vol_scale    = clip(0.20 / realized_vol, 0.3, 2.0)
position_size = base_weight × vol_scale × equity
```

高波动期（如 2022Q1）自动减仓最多到 30%，低波动期最多放大到 200%。
效果：6 年 Sharpe 从 0.89 提升到 1.18，最大回撤从 -43% 降到 -26%。

### 8. 风控

- **个股止损**：`stop = entry × (1 - max(5%, 1.5 × ATR14/entry))`
  - 5% 是地板，避免低波动股票止损被压得太紧
  - ATR 倍数让高波动股票有更宽容忍度
- **最长持仓**：20 个交易日（`MAX_HOLD_DAYS`），到期强平
- **三类出场**：
  1. `stop_loss`：触发止损价
  2. `max_hold`：到期持仓
  3. `signal_exit`：composite 跌出 hysteresis 带

### 9. 仓位分配

- **总杠杆** `GROSS_LEVERAGE = 1.0`（满仓不加杠杆）
- **多头权重** `LONG_WEIGHT_FRAC × GROSS = 100%` of gross（默认纯多头）
- **每只权重** = `100% / K_LONG = 12.5%` × `vol_scale`
- $1M 本金例：每只多头开仓金额 ≈ $80k–$250k（视 vol_scale）

### 10. 交易成本（Longport 美股口径）

| 项 | 费率 | 计费方 |
|---|---|---|
| 佣金 | $0 | — |
| 平台费 | $0.005/股，每单最低 $1 | 双边 |
| SEC 费 | 0.0000278% × notional | 仅卖出 |
| TAF | $0.000166/股，每单最高 $8.3 | 仅卖出 |
| **滑点（估算）** | **5 bps/侧**（默认；NAS100 大盘股流动性好可调到 2 bps） | 双边 |

---

## 实测结果

### 2020-01 至 2026-05（6.33 年）含 2020 疫情、2022 熊市、2023-25 AI 牛市

| 配置 | CAGR | Sharpe | MDD | 终值 | vs QQQ |
|---|---|---|---|---|---|
| 理想（无成本无滑点） | 28.45% | 1.18 | -26% | $4.88M | +7.4% ✅ |
| **2 bps 滑点 + Longport 平台费** | 14.65% | 0.79 | -37% | $2.38M | -6.4% ❌ |
| **5 bps 滑点 + Longport 平台费**（默认） | 11.48% | 0.67 | -30% | $1.99M | -9.5% ❌ |
| QQQ 买入持有 | 21.01% | 0.89 | -35% | $3.34M | — |

### 2022 熊市（regime filter 是核心保护）

| 指标 | 策略（含成本） | QQQ |
|---|---|---|
| 全年收益 | **-11.28%** | -33.22% |
| MDD | -14.23% | -34.83% |
| 交易笔数 | 59（regime 关掉了大部分新仓） | — |

---

## 关键发现与诚实结论

1. **策略本身有 alpha**：理想条件下 6 年 CAGR 28.45%，比 QQQ 多 7.4%。
2. **但高换手吞噬 alpha**：6 年 ~900 笔交易，5 bps 滑点 + 平台费总成本约 $170k = 17% drag。净化后 CAGR 跌到 11.48%，**跑不赢 QQQ buy-and-hold**。
3. **熊市保护非常实**：含成本下 2022 仅 -11.28%（QQQ -33.22%），这是策略最大价值所在。
4. **不是免费午餐**：单独看 6 年总收益 11.48% < QQQ 21%，但 Sharpe 0.67 vs 0.89 差距比想象小，且回撤更可控。
5. **下一步优化方向**（按 ROI 排序）：
   - **降低换手**：把 `HYSTERESIS_MULT` 从 4 调到 6 或 8，或改为周频再平衡 → 直接对冲滑点 drag
   - **提高 K_LONG** 到 15+：分散 → 单笔位置变小 → 滑点占比降低
   - **加入更弱相关的因子**（基本面、情绪）增厚 alpha 来扛得起成本
   - **参数滚动训练**：mom_weight 在熊市可能要降回 0.5

---

## 交易时点 / 实盘对齐（重要）

回测的执行语义是 **MOC（Market On Close，按官方收盘价成交）**：

```
T 日时序：
  15:30 ET (盘前 30min)  根据接近收盘的实时价计算信号、composite 排名、止损价
  15:45 ET               提交本日所有的 MOC（开仓/平仓）订单
  16:00 ET 官方收盘       订单按官方 close 成交  ← 回测里 entry_price/exit_price 用此值
  16:00 ET 之后          已持仓的止损价提交为下一交易日的 GTC 止损单
T+1 起：
  盘中任意时刻          如果 low ≤ stop（多）/ high ≥ stop（空），止损单触发成交
```

**实盘要做到与回测一致**，按以下步骤：

1. **每日 15:30 ET**（或夏令时 15:30 EST）跑 `python backtest.py` 的"今日决策"模式（待加，目前回测模式跑全期）
2. **15:45 ET 前提交 MOC 订单**：
   - 平仓单：当前持仓里 composite 跌出 hysteresis 带（默认 top 32 外）的全部
   - 开仓单：composite 在新 top 8 内、当前未持有的
3. **盘后下次日的止损单**：每只持仓提交价格为 `entry × (1 - max(5%, 1.5×ATR/entry))` 的 GTC stop-loss
4. **持仓满 20 个交易日强平**：到期日提交 MOC 平仓

**已知差距（回测略乐观）**：

- 止损成交价：回测用 `today_close` 近似，实盘用 `stop_price`。如果 stop 在盘中触发后股价继续下跌，实盘会比回测**亏更多**；反弹收高则**赚更少**。整体偏中性，但波动较大的股票上偏差更大。
- MOC 滑点：回测默认 5 bps/侧，已偏保守。NAS100 大盘股 MOC 实际滑点常在 1–3 bps，可在 [`backtest.py`](backtest.py) 把 `SLIPPAGE_BPS` 调到 2 看更真实情况。
- 数据时滞：回测用日 K 收盘价计算指标，实盘需在 15:30 ET 用接近收盘的实时价代替——这点很关键，否则就是"未来函数"（看到 16:00 才能确定的 close 然后下 16:00 的单）。

**不推荐的实盘方式（与回测不一致，会引入额外噪声）**：

- ❌ T 日盘后看 close 算信号，T+1 日开盘下市价单 → 引入隔夜跳空风险（回测不反映）
- ❌ T 日盘中任意时刻随机执行 → 信号与价格脱钩

---

## 项目结构

```
.
├── backtest.py              # 策略与回测主程序（入口）
├── nas100_universe.py       # NAS100 成分股清单
├── longport_api.py          # Longport 日线数据接口
├── daily_cache.py           # 日线 CSV 本地缓存
├── data_cache/daily/        # 缓存数据（自动维护）
├── requirements.txt
├── .env                     # Longport 凭证（不要提交）
└── README.md
```

## 配置项一览（[`backtest.py`](backtest.py) 顶部）

```python
BACKTEST_START = "2020-01-01"        # 回测起点
BACKTEST_END   = "today"
STARTING_CAPITAL = 1_000_000

K_LONG  = 8                          # 多头持仓数
K_SHORT = 0                          # 空头持仓数（0 = 纯多头）
LONG_WEIGHT_FRAC = 1.0               # 多头占 gross 比例
GROSS_LEVERAGE = 1.0

MOM_WEIGHT  = 0.7                    # 动量信号权重
BIAS_WEIGHT = 0.2                    # EMA 趋势 bias

HYSTERESIS_MULT = 4.0                # 持仓退出 top (4×K) 才平仓

STOP_LOSS_PCT      = 0.05            # 止损 = max(5%, 1.5×ATR)
STOP_LOSS_ATR_MULT = 1.5
MAX_HOLD_DAYS      = 20
MIN_DOLLAR_VOLUME  = 5e7

REGIME_FILTER = True                 # SPY 200DMA 大盘过滤

VOL_TARGET_ANNUAL = 0.20             # 目标年化波动；0=关
VOL_TARGET_LOOKBACK = 20
VOL_SCALE_MIN = 0.3
VOL_SCALE_MAX = 2.0

ENABLE_COSTS         = True          # 交易成本与滑点开关
PLATFORM_FEE_PER_SHARE = 0.005       # Longport 平台费
PLATFORM_FEE_MIN     = 1.0
SEC_FEE_RATE         = 0.0000278
TAF_PER_SHARE        = 0.000166
TAF_MAX_PER_ORDER    = 8.3
SLIPPAGE_BPS         = 5.0           # 单侧滑点；NAS100 流动性好可调到 2

VERBOSE_TRADES = True                # 控制台打印每笔交易
PRINT_DAILY_POSITIONS = False
```
