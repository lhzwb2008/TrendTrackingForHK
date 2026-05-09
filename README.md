# nas100-quant

NAS100 短中线动量策略（**分钟级日内决策**，含 regime 过滤与波动率目标仓位）

一个基于 NAS100 成分股的横截面打分日频选股回测系统。每个交易日在**美东 15:50** 用「日线历史 + 当日截至 15:50 的分钟数据」计算横截面 composite 分数，挑前 8 名做多，配大盘 regime 过滤与波动率目标仓位，挂止损。回测与实盘**使用同一份决策逻辑、同一个时点**——可直接对接 Longport 实盘。

---

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 在 .env 中填入 Longport 凭证
# LONGPORT_APP_KEY=...
# LONGPORT_APP_SECRET=...
# LONGPORT_ACCESS_TOKEN=...

# 3. 双模式回测：DAILY (2020+) + INTRADAY (2024-05+) 同时跑
python backtest.py
```

> 首次运行会拉取约 100 只股票的 2 年 5-min K 线，约 5–10 分钟，之后走本地缓存秒级加载。

---

## 关键改动（2026-05 版）

之前的版本用日 K 收盘价作为决策与成交价，存在**前瞻偏差**——实盘里你拿到 close 时已经无法下单。本版本切换为：

1. **数据**：日线 + 5-min K（Longport 分钟数据回溯上限 ~2 年，所以默认起点 2024-05-08）
2. **决策时点**：每天美东 15:50（参数 `DECISION_TIME_ET`），收盘前 10 分钟
3. **决策价 = 成交价**：用 15:45-15:50 那根 5-min bar 的收盘价。实盘 15:50 跑脚本，2-3 分钟内提交订单完全来得及
4. **止损**：精确到 5 分钟级别。若日内 `pd_low ≤ stop_price` 则按 `stop_price` 成交；若开盘 `gap_open` 已穿透 stop，则按 `gap_open` 成交（模拟跳空真实损失）

---

## 策略原理

### 1. 股票池

NAS100 当期成分股共 100 只（[`nas100_universe.py`](nas100_universe.py)）。
每天动态过滤：20 日平均成交额 ≥ $50M，且当日有完整 OHLC + 分钟数据。

### 2. 信号

每只股票每天计算 6 个信号：

| 类别 | 信号 | 含义 |
|---|---|---|
| 动量 | `mom_20 = pd_close / close_20d_ago - 1` | 20 日涨幅（分子用决策时点价） |
| 动量 | `mom_60 = pd_close / close_60d_ago - 1` | 60 日涨幅 |
| 反转 | `IBS = (pd_close - pd_low) / (pd_high - pd_low)` | 日内位置 |
| 反转 | `Williams%R(14)` | 14 日相对位置 |
| 反转 | `rev_5 = -(pd_close / close_5d_ago - 1)` | 5 日反向涨幅 |
| Bias | `EMA9 > EMA21`（用前一日值，避免日内偏差） | 短均线趋势 |

> `pd_close` / `pd_high` / `pd_low` 是当日截至决策时点（默认 15:50）的累计 OHLC，由 5-min bar 聚合而来。
> 历史日的指标仍用真实日 K 全天值——只有"今天"这一行会用决策时点的"代理值"覆盖重算。

### 3. 横截面打分（每日）

每个信号在 100 只股票上做 rank，归一到 [-1, +1]：

```
momentum_block = mean(rank(mom_20), rank(mom_60))
reversal_block = mean(-rank(IBS), -rank(-Williams%R), rank(rev_5))
bias           = (trend_up - 0.5) × 2          # ∈ {-1, +1}
composite      = 0.7 × momentum_block + 0.3 × reversal_block + 0.2 × bias
```

`composite` 越大越想做多。NAS100 是趋势市场，动量权重 0.7 优于 0.5。

### 4. 选股 + Hysteresis 滞后带

每日按 composite 排序：

- **新开仓**：composite 进入 top **K_LONG (默认 8)** 才开
- **维持持仓**：composite 仍在 top **(HYSTERESIS_MULT × K_LONG) = 32** 内就保留
- **跌出 top 32 才平仓**

这把日均换手从约 50% 降到 8%。`K_SHORT = 0` 时为纯多头（默认）。

### 5. 大盘 Regime 过滤

`REGIME_FILTER=True` 时，每日计算 SPY 200 日均线：

- SPY > 200DMA：允许开新多头
- SPY < 200DMA：暂停开新多头（已持仓继续按信号/止损管理）

历史回测显示 2022 熊市该过滤把回撤从 -34.7% 压到 -12.8%。

### 6. 波动率目标仓位

`VOL_TARGET_ANNUAL=0.20` 时，按近 20 日组合实际波动调节仓位规模：

```
realized_vol  = std(daily_returns_last_20d) × sqrt(252)
vol_scale     = clip(0.20 / realized_vol, 0.3, 2.0)
position_size = (gross / K_LONG) × vol_scale × equity
```

高波动期自动减仓最多到 30%，低波动期最多放大到 200%。

### 7. 风控

- **个股止损**：`stop = entry × (1 - max(5%, 1.5 × ATR14/entry))`（多头；空头对称）
- **最长持仓**：20 个交易日，到期强平
- **三类出场**：
  1. `stop_loss`：日内触发止损（按 `stop_price` 或 `gap_open` 成交）
  2. `max_hold`：到期持仓
  3. `signal_exit`：composite 跌出 hysteresis 带

### 8. 仓位分配

- **总杠杆** `GROSS_LEVERAGE = 1.0`（满仓不加杠杆）
- **每只权重** = `100% / K_LONG = 12.5%` × `vol_scale`
- $1M 本金例：每只多头开仓 ≈ $40k–$250k

### 9. 交易成本（Longport 美股口径）

| 项 | 费率 | 计费方 |
|---|---|---|
| 佣金 | $0 | — |
| 平台费 | $0.005/股，每单最低 $1 | 双边 |
| SEC 费 | 0.0000278% × notional | 仅卖出 |
| TAF | $0.000166/股，每单最高 $8.3 | 仅卖出 |
| 滑点 | **5 bps/侧**（默认） | 双边 |

---

## 每日执行时序（回测 = 实盘）

```
T 日：
  09:30 ET   美股开盘
  ─────── Phase A ───────
  09:30 → 15:50：监控存量持仓
    ├─ 任一时刻 low ≤ stop（多头）/ high ≥ stop（空头） → 触发止损
    │   - 一般情况：按 stop_price 成交（GTC stop 限价单）
    │   - 极端跳空：开盘已穿透 stop → 按 gap_open 成交（市价被动接受）
    └─ 未触发的持仓继续持有
  
  ─────── Phase B (15:50 决策点) ───────
  15:50 ET   单一决策时点
    1. 用「日 K 历史 + 今日 09:30~15:50 的 5-min 聚合」算 composite
    2. 平仓（在 15:50 价立即成交）：
       - 持仓满 20 天 → max_hold
       - 跌出 top 32 → signal_exit
       - regime 翻转且持仓方向相反 → signal_exit
    3. 开仓（在 15:50 价立即成交）：
       - top 8 中尚未持有的标的 → 开多
       - 同时挂当日及次日的 GTC stop-loss 限价单
  
  ─────── Phase C ───────
  15:50 → 16:00：剩余 10 分钟
    存量持仓继续持有，价格波动正常 MTM 至 16:00 真收盘

T+1 日：重复上面流程
```

**实盘对应步骤**：

1. 服务器配定时任务，每日美东 **15:50** 跑 `python backtest.py`（或专门的实盘单日决策脚本，待加）
2. 脚本输出今日的开/平仓清单与 stop 价
3. 立即通过 Longport `submit_order` 提交：
   - 平仓：限价单贴近 last（10 分钟内成交）
   - 开仓：限价单贴近 last + 同步挂 GTC stop-loss
4. 16:00 前确认所有订单已成交；未成的 IOC 可以追到收盘

回测和实盘**完全用同一份决策逻辑、同一个时点**，没有任何"未来函数"。

---

## 双模式回测

`backtest.py` 一次同时跑两个：

| 模式 | 起点 | 数据 | 决策/成交价 | 用途 |
|---|---|---|---|---|
| **INTRADAY** | 2024-05-08 | 5-min K | 当日 15:50 ET | 模拟实盘可执行的真实回测 |
| **DAILY** | 2020-01-01 | 日 K | 当日收盘价（理论参考） | 跨牛熊验证策略稳健性（含 2022 熊市） |

DAILY 模式因为没有分钟数据，决策/成交都是同一根日 K 收盘——这相当于"完美信息"基线，主要看年度形态、回撤、是否能扛过熊市，**不用作可执行收益估计**；INTRADAY 模式才代表实盘水平。

---

## 实测结果

### 默认参数（v1：k_long=8, max_hold=20）

| 指标 | INTRADAY (24-05~26-05) | DAILY (20-01~26-05) |
|---|---|---|
| CAGR | **+40.15%** | +18.83% |
| Sharpe | **+1.44** | +0.94 |
| MDD | -17.29% | -39.93% |

DAILY 的 -39.9% MDD 主要来自 2022 熊市，胜在能扛过来；2024 起切换到 INTRADAY 后明显改善。

### 参数扫描结论（25 组随机抽样 × DAILY + INTRADAY 双段评分）

抽样空间：`k_long ∈ {6,8,10,12,15}`、`mom_weight ∈ {0.5,0.6,0.7,0.8}`、`bias_weight ∈ {0,0.1,0.2,0.3}`、`hysteresis_mult ∈ {2.5,3,4,5,6}`、`stop_loss_pct ∈ {4,5,6,8}%`、`stop_loss_atr_mult ∈ {1,1.5,2,2.5,3}`、`max_hold_days ∈ {20,30,40,60,80,120}`。

排名前 3（按 min(DAILY Sharpe, INTRADAY Sharpe)，越高越说明两段都稳）：

| 组合 | k_long | mom_w | bias_w | hyst | sl% | atr | hold | DAILY CAGR/Sh/MDD | INTRADAY CAGR/Sh/MDD |
|---|---|---|---|---|---|---|---|---|---|
| **#20** | 10 | 0.8 | 0.1 | 5.0 | 4 | 2.0 | 80  | +27.3% / 1.15 / -21.0% | +36.9% / 1.35 / -23.9% |
| **#16** | 10 | 0.8 | 0.3 | 4.0 | 5 | 1.5 | 80  | +25.1% / 1.10 / -24.2% | +40.3% / 1.40 / -21.7% |
| #17     | 12 | 0.5 | 0.0 | 2.5 | 5 | 1.5 | 120 | +20.9% / 1.08 / -27.3% | +32.3% / 1.38 / -16.4% |
| DEFAULT |  8 | 0.7 | 0.2 | 4.0 | 5 | 1.5 | 20  | +18.8% / 0.94 / -39.9% | +40.2% / 1.44 / -17.3% |

**关键发现**：
1. **`max_hold_days=20` 确实偏紧**：放宽到 60-120 时 DAILY 表现普遍更好（少触发 max_hold 强平错杀）。
2. **k_long 适度增大（10-12）+ hysteresis_mult≥4** 更稳：分散度↑、换手↓、Sharpe ↑。
3. **mom_weight=0.8、bias_weight 0.1-0.3** 略优于默认 0.7/0.2，但差异有限。
4. **止损 4-5% + ATR 1.5-2.0** 区间内表现都接近，过宽（8%）反而拖累 INTRADAY。
5. INTRADAY Sharpe 单项最高 1.63（#13: k=10, mom=0.5, hyst=2.5, sl=6%, atr=2.5, hold=40），但 DAILY Sharpe 只有 1.02——单点过拟合 2024-26 牛市的风险更大。

**推荐**：把 DEFAULT 升级为 **#16 或 #20**，DAILY MDD 从 -40% 收敛到 -24%/-21%，INTRADAY 几乎无损。最保守起见可只改 `MAX_HOLD_DAYS = 60` 这一项，DAILY/INTRADAY 都能受益。

> 25 组随机抽样统计意义有限（且 INTRADAY 区间正好踩在牛市），**Sharpe ~1.0 / CAGR 15-25% 才是长期合理预期**；超过这个数都要警惕 lucky regime。

---

## 项目结构

```
.
├── backtest.py              # 策略与双模式回测主程序（DAILY + INTRADAY）
├── nas100_universe.py       # NAS100 成分股清单（含中文名映射）
├── longport_api.py          # Longport 日线数据接口
├── intraday_api.py          # Longport 分钟级数据接口（含 RTH 过滤、HKT 时区修正）
├── daily_cache.py           # 日线 CSV 本地缓存
├── data_cache/
│   ├── daily/               # 日线 CSV
│   └── intraday/5min/       # 分钟 parquet
├── requirements.txt
├── .env                     # Longport 凭证（不要提交）
└── README.md
```

---

## 关键参数（[`backtest.py`](backtest.py) 顶部）

```python
# 回测窗口（双模式）
DAILY_START   = "2020-01-01"         # DAILY 模式起点（跨牛熊参考）
INTRA_START   = "2024-05-08"         # INTRADAY 模式起点；分钟数据上限 ~2 年
BACKTEST_END  = "today"
STARTING_CAPITAL = 1_000_000

# 日内决策（核心新增）
INTRADAY_PERIOD  = "5min"            # 1min / 5min / 15min / 30min / 60min
DECISION_TIME_ET = "15:50"           # 美东时间 HH:MM；收盘前留 10 分钟下单

# 持仓结构
K_LONG  = 8
K_SHORT = 0
LONG_WEIGHT_FRAC = 1.0
GROSS_LEVERAGE = 1.0

# 信号
MOM_WEIGHT  = 0.7                    # 动量/反转权重
BIAS_WEIGHT = 0.2

# 滞后带
HYSTERESIS_MULT = 4.0                # 跌出 top (4×K) 才平仓

# 风控
STOP_LOSS_PCT      = 0.05
STOP_LOSS_ATR_MULT = 1.5
MAX_HOLD_DAYS      = 20
MIN_DOLLAR_VOLUME  = 5e7
REGIME_FILTER      = True

# 波动率目标
VOL_TARGET_ANNUAL   = 0.20
VOL_TARGET_LOOKBACK = 20
VOL_SCALE_MIN       = 0.3
VOL_SCALE_MAX       = 2.0

# 成本
ENABLE_COSTS           = True
PLATFORM_FEE_PER_SHARE = 0.005
PLATFORM_FEE_MIN       = 1.0
SEC_FEE_RATE           = 0.0000278
TAF_PER_SHARE          = 0.000166
TAF_MAX_PER_ORDER      = 8.3
SLIPPAGE_BPS           = 5.0
```

---

## 已知限制与注意事项

1. **分钟数据上限 2 年**：Longport `history_candlesticks_by_date` 在分钟周期下仅回溯到约 2024-05；早于此抛 `301600 out of minute kline begin date`。
2. **时区**：Longport 分钟 K 的 timestamp 是 **HKT (UTC+8) 但 tz-naive**，[`intraday_api.py`](intraday_api.py) 会显式 localize 后转 UTC，再转 ET 过滤 RTH。已封装好，无需关心。
3. **首次拉数据耗时**：100 股 × 2 年 × 5min ≈ 4M 根 bar，Longport 限速下首次约 5-10 分钟。之后增量缓存只补当天最新数据。
4. **Sharpe 训练→验证 大幅提升**不代表泛化好，反而要警惕 lucky regime：验证期是趋势市，长期 Sharpe ~1.0 才是合理预期。
5. **止损价的近似性**：`pd_low ≤ stop_price` 触发后假设按 `stop_price` 成交。极端波动股票（如 LCID、TTD）实盘可能比回测亏更多，gap-down 已用 `gap_open` 兜底。
