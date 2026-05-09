# crossrank-quant

美股**横截面排名（Cross-sectional Rank）选股**策略，**分钟级日内决策**，回测与实盘**完全同时点同逻辑**。

每个交易日美东 **15:50** 用「日 K 历史 + 当日截至 15:50 的分钟数据」算横截面 composite 分数，从 NAS100 ∪ S&P 500（~516 只）的池子里挑前 10 名做多，叠加 SPY 200DMA regime 过滤、波动率目标仓位与 max(5%, 1.5×ATR) 个股止损。回测产生的开/平仓清单可以直接对接 Longport 实盘。

> 项目曾用名 `nas100-quant`；初版仅用 NAS100 (101 只) 池子，扩展到 SP500 后 DAILY Sharpe 从 1.12 提升到 1.51（详见附录 C），现已升级为默认。

---

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 在 .env 中填入 Longport 凭证
# LONGPORT_APP_KEY=...
# LONGPORT_APP_SECRET=...
# LONGPORT_ACCESS_TOKEN=...

# 3. 双模式回测（DAILY 跨牛熊 + INTRADAY 实盘对齐，一次跑完）
python backtest.py
```

> **首次运行耗时**：日线 ~5-10 分钟（516 只 × 6.3 年）；分钟数据 **~1-2 小时**（516 只 × 2 年 5-min ≈ 20M 根 bar，受 Longport 限速）。之后走本地缓存（`data_cache/`）秒级加载。
>
> 想先快速验证策略代码，可临时把 `INTRA_START` 改到近 1 个月（如 `"2026-04-01"`），10 分钟内跑完两段；确认无误后再改回 `"2024-05-08"` 拉完整 2 年分钟数据。

---

## 双模式回测

`backtest.py` 一次同时跑两个互补的回测：

| 模式 | 区间 | 数据 | 决策 / 成交 | 用途 |
|---|---|---|---|---|
| **INTRADAY** | 2024-05-08 ~ today | 5-min K | 当日 15:50 ET（同价） | 与实盘 1:1 对齐，给出可执行收益 |
| **DAILY** | 2020-01-01 ~ today | 日 K | 当日 close（同价） | 跨牛熊压力测试，含 2022 熊市验证稳健性 |

DAILY 模式因为没有分钟数据，决策点和成交点都用同一根日 K 收盘——这是"完美信息"理论基线，**不可外推为可执行收益**，但年度形态、回撤、能否扛过熊市都是真实的；INTRADAY 才代表实盘能拿到的水平。

---

## 策略原理

### 1. 股票池

NAS100 ∪ S&P 500 静态合集（[`universe.py`](universe.py)）共 **516 只**。每天动态过滤：20 日平均成交额 ≥ $50M、当日 OHLC + 分钟数据完整。也提供 `get_nas100_universe()` 仅返回 101 只 NAS100 子集，用于做对照实验。

> Longport 暂不支持类股代号（BF-B、BRK-B），已在 universe 里预先剔除。

### 2. 信号（每只股票每天 6 个）

| 类别 | 信号 | 含义 |
|---|---|---|
| 动量 | `mom_20 = pd_close / close_20d_ago - 1` | 20 日涨幅（分子用决策时点价） |
| 动量 | `mom_60 = pd_close / close_60d_ago - 1` | 60 日涨幅 |
| 反转 | `IBS = (pd_close - pd_low) / (pd_high - pd_low)` | 当日相对位置 |
| 反转 | `Williams%R(14)` | 14 日相对位置 |
| 反转 | `rev_5 = -(pd_close / close_5d_ago - 1)` | 5 日反向涨幅 |
| Bias | `EMA9 > EMA21`（用前一日值） | 短均线趋势 |

> `pd_close` / `pd_high` / `pd_low` 是当日 09:30 ~ 决策时点的累计 OHLC，由 5-min bar 聚合而来。历史日的指标用真实日 K 全天值，只有"今天"这一行被决策时点的代理值覆盖重算——彻底消除前瞻偏差。

### 3. 横截面打分（每日）

每个信号在股票池里做 rank，归一到 [-1, +1]：

```
momentum_block = mean(rank(mom_20), rank(mom_60))
reversal_block = mean(-rank(IBS), -rank(-Williams%R), rank(rev_5))
bias           = (trend_up - 0.5) × 2          # ∈ {-1, +1}
composite      = 0.8 × momentum_block + 0.2 × reversal_block + 0.3 × bias
```

`composite` 越大越想做多。趋势市场动量权重 0.8 略优于 0.5/0.7。

### 4. 选股 + Hysteresis 滞后带

- **新开仓**：composite 进入 top **K_LONG (=10)** 才开
- **维持持仓**：仍在 top **(HYSTERESIS_MULT × K_LONG) = 40** 内就保留
- **跌出 top 40 才平仓**

把日均换手压到 ~8%，显著节省成本。`K_SHORT = 0` 为纯多头（默认）。

### 5. 大盘 Regime 过滤

每日计算 SPY 200DMA：

- SPY > 200DMA：允许开新多头
- SPY < 200DMA：暂停开新多头（已持仓继续按信号 / 止损管理）

历史回测显示 2022 熊市该过滤把组合回撤压到 -11.8%、全年 -6.5%（同期 QQQ -33%）。

### 6. 波动率目标仓位

按近 20 日组合实际波动调节仓位规模：

```
realized_vol  = std(daily_returns_last_20d) × sqrt(252)
vol_scale     = clip(0.20 / realized_vol, 0.3, 2.0)
position_size = (gross / K_LONG) × vol_scale × equity
```

高波动期自动减仓（最多压到 30%），低波动期最多放大到 200%。

### 7. 风控

- **个股止损**：`stop = entry × (1 - max(5%, 1.5 × ATR14/entry))`，多头；空头对称
- **最大持仓**：80 个交易日到期强平
- **三类出场**：
  1. `stop_loss`：日内触发止损（按 `stop_price` 或 `gap_open` 成交）
  2. `max_hold`：到期持仓
  3. `signal_exit`：composite 跌出 hysteresis 带

### 8. 仓位分配

- 总杠杆 1.0（满仓不加杠杆）
- 每只权重 = `100% / 10 = 10%` × `vol_scale`
- 默认 $100,000 本金：每只多头开仓 ≈ $3k - $20k

### 9. 交易成本（Longport 美股口径）

| 项 | 费率 | 计费方 |
|---|---|---|
| 佣金 | $0 | — |
| 平台费 | $0.005/股，每单最低 $1 | 双边 |
| SEC 费 | 0.0000278% × notional | 仅卖出 |
| TAF | $0.000166/股，每单最高 $8.3 | 仅卖出 |
| 滑点 | 5 bps/侧（默认） | 双边 |

---

## 收益来源（为什么这个策略能赚）

1. **横截面动量**（mom_20 + mom_60，权重 0.8）
   美股大盘股有显著**强者恒强**结构——20/60 日跑赢同侪的股票，下个月平均仍跑赢。这是组合主要的 alpha 来源。扩到 SP500 后中盘成长（CVNA、SMCI、APP、PLTR、SNDK）贡献了大量 outsized winners。

2. **横截面反转作为补丁**（IBS / W%R / 5 日反转，权重 0.2）
   纯动量在拐点附近吃亏。三个反转因子在动量逻辑之上挑"被错杀的"，对短期回吐起到对冲作用。

3. **EMA Bias 加权**（权重 0.3）
   只在短均线已经向上的标的里挑——避免抄底正在掉的"低 IBS"。

4. **Regime 过滤防熊市**
   2022 那种系统性下跌里，横截面 alpha 被 beta 完全吞掉。SPY 200DMA 一旦跌破就停止开仓，把单年最差从 -25% 量级压到 -7% 量级。

5. **波动率目标控制回撤**
   组合 vol 高了自动减仓，能把 max drawdown 收敛到比 buy-and-hold 一半还低。

6. **Hysteresis 减摩擦**
   横截面排名每天都在抖动，不加滞后带 turnover 50% 起步，年成本 8-12% 直接吞掉一半 alpha。滞后到 4×K 后 NAS100 段年成本降到 ~3-5%（INTRADAY 模式 NAS100 实测 2.98%）；池子扩到 516 后换手翻倍、年成本约 30%（仍能净盈利）。

---

## 每日执行时序（回测 = 实盘）

```
T 日：
  09:30 ET   美股开盘
  ─────── Phase A ───────
  09:30 → 15:50：监控存量持仓
    任一时刻 low ≤ stop（多头）/ high ≥ stop（空头） → 触发止损
      ├─ 一般情况：按 stop_price 成交（GTC stop 限价单）
      └─ 极端跳空：开盘已穿透 stop → 按 gap_open 成交（被动接受）
    未触发的持仓继续持有

  ─────── Phase B (15:50 决策点) ───────
  15:50 ET   单一决策时点
    1. 用「日 K 历史 + 今日 09:30~15:50 的 5-min 聚合」算 composite
    2. 平仓（在 15:50 价立即成交）：
       - 持仓满 80 天 → max_hold
       - 跌出 top 40 → signal_exit
       - regime 翻转且持仓方向相反 → signal_exit
    3. 开仓（在 15:50 价立即成交）：
       - top 10 中尚未持有的标的 → 开多
       - 同步挂当日及次日的 GTC stop-loss 限价单

  ─────── Phase C ───────
  15:50 → 16:00：剩余 10 分钟
    存量持仓继续持有，价格波动正常 MTM 至 16:00 真收盘

T+1 日：重复
```

**实盘对应**：

1. 服务器配定时任务，每日美东 **15:50** 跑 `python backtest.py`（或单日决策脚本，待加）
2. 输出今日开 / 平仓清单与 stop 价
3. 立即通过 Longport `submit_order` 提交：
   - 平仓：限价单贴近 last（10 分钟内成交）
   - 开仓：限价单贴近 last + 同步挂 GTC stop-loss
4. 16:00 前确认全部成交；未成的可 IOC 追到收盘

---

## 实测结果（默认参数，起始本金 $100,000）

### DAILY 模式（2020-01 ~ 2026-05，6.33 年，跨牛熊）

| 指标 | us-equity (NAS100 ∪ SP500, 516) | NAS100 子集 (101) | QQQ 基准 |
|---|---|---|---|
| 累计收益 | **+651.14%** | +323.33% | +241.92% |
| CAGR | **37.49%** | 25.59% | 21.44% |
| Sharpe | **1.51** | 1.12 | 0.90 |
| 最大回撤 | -25.23% | -24.22% | -35.12% |
| Calmar | 1.49 | 1.06 | — |
| 胜率 | 42.8% | 38.9% | — |
| 平均持仓天数 | 9.6 | 16.9 | — |
| 总交易笔数 | 1,303 | 733 | — |
| 总交易成本 | $30,240 (30.24%) | $14,414 (14.41%) | — |

**池子扩到 516 后**：Sharpe **+0.39**、CAGR **+12pp**、累计收益翻倍，MDD 仅恶化 1pp。代价是换手翻倍、年成本从 14% → 30%（仍能净盈利）。SP500 中的中盘成长股提供了大量 outsized winners（SNDK +160%、CVNA +240%、APP +152%、SMCI +123%、COIN +63%、PLTR +90%、LITE +74% …）。

### INTRADAY 模式（2024-05 ~ today，~2 年，实盘对齐）

> 当前数据为 NAS100 子集旧结果；扩展到 516 只的 INTRADAY 需首次拉 ~1-2 小时分钟数据，本地跑完后会更新。

| 指标 | NAS100 子集 (101) | QQQ 基准 |
|---|---|---|
| 累计收益 | +96.19% | +63.40% |
| CAGR | 40.26% | 28.02% |
| Sharpe | 1.40 | 1.27 |
| 最大回撤 | -21.71% | -22.77% |
| 胜率 | 33.9% | — |
| 总交易笔数 | 230 | — |
| 总交易成本 | $2,983 (2.98%) | — |

### DAILY 逐年分解（516 只 universe）

| 年份 | 收益 | MDD | Sharpe | 备注 |
|---|---|---|---|---|
| 2020 | 强劲 | — | — | 疫情 V 反弹 |
| 2021 | 正收益 | — | — | 牛市 |
| **2022** | **熊市保护** | **小幅** | — | **regime 过滤救命，QQQ -33%** |
| 2023-2024 | 持续正收益 | 中等 | — | 震荡中跑赢 |
| 2025-2026 | 加速 | 较小 | — | 强趋势期，**不可外推** |

> 长期合理预期仍是 **Sharpe ~1.0-1.3、CAGR 20-30%**；显著超过此区间的样本（如 2026 YTD）大概率是 lucky regime。

---

## 关键参数（[`backtest.py`](backtest.py) 顶部）

```python
# 回测窗口（双模式）
DAILY_START   = "2020-01-01"         # DAILY 起点（跨牛熊参考）
INTRA_START   = "2024-05-08"         # INTRADAY 起点（分钟数据上限 ~2 年）
BACKTEST_END  = "today"
STARTING_CAPITAL = 100_000

# 日内决策
INTRADAY_PERIOD  = "5min"            # 1min / 5min / 15min / 30min / 60min
DECISION_TIME_ET = "15:50"           # 美东 HH:MM；收盘前 10 分钟

# 持仓结构
K_LONG  = 10
K_SHORT = 0
GROSS_LEVERAGE = 1.0

# 信号
MOM_WEIGHT  = 0.8
BIAS_WEIGHT = 0.3

# 滞后带 / 风控
HYSTERESIS_MULT     = 4.0
STOP_LOSS_PCT       = 0.05
STOP_LOSS_ATR_MULT  = 1.5
MAX_HOLD_DAYS       = 80
MIN_DOLLAR_VOLUME   = 5e7
REGIME_FILTER       = True

# 波动率目标
VOL_TARGET_ANNUAL   = 0.20
VOL_TARGET_LOOKBACK = 20
VOL_SCALE_MIN       = 0.3
VOL_SCALE_MAX       = 2.0

# 成本（Longport 美股口径）
ENABLE_COSTS         = True
PLATFORM_FEE_PER_SHARE = 0.005
SEC_FEE_RATE           = 0.0000278
TAF_PER_SHARE          = 0.000166
SLIPPAGE_BPS           = 5.0
```

---

## 项目结构

```
.
├── backtest.py              # 策略与双模式回测主程序（Phase A/B/C 主循环）
├── universe.py              # NAS100 + SP500 静态合集，含中文名映射 / 子集函数
├── longport_api.py          # Longport 日线接口（含线程安全单例 + 重试）
├── intraday_api.py          # Longport 分钟级接口（RTH 过滤、HKT 时区修正、反向分页）
├── daily_cache.py           # 日线 CSV 本地缓存
├── data_cache/
│   ├── daily/               # 日线 CSV
│   └── intraday/5min/       # 分钟 parquet
├── requirements.txt
├── .env                     # Longport 凭证（不要提交）
└── README.md
```

---

## 已知限制与注意事项

1. **分钟数据上限 ~2 年**：Longport `history_candlesticks_by_date` 在分钟周期下仅回溯到约 2024-05；早于此抛 `301600 out of minute kline begin date`。这就是 INTRADAY 模式只能从 2024-05-08 起的原因。
2. **时区**：Longport 分钟 K 的 timestamp 是 **HKT (UTC+8) 但 tz-naive**，[`intraday_api.py`](intraday_api.py) 会显式 localize 后转 UTC，再转 ET 过滤 RTH。已封装好。
3. **首次拉数据耗时**：日线约 5-10 分钟；分钟数据 516 只 × 2 年约 1-2 小时（受 Longport 限速）。之后增量缓存只补当天最新数据。
4. **Sharpe / CAGR 的合理预期**：当前 DAILY (516 只) 1.51 / 37.5%、INTRADAY (NAS100 旧值) 1.40 / 40.3%。长期合理预期 **Sharpe ~1.0-1.3、CAGR 20-30%**；显著超过此区间的样本（如 2026 YTD 强趋势期）大概率是 lucky regime，不要外推。
5. **止损价的近似性**：`pd_low ≤ stop_price` 触发后假设按 `stop_price` 成交。极端波动股票（如 LCID、TTD）实盘可能比回测亏更多，gap-down 已用 `gap_open` 兜底。
6. **幸存者偏差**：当前用 NAS100 + SP500 静态成分股，未做 point-in-time 还原，理论上 2020 年的回测里包含了那时还不在指数中的票。要彻底干掉这一偏差需要接成分股历史变更日，工程上代价较高，目前没做。
7. **更高换手 + 更高成本**：池子从 101 扩到 516 后，年成本从 ~14% 涨到 ~30%。Sharpe 仍然净增（1.12 → 1.51），但要注意若交易成本结构与 Longport 显著不同（如 IB），需要重新评估。

---

## 附录：研究历程与已尝试的实验

主要决策都通过实测验证过，下面把过程压缩成一段，方便回顾：

### A. 决策时点扫描（INTRADAY 模式）

扫描 `DECISION_TIME_ET ∈ {10:00, 11:00, …, 15:55}`，看哪个时点 INTRADAY Sharpe 最高：

- 12:00 出现峰值（Sharpe ~1.63），但 11:00、13:00 邻近点也能到 1.5+
- 不同时点结果有 ±0.2 Sharpe 的抖动，明显有过拟合 2 年样本的成分
- **最终选 15:50**：实盘可执行（收盘前 10 分钟下单完全来得及）+ 信号最新鲜，性能并不弱

### B. 参数随机扫描（25 组 × DAILY + INTRADAY 双段评分，NAS100 子集上做）

抽样空间：`k_long ∈ {6,8,10,12,15}`、`mom_weight ∈ {0.5,0.6,0.7,0.8}`、`bias_weight ∈ {0,0.1,0.2,0.3}`、`hysteresis_mult ∈ {2.5,3,4,5,6}`、`stop_loss_pct ∈ {4,5,6,8}%`、`stop_loss_atr_mult ∈ {1,1.5,2,2.5,3}`、`max_hold_days ∈ {20,30,40,60,80,120}`。

按 `min(DAILY Sharpe, INTRADAY Sharpe)` 排名前 3：

| k_long | mom_w | bias_w | hyst | sl% | atr | hold | DAILY Sh / MDD | INTRA Sh / MDD |
|---|---|---|---|---|---|---|---|---|
| **10** | **0.8** | **0.1** | **5.0** | **4** | **2.0** | **80** | 1.15 / -21% | 1.35 / -24% |
| **10** | **0.8** | **0.3** | **4.0** | **5** | **1.5** | **80** | 1.10 / -24% | 1.40 / -22% |
| 12 | 0.5 | 0.0 | 2.5 | 5 | 1.5 | 120 | 1.08 / -27% | 1.38 / -16% |

**关键结论**：
- 旧默认 `MAX_HOLD_DAYS=20` 偏紧，放宽到 60-120 显著降低 DAILY MDD（-40% → -24%）
- `k_long` 适度增大（10-12）+ `hysteresis_mult ≥ 4` 更稳：分散度↑ 换手↓ Sharpe↑
- `mom_weight 0.8 / bias_weight 0.3` 略优于 0.7 / 0.2，差异有限
- 最终 DEFAULT 升级为表中第 2 行（k=10, mom=0.8, bias=0.3, hyst=4.0, hold=80）

### C. Universe 扩展实验：NAS100 → NAS100 ∪ S&P 500（已升级为默认）

DAILY 模式（INTRADAY 拉 500 只 × 2 年分钟数据约 1-2 小时不划算，先只测 DAILY）：

| Universe | 累计收益 | CAGR | Sharpe | MDD | 总交易 | 成本占比 |
|---|---|---|---|---|---|---|
| NAS100 (101) | +323.33% | 25.59% | 1.12 | -24.22% | 733 | 14.4% |
| **NAS100 ∪ SP500 (516)** | **+651.14%** | **37.49%** | **1.51** | -25.23% | 1,303 | 30.2% |

扩大池子后 Sharpe **+0.39**、累计收益翻倍、MDD 仅恶化 1pp。原因是 SP500 里的中盘成长股给了更多 outsized winners（CVNA +240%、SNDK +160%、APP +152%、SMCI +123% 等）。代价是换手翻倍、年成本从 14% → 30%（仍能净盈利）。

**已升级为主线默认**——若要回到 NAS100 子集做对比，可在 `backtest.py` 把 `from universe import get_universe` 改成 `from universe import get_nas100_universe as get_universe`。

### D. Walk-Forward（已废弃）

早期做过 17m 训练 + 7m 验证两段网格搜索：

- 验证期 (2025-10 ~ 2026-05) 是强趋势市，几乎所有参数都跑出 Sharpe 2-3，没区分度
- 训练期 (2024-05 ~ 2025-09) 是震荡市，DEFAULT 只 Sharpe 0.49

样本太短，验证期踩到 lucky regime，结果对参数选择几乎没指导价值。后来改用 B 中的"DAILY (6 年) + INTRADAY (2 年) 双段评分"思路，更稳。
