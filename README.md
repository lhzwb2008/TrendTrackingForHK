# crossrank-quant

美股**横截面排名（Cross-sectional Rank）选股**策略，**分钟级日内决策**，回测与实盘**完全同时点同逻辑**。

每个交易日美东 **15:50** 用「日 K 历史 + 当日截至 15:50 的分钟数据」算横截面 composite 分数，从 NAS100 ∪ S&P 500（~516 只）的池子里挑前 10 名做多，叠加 SPY 200DMA regime 过滤、波动率目标仓位与 max(5%, 1.5×ATR) 个股止损。回测产生的开/平仓清单可以直接对接 Longport 实盘。

---

## 策略一句话

> 在 NAS100∪SP500 中按动量+反转+均线 bias 做横截面 rank，每天 15:50 选 top 10 做多；hysteresis 滞后带降换手；SPY 200DMA 下方停止开新仓；按近 20 日组合波动反向缩放仓位；个股 max(5%, 1.5×ATR) 止损。

### 6 个信号（每只股票每天）

| 类别 | 信号 | 含义 |
|---|---|---|
| 动量 | `mom_20 = pd_close / close_20d_ago - 1` | 20 日涨幅 |
| 动量 | `mom_60 = pd_close / close_60d_ago - 1` | 60 日涨幅 |
| 反转 | `IBS = (pd_close - pd_low) / (pd_high - pd_low)` | 当日相对位置 |
| 反转 | `Williams%R(14)` | 14 日相对位置 |
| 反转 | `rev_5 = -(pd_close / close_5d_ago - 1)` | 5 日反向涨幅 |
| Bias | `EMA9 > EMA21` | 短均线趋势 |

### 横截面打分

```
momentum_block = mean(rank(mom_20), rank(mom_60))
reversal_block = mean(-rank(IBS), -rank(-Williams%R), rank(rev_5))
bias           = (trend_up - 0.5) × 2
composite      = 0.8 × momentum_block + 0.2 × reversal_block + 0.3 × bias
```

### 关键风控

- **Hysteresis**：top 10 才开仓，跌出 top 40 才平仓 → 日均换手 ~8%
- **Regime**：SPY < 200DMA 时停止开新多头
- **Vol target**：按近 20 日组合波动调节仓位，0.20 年化目标
- **个股止损**：`max(5%, 1.5 × ATR14)`，多头跌穿即止损
- **最大持仓 80 个交易日**，到期强平

---

## 回测结果（2026-05 最新）

### 1. 主策略（`backtest.py`，DAILY 跨牛熊 + INTRADAY 实盘对齐）

| 指标 | DAILY (2020-01 ~ 2026-05, 6.33y) | INTRADAY (2024-05 ~ 2026-05, 1.99y) | QQQ (DAILY 区间) |
|---|---|---|---|
| 累计收益 | **+641.54%** | **+141.93%** | +241.92% |
| CAGR | **37.21%** | **55.81%** | 21.44% |
| 年化波动 | 24.72% | 28.69% | — |
| Sharpe | **1.50** | **1.77** | 0.90 |
| 最大回撤 | -24.89% | -25.37% | -35.12% |
| Calmar | 1.49 | 2.20 | — |
| 胜率 | 43.0% | 40.5% | — |
| 平均持仓天 | 9.6 | 10.6 | — |
| 总交易笔 | 1,303 | 410 | — |
| 总交易成本 | $30,077 (30.08%) | $5,185 (5.19%) | — |

**逐年（DAILY / INTRADAY 收益%）**

| 年份 | DAILY | INTRADAY | DAILY MDD | 备注 |
|---|---|---|---|---|
| 2020 | +31.83% | — | -16.62% | 疫情 V 反弹 |
| 2021 | +17.58% | — | -16.02% | 牛市 |
| 2022 | +3.80% | — | -13.37% | **熊市保护：QQQ -33% 而策略 +3.8%** |
| 2023 | +74.75% | — | -14.46% | 反弹 |
| 2024 | +13.45% | +6.06% | -24.89% | 震荡 |
| 2025 | +42.82% | +38.01% | -14.95% | 强趋势 |
| 2026 YTD | +89.94% | +73.29% | -9.39% | 加速，**不可外推** |

> 长期合理预期 **Sharpe 1.0~1.3、CAGR 20~30%**；2025-2026 YTD 大概率是 lucky regime。

### 2. 对冲实验（`backtest_hedged.py`，DAILY 10 年区间）

> **结论：纯多头优于做空 QQQ 对冲**。横截面 alpha 与 QQQ 残留相关性 0.43，强行 zero-beta 把策略 beta 压到 0 的同时也吃掉了趋势 beta，10 年 Sharpe 反而从 1.19 → 0.73。

区间 **2016-05-01 ~ 2026-05-01（10 年）**，起始本金 $100,000，三个变体共享同一份多头组合（2,076 笔），仅叠加 QQQ 短头寸 overlay：

| 变体 | 累计 | CAGR | 年化波动 | Sharpe | MDD | Calmar | Corr(QQQ) | Beta(QQQ) | 总成本% |
|---|---|---|---|---|---|---|---|---|---|
| **baseline (no hedge)** | **+799.3%** | **24.62%** | 22.96% | **1.19** | **-24.88%** | **0.99** | +0.43 | +0.44 | 56.3% |
| rolling-beta QQQ 0.5x | +674.0% | 22.76% | 24.99% | 0.95 | -32.55% | 0.70 | +0.25 | +0.29 | 64.7% |
| rolling-beta QQQ 1.0x | +550.9% | 20.65% | 33.62% | 0.73 | -48.62% | 0.42 | +0.00 | +0.01 | 71.0% |
| (基准) QQQ buy&hold | +579.0% | 21.21% | 22.32% | 0.97 | -35.12% | 0.60 | 1.00 | 1.00 | 0% |

**逐年收益% 对比**

| 年份 | baseline | QQQ 0.5x | QQQ 1.0x |
|---|---|---|---|
| 2016 | +6.42 | +0.61 | -2.48 |
| 2017 | +31.02 | +4.87 | -17.91 |
| 2018 | +6.56 | -0.54 | -6.53 |
| 2019 | +3.61 | -7.66 | -21.78 |
| 2020 | +34.56 | +37.66 | +51.99 |
| 2021 | +17.74 | +9.77 | +0.79 |
| **2022** | **+4.29** | **+13.30** | **+37.54** |
| 2023 | +74.75 | +68.88 | +67.51 |
| 2024 | +13.45 | +5.44 | -3.39 |
| 2025 | +40.96 | +48.12 | +67.49 |
| 2026 YTD | +66.42 | +76.88 | +94.74 |

**关键观察**：
- 对冲只在 **2020 V 反弹、2022 熊市、2025-2026 强势期** 这种「策略有大量 long alpha 而 QQQ 大跌或大涨」的极端窗口里跑赢，长期是负贡献
- regime 过滤已经在熊市里压低多头敞口，叠加做空属于双重对冲，吃掉了上涨期 beta 收益
- **结论**：保持纯多头 baseline；如希望降低 beta，可以在已知风险事件前临时挂 0.5x，长期不开

---

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env 中填入 Longport 凭证：
# LONGPORT_APP_KEY=...
# LONGPORT_APP_SECRET=...
# LONGPORT_ACCESS_TOKEN=...

python backtest.py            # 主策略：DAILY + INTRADAY 双段对照
python backtest_hedged.py     # 对冲实验：baseline vs QQQ 0.5x / 1.0x（10 年）
```

数据本地缓存在 `data_cache/`（首次拉取日线 ~5-10 分钟、分钟数据 ~1-2 小时）。

---

## 每日执行时序（回测 = 实盘）

```
T 日：
  09:30 ET   开盘
  ─── Phase A：09:30 → 15:50 监控存量持仓
       low ≤ stop  → 按 stop_price 成交（gap-down 时按 gap_open）

  ─── Phase B (15:50 决策点)：
       1. 用「日 K 历史 + 09:30~15:50 的 5-min 聚合」算 composite
       2. 平仓：max_hold / 跌出 top 40 / regime 翻转 → pd_close 立即成交
       3. 开仓：top 10 中尚未持有的 → pd_close 开多 + 挂 GTC stop

  ─── Phase C：15:50 → 16:00 MTM 至真收盘
```

实盘对应：服务器定时任务每日 15:50 跑脚本 → Longport `submit_order`。

---

## 项目结构

```
.
├── backtest.py              # 主策略（DAILY + INTRADAY 双段回测）
├── backtest_hedged.py       # 对冲实验（DAILY，叠加 QQQ 短头寸 overlay）
├── universe.py              # NAS100 ∪ SP500 静态合集（516 只）
├── longport_api.py          # 日线 API（缓存 + 单例 + 重试）
├── intraday_api.py          # 分钟级 API（RTH 过滤 / HKT 时区修正）
├── daily_cache.py           # 日线 CSV 增量缓存
├── data_cache/              # 本地数据缓存
├── requirements.txt
└── .env                     # Longport 凭证（不要提交）
```

---

## 关键参数

详见 [`backtest.py`](backtest.py) 顶部 CONFIG。常调整：

```python
DAILY_START   = "2020-01-01"
INTRA_START   = "2024-05-08"     # 分钟数据 Longport 仅回溯 ~2 年
K_LONG  = 10
HYSTERESIS_MULT = 4.0
MOM_WEIGHT  = 0.8
BIAS_WEIGHT = 0.3
STOP_LOSS_PCT      = 0.05
STOP_LOSS_ATR_MULT = 1.5
MAX_HOLD_DAYS      = 80
REGIME_FILTER      = True
VOL_TARGET_ANNUAL  = 0.20
DECISION_TIME_ET   = "15:50"
```

成本默认按 Longport 美股口径：平台费 $0.005/股(min $1)、SEC 0.0000278%、TAF $0.000166/股、滑点 5 bps/侧。

---

## 已知限制

1. **分钟数据 ~2 年上限**：Longport `history_candlesticks_by_date` 分钟周期仅回溯到约 2024-05，所以 INTRADAY 模式起点固定 2024-05-08。
2. **幸存者偏差**：用 NAS100+SP500 静态成分股，未做 point-in-time 还原。
3. **止损价近似**：`pd_low ≤ stop_price` 触发后假设按 `stop_price` 成交，极端波动股票实盘可能更差，gap-down 已用 `gap_open` 兜底。
4. **池子换手 + 成本**：516 只 universe 年成本 ~30%（仍能净盈利）；若用 IB 等不同费率结构需重新评估。
5. **2025-2026 lucky regime**：YTD 收益显著超过 Sharpe 1.0-1.3 长期预期，不可外推。

---

# 附录：研究历程与已尝试的实验

## A. Universe 扩展：NAS100 (101) → NAS100 ∪ SP500 (516)（已升级为默认）

DAILY 模式（2020-01 ~ 2026-05）：

| Universe | 累计 | CAGR | Sharpe | MDD | 总交易 | 成本% |
|---|---|---|---|---|---|---|
| NAS100 (101) | +323.33% | 25.59% | 1.12 | -24.22% | 733 | 14.4% |
| **NAS100 ∪ SP500 (516)** | **+641.54%** | **37.21%** | **1.50** | -24.89% | 1,303 | 30.1% |

扩大池子后 Sharpe **+0.38**、累计收益翻倍、MDD 仅恶化 0.7pp。SP500 中盘成长股给了大量 outsized winners（CVNA +240%、SNDK +160%、APP +152%、SMCI +123%、PLTR +90% 等）。代价是换手翻倍、年成本 14% → 30%。

## B. 决策时点扫描（INTRADAY 模式）

扫描 `DECISION_TIME_ET ∈ {10:00, 11:00, …, 15:55}`：

- 12:00 出现峰值（Sharpe ~1.63），11:00、13:00 邻近也能到 1.5+
- 不同时点结果有 ±0.2 Sharpe 的抖动，明显有过拟合 2 年样本的成分
- **最终选 15:50**：实盘可执行（收盘前 10 分钟下单完全来得及）+ 信号最新鲜

## C. 参数随机扫描（25 组 × DAILY + INTRADAY 双段评分，NAS100 子集上做）

抽样空间：`k_long ∈ {6,8,10,12,15}`、`mom_weight ∈ {0.5,0.6,0.7,0.8}`、`bias_weight ∈ {0,0.1,0.2,0.3}`、`hysteresis_mult ∈ {2.5,3,4,5,6}`、`stop_loss_pct ∈ {4,5,6,8}%`、`stop_loss_atr_mult ∈ {1,1.5,2,2.5,3}`、`max_hold_days ∈ {20,30,40,60,80,120}`。

按 `min(DAILY Sharpe, INTRADAY Sharpe)` 排名前 3：

| k_long | mom_w | bias_w | hyst | sl% | atr | hold | DAILY | INTRADAY |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.8 | 0.1 | 5.0 | 4 | 2.0 | 80 | 1.15 / -21% | 1.35 / -24% |
| **10** | **0.8** | **0.3** | **4.0** | **5** | **1.5** | **80** | 1.10 / -24% | 1.40 / -22% |
| 12 | 0.5 | 0.0 | 2.5 | 5 | 1.5 | 120 | 1.08 / -27% | 1.38 / -16% |

**关键结论**：
- 旧默认 `MAX_HOLD_DAYS=20` 偏紧，放宽到 60-120 显著降低 DAILY MDD（-40% → -24%）
- `k_long` 适度增大（10-12）+ `hysteresis_mult ≥ 4` 更稳：分散度↑ 换手↓ Sharpe↑
- `mom_weight 0.8 / bias_weight 0.3` 略优于 0.7 / 0.2，差异有限

最终 DEFAULT 升级为表中第 2 行。

## D. 对冲实验（详见 `backtest_hedged.py`）

研究问题：在多头之上叠加 QQQ 做空，能否降低与大盘的相关性、提升风险调整后收益？

实现方式：双账户解耦——多头层与原 backtest.py DAILY 完全等价；对冲层独立做 QQQ 短头寸的 MTM + 调仓成本，按 `target_short = ratio × rolling_beta(60d) × long_mv` 调仓，纯后处理。

10 年（2016-05 ~ 2026-05）结果（见正文表格）：所有对冲档位 Sharpe / CAGR / MDD 全面变差，仅在 2022 熊市等极端窗口里短期反超。原因：

1. regime 过滤已经把 2022 这种系统性下跌期的 long 敞口压低了
2. 横截面 alpha 与 QQQ 的相关性只有 0.43，并不需要严格 zero-beta
3. QQQ 长期 +21%/年，做空一份就稳定吃掉这部分 beta 收益

**结论**：保持纯多头 baseline。如未来想短期降 beta（如已知会议/财报窗口），用 0.5x 比 1.0x 更稳。

## E. Walk-Forward（已废弃）

早期做过 17m 训练 + 7m 验证两段网格搜索：

- 验证期 (2025-10 ~ 2026-05) 是强趋势市，几乎所有参数都跑出 Sharpe 2-3，没区分度
- 训练期 (2024-05 ~ 2025-09) 是震荡市，DEFAULT 只 Sharpe 0.49

样本太短，验证期踩到 lucky regime，结果对参数选择几乎没指导价值。后来改用 C 中的"DAILY (6 年) + INTRADAY (2 年) 双段评分"思路，更稳。
