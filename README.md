# 港股 + 美股 趋势突破回测

单一入口脚本 **`backtest.py`**：在文件**顶部配置区**修改回测区间、股票池与策略参数，然后运行：

```bash
python backtest.py
```

依赖：`.env` 中配置 Longport；Python 依赖见 `requirements.txt`。

---

## 策略说明（当前默认）

本策略为**日频、纯技术面、多头趋势突破**，**不含基本面**。以下为逻辑要点，与 `backtest.py` 内默认参数一致。

### 1. 「选股」与动态可交易池

- **推荐默认：`UNIVERSE_MODE = 'hsi_hstech_ipo'`**  
  - **恒指 / 恒生科技成分**：由 `data/hsi_constituents.csv`、`data/hstech_constituents.csv` 维护（若无则自动用同目录 `*.example.csv` 样例）。**成分名单需你按恒生指数公司披露定期更新**（本仓库样例仅作格式参考，可能非当日完整成分）。  
  - **近一年新股**：对 `hk_all_stocks.csv` 中全部 5 位代码批量调用 Longport `static_info`，用 `listing_date` 筛出「上市日至多 `IPO_LISTING_MAX_AGE_DAYS` 天前」的标的，与上面成分取**并集**。首次会多打若干次 `static_info`（每批最多 500 个代码），之后仍主要吃日线缓存。  
  - **规模粗估**：恒指约 80 只、恒生科技约 30 只（与恒指大量重叠），并集约 **90±** 只量级；近一年港股新股常见 **几十只** 量级。合计通常 **约一百多只**，远小于全市场两千多只，也**不是**「全表前 200 行」这种任意截断。
- **`UNIVERSE_MODE = 'csv'`**：仅交易你在自选 CSV 里列出的标的。  
- **`UNIVERSE_MODE = 'hk_all'`**（不推荐）：仍支持从全表取前 `HK_ALL_MAX` 只，仅兼容旧用法。  
- **每日动态筛选**：在**已载入日线**的候选集合上，每个交易日用**截至昨收**的流动性条件过滤；未载入历史的代码**不会**出现在当日可买池里。

### 2. 数据与时间对齐（防止未来函数）

- 回测日历日 \(T\) 上，任何决策仅使用**不晚于 \(T-1\)** 的日线数据（通过 `HistoricalDataManager` 截断）。
- **突破**：收盘价突破「前一日为止的 \(N\) 日最高价」，即 `close > rolling_max(high, N).shift(1)`，当日最高价不参与比较。
- **趋势确认**：收盘价高于趋势均线（默认 50 日）。
- **量能**：当日量相对成交量均线之比 ≥ 阈值（默认 1.2）。
- **流动性**：近若干日平均成交额落在港股/美股各自区间内（见配置常量）。
- **成交假设**：信号在 \(T-1\) 收盘形成，成交价用该收盘价，等价于简化「按昨收成交」；**未模拟**盘中滑点、跳空与冲击成本，实盘中通常更差。

### 3. 大盘过滤（默认开启）

- 仅在 **恒生指数 `HSI.HK`** 与 **标普 ETF `SPY.US`** 的收盘价均高于各自 **\(M\) 日移动平均**（默认 \(M=200\)）时，允许**新开仓**；持仓仍可按下方规则卖出。
- `REGIME_MODE = 'all'` 表示两个基准都要满足；改为 `'any'` 则任一满足即可（更宽松，慎用）。

### 4. 波动率目标仓位（默认开启）

- 用标的近 `VOL_LOOKBACK` 日对数收益估计**年化波动**，将单笔目标名义仓位按「目标年化波动 / 实现波动」缩放，并夹在 `VOL_SCALE_MIN`～`VOL_SCALE_MAX` 之间。
- 目的：高波动标的少买、低波动略多买，使贡献更均匀；设为 `VOL_TARGET_ANNUAL = 0` 可关闭。

### 5. 出场规则（按顺序检查）

1. **固定止损**：相对买入价亏损超过 `STOP_LOSS_PCT`（默认 10%）清仓。
2. **移动止盈**（默认开启）：浮盈 ≥ `TRAILING_ACTIVATION_PCT` 后，若收盘价自持仓以来「最高价」（用日线收盘价递推的峰值）回撤 ≥ `TRAILING_STOP_PCT`，则清仓。
3. **唐奇安下轨**：收盘价跌破「前一日为止的 \(D\) 日最低价」（默认 \(D=20\)），`shift(1)`，清仓。
4. **亏损 + MA60**：若仍开启 `USE_MA60_LOSS_EXIT`，收盘价跌破 60 日均线且当前为亏损，则清仓。

### 6. 仓位与交易单位

- 最多同时持有 `MAX_POSITIONS` 只；单笔目标为总权益的 `POSITION_SIZE_PCT`，再乘以波动率缩放。
- 港股按**整手 100 股**近似；美股按**整股**。
- `ONE_WAY_COST_RATE` 为单边费率（买卖各计一次需自行理解；默认 0）。

### 7. 基准

- 报告中「基准」为 **恒指与 SPY 收盘价等权、各自归一化到起点 100 后再取平均**得到的组合曲线，用于粗略对比跨市场买入持有风格。

### 8. 局限与免责声明

- 历史回测不预示未来；股票池过小或时段过短会**严重扭曲**夏普、回撤与年度胜率。
- 年度收益若首段数据始于某年中，该「首年」可能是**非完整自然年**。
- 策略参数曾在探索阶段比较过多组组合，当前默认采用**风险调整向**的一组（大盘过滤 + 波动目标 + 移动止盈 + 10% 止损）；**不保证**样本外最优。

---

## 数据缓存与 API 调用

- 日线通过 **`hk_stock_api.fetch_daily_bars`** 写入 **`data_cache/daily/*.csv`**（`.gitignore` 已忽略该目录）。
- **若缓存已覆盖请求区间，不会初始化 Longport、也不会再请求网络**（仅读本地 CSV）。
- 仅当**缺文件、或需向前/向后补数据**时，才会懒加载 API 并请求。
- 若曾出现「终端显示拉了几千根但成功加载 0 只」，多为旧缓存索引与日期比较异常（已修复）；可删除 `data_cache/daily/` 下对应 CSV 后重跑。
- 强制关闭缓存：`TREND_DISABLE_DAILY_CACHE=1`
- 自定义目录：`TREND_DAILY_CACHE_DIR=/path/to/dir`

其它：`LONGPORT_REQUEST_PAUSE` 控制请求间隔。

### 终端里常见现象

- **未找到 data/hsi_constituents.csv，改用样例**：预期行为；正式回测请复制为 `hsi_constituents.csv` 并按恒生指数官网更新。  
- **新股 0 只**：若随后出现「static_info 中几乎无 listing_date」提示，多为 API/SDK 未返回上市日字段，可设 `INCLUDE_IPO_UNIVERSE=False` 跳过全表扫描。若有 `listing_date` 但仍为 0，表示窗口内确实无近一年上市标的。  
- **日志级别为 INFO 时**：不应再对「某一子区间无 K 线」刷屏 WARNING（已改为 debug）；若最终该标的仍无日线，会在「成功加载」里体现。

---

## 仓库文件说明

| 文件 | 说明 |
|------|------|
| `backtest.py` | 唯一回测入口与策略实现 |
| `hk_universe.py` | 恒指/恒生科技/新股候选池合并逻辑 |
| `hk_stock_api.py` | Longport 日线封装（含分段合并突破约 1000 根限制） |
| `daily_cache.py` | 日线 CSV 缓存合并逻辑 |
| `data/hsi_constituents.example.csv` 等 | 成分表样例；正式回测请复制为 `hsi_constituents.csv` 并按官网更新 |
| `dual_universe.example.csv` | 示例股票池（`symbol` 列） |
| `dual_universe.csv` | 若存在则优先于 example 被读取（需自行创建） |
| `hk_all_stocks.csv` | 港股代码表；`UNIVERSE_MODE = 'hk_all'` 时使用 |
| `data_cache/` | 已拉取的日线缓存（可复用、可删后全量重拉） |

---

## 配置项速查（均在 `backtest.py` 顶部）

- **时间与数据**：`BACKTEST_START`、`BACKTEST_END`、`DATA_WARMUP_DAYS_BEFORE_START`（大盘 200 日均线时建议 warmup≥400 自然日）  
- **候选池**：`UNIVERSE_MODE`、`UNIVERSE_CSV`（csv 模式）、`HSI_CONSTITUENTS_CSV`、`HSTECH_CONSTITUENTS_CSV`、`*_EXAMPLE`、`HK_ALL_STOCKS_CSV`、`INCLUDE_IPO_UNIVERSE`、`IPO_LISTING_MAX_AGE_DAYS`、`HK_ALL_MAX`（仅 hk_all 模式）
- **账户**：`INITIAL_CAPITAL`、`MAX_POSITIONS`、`POSITION_SIZE_PCT`
- **信号**：`BREAKOUT_LOOKBACK`、`TREND_MA_PERIOD`、`VOLUME_RATIO_THRESHOLD`、`EXIT_DONCHIAN_DAYS`
- **风控**：`STOP_LOSS_PCT`、`USE_MA60_LOSS_EXIT`、`ONE_WAY_COST_RATE`
- **大盘**：`USE_REGIME_FILTER`、`REGIME_BENCHMARKS`、`REGIME_MODE`、`REGIME_MA_DAYS`
- **波动目标**：`VOL_TARGET_ANNUAL`、`VOL_LOOKBACK`、`VOL_SCALE_MIN`、`VOL_SCALE_MAX`
- **移动止盈**：`TRAILING_ACTIVATION_PCT`、`TRAILING_STOP_PCT`（任一为 0 则关闭）
- **流动性上下界**：`HK_AVG_TURNOVER_*`、`US_AVG_TURNOVER_*`

修改后保存，直接运行 `python backtest.py` 即可。
