#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAS100 短中线策略回测

策略简介：
- 池子：NAS100 成分股，每日按 20D 平均成交额过滤 ≥ $50M
- 信号：动量 (mom_20/mom_60) + 反转 (IBS / Williams%R / rev_5)，rank 化聚合
- 入场：横截面 composite 分数 top K_LONG 做多
- 出场：止损（max(5%, 1.5×ATR)）/ 持仓 20 日到期 / 信号退出 hysteresis 带（4×K）
- 大盘 regime 过滤：仅 SPY > 200DMA 时开新多头
- 波动率目标仓位：按近 20D 组合波动反向缩放仓位（高波动减仓、低波动加仓）
- 交易成本：Longport 美股平台费 $0.005/股 (min $1) + SEC 0.0000278% + TAF $0.000166/股
- 滑点：默认 5 bps/侧

详细介绍见 README.md。所有可调参数集中在文件顶部的 CONFIG 区块。
运行：  python backtest.py
"""

from __future__ import annotations

# ============================================================================
#                              用户可调参数
# ============================================================================

# ---- 回测区间与本金 ----
BACKTEST_START = "2020-01-01"      # 回测起点；股票在该日之前未上市的，从其上市日起参与
BACKTEST_END   = "today"            # "today" 或 "YYYY-MM-DD"
STARTING_CAPITAL = 1_000_000        # 起始本金（美元），仅用于显示与 P&L 美元金额

# ---- 持仓结构 ----
K_LONG  = 8                         # 最多同时持有的多头数
K_SHORT = 0                         # 最多同时持有的空头数（0 = 纯多头，默认）
LONG_WEIGHT_FRAC = 1.0              # 多头占 gross 的比例；空头占 (1 - 此值)
                                    #   1.0  = 纯多头（默认；建议同时 K_SHORT=0）
                                    #   2/3  ≈ 多空 2:1
                                    #   0.5  = 多空 1:1
GROSS_LEVERAGE = 1.0                # 总毛敞口（long + |short|）

# ---- 信号聚合 ----
MOM_WEIGHT  = 0.7                   # 动量信号权重；反转 = 1 - 此值
                                    # NAS100 趋势市场实验显示 0.7 显著优于 0.5
BIAS_WEIGHT = 0.2                   # EMA9/21 趋势 bias 系数

# ---- Hysteresis 与持仓维持 ----
HYSTERESIS_MULT = 4.0               # 持仓需 score 退出 top/bottom (mult * K) 才平仓
                                    # 越大换手越低；2 几乎不滞后，4 月换手降到合理水平

# ---- 风控 ----
STOP_LOSS_PCT      = 0.05           # 个股止损百分比下限
STOP_LOSS_ATR_MULT = 1.5            # 个股止损 ATR 倍数；实际止损 = max(pct, ATR mult)
MAX_HOLD_DAYS      = 20             # 最大持仓天数
MIN_DOLLAR_VOLUME  = 5e7            # 流动性过滤：20D 平均成交额下限（美元）

# ---- 大盘 regime 过滤 ----
REGIME_FILTER = True                # True 时启用 SPY 200DMA：上方仅开多，下方仅开空（关闭时全开）

# ---- 波动率目标仓位 ----
VOL_TARGET_ANNUAL = 0.20            # 目标年化波动；0 = 关闭。开仓时按 (target / realized_vol) 缩放
VOL_TARGET_LOOKBACK = 20            # 计算已实现组合波动的滚动窗口
VOL_SCALE_MIN = 0.3                 # 缩放下限（高波动期最多压到原 30%）
VOL_SCALE_MAX = 2.0                 # 缩放上限（低波动期最多放大到 200%）

# ---- 交易成本与滑点（Longport 美股口径） ----
ENABLE_COSTS         = True
PLATFORM_FEE_PER_SHARE = 0.005      # 平台费 $0.005/股，每单最低 $1（双边收取）
PLATFORM_FEE_MIN     = 1.0
SEC_FEE_RATE         = 0.0000278    # SEC 费率（仅卖出方收取）
TAF_PER_SHARE        = 0.000166     # 交易活动费（仅卖出方收取）
TAF_MAX_PER_ORDER    = 8.3
SLIPPAGE_BPS         = 5.0          # 单侧滑点（基点；买入抬价 / 卖出压价）

# ---- 输出 ----
VERBOSE_TRADES = True               # 控制台打印每一笔交易的开/平仓
PRINT_DAILY_POSITIONS = False       # True 时每日打印当前持仓快照（很啰嗦，调试用）

# ============================================================================
#                       以下为实现，一般不需要修改
# ============================================================================

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from longport_api import fetch_daily_bars, get_api_singleton
from nas100_universe import get_universe


@dataclass
class Config:
    start: date
    end: date
    starting_capital: float = STARTING_CAPITAL
    k_long: int = K_LONG
    k_short: int = K_SHORT
    long_weight_frac: float = LONG_WEIGHT_FRAC
    gross_leverage: float = GROSS_LEVERAGE
    hysteresis_mult: float = HYSTERESIS_MULT
    stop_loss_pct: float = STOP_LOSS_PCT
    stop_loss_atr_mult: float = STOP_LOSS_ATR_MULT
    max_hold_days: int = MAX_HOLD_DAYS
    min_dollar_volume: float = MIN_DOLLAR_VOLUME
    mom_weight: float = MOM_WEIGHT
    bias_weight: float = BIAS_WEIGHT
    regime_filter: bool = REGIME_FILTER
    vol_target_annual: float = VOL_TARGET_ANNUAL
    vol_target_lookback: int = VOL_TARGET_LOOKBACK
    vol_scale_min: float = VOL_SCALE_MIN
    vol_scale_max: float = VOL_SCALE_MAX
    enable_costs: bool = ENABLE_COSTS
    platform_fee_per_share: float = PLATFORM_FEE_PER_SHARE
    platform_fee_min: float = PLATFORM_FEE_MIN
    sec_fee_rate: float = SEC_FEE_RATE
    taf_per_share: float = TAF_PER_SHARE
    taf_max_per_order: float = TAF_MAX_PER_ORDER
    slippage_bps: float = SLIPPAGE_BPS
    verbose_trades: bool = VERBOSE_TRADES
    print_daily_positions: bool = PRINT_DAILY_POSITIONS


# ---------------- 数据加载 ----------------

def load_all_data(symbols: List[str], start: date, end: date,
                  warmup_days: int = 120) -> Dict[str, pd.DataFrame]:
    fetch_start = start - timedelta(days=warmup_days * 2)
    out: Dict[str, pd.DataFrame] = {}
    max_workers = int(os.getenv("NAS100_FETCH_WORKERS", "8"))

    # 预热单例：避免 8 个 worker 并发首次访问时各自创建连接（触发 limit=10）
    # 仅当存在缓存未命中需调 API 时才会真正实例化；若全部命中缓存，单例不会创建
    # 这里通过让主线程先尝试一次（带 try）来完成初始化
    try:
        get_api_singleton()
    except Exception as e:
        print(f"[警告] Longport 初始化失败（若数据已全部缓存可忽略）: {e}")

    def _fetch(sym: str):
        try:
            df = fetch_daily_bars(sym, fetch_start, end, log_cache=False)
            return sym, df
        except Exception as e:
            print(f"[警告] {sym} 拉取失败: {e}")
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch, s) for s in symbols]
        for i, f in enumerate(as_completed(futs), 1):
            sym, df = f.result()
            if len(df) > 0:
                out[sym] = df
            if i % 20 == 0:
                print(f"  已加载 {i}/{len(symbols)}")
    print(f"[数据] 成功加载 {len(out)}/{len(symbols)} 个标的")
    return out


# ---------------- 指标 ----------------

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _williams_r(h, l, c, n=14):
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h, l, c = out["high"], out["low"], out["close"]
    out["ema9"] = _ema(c, 9)
    out["ema21"] = _ema(c, 21)
    out["trend_up"] = (out["ema9"] > out["ema21"]).astype(float)
    out["mom_20"] = c / c.shift(20) - 1
    out["mom_60"] = c / c.shift(60) - 1
    out["rev_5"] = -(c / c.shift(5) - 1)
    rng = (h - l).replace(0, np.nan)
    out["ibs"] = (c - l) / rng
    out["wr14"] = _williams_r(h, l, c, 14)
    out["atr14"] = _atr(h, l, c, 14)
    out["dollar_vol_20"] = (c * out["volume"]).rolling(20).mean()
    return out


def build_panel(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {sym: compute_indicators(df) for sym, df in data.items()}


def _csrank(s):
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    r = valid.rank(method="average", pct=True)
    return (r * 2 - 1).reindex(s.index)


def composite_score(day_panel: pd.DataFrame, mom_w=0.7, bias_w=0.2) -> pd.Series:
    s_mom20 = _csrank(day_panel["mom_20"])
    s_mom60 = _csrank(day_panel["mom_60"])
    s_ibs = -_csrank(day_panel["ibs"])
    s_wr = -_csrank(-day_panel["wr14"])
    s_rev = _csrank(day_panel["rev_5"])
    bias = day_panel["trend_up"].fillna(0.5) * 2 - 1
    momentum_block = (s_mom20 + s_mom60) / 2
    reversal_block = (s_ibs + s_wr + s_rev) / 3
    return mom_w * momentum_block + (1 - mom_w) * reversal_block + bias_w * bias


# ---------------- 回测引擎 ----------------

@dataclass
class Position:
    side: int
    entry_date: pd.Timestamp
    entry_price: float
    weight: float
    shares: float          # 用于美元 P&L 计算
    stop_price: float
    open_cost: float = 0.0 # 开仓时支付的滑点+平台费(+SEC/TAF 若开空)
    days_held: int = 0


@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    trades: List[dict]
    long_count: pd.Series
    short_count: pd.Series


def _fmt_usd(x): return f"${x:,.0f}"


def _open_cost(cfg: 'Config', shares: float, price: float, side: int) -> float:
    """开仓总成本（美元）：滑点 + 平台费 (+ side=-1 即开空时还有 SEC/TAF)。"""
    if not cfg.enable_costs:
        return 0.0
    notional = shares * price
    slip = notional * cfg.slippage_bps / 10000.0
    plat = max(shares * cfg.platform_fee_per_share, cfg.platform_fee_min)
    cost = slip + plat
    if side == -1:  # 开空 = 卖出，触发 SEC/TAF
        cost += notional * cfg.sec_fee_rate
        cost += min(shares * cfg.taf_per_share, cfg.taf_max_per_order)
    return cost


def _close_cost(cfg: 'Config', shares: float, price: float, side: int) -> float:
    """平仓总成本（美元）：滑点 + 平台费 (+ side=+1 即平多卖出时还有 SEC/TAF)。"""
    if not cfg.enable_costs:
        return 0.0
    notional = shares * price
    slip = notional * cfg.slippage_bps / 10000.0
    plat = max(shares * cfg.platform_fee_per_share, cfg.platform_fee_min)
    cost = slip + plat
    if side == 1:  # 平多 = 卖出
        cost += notional * cfg.sec_fee_rate
        cost += min(shares * cfg.taf_per_share, cfg.taf_max_per_order)
    return cost


def run_backtest(panel: Dict[str, pd.DataFrame], cfg: Config,
                 regime_series: Optional[pd.Series] = None) -> BacktestResult:
    all_dates = sorted({d for df in panel.values() for d in df.index})
    all_dates = [d for d in all_dates
                 if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)]
    if not all_dates:
        raise RuntimeError("无可用交易日")

    symbols = list(panel.keys())
    positions: Dict[str, Position] = {}
    equity = float(cfg.starting_capital)
    equity_curve, daily_rets = [], []
    long_counts, short_counts = [], []
    trades: List[dict] = []

    long_w = cfg.gross_leverage * cfg.long_weight_frac
    short_w = cfg.gross_leverage * (1 - cfg.long_weight_frac)
    long_per_pos_base = long_w / cfg.k_long if cfg.k_long > 0 else 0
    short_per_pos_base = short_w / cfg.k_short if cfg.k_short > 0 else 0

    prev_date = None
    v = cfg.verbose_trades

    def _log_open(side, sym, date_, px, stop_px, weight, equity_now, long_n, short_n):
        if not v:
            return
        s = "LONG " if side == 1 else "SHORT"
        notional = equity_now * weight
        print(f"  [{date_.strftime('%Y-%m-%d')}] OPEN  {s} {sym:<8} "
              f"@ ${px:7.2f}  stop=${stop_px:7.2f}  "
              f"size={_fmt_usd(notional)} ({weight*100:.1f}%)  "
              f"holdings: L{long_n} S{short_n}")

    def _log_close(side, sym, entry_d, exit_d, entry_px, exit_px,
                   shares, days_held, reason):
        if not v:
            return
        pnl_pct = (exit_px / entry_px - 1) * side
        pnl_usd = (exit_px - entry_px) * shares * side
        s = "LONG " if side == 1 else "SHORT"
        print(f"  [{exit_d.strftime('%Y-%m-%d')}] CLOSE {s} {sym:<8} "
              f"@ ${exit_px:7.2f}  entry=${entry_px:7.2f}  "
              f"pnl={pnl_pct*100:+6.2f}% / {_fmt_usd(pnl_usd):>10s}  "
              f"held={days_held}d  ({reason})")

    for i, today in enumerate(all_dates):
        # 1) 横截面 panel
        today_data = {}
        for sym in symbols:
            df = panel[sym]
            if today not in df.index:
                continue
            row = df.loc[today]
            if pd.isna(row.get("mom_60")) or pd.isna(row.get("atr14")):
                continue
            if row.get("dollar_vol_20", 0) < cfg.min_dollar_volume:
                continue
            today_data[sym] = row

        if len(today_data) < cfg.k_long + cfg.k_short:
            equity_curve.append(equity)
            daily_rets.append(0.0)
            long_counts.append(0)
            short_counts.append(0)
            prev_date = today
            continue

        day_panel = pd.DataFrame(today_data).T

        # 2) 计算今日 P&L
        day_pnl = 0.0
        if prev_date is not None:
            for sym, pos in positions.items():
                df = panel[sym]
                if today in df.index and prev_date in df.index:
                    p_prev = df.loc[prev_date, "close"]
                    p_today = df.loc[today, "close"]
                    if p_prev > 0 and not np.isnan(p_today):
                        ret = (p_today / p_prev - 1) * pos.side
                        day_pnl += pos.weight * ret
                pos.days_held += 1
        equity *= (1 + day_pnl)
        daily_rets.append(day_pnl)
        equity_curve.append(equity)

        # 波动率目标缩放：基于近 N 日组合日收益的年化波动决定本日新仓 scale
        vol_scale = 1.0
        if cfg.vol_target_annual > 0 and len(daily_rets) >= cfg.vol_target_lookback:
            recent = np.asarray(daily_rets[-cfg.vol_target_lookback:])
            sd = recent.std()
            if sd > 1e-6:
                rv = sd * np.sqrt(252)
                vol_scale = float(np.clip(cfg.vol_target_annual / rv,
                                           cfg.vol_scale_min, cfg.vol_scale_max))
        long_per_pos = long_per_pos_base * vol_scale
        short_per_pos = short_per_pos_base * vol_scale

        # 3) 止损 / 持有期到期
        to_close = []
        for sym, pos in positions.items():
            df = panel[sym]
            if today not in df.index:
                continue
            today_low = df.loc[today, "low"]
            today_high = df.loc[today, "high"]
            today_close = df.loc[today, "close"]
            triggered = False
            reason = ""
            if pos.side == 1 and today_low <= pos.stop_price:
                triggered, reason = True, "stop_loss"
            elif pos.side == -1 and today_high >= pos.stop_price:
                triggered, reason = True, "stop_loss"
            if pos.days_held >= cfg.max_hold_days:
                triggered, reason = True, "max_hold"
            if triggered:
                to_close.append((sym, today_close, reason))
        for sym, px, reason in to_close:
            pos = positions.pop(sym)
            close_c = _close_cost(cfg, pos.shares, px, pos.side)
            equity -= close_c
            gross_pnl = (px - pos.entry_price) * pos.shares * pos.side
            pnl_usd = gross_pnl - pos.open_cost - close_c
            pnl_pct = pnl_usd / (pos.entry_price * pos.shares) if pos.shares > 0 else 0
            trades.append({
                "symbol": sym, "side": pos.side,
                "entry_date": pos.entry_date, "exit_date": today,
                "entry_price": pos.entry_price, "exit_price": px,
                "shares": pos.shares, "days_held": pos.days_held,
                "reason": reason, "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                "costs": pos.open_cost + close_c,
            })
            _log_close(pos.side, sym, pos.entry_date, today, pos.entry_price,
                       px, pos.shares, pos.days_held, reason)

        # 4) 重平衡
        scores = composite_score(
            day_panel, mom_w=cfg.mom_weight, bias_w=cfg.bias_weight,
        ).dropna().sort_values(ascending=False)
        top_k = set(scores.head(cfg.k_long).index)
        bot_k = set(scores.tail(cfg.k_short).index) if cfg.k_short > 0 else set()
        top_2k = set(scores.head(int(cfg.k_long * cfg.hysteresis_mult)).index)
        bot_2k = (set(scores.tail(int(cfg.k_short * cfg.hysteresis_mult)).index)
                  if cfg.k_short > 0 else set())

        allow_long = allow_short = True
        if cfg.regime_filter and regime_series is not None and today in regime_series.index:
            up = bool(regime_series.loc[today])
            allow_long, allow_short = up, not up

        # 4a) 信号反转出场（hysteresis 边界外）
        for sym in list(positions.keys()):
            pos = positions[sym]
            should_close = ((pos.side == 1 and sym not in top_2k) or
                            (pos.side == -1 and sym not in bot_2k))
            if not should_close:
                continue
            df = panel[sym]
            if today in df.index:
                px = df.loc[today, "close"]
                close_c = _close_cost(cfg, pos.shares, px, pos.side)
                equity -= close_c
                gross_pnl = (px - pos.entry_price) * pos.shares * pos.side
                pnl_usd = gross_pnl - pos.open_cost - close_c
                pnl_pct = pnl_usd / (pos.entry_price * pos.shares) if pos.shares > 0 else 0
                trades.append({
                    "symbol": sym, "side": pos.side,
                    "entry_date": pos.entry_date, "exit_date": today,
                    "entry_price": pos.entry_price, "exit_price": px,
                    "shares": pos.shares, "days_held": pos.days_held,
                    "reason": "signal_exit", "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                    "costs": pos.open_cost + close_c,
                })
                _log_close(pos.side, sym, pos.entry_date, today, pos.entry_price,
                           px, pos.shares, pos.days_held, "signal_exit")
            del positions[sym]

        # 4b) 开新仓
        cur_long_n = sum(1 for p in positions.values() if p.side == 1)
        cur_short_n = sum(1 for p in positions.values() if p.side == -1)
        need_long = top_k - {s for s, p in positions.items() if p.side == 1} if allow_long else set()
        need_short = bot_k - {s for s, p in positions.items() if p.side == -1} if allow_short else set()

        for sym in need_long:
            if cur_long_n >= cfg.k_long:
                break
            if sym in positions or sym not in day_panel.index:
                continue
            row = day_panel.loc[sym]
            px = row["close"]; atr = row["atr14"]
            if pd.isna(px) or pd.isna(atr) or px <= 0:
                continue
            stop_dist = max(cfg.stop_loss_pct * px, cfg.stop_loss_atr_mult * atr)
            notional = equity * long_per_pos
            shares = notional / px
            stop_px = px - stop_dist
            open_c = _open_cost(cfg, shares, px, side=1)
            equity -= open_c
            positions[sym] = Position(
                side=1, entry_date=today, entry_price=px,
                weight=long_per_pos, shares=shares, stop_price=stop_px,
                open_cost=open_c,
            )
            cur_long_n += 1
            _log_open(1, sym, today, px, stop_px, long_per_pos,
                      equity, cur_long_n, cur_short_n)

        for sym in need_short:
            if cur_short_n >= cfg.k_short:
                break
            if sym in positions or sym not in day_panel.index:
                continue
            row = day_panel.loc[sym]
            px = row["close"]; atr = row["atr14"]
            if pd.isna(px) or pd.isna(atr) or px <= 0:
                continue
            stop_dist = max(cfg.stop_loss_pct * px, cfg.stop_loss_atr_mult * atr)
            notional = equity * short_per_pos
            shares = notional / px
            stop_px = px + stop_dist
            open_c = _open_cost(cfg, shares, px, side=-1)
            equity -= open_c
            positions[sym] = Position(
                side=-1, entry_date=today, entry_price=px,
                weight=short_per_pos, shares=shares, stop_price=stop_px,
                open_cost=open_c,
            )
            cur_short_n += 1
            _log_open(-1, sym, today, px, stop_px, short_per_pos,
                      equity, cur_long_n, cur_short_n)

        long_counts.append(cur_long_n)
        short_counts.append(cur_short_n)

        if cfg.print_daily_positions and (cur_long_n + cur_short_n) > 0:
            longs = [s for s, p in positions.items() if p.side == 1]
            shorts = [s for s, p in positions.items() if p.side == -1]
            print(f"  [{today.strftime('%Y-%m-%d')}] equity={_fmt_usd(equity)}  "
                  f"L({len(longs)}): {','.join(longs)}  "
                  f"S({len(shorts)}): {','.join(shorts)}")
        prev_date = today

    idx = pd.DatetimeIndex(all_dates)
    return BacktestResult(
        equity=pd.Series(equity_curve, index=idx, name="equity"),
        daily_returns=pd.Series(daily_rets, index=idx, name="ret"),
        trades=trades,
        long_count=pd.Series(long_counts, index=idx),
        short_count=pd.Series(short_counts, index=idx),
    )


# ---------------- 评估与打印 ----------------

def summarize(result: BacktestResult, cfg: Config,
              benchmark: Optional[pd.Series] = None) -> dict:
    rets = result.daily_returns.dropna()
    n = len(rets)
    eq = result.equity
    cum_ret = eq.iloc[-1] / cfg.starting_capital - 1 if n > 0 else 0
    years = n / 252
    cagr = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    sharpe = rets.mean() * 252 / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = dd.min() if n else 0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0

    trades = result.trades
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_hold = np.mean([t["days_held"] for t in trades]) if trades else 0
    long_trades = [t for t in trades if t["side"] == 1]
    short_trades = [t for t in trades if t["side"] == -1]
    long_pnl = sum(t["pnl_usd"] for t in long_trades)
    short_pnl = sum(t["pnl_usd"] for t in short_trades)
    total_costs = sum(t.get("costs", 0) for t in trades)

    out = {
        "区间": f"{result.equity.index[0].strftime('%Y-%m-%d')} ~ "
                 f"{result.equity.index[-1].strftime('%Y-%m-%d')} ({years:.2f}年)",
        "起始本金": _fmt_usd(cfg.starting_capital),
        "终值": _fmt_usd(eq.iloc[-1]) if n else "-",
        "累计收益": f"{cum_ret*100:+.2f}%  ({_fmt_usd(eq.iloc[-1] - cfg.starting_capital)})" if n else "-",
        "年化收益(CAGR)": f"{cagr*100:.2f}%",
        "年化波动": f"{vol*100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd*100:.2f}%",
        "Calmar": f"{calmar:.2f}",
        "总交易笔数": str(len(trades)),
        "  其中多头": f"{len(long_trades)}笔, 累计 {_fmt_usd(long_pnl)}",
        "  其中空头": f"{len(short_trades)}笔, 累计 {_fmt_usd(short_pnl)}",
        "总交易成本": f"{_fmt_usd(total_costs)} ({total_costs/cfg.starting_capital*100:.2f}%)",
        "胜率": f"{win_rate*100:.1f}%",
        "平均持仓天数": f"{avg_hold:.1f}",
        "平均多头持仓数": f"{result.long_count.mean():.1f}",
        "平均空头持仓数": f"{result.short_count.mean():.1f}",
    }
    if benchmark is not None and len(benchmark) > 1:
        bret = benchmark.pct_change().dropna()
        bcum = (1 + bret).prod()
        byears = len(bret) / 252
        bcagr = bcum ** (1 / byears) - 1 if byears > 0 else 0
        bvol = bret.std() * np.sqrt(252)
        bsharpe = bret.mean() * 252 / bvol if bvol > 0 else 0
        bdd = (benchmark / benchmark.cummax() - 1).min()
        out["基准(QQQ)累计"] = f"{(bcum-1)*100:+.2f}%"
        out["基准(QQQ)CAGR"] = f"{bcagr*100:.2f}%"
        out["基准(QQQ)Sharpe"] = f"{bsharpe:.2f}"
        out["基准(QQQ)MDD"] = f"{bdd*100:.2f}%"
    return out


def print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("                        回测汇总")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<20s} {v}")
    print("=" * 60)


def print_top_trades(trades: List[dict], n: int = 10):
    if not trades:
        return
    sorted_t = sorted(trades, key=lambda t: t["pnl_usd"], reverse=True)
    print("\n----- 盈利 Top 10 -----")
    for t in sorted_t[:n]:
        s = "LONG " if t["side"] == 1 else "SHORT"
        print(f"  {t['symbol']:<8} {s} {t['entry_date'].strftime('%Y-%m-%d')} → "
              f"{t['exit_date'].strftime('%Y-%m-%d')}  "
              f"${t['entry_price']:7.2f} → ${t['exit_price']:7.2f}  "
              f"{t['pnl_pct']*100:+6.2f}% / {_fmt_usd(t['pnl_usd']):>10s}  "
              f"({t['days_held']}d, {t['reason']})")
    print("\n----- 亏损 Top 10 -----")
    for t in sorted_t[-n:][::-1]:
        s = "LONG " if t["side"] == 1 else "SHORT"
        print(f"  {t['symbol']:<8} {s} {t['entry_date'].strftime('%Y-%m-%d')} → "
              f"{t['exit_date'].strftime('%Y-%m-%d')}  "
              f"${t['entry_price']:7.2f} → ${t['exit_price']:7.2f}  "
              f"{t['pnl_pct']*100:+6.2f}% / {_fmt_usd(t['pnl_usd']):>10s}  "
              f"({t['days_held']}d, {t['reason']})")


# ---------------- 入口 ----------------

def main():
    end_date = date.today() if BACKTEST_END == "today" else date.fromisoformat(BACKTEST_END)
    cfg = Config(start=date.fromisoformat(BACKTEST_START), end=end_date)

    print("=" * 60)
    print(f"  策略配置")
    print("=" * 60)
    print(f"  区间          {cfg.start} ~ {cfg.end}")
    print(f"  起始本金      {_fmt_usd(cfg.starting_capital)}")
    print(f"  最大持仓      多 {cfg.k_long} / 空 {cfg.k_short}  "
          f"(共 {cfg.k_long + cfg.k_short} 只)")
    print(f"  多头权重      {cfg.long_weight_frac*100:.0f}% of gross "
          f"({cfg.long_weight_frac*cfg.gross_leverage*100/cfg.k_long if cfg.k_long else 0:.1f}%/只)")
    print(f"  空头权重      {(1-cfg.long_weight_frac)*100:.0f}% of gross "
          f"({(1-cfg.long_weight_frac)*cfg.gross_leverage*100/cfg.k_short if cfg.k_short else 0:.1f}%/只)")
    print(f"  动量/反转     {cfg.mom_weight:.2f} / {1-cfg.mom_weight:.2f}")
    print(f"  Hysteresis    {cfg.hysteresis_mult}x K")
    print(f"  止损          max({cfg.stop_loss_pct*100:.0f}%, {cfg.stop_loss_atr_mult}×ATR14)")
    print(f"  最大持仓天    {cfg.max_hold_days}")
    print(f"  Regime过滤    {'开启' if cfg.regime_filter else '关闭'}")
    print(f"  逐笔打印      {'开启' if cfg.verbose_trades else '关闭'}")

    syms = get_universe()
    print(f"\n[数据] 加载 {len(syms)} 只 NAS100 成分股...")
    data = load_all_data(syms, cfg.start, cfg.end)
    panel = build_panel(data)

    regime_series = None
    if cfg.regime_filter:
        spy_df = fetch_daily_bars("SPY.US", cfg.start - timedelta(days=400), cfg.end,
                                   log_cache=False)
        spy_close = spy_df["close"]
        regime_series = (spy_close > spy_close.rolling(200).mean())
        up_days = int(regime_series.sum())
        print(f"[regime] SPY 200DMA：{up_days}/{len(regime_series)} 日上行")

    if cfg.verbose_trades:
        print("\n----- 交易明细 -----")
    result = run_backtest(panel, cfg, regime_series=regime_series)

    qqq_df = fetch_daily_bars("QQQ.US", cfg.start - timedelta(days=10), cfg.end,
                               log_cache=False)
    qqq = qqq_df.loc[
        (qqq_df.index >= pd.Timestamp(cfg.start)) & (qqq_df.index <= pd.Timestamp(cfg.end)),
        "close",
    ] if len(qqq_df) else None

    summary = summarize(result, cfg, qqq)
    print_summary(summary)
    print_top_trades(result.trades)


if __name__ == "__main__":
    main()
