#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crossrank 对冲实验：在 DAILY 模式基础上叠加 SPY/QQQ 做空对冲，研究市场中性化版本。

设计要点：
  - 仅 DAILY 模式（日 K，不依赖分钟数据，方便快速实验）
  - 完全复用 backtest.py 的指标计算 / panel 构造 / Phase A-B-C 主循环
  - 在每个交易日额外维护一笔 benchmark 短头寸（SPY 或 QQQ）：
      * 每日开始：用昨日 close → 今日 close 给已有空头做 MTM
      * 每日结束（Phase C 之后）：根据当日多头实际 market value 调整 hedge_short_notional
        到 target = ratio × beta × long_mv，并为 |delta| 部分计提调仓成本（滑点 +
        平台费 + 卖出方向的 SEC/TAF）
  - 三种对冲模式：
      none           : 纯多头基准（与原 backtest.py DAILY 一致，用于复现）
      static[r]      : 固定 hedge_ratio = r 倍多头 notional
      rolling_beta   : 动态对冲，beta 用过去 N 日策略 pre-hedge returns 与 benchmark
                       returns 的协方差/方差估计；首段未到 lookback 时退化为 1.0

输出：每个变体的 Sharpe / CAGR / MDD / Calmar / 与基准 QQQ 相关性，再做并排对比。

运行：  python3 backtest_hedged.py
"""

from __future__ import annotations

# ============================================================================
#                              用户可调参数
# ============================================================================

# ---- 区间（沿用主 backtest 的 DAILY 区间） ----
DAILY_START   = "2020-01-01"
BACKTEST_END  = "today"
STARTING_CAPITAL = 100_000

# ---- 对冲实验配置 ----
# 跑哪些变体，列表里每项 = (name, instrument, mode, ratio)
#   instrument: "QQQ.US" / "SPY.US"
#   mode      : "none" / "static" / "rolling_beta"
#   ratio     : static 模式下 = hedge_notional / long_mv
#               rolling_beta 模式下 = 乘到估计 beta 上的额外 scale
HEDGE_VARIANTS = [
    ("baseline (no hedge)",           None,      "none",          0.0),
    ("static QQQ 0.5x",               "QQQ.US",  "static",        0.5),
    ("static QQQ 1.0x",               "QQQ.US",  "static",        1.0),
    ("static SPY 0.5x",               "SPY.US",  "static",        0.5),
    ("static SPY 1.0x",               "SPY.US",  "static",        1.0),
    ("rolling-beta QQQ 1.0x",         "QQQ.US",  "rolling_beta",  1.0),
    ("rolling-beta SPY 1.0x",         "SPY.US",  "rolling_beta",  1.0),
]

# 滚动 beta 估计窗口
BETA_LOOKBACK = 60       # 过去 60 个交易日（≈3 个月）
BETA_MIN_OBS  = 20       # 至少累计 20 日 returns 才开始估计（之前用 1.0）
BETA_MIN, BETA_MAX = 0.0, 2.0   # 限制 beta 估计的合理区间，避免极值

# ---- 是否使用 regime 过滤（保持与原 backtest 一致：开启） ----
REGIME_FILTER = True

# ============================================================================
#                       以下为实现，一般不需要修改
# ============================================================================

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 复用 backtest.py 全部基础组件（指标、Position、Config、辅助函数）
from backtest import (
    Config, Position, BacktestResult,
    load_all_data, build_daily_panel, build_panel,
    composite_score, compute_indicators,
    _open_cost, _close_cost, _fmt_usd,
    summarize, print_summary, print_top_trades,
    K_LONG, K_SHORT, LONG_WEIGHT_FRAC, GROSS_LEVERAGE,
    HYSTERESIS_MULT, MOM_WEIGHT, BIAS_WEIGHT,
    STOP_LOSS_PCT, STOP_LOSS_ATR_MULT, MAX_HOLD_DAYS, MIN_DOLLAR_VOLUME,
    VOL_TARGET_ANNUAL, VOL_TARGET_LOOKBACK, VOL_SCALE_MIN, VOL_SCALE_MAX,
    ENABLE_COSTS, PLATFORM_FEE_PER_SHARE, PLATFORM_FEE_MIN,
    SEC_FEE_RATE, TAF_PER_SHARE, TAF_MAX_PER_ORDER, SLIPPAGE_BPS,
    INTRADAY_PERIOD, DECISION_TIME_ET,
)
from longport_api import fetch_daily_bars
from universe import get_universe


# ---------------- 对冲配置 / 状态 ----------------

@dataclass
class HedgeCfg:
    name: str
    instrument: Optional[str]      # "QQQ.US" / "SPY.US" / None
    mode: str                      # "none" / "static" / "rolling_beta"
    ratio: float = 1.0
    beta_lookback: int = BETA_LOOKBACK
    beta_min_obs: int = BETA_MIN_OBS
    beta_clip: Tuple[float, float] = (BETA_MIN, BETA_MAX)


@dataclass
class HedgeState:
    short_notional: float = 0.0    # 当前对冲的市值（始终 ≥ 0；short = -short_notional 美元敞口）
    last_close: float = 0.0        # 上一次 MTM 时的 benchmark close
    total_costs: float = 0.0       # 累计调仓成本
    rebalance_count: int = 0       # 调仓次数（包括首次开仓）
    pnl_dollars: float = 0.0       # 累计对冲层 PnL（不含调仓成本）


# ---------------- 工具 ----------------

def rolling_beta(strat_rets: List[float], bench_rets: List[float],
                 window: int, min_obs: int,
                 clip: Tuple[float, float]) -> float:
    """用过去 `window` 日的策略 pre-hedge returns 与 benchmark returns 估计 beta。"""
    n = min(len(strat_rets), len(bench_rets))
    if n < min_obs:
        return 1.0
    n = min(n, window)
    s = np.asarray(strat_rets[-n:])
    b = np.asarray(bench_rets[-n:])
    var_b = b.var(ddof=0)
    if var_b < 1e-12:
        return 1.0
    beta = float(np.cov(s, b, ddof=0)[0, 1] / var_b)
    return float(np.clip(beta, clip[0], clip[1]))


def hedge_rebalance_cost(cfg: 'Config', delta_notional: float,
                          price: float, side_of_trade: int) -> float:
    """
    benchmark 调仓成本（绝对值 delta_notional）。
    side_of_trade: +1 = 卖出（增加空头 / 平多）触发 SEC/TAF
                   -1 = 买入（减少空头 / 开多）不触发 SEC/TAF
    """
    if not cfg.enable_costs or delta_notional <= 0 or price <= 0:
        return 0.0
    shares = delta_notional / price
    slip = delta_notional * cfg.slippage_bps / 10000.0
    plat = max(shares * cfg.platform_fee_per_share, cfg.platform_fee_min)
    cost = slip + plat
    if side_of_trade == 1:
        cost += delta_notional * cfg.sec_fee_rate
        cost += min(shares * cfg.taf_per_share, cfg.taf_max_per_order)
    return cost


# ---------------- 带对冲层的回测引擎 ----------------

def run_backtest_hedged(panel: Dict[str, pd.DataFrame], cfg: Config,
                         hedge_cfg: HedgeCfg,
                         regime_series: Optional[pd.Series] = None,
                         bench_close: Optional[pd.Series] = None
                         ) -> Tuple[BacktestResult, HedgeState, pd.Series]:
    """
    在 DAILY 模式上叠加一层 SPY/QQQ 做空对冲。
    与原 run_backtest 的主体逻辑一致；新增：
      - 每日开盘前：MTM 已有 hedge 短仓
      - 每日 Phase C 后：按当日 long_mv 重设 target hedge 并扣调仓成本
    返回 (BacktestResult, HedgeState, gross_strategy_returns_series)
    其中 gross_strategy_returns 用来校验 / 对比 pre-hedge 的策略表现。
    """
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

    # DAILY 模式 panel 已经是 build_daily_panel 的结果，本身就是 proxy 指标
    proxy_panel = panel

    # ---- 对冲层状态 ----
    hstate = HedgeState()
    bench_returns_history: List[float] = []
    strat_pre_hedge_returns: List[float] = []

    def _realize_close(sym: str, exit_px: float, reason: str,
                       exit_date: pd.Timestamp) -> float:
        nonlocal equity
        pos = positions.pop(sym)
        close_c = _close_cost(cfg, pos.shares, exit_px, pos.side)
        equity -= close_c
        gross_pnl = (exit_px - pos.entry_price) * pos.shares * pos.side
        pnl_usd = gross_pnl - pos.open_cost - close_c
        pnl_pct = pnl_usd / (pos.entry_price * pos.shares) if pos.shares > 0 else 0
        trades.append({
            "symbol": sym, "side": pos.side,
            "entry_date": pos.entry_date, "exit_date": exit_date,
            "entry_price": pos.entry_price, "exit_price": exit_px,
            "shares": pos.shares, "days_held": pos.days_held,
            "reason": reason, "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
            "costs": pos.open_cost + close_c,
        })
        if pos.last_mark > 0:
            return (exit_px / pos.last_mark - 1) * pos.side * pos.weight
        return 0.0

    for i, today in enumerate(all_dates):
        day_pnl_strat = 0.0     # 策略本身（多头层）的当日 ratio
        day_pnl_hedge = 0.0     # 对冲层当日 ratio

        def _row(sym):
            df = panel[sym]
            return df.loc[today] if today in df.index else None

        def _proxy_row(sym):
            df = proxy_panel[sym]
            return df.loc[today] if today in df.index else None

        # ============ 对冲 MTM（用昨 close → 今 close）============
        bench_close_today = (float(bench_close.loc[today])
                             if bench_close is not None and today in bench_close.index
                             else None)
        bench_ret_today = 0.0
        if bench_close_today is not None and hstate.last_close > 0:
            bench_ret_today = bench_close_today / hstate.last_close - 1
            if hstate.short_notional > 0:
                pnl_dollar = -hstate.short_notional * bench_ret_today
                day_pnl_hedge += pnl_dollar / equity
                hstate.pnl_dollars += pnl_dollar

        # ============ Phase A: 日内止损扫描 ============
        for sym in list(positions.keys()):
            pos = positions[sym]
            row = _row(sym)
            if row is None:
                continue
            gap_open = row.get("gap_open", np.nan)
            pd_low = row.get("pd_low", np.nan)
            pd_high = row.get("pd_high", np.nan)
            if pd.isna(gap_open):
                continue
            stop_hit = False
            exit_px = np.nan
            if pos.side == 1:
                if gap_open <= pos.stop_price:
                    stop_hit, exit_px = True, gap_open
                elif not pd.isna(pd_low) and pd_low <= pos.stop_price:
                    stop_hit, exit_px = True, pos.stop_price
            else:
                if gap_open >= pos.stop_price:
                    stop_hit, exit_px = True, gap_open
                elif not pd.isna(pd_high) and pd_high >= pos.stop_price:
                    stop_hit, exit_px = True, pos.stop_price
            if stop_hit:
                day_pnl_strat += _realize_close(sym, float(exit_px), "stop_loss", today)

        # ============ Phase B: 决策时点 ============
        today_data = {}
        for sym in symbols:
            prow = _proxy_row(sym)
            if prow is None:
                continue
            if pd.isna(prow.get("pd_close")) or pd.isna(prow.get("mom_60")) \
               or pd.isna(prow.get("atr14")):
                continue
            if prow.get("dollar_vol_20", 0) < cfg.min_dollar_volume:
                continue
            today_data[sym] = prow

        has_panel = len(today_data) >= cfg.k_long + cfg.k_short
        if has_panel:
            day_panel = pd.DataFrame(today_data).T
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

            # 波动率目标缩放（用策略 pre-hedge 的 returns）
            vol_scale = 1.0
            if cfg.vol_target_annual > 0 and len(strat_pre_hedge_returns) >= cfg.vol_target_lookback:
                recent = np.asarray(strat_pre_hedge_returns[-cfg.vol_target_lookback:])
                sd = recent.std()
                if sd > 1e-6:
                    rv = sd * np.sqrt(252)
                    vol_scale = float(np.clip(cfg.vol_target_annual / rv,
                                               cfg.vol_scale_min, cfg.vol_scale_max))
            long_per_pos = long_per_pos_base * vol_scale
            short_per_pos = short_per_pos_base * vol_scale

            # B.2 信号/regime/max_hold 平仓
            for sym in list(positions.keys()):
                pos = positions[sym]
                prow = _proxy_row(sym)
                if prow is None or pd.isna(prow.get("pd_close")):
                    continue
                reason = None
                if pos.days_held >= cfg.max_hold_days:
                    reason = "max_hold"
                elif pos.side == 1 and (sym not in top_2k or not allow_long):
                    reason = "signal_exit"
                elif pos.side == -1 and (sym not in bot_2k or not allow_short):
                    reason = "signal_exit"
                if reason is None:
                    continue
                exit_px = float(prow["pd_close"])
                day_pnl_strat += _realize_close(sym, exit_px, reason, today)

            # B.3 开仓
            cur_long_n = sum(1 for p in positions.values() if p.side == 1)
            cur_short_n = sum(1 for p in positions.values() if p.side == -1)
            held = set(positions.keys())

            if allow_long and cur_long_n < cfg.k_long:
                for sym in scores.index:
                    if cur_long_n >= cfg.k_long:
                        break
                    if sym not in top_k or sym in held:
                        continue
                    prow = day_panel.loc[sym]
                    px = float(prow["pd_close"]); atr = float(prow["atr14"])
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
                        open_cost=open_c, last_mark=px,
                    )
                    cur_long_n += 1

            if allow_short and cur_short_n < cfg.k_short:
                for sym in scores.index[::-1]:
                    if cur_short_n >= cfg.k_short:
                        break
                    if sym not in bot_k or sym in held:
                        continue
                    prow = day_panel.loc[sym]
                    px = float(prow["pd_close"]); atr = float(prow["atr14"])
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
                        open_cost=open_c, last_mark=px,
                    )
                    cur_short_n += 1

        # ============ Phase C: MTM 到当日真收盘 ============
        long_mv_today = 0.0
        short_mv_today = 0.0
        for sym, pos in positions.items():
            row = _row(sym)
            if row is None:
                continue
            today_close = row.get("close", np.nan)
            if pd.isna(today_close) or pos.last_mark <= 0:
                continue
            day_pnl_strat += (today_close / pos.last_mark - 1) * pos.side * pos.weight
            pos.last_mark = float(today_close)
            pos.days_held += 1
            mv = pos.shares * float(today_close)
            if pos.side == 1:
                long_mv_today += mv
            else:
                short_mv_today += mv

        # ============ 综合 day_pnl 并更新 equity ============
        day_pnl = day_pnl_strat + day_pnl_hedge
        equity *= (1 + day_pnl)
        daily_rets.append(day_pnl)
        equity_curve.append(equity)

        # 记录用于 beta 估计的 returns 与 benchmark returns（必须同步）
        if bench_close_today is not None:
            strat_pre_hedge_returns.append(day_pnl_strat)
            bench_returns_history.append(bench_ret_today)

        # ============ 调整 hedge 至今日目标 ============
        if hedge_cfg.mode != "none" and bench_close_today is not None and bench_close_today > 0:
            if hedge_cfg.mode == "static":
                hedge_factor = hedge_cfg.ratio
            elif hedge_cfg.mode == "rolling_beta":
                beta = rolling_beta(
                    strat_pre_hedge_returns, bench_returns_history,
                    hedge_cfg.beta_lookback, hedge_cfg.beta_min_obs,
                    hedge_cfg.beta_clip,
                )
                hedge_factor = max(0.0, hedge_cfg.ratio * beta)
            else:
                hedge_factor = 0.0
            target_short = hedge_factor * long_mv_today
            delta = target_short - hstate.short_notional
            if abs(delta) > 1e-6:
                # delta>0: 增加空头 = 卖出（触发 SEC/TAF）
                # delta<0: 减少空头 = 买入回补
                side_of_trade = 1 if delta > 0 else -1
                cost = hedge_rebalance_cost(cfg, abs(delta), bench_close_today, side_of_trade)
                equity -= cost
                hstate.total_costs += cost
                hstate.rebalance_count += 1
            hstate.short_notional = max(0.0, target_short)
            hstate.last_close = bench_close_today
        elif bench_close_today is not None:
            hstate.last_close = bench_close_today

        cur_long_n = sum(1 for p in positions.values() if p.side == 1)
        cur_short_n = sum(1 for p in positions.values() if p.side == -1)
        long_counts.append(cur_long_n)
        short_counts.append(cur_short_n)

    idx = pd.DatetimeIndex(all_dates)
    result = BacktestResult(
        equity=pd.Series(equity_curve, index=idx, name="equity"),
        daily_returns=pd.Series(daily_rets, index=idx, name="ret"),
        trades=trades,
        long_count=pd.Series(long_counts, index=idx),
        short_count=pd.Series(short_counts, index=idx),
    )
    # 把 pre-hedge 策略 returns 也返回，便于 beta / 相关性后处理
    pre_hedge_idx = idx[:len(strat_pre_hedge_returns)]
    pre_hedge_series = pd.Series(strat_pre_hedge_returns, index=pre_hedge_idx,
                                  name="strat_pre_hedge_ret")
    return result, hstate, pre_hedge_series


# ---------------- 评估辅助 ----------------

def _ann_stats(rets: pd.Series) -> Dict[str, float]:
    rets = rets.dropna()
    if len(rets) == 0:
        return dict(cum=0.0, cagr=0.0, vol=0.0, sharpe=0.0, mdd=0.0, calmar=0.0)
    eq = (1 + rets).cumprod()
    cum = float(eq.iloc[-1] - 1)
    years = len(rets) / 252.0
    cagr = (eq.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0
    sd = rets.std(ddof=0)
    vol = float(sd * np.sqrt(252))
    sharpe = float(rets.mean() * 252 / vol) if vol > 1e-9 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    mdd = float(dd) if not np.isnan(dd) else 0.0
    calmar = (cagr / abs(mdd)) if mdd < 0 else 0.0
    return dict(cum=cum, cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar)


def _corr_with(rets: pd.Series, bench_rets: pd.Series) -> float:
    rets = rets.dropna()
    bench_rets = bench_rets.dropna()
    common = rets.index.intersection(bench_rets.index)
    if len(common) < 30:
        return float("nan")
    a = rets.reindex(common).values
    b = bench_rets.reindex(common).values
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _empirical_beta(rets: pd.Series, bench_rets: pd.Series) -> float:
    common = rets.dropna().index.intersection(bench_rets.dropna().index)
    if len(common) < 30:
        return float("nan")
    a = rets.reindex(common).values
    b = bench_rets.reindex(common).values
    var_b = b.var(ddof=0)
    if var_b < 1e-12:
        return float("nan")
    return float(np.cov(a, b, ddof=0)[0, 1] / var_b)


def print_variant_summary(name: str, result: BacktestResult, hstate: HedgeState,
                           cfg: Config, qqq_close: pd.Series):
    s = _ann_stats(result.daily_returns)
    qqq_rets = qqq_close.pct_change()
    corr_qqq = _corr_with(result.daily_returns, qqq_rets)
    beta_qqq = _empirical_beta(result.daily_returns, qqq_rets)
    n_trades = len(result.trades)
    long_pnl = sum(t["pnl_usd"] for t in result.trades if t["side"] == 1)
    total_strat_costs = sum(t.get("costs", 0) for t in result.trades)
    total_costs = total_strat_costs + hstate.total_costs

    print(f"\n----- {name} -----")
    print(f"  累计收益    {s['cum']*100:+8.2f}%   "
          f"CAGR {s['cagr']*100:6.2f}%   "
          f"年化波动 {s['vol']*100:5.2f}%   "
          f"Sharpe {s['sharpe']:5.2f}   "
          f"MDD {s['mdd']*100:6.2f}%   "
          f"Calmar {s['calmar']:.2f}")
    print(f"  Corr(QQQ)   {corr_qqq:+5.2f}   "
          f"Beta(QQQ) {beta_qqq:+5.2f}   ", end="")
    print(f"对冲累计 PnL ${hstate.pnl_dollars:>+10,.0f}   "
          f"调仓次数 {hstate.rebalance_count}")
    print(f"  策略多头交易 {n_trades} 笔, 累计 PnL ${long_pnl:>+10,.0f}   "
          f"策略成本 ${total_strat_costs:>+9,.0f}   "
          f"对冲成本 ${hstate.total_costs:>+8,.0f}   "
          f"合计成本 ${total_costs:>+9,.0f} ({total_costs/cfg.starting_capital*100:.1f}%)")


def print_compare_table(rows: List[Tuple[str, dict]]):
    """rows: [(name, stats_dict), ...]，stats_dict 包含统计指标"""
    headers = ["变体", "累计%", "CAGR%", "Vol%", "Sharpe", "MDD%", "Calmar",
               "Corr(QQQ)", "Beta(QQQ)", "成本%"]
    print("\n" + "=" * 130)
    print("                                  对冲变体并排对比")
    print("=" * 130)
    fmt = "{:<26s}{:>9}{:>9}{:>9}{:>9}{:>10}{:>9}{:>11}{:>11}{:>9}"
    print(fmt.format(*headers))
    print("-" * 130)
    for name, st in rows:
        print(fmt.format(
            name[:26],
            f"{st['cum']*100:+.1f}",
            f"{st['cagr']*100:+.1f}",
            f"{st['vol']*100:.1f}",
            f"{st['sharpe']:+.2f}",
            f"{st['mdd']*100:+.1f}",
            f"{st['calmar']:.2f}",
            f"{st['corr']:+.2f}",
            f"{st['beta']:+.2f}",
            f"{st['cost']*100:.1f}",
        ))
    print("=" * 130)


def print_yearly_grid(yearly: Dict[str, Dict[int, dict]]):
    years = sorted({y for v in yearly.values() for y in v.keys()})
    names = list(yearly.keys())
    print("\n" + "=" * 96)
    print("                                  逐年收益% 对比（{} 个变体）".format(len(names)))
    print("=" * 96)
    head = "{:<6}".format("年份") + "".join(f"{n[:18]:>20s}" for n in names)
    print(head)
    print("-" * 96)
    for y in years:
        row = f"{y:<6}"
        for n in names:
            d = yearly.get(n, {}).get(y)
            row += f"{d['ret']:>+19.2f} " if d else f"{'--':>20s}"
        print(row)
    print("=" * 96)


def _yearly_stats(result: BacktestResult) -> Dict[int, dict]:
    rets = result.daily_returns.dropna()
    eq = result.equity
    out: Dict[int, dict] = {}
    for y, r in rets.groupby(rets.index.year):
        eq_y = eq.loc[eq.index.year == y]
        dd = (eq_y / eq_y.cummax() - 1).min() if len(eq_y) else 0.0
        sd = r.std(ddof=0)
        out[int(y)] = {
            "ret":    float((1 + r).prod() - 1) * 100,
            "mdd":    float(dd) * 100,
            "sharpe": float(r.mean() * 252 / (sd * np.sqrt(252))) if sd > 1e-12 else 0.0,
        }
    return out


# ---------------- 主入口 ----------------

def _make_cfg(start: date, end: date) -> Config:
    return Config(
        start=start, end=end, mode="daily",
        starting_capital=STARTING_CAPITAL,
        regime_filter=REGIME_FILTER,
    )


def main():
    end_date = date.today() if BACKTEST_END == "today" else date.fromisoformat(BACKTEST_END)
    start_date = date.fromisoformat(DAILY_START)
    cfg = _make_cfg(start_date, end_date)

    print("=" * 70)
    print("  crossrank 对冲实验（DAILY 日 K 模式）")
    print("=" * 70)
    print(f"  区间        {cfg.start} ~ {cfg.end}")
    print(f"  起始本金    {_fmt_usd(cfg.starting_capital)}")
    print(f"  K_LONG/K_SHORT  {cfg.k_long} / {cfg.k_short}")
    print(f"  动量/反转   {cfg.mom_weight:.2f} / {1-cfg.mom_weight:.2f}   bias {cfg.bias_weight:.2f}")
    print(f"  Regime过滤  {'开启 (SPY 200DMA)' if cfg.regime_filter else '关闭'}")
    print(f"  对冲变体    {len(HEDGE_VARIANTS)} 组：")
    for v in HEDGE_VARIANTS:
        print(f"              - {v[0]}  (instrument={v[1]}, mode={v[2]}, ratio={v[3]})")

    syms = get_universe()
    print(f"\n[数据] 加载 {len(syms)} 只成分股日线 ({cfg.start} ~ {cfg.end})...")
    data = load_all_data(syms, cfg.start, cfg.end)
    print("[数据] 构造 DAILY panel...")
    daily_panel = build_daily_panel(data)

    # SPY regime + benchmark
    spy_df = fetch_daily_bars("SPY.US",
                               cfg.start - timedelta(days=400),
                               cfg.end, log_cache=False)
    spy_close = spy_df["close"]
    regime_series = (spy_close > spy_close.rolling(200).mean())

    qqq_df = fetch_daily_bars("QQQ.US",
                               cfg.start - timedelta(days=10),
                               cfg.end, log_cache=False)
    qqq_close = qqq_df["close"]
    spy_close_ranged = spy_df["close"].loc[
        (spy_df.index >= pd.Timestamp(cfg.start)) & (spy_df.index <= pd.Timestamp(cfg.end))
    ]
    qqq_close_ranged = qqq_df["close"].loc[
        (qqq_df.index >= pd.Timestamp(cfg.start)) & (qqq_df.index <= pd.Timestamp(cfg.end))
    ]

    # 不同 instrument 的 close 字典，方便循环
    bench_close_map = {
        "QQQ.US": qqq_close_ranged,
        "SPY.US": spy_close_ranged,
    }

    # ---------- 跑所有变体 ----------
    rows_for_table: List[Tuple[str, dict]] = []
    yearly_grid: Dict[str, Dict[int, dict]] = {}
    detailed_results: Dict[str, BacktestResult] = {}

    for name, instrument, mode, ratio in HEDGE_VARIANTS:
        print(f"\n\n========== 跑变体: {name} ==========")
        hcfg = HedgeCfg(name=name, instrument=instrument, mode=mode, ratio=ratio)
        bclose = bench_close_map.get(instrument) if instrument else None
        result, hstate, pre_hedge = run_backtest_hedged(
            daily_panel, cfg, hcfg,
            regime_series=regime_series,
            bench_close=bclose,
        )
        print_variant_summary(name, result, hstate, cfg, qqq_close_ranged)

        s = _ann_stats(result.daily_returns)
        s["corr"] = _corr_with(result.daily_returns, qqq_close_ranged.pct_change())
        s["beta"] = _empirical_beta(result.daily_returns, qqq_close_ranged.pct_change())
        s["cost"] = (sum(t.get("costs", 0) for t in result.trades) + hstate.total_costs) \
                    / cfg.starting_capital
        rows_for_table.append((name, s))
        yearly_grid[name] = _yearly_stats(result)
        detailed_results[name] = result

    # ---------- benchmark QQQ / SPY 自身指标 ----------
    qqq_rets = qqq_close_ranged.pct_change().dropna()
    spy_rets = spy_close_ranged.pct_change().dropna()
    qqq_stats = _ann_stats(qqq_rets); qqq_stats["corr"] = 1.0; qqq_stats["beta"] = 1.0; qqq_stats["cost"] = 0.0
    spy_stats = _ann_stats(spy_rets); spy_stats["corr"] = _corr_with(spy_rets, qqq_rets); spy_stats["beta"] = _empirical_beta(spy_rets, qqq_rets); spy_stats["cost"] = 0.0
    rows_for_table.append(("(基准) QQQ buy&hold", qqq_stats))
    rows_for_table.append(("(基准) SPY buy&hold", spy_stats))

    # ---------- 打印汇总 ----------
    print_compare_table(rows_for_table)
    print_yearly_grid(yearly_grid)

    # ---------- 重点变体的盈/亏 Top 10（仅 baseline 与最佳对冲） ----------
    print("\n\n========== 进一步细节：baseline vs 最佳对冲变体 ==========")
    # 选出 Sharpe 最高的对冲变体（排除 baseline）
    hedged_rows = [r for r in rows_for_table[1:len(HEDGE_VARIANTS)]]
    if hedged_rows:
        best = max(hedged_rows, key=lambda r: r[1]["sharpe"])
        print(f"\n按 Sharpe 排序，最佳对冲变体：{best[0]}（Sharpe {best[1]['sharpe']:.2f}）")
        print("baseline 多头交易 Top 5 / 亏损 Top 5：")
        if "baseline (no hedge)" in detailed_results:
            print_top_trades(detailed_results["baseline (no hedge)"].trades, n=5)


if __name__ == "__main__":
    main()
