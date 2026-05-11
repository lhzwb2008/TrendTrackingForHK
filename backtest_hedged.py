#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crossrank 对冲实验：在 DAILY 模式基础上叠加 SPY/QQQ 做空对冲，研究市场中性化版本。

设计要点：
  - 仅 DAILY 模式（日 K，不依赖分钟数据，方便快速实验）
  - 完全复用 backtest.py 的指标计算 / panel 构造 / Phase A-B-C 主循环
  - **两个独立账户**：
      多头账户：与 backtest.py DAILY 的 baseline 100% 一致 —— 多头交易、笔数、
              累计 PnL、equity 滚动完全相同，不受对冲影响
      对冲账户：独立维护 benchmark 短头寸 PnL 与调仓成本
      总 equity = baseline_equity + 累积对冲 PnL - 累积对冲成本
    这样保证对冲只是「叠加在原策略之上」，对原策略零干扰。
  - 三种对冲模式：
      none           : 纯多头（与原 backtest.py DAILY 完全一致）
      static[r]      : 固定 hedge_short = r × 多头当日真实市值
      rolling_beta   : hedge_short = ratio × rolling_beta × 多头当日真实市值
                       beta 用过去 N 日多头策略 returns 与 benchmark returns 的
                       协方差/方差估计；首段未到 lookback 时退化为 1.0
  - 对冲调仓 size 基于 baseline 多头当日真实市值（按当日真收盘价 MTM 后）
    所以相同多头组合对应同一份对冲规模，不会因对冲而漂移

输出：每个变体的 Sharpe / CAGR / MDD / Calmar / 与基准 QQQ 相关性，再做并排对比。

运行：  python backtest_hedged.py
"""

from __future__ import annotations

# ============================================================================
#                              用户可调参数
# ============================================================================

# ---- 区间：默认跑近 10 年（Longport 日 K 数据有多少用多少） ----
DAILY_START   = "2016-05-01"
BACKTEST_END  = "2026-05-01"
STARTING_CAPITAL = 100_000

# 默认对比 baseline + 两档 QQQ 滚动 beta 对冲（0.5x = 半对冲、1.0x = 严格 zero-beta）
HEDGE_VARIANTS = [
    ("baseline (no hedge)",     None,      "none",          0.0),
    ("rolling-beta QQQ 0.5x",   "QQQ.US",  "rolling_beta",  0.5),
    ("rolling-beta QQQ 1.0x",   "QQQ.US",  "rolling_beta",  1.0),
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
    _yearly_stats as _yearly_stats_basic,
)
from longport_api import fetch_daily_bars
from universe import get_universe, label as sym_label


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
class StratResult:
    """baseline 多头层的回测结果（独立于对冲，完全等价于 backtest.py 的 run_backtest DAILY 模式）。"""
    equity: pd.Series                # 多头层 equity（不含对冲）
    daily_returns: pd.Series         # 多头层日收益（不含对冲）
    trades: List[dict]
    long_count: pd.Series
    short_count: pd.Series
    long_mv: pd.Series                # 每日多头实际总市值（按当日真收盘 close MTM 后；用于决定对冲 size）

    def to_basic(self) -> BacktestResult:
        return BacktestResult(
            equity=self.equity, daily_returns=self.daily_returns,
            trades=self.trades, long_count=self.long_count,
            short_count=self.short_count,
        )


@dataclass
class HedgeOverlayResult:
    """对冲 overlay 后的结果（叠加在 baseline 之上）。"""
    name: str
    cfg: HedgeCfg
    # 总 equity 与日 returns（含对冲）
    equity_total: pd.Series
    daily_returns_total: pd.Series
    # 对冲层独立累计 PnL / 成本
    hedge_pnl_cum: pd.Series          # 累积对冲 PnL（mark-to-market，不含调仓成本）
    hedge_cost_cum: pd.Series         # 累积对冲调仓成本
    short_notional: pd.Series         # 每日对冲短头寸大小
    rebalance_count: int
    # 每日 hedge ratio 实际值（调仓 size 占当日 long_mv 的比例）
    effective_factor: pd.Series


# ---------------- 工具 ----------------

def rolling_beta(strat_rets: List[float], bench_rets: List[float],
                 window: int, min_obs: int,
                 clip: Tuple[float, float]) -> float:
    """用过去 `window` 日的策略多头 returns 与 benchmark returns 估计 beta。"""
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
    """benchmark 调仓成本（绝对值 delta_notional）。
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


# ---------------- baseline 多头回测（与 backtest.py run_backtest DAILY 完全等价，外加 long_mv 记录） ----------------

def run_strategy_baseline(panel: Dict[str, pd.DataFrame], cfg: Config,
                           regime_series: Optional[pd.Series] = None) -> StratResult:
    """
    DAILY 模式 baseline 多头回测；逻辑与 backtest.py 中 run_backtest 等价
    （cfg.mode 视作 'daily'，proxy_panel = panel），额外返回每日多头总市值
    long_mv（按当日真收盘 close MTM 后），供对冲 overlay 使用。
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
    long_mv_curve: List[float] = []
    trades: List[dict] = []

    long_w = cfg.gross_leverage * cfg.long_weight_frac
    short_w = cfg.gross_leverage * (1 - cfg.long_weight_frac)
    long_per_pos_base = long_w / cfg.k_long if cfg.k_long > 0 else 0
    short_per_pos_base = short_w / cfg.k_short if cfg.k_short > 0 else 0

    proxy_panel = panel

    def _row(sym, today):
        df = panel[sym]
        return df.loc[today] if today in df.index else None

    def _proxy_row(sym, today):
        df = proxy_panel[sym]
        return df.loc[today] if today in df.index else None

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
        day_pnl = 0.0

        # ============ Phase A: 日内止损扫描 ============
        for sym in list(positions.keys()):
            pos = positions[sym]
            row = _row(sym, today)
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
                day_pnl += _realize_close(sym, float(exit_px), "stop_loss", today)

        # ============ Phase B: 决策时点 ============
        today_data = {}
        for sym in symbols:
            prow = _proxy_row(sym, today)
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

            for sym in list(positions.keys()):
                pos = positions[sym]
                prow = _proxy_row(sym, today)
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
                day_pnl += _realize_close(sym, exit_px, reason, today)

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
        for sym, pos in positions.items():
            row = _row(sym, today)
            if row is None:
                continue
            today_close = row.get("close", np.nan)
            if pd.isna(today_close) or pos.last_mark <= 0:
                continue
            day_pnl += (today_close / pos.last_mark - 1) * pos.side * pos.weight
            pos.last_mark = float(today_close)
            pos.days_held += 1
            if pos.side == 1:
                long_mv_today += pos.shares * float(today_close)

        equity *= (1 + day_pnl)
        daily_rets.append(day_pnl)
        equity_curve.append(equity)
        long_mv_curve.append(long_mv_today)

        cur_long_n = sum(1 for p in positions.values() if p.side == 1)
        cur_short_n = sum(1 for p in positions.values() if p.side == -1)
        long_counts.append(cur_long_n)
        short_counts.append(cur_short_n)

    idx = pd.DatetimeIndex(all_dates)
    return StratResult(
        equity=pd.Series(equity_curve, index=idx, name="equity"),
        daily_returns=pd.Series(daily_rets, index=idx, name="ret"),
        trades=trades,
        long_count=pd.Series(long_counts, index=idx),
        short_count=pd.Series(short_counts, index=idx),
        long_mv=pd.Series(long_mv_curve, index=idx, name="long_mv"),
    )


# ---------------- 对冲 overlay（纯后处理） ----------------

def simulate_hedge_overlay(strat: StratResult, bench_close: pd.Series,
                            hedge_cfg: HedgeCfg, cfg: Config) -> HedgeOverlayResult:
    """
    在 baseline 多头层之上叠加 benchmark 短头寸 overlay。
    注意：不修改 baseline equity 滚动；只在最终结果里把对冲 PnL/成本叠加上去。
    每日：
      1) 用昨日 close → 今日 close 给已有空仓做 MTM（计入 hedge_pnl）
      2) 算 target = ratio × beta × baseline.long_mv[today]，调仓到 target
         （rolling_beta 用 baseline.daily_returns 与 benchmark returns 的协方差）
      3) 调仓的 |delta| 部分按卖空 / 平空方向计提滑点 + 平台费 + （卖空时）SEC/TAF
    """
    idx = strat.equity.index
    bench_close_aligned = bench_close.reindex(idx).ffill()

    short_notional = 0.0
    last_bench_close: Optional[float] = None
    cum_pnl = 0.0
    cum_cost = 0.0
    rebal_count = 0

    cum_pnl_curve, cum_cost_curve = [], []
    short_notional_curve, factor_curve = [], []

    strat_rets_history: List[float] = []
    bench_rets_history: List[float] = []

    for i, today in enumerate(idx):
        bench_close_today = bench_close_aligned.iloc[i]

        # 1) MTM
        bench_ret_today = 0.0
        if last_bench_close is not None and not np.isnan(bench_close_today) and last_bench_close > 0:
            bench_ret_today = bench_close_today / last_bench_close - 1
            if short_notional > 0:
                pnl = -short_notional * bench_ret_today
                cum_pnl += pnl

        # 把当日 strat returns 与 bench returns 入 history（用于 beta 估计）
        strat_rets_history.append(float(strat.daily_returns.iloc[i]))
        bench_rets_history.append(bench_ret_today if last_bench_close is not None else 0.0)

        # 2) 调仓 target
        long_mv_today = float(strat.long_mv.iloc[i])
        if hedge_cfg.mode == "static":
            hedge_factor = hedge_cfg.ratio
        elif hedge_cfg.mode == "rolling_beta":
            beta = rolling_beta(
                strat_rets_history, bench_rets_history,
                hedge_cfg.beta_lookback, hedge_cfg.beta_min_obs,
                hedge_cfg.beta_clip,
            )
            hedge_factor = max(0.0, hedge_cfg.ratio * beta)
        else:
            hedge_factor = 0.0

        target_short = hedge_factor * long_mv_today
        if not np.isnan(bench_close_today) and bench_close_today > 0:
            delta = target_short - short_notional
            if abs(delta) > 1e-6:
                side = 1 if delta > 0 else -1
                cost = hedge_rebalance_cost(cfg, abs(delta), bench_close_today, side)
                cum_cost += cost
                rebal_count += 1
            short_notional = max(0.0, target_short)
            last_bench_close = float(bench_close_today)

        cum_pnl_curve.append(cum_pnl)
        cum_cost_curve.append(cum_cost)
        short_notional_curve.append(short_notional)
        factor_curve.append(hedge_factor)

    cum_pnl_s = pd.Series(cum_pnl_curve, index=idx, name="hedge_pnl_cum")
    cum_cost_s = pd.Series(cum_cost_curve, index=idx, name="hedge_cost_cum")
    short_s = pd.Series(short_notional_curve, index=idx, name="short_notional")
    factor_s = pd.Series(factor_curve, index=idx, name="effective_factor")

    # 总 equity = baseline equity + 累积 hedge PnL - 累积 hedge cost
    equity_total = strat.equity + cum_pnl_s - cum_cost_s
    equity_total.name = "equity_total"
    daily_rets_total = equity_total.pct_change().fillna(
        equity_total.iloc[0] / cfg.starting_capital - 1)

    return HedgeOverlayResult(
        name=hedge_cfg.name, cfg=hedge_cfg,
        equity_total=equity_total, daily_returns_total=daily_rets_total,
        hedge_pnl_cum=cum_pnl_s, hedge_cost_cum=cum_cost_s,
        short_notional=short_s, rebalance_count=rebal_count,
        effective_factor=factor_s,
    )


# ---------------- 评估辅助 ----------------

def _ann_stats(rets: pd.Series, equity: Optional[pd.Series] = None,
               starting: Optional[float] = None) -> Dict[str, float]:
    rets = rets.dropna()
    if len(rets) == 0:
        return dict(cum=0.0, cagr=0.0, vol=0.0, sharpe=0.0, mdd=0.0, calmar=0.0)
    if equity is not None and starting is not None:
        cum = float(equity.iloc[-1] / starting - 1)
    else:
        eq_imp = (1 + rets).cumprod()
        cum = float(eq_imp.iloc[-1] - 1)
    years = len(rets) / 252.0
    cagr = (1 + cum) ** (1 / years) - 1 if years > 0 else 0.0
    sd = rets.std(ddof=0)
    vol = float(sd * np.sqrt(252))
    sharpe = float(rets.mean() * 252 / vol) if vol > 1e-9 else 0.0
    eq_for_dd = equity if equity is not None else (1 + rets).cumprod()
    dd = (eq_for_dd / eq_for_dd.cummax() - 1).min()
    mdd = float(dd) if not np.isnan(dd) else 0.0
    calmar = (cagr / abs(mdd)) if mdd < 0 else 0.0
    return dict(cum=cum, cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar)


def _corr_with(rets: pd.Series, bench_rets: pd.Series) -> float:
    rets = rets.dropna(); bench_rets = bench_rets.dropna()
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


def print_variant_summary(name: str, equity_total: pd.Series, daily_rets: pd.Series,
                           strat: StratResult, hedge_pnl: float, hedge_cost: float,
                           rebal_count: int, cfg: Config, qqq_close: pd.Series,
                           bench_label: str = ""):
    s = _ann_stats(daily_rets, equity=equity_total, starting=cfg.starting_capital)
    qqq_rets = qqq_close.pct_change()
    corr_qqq = _corr_with(daily_rets, qqq_rets)
    beta_qqq = _empirical_beta(daily_rets, qqq_rets)
    n_trades = len(strat.trades)
    long_pnl = sum(t["pnl_usd"] for t in strat.trades if t["side"] == 1)
    total_strat_costs = sum(t.get("costs", 0) for t in strat.trades)
    total_costs = total_strat_costs + hedge_cost

    print(f"\n----- {name} -----")
    print(f"  累计收益    {s['cum']*100:+8.2f}%   "
          f"CAGR {s['cagr']*100:6.2f}%   "
          f"年化波动 {s['vol']*100:5.2f}%   "
          f"Sharpe {s['sharpe']:5.2f}   "
          f"MDD {s['mdd']*100:6.2f}%   "
          f"Calmar {s['calmar']:.2f}")
    print(f"  Corr(QQQ)   {corr_qqq:+5.2f}   "
          f"Beta(QQQ) {beta_qqq:+5.2f}   "
          f"对冲累计 PnL ${hedge_pnl:>+10,.0f}   "
          f"调仓次数 {rebal_count}")
    print(f"  策略多头交易 {n_trades} 笔, 累计 PnL ${long_pnl:>+10,.0f}   "
          f"策略成本 ${total_strat_costs:>+9,.0f}   "
          f"对冲成本 ${hedge_cost:>+8,.0f}   "
          f"合计成本 ${total_costs:>+9,.0f} ({total_costs/cfg.starting_capital*100:.1f}%)"
          + (f"   [bench={bench_label}]" if bench_label else ""))


def print_compare_table(rows: List[Tuple[str, dict]]):
    headers = ["变体", "累计%", "CAGR%", "Vol%", "Sharpe", "MDD%", "Calmar",
               "Corr(QQQ)", "Beta(QQQ)", "成本%"]
    print("\n" + "=" * 132)
    print("                                  对冲变体并排对比")
    print("=" * 132)
    fmt = "{:<28s}{:>9}{:>9}{:>9}{:>9}{:>10}{:>9}{:>11}{:>11}{:>9}"
    print(fmt.format(*headers))
    print("-" * 132)
    for name, st in rows:
        print(fmt.format(
            name[:28],
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
    print("=" * 132)


def _yearly_stats(equity: pd.Series, daily_rets: pd.Series) -> Dict[int, dict]:
    rets = daily_rets.dropna()
    eq = equity
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


def print_yearly_grid(yearly: Dict[str, Dict[int, dict]]):
    years = sorted({y for v in yearly.values() for y in v.keys()})
    names = list(yearly.keys())
    width = 96 if len(names) <= 4 else 96 + (len(names) - 4) * 18
    print("\n" + "=" * width)
    print(f"                                  逐年收益% 对比（{len(names)} 个变体）")
    print("=" * width)
    head = "{:<6}".format("年份") + "".join(f"{n[:18]:>20s}" for n in names)
    print(head)
    print("-" * width)
    for y in years:
        row = f"{y:<6}"
        for n in names:
            d = yearly.get(n, {}).get(y)
            row += f"{d['ret']:>+19.2f} " if d else f"{'--':>20s}"
        print(row)
    print("=" * width)


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
    print("  crossrank 对冲实验（DAILY 日 K 模式 - 双账户解耦版）")
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

    spy_df = fetch_daily_bars("SPY.US",
                               cfg.start - timedelta(days=400),
                               cfg.end, log_cache=False)
    spy_close = spy_df["close"]
    regime_series = (spy_close > spy_close.rolling(200).mean())

    qqq_df = fetch_daily_bars("QQQ.US",
                               cfg.start - timedelta(days=10),
                               cfg.end, log_cache=False)
    qqq_close_ranged = qqq_df["close"].loc[
        (qqq_df.index >= pd.Timestamp(cfg.start)) & (qqq_df.index <= pd.Timestamp(cfg.end))
    ]
    bench_close_map = {"QQQ.US": qqq_close_ranged}

    # ---------- 跑一次 baseline 多头（所有变体共享）----------
    print(f"\n========== 跑 baseline 多头层（与 backtest.py DAILY 100% 等价） ==========")
    strat = run_strategy_baseline(daily_panel, cfg, regime_series=regime_series)
    print(f"  baseline 多头层完成：{len(strat.trades)} 笔多头交易，"
          f"终值 {_fmt_usd(strat.equity.iloc[-1])}")

    # ---------- 对每个变体做对冲 overlay（纯后处理） ----------
    rows_for_table: List[Tuple[str, dict]] = []
    yearly_grid: Dict[str, Dict[int, dict]] = {}
    detailed_results: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    for name, instrument, mode, ratio in HEDGE_VARIANTS:
        hcfg = HedgeCfg(name=name, instrument=instrument, mode=mode, ratio=ratio)
        if hcfg.mode == "none" or hcfg.instrument is None:
            equity_total = strat.equity
            daily_rets_total = strat.daily_returns
            hedge_pnl_total, hedge_cost_total, rebal_count = 0.0, 0.0, 0
            bench_label = "-"
        else:
            bclose = bench_close_map.get(instrument)
            if bclose is None:
                print(f"[警告] 未识别的对冲标的 {instrument}，跳过 {name}")
                continue
            overlay = simulate_hedge_overlay(strat, bclose, hcfg, cfg)
            equity_total = overlay.equity_total
            daily_rets_total = overlay.daily_returns_total
            hedge_pnl_total = float(overlay.hedge_pnl_cum.iloc[-1])
            hedge_cost_total = float(overlay.hedge_cost_cum.iloc[-1])
            rebal_count = overlay.rebalance_count
            bench_label = instrument

        print_variant_summary(name, equity_total, daily_rets_total, strat,
                               hedge_pnl_total, hedge_cost_total, rebal_count,
                               cfg, qqq_close_ranged, bench_label=bench_label)

        s = _ann_stats(daily_rets_total, equity=equity_total,
                       starting=cfg.starting_capital)
        s["corr"] = _corr_with(daily_rets_total, qqq_close_ranged.pct_change())
        s["beta"] = _empirical_beta(daily_rets_total, qqq_close_ranged.pct_change())
        s["cost"] = (sum(t.get("costs", 0) for t in strat.trades) + hedge_cost_total) \
                    / cfg.starting_capital
        rows_for_table.append((name, s))
        yearly_grid[name] = _yearly_stats(equity_total, daily_rets_total)
        detailed_results[name] = (equity_total, daily_rets_total)

    # ---------- benchmark 对照 ----------
    qqq_rets = qqq_close_ranged.pct_change().dropna()
    qqq_stats = _ann_stats(qqq_rets); qqq_stats["corr"] = 1.0; qqq_stats["beta"] = 1.0; qqq_stats["cost"] = 0.0
    rows_for_table.append(("(基准) QQQ buy&hold", qqq_stats))

    print_compare_table(rows_for_table)
    print_yearly_grid(yearly_grid)

    # ---------- 多头交易明细（baseline 共享） ----------
    print("\n========== baseline 多头交易明细（所有变体共享同一组多头） ==========")
    print_top_trades(strat.trades, n=5)


if __name__ == "__main__":
    main()
