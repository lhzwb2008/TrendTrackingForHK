#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股 + 美股 日K 趋势突破回测（纯技术面）

无未来函数要点：回测日 T 仅用截止 T-1 的日线；突破与通道均对前一日指标 shift(1)；
成交价按昨收简化，实盘中通常更差。详见项目 README。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# =============================================================================
# 用户配置（修改此处即可）
# =============================================================================

# 主回测区间（推理；训练期仅在 train_params.py 配置）
# 可用环境变量覆盖（便于一次性实验）：BACKTEST_START / BACKTEST_END，格式 YYYY-MM-DD
_DEFAULT_BACKTEST_START = date(2024, 1, 1)
_DEFAULT_BACKTEST_END = date(2025, 12, 31)
BACKTEST_START = (
    date.fromisoformat(os.environ['BACKTEST_START'])
    if os.environ.get('BACKTEST_START')
    else _DEFAULT_BACKTEST_START
)
BACKTEST_END = (
    date.fromisoformat(os.environ['BACKTEST_END'])
    if os.environ.get('BACKTEST_END')
    else _DEFAULT_BACKTEST_END
)  # 不设环境变量时也可在代码中改为 None 表示到今天

# 股票池 CSV（symbol 列）。空字符串则依次尝试 dual_universe.csv、dual_universe.example.csv
UNIVERSE_CSV = ""

# 候选池来源：
# - "csv"：仅使用 UNIVERSE_CSV 中的标的（或默认 example）
# - "hsi_hstech"：恒指成分 ∪ 恒生科技成分（仅下方成分 CSV）
UNIVERSE_MODE = 'hsi_hstech'

# 恒指 / 恒生科技成分表（正式文件优先；若无则用 *.example.csv 样例，请定期从恒生指数官网更新）
HSI_CONSTITUENTS_CSV = 'data/hsi_constituents.csv'
HSTECH_CONSTITUENTS_CSV = 'data/hstech_constituents.csv'
HSI_CONSTITUENTS_EXAMPLE = 'data/hsi_constituents.example.csv'
HSTECH_CONSTITUENTS_EXAMPLE = 'data/hstech_constituents.example.csv'

# 可选：含「代码」「中文名称」列的 CSV，仅用于成交明细展示；不设则依赖 Longport 补全（见 enrich_cn_map_for_trades）
HK_CN_NAMES_CSV = os.environ.get('HK_CN_NAMES_CSV', '').strip()

# 回测起点再往前多取的自然日（用于指标预热；REGIME_MA_DAYS=200 时建议 ≥400）
DATA_WARMUP_DAYS_BEFORE_START = 400

# 账户
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.20  # 单笔目标占当时总权益比例（会再乘以下波动缩放）

# —— 策略参数（若存在 trained_strategy_params.json 则由训练覆盖；否则用此处默认）——
STOP_LOSS_PCT = 0.10
BREAKOUT_LOOKBACK = 40
TREND_MA_PERIOD = 50
VOL_MA_PERIOD = 20
VOLUME_RATIO_THRESHOLD = 1.2
EXIT_DONCHIAN_DAYS = 20
USE_MA60_LOSS_EXIT = True
ONE_WAY_COST_RATE = 0.0  # 单边费率，如万 3 填 0.00015（买/卖各收一次需自行理解口径）

# 大盘过滤：恒指、SPY 均在长均线上方才允许新开仓（训练选优为关）
USE_REGIME_FILTER = False
REGIME_BENCHMARKS: List[str] = ['HSI.HK', 'SPY.US']
REGIME_MODE = 'all'  # 'all' = 全部在均线上；'any' = 任一在均线上
REGIME_MA_DAYS = 200

# 波动率目标：按标的实现波动缩放单笔仓位（0 表示关闭）
VOL_TARGET_ANNUAL = 0.15
VOL_LOOKBACK = 20
VOL_SCALE_MIN = 0.35
VOL_SCALE_MAX = 2.5

# 移动止盈：浮盈达到比例后，自持仓以来最高价回撤超过比例则清仓（任一为 0 则关闭）
TRAILING_ACTIVATION_PCT = 0.12
TRAILING_STOP_PCT = 0.09

# 流动性区间（近 lookback 日均成交额，港元/美元视市场而定）
HK_AVG_TURNOVER_MIN = 5e6
HK_AVG_TURNOVER_MAX = 50e9
US_AVG_TURNOVER_MIN = 2e6
US_AVG_TURNOVER_MAX = 50e9

# 数据管理器：最少载入日线根数（过短则跳过该标的）
MIN_HISTORY_DAYS_LOAD = 120
MIN_HISTORY_DAYS_DUAL = 80  # 双市场池内有效数据下限（可略低于 LOAD）

# 由 train_params.py 训练后写入；回测时若文件存在则覆盖下方策略常量（训练一次，多次推理）
STRATEGY_PARAMS_JSON = os.environ.get('STRATEGY_PARAMS_JSON', 'trained_strategy_params.json')

# =============================================================================


class HistoricalDataManager:
    """历史数据管理器：任意时刻 T 仅可见 T-1 及以前数据。"""

    def __init__(self) -> None:
        self._all_data: Dict[str, pd.DataFrame] = {}
        self._current_date: Optional[date] = None
        self._min_history_days: int = MIN_HISTORY_DAYS_LOAD

    def load_stock_data(self, symbols: List[str], start_date: date, end_date: date) -> int:
        from daily_cache import normalize_df_index
        from hk_stock_api import fetch_daily_bars

        verbose_daily = os.getenv('TREND_VERBOSE_DAILY_LOG', '').lower() in ('1', 'true', 'yes')
        loaded = 0
        pause = float(os.getenv('LONGPORT_REQUEST_PAUSE', '0.15'))
        for i, symbol in enumerate(symbols):
            try:
                df = fetch_daily_bars(
                    symbol,
                    start_date,
                    end_date,
                    log_cache=verbose_daily,
                    progress=(i + 1, len(symbols)) if verbose_daily else None,
                )
                if df is not None and len(df) > 0:
                    df = normalize_df_index(df)
                if df is not None and len(df) >= self._min_history_days:
                    self._all_data[symbol] = df
                    loaded += 1
            except Exception as e:
                print(f'[日线] {symbol} 加载异常: {e}', flush=True)
            finally:
                if pause > 0 and i + 1 < len(symbols):
                    time.sleep(pause)
        if not verbose_daily:
            print(
                f'[日线] 批量加载完成: {loaded}/{len(symbols)} 只有效（{start_date}～{end_date}）；'
                f'单笔拉取日志已关闭，需要时设 TREND_VERBOSE_DAILY_LOG=1',
                flush=True,
            )
        else:
            print(f'\n成功加载: {loaded}/{len(symbols)} 只股票')
        if loaded == 0:
            print(
                '[提示] 若缓存 CSV 为旧版本导致切片为空，可删除 data_cache/daily/ 后重跑，'
                '或设 TREND_DISABLE_DAILY_CACHE=1 强制走 API。',
                flush=True,
            )
        return loaded

    def set_current_date(self, current_date) -> None:
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date).date()
        elif isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
        self._current_date = current_date

    def _get_cutoff_date(self) -> date:
        if self._current_date is None:
            raise ValueError('必须先调用 set_current_date()')
        return self._current_date - timedelta(days=1)

    def get_history(self, symbol: str, lookback_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        if symbol not in self._all_data:
            return None
        cutoff = self._get_cutoff_date()
        df = self._all_data[symbol]
        df_filtered = df[df.index.date <= cutoff].copy()
        if len(df_filtered) < self._min_history_days:
            return None
        if lookback_days is not None:
            df_filtered = df_filtered.tail(lookback_days)
        return df_filtered

    def get_latest_price(self, symbol: str) -> Optional[float]:
        df = self.get_history(symbol, lookback_days=1)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None

    def get_tradable_pool(
        self,
        min_price: float = 1.0,
        min_avg_turnover: float = 5000000,
        max_avg_turnover: float = 500000000,
        lookback_days: int = 20,
        symbols_subset: Optional[Union[Set[str], List[str]]] = None,
    ) -> List[str]:
        del min_avg_turnover, max_avg_turnover
        tradable = []
        subset = set(symbols_subset) if symbols_subset is not None else None
        keys = self._all_data.keys()
        if subset is not None:
            keys = (s for s in keys if s in subset)
        for symbol in keys:
            df = self.get_history(symbol, lookback_days=lookback_days)
            if df is None or len(df) < lookback_days // 2:
                continue
            latest_price = df['close'].iloc[-1]
            avg_turnover = (df['close'] * df['volume']).mean()
            if latest_price >= min_price and 5000000 <= avg_turnover <= 500000000:
                tradable.append(symbol)
        return tradable

    def get_all_trading_dates(self) -> List[date]:
        all_dates = set()
        for df in self._all_data.values():
            all_dates.update(df.index.date)
        return sorted(all_dates)

    def is_regime_bull(self, benchmark_symbol: str, ma_days: int = 200) -> bool:
        df = self.get_history(benchmark_symbol)
        if df is None or len(df) < ma_days + 2:
            return False
        close = df['close'].astype(float)
        ma = close.rolling(ma_days).mean()
        last_c = close.iloc[-1]
        last_ma = ma.iloc[-1]
        if pd.isna(last_ma) or last_ma <= 0:
            return False
        return bool(last_c > last_ma)


class DualMarketDataManager(HistoricalDataManager):
    """港股 / 美股分档成交额过滤。"""

    def __init__(
        self,
        hk_turnover: tuple = None,
        us_turnover: tuple = None,
        min_history_days: int = None,
    ) -> None:
        super().__init__()
        hk_turnover = hk_turnover or (HK_AVG_TURNOVER_MIN, HK_AVG_TURNOVER_MAX)
        us_turnover = us_turnover or (US_AVG_TURNOVER_MIN, US_AVG_TURNOVER_MAX)
        self._hk_lo, self._hk_hi = hk_turnover
        self._us_lo, self._us_hi = us_turnover
        self._min_history_days = min_history_days if min_history_days is not None else MIN_HISTORY_DAYS_DUAL

    def _turn_bounds(self, symbol: str) -> tuple:
        return (self._us_lo, self._us_hi) if symbol.endswith('.US') else (self._hk_lo, self._hk_hi)

    def get_tradable_pool(
        self,
        min_price: float = 1.0,
        min_avg_turnover: float = 5000000,
        max_avg_turnover: float = 500000000,
        lookback_days: int = 20,
        symbols_subset: Optional[Union[Set[str], List[str]]] = None,
    ) -> List[str]:
        del min_avg_turnover, max_avg_turnover
        tradable = []
        subset = set(symbols_subset) if symbols_subset is not None else None
        keys = self._all_data.keys()
        if subset is not None:
            keys = (s for s in keys if s in subset)
        for symbol in keys:
            if not (symbol.endswith('.HK') or symbol.endswith('.US')):
                continue
            df = self.get_history(symbol, lookback_days=lookback_days)
            if df is None or len(df) < max(10, lookback_days // 2):
                continue
            latest_price = float(df['close'].iloc[-1])
            avg_turnover = float((df['close'] * df['volume']).mean())
            lo, hi = self._turn_bounds(symbol)
            if latest_price >= min_price and lo <= avg_turnover <= hi:
                tradable.append(symbol)
        return tradable

    def calculate_indicators(
        self,
        symbol: str,
        breakout_lookback: int = 55,
        trend_ma_period: int = 50,
        vol_ma_period: int = 20,
        exit_donchian_days: int = 20,
    ) -> Optional[pd.DataFrame]:
        df = self.get_history(symbol)
        if df is None:
            return None
        df = df.copy()
        df['ma_trend'] = df['close'].rolling(trend_ma_period).mean()
        df['vol_ma'] = df['volume'].rolling(vol_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
        df['high_nd'] = df['high'].rolling(breakout_lookback).max()
        df['is_breakout'] = df['close'] > df['high_nd'].shift(1)
        df['trend_ok'] = df['close'] > df['ma_trend']
        df['low_exit_level'] = df['low'].rolling(exit_donchian_days).min().shift(1)
        df['turnover'] = df['close'] * df['volume']
        df['avg_turnover_20d'] = df['turnover'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        return df


class DualBreakoutEngine:
    def __init__(self, dm: DualMarketDataManager, config: Optional[dict] = None) -> None:
        self.dm = dm
        self.config = config or {}
        self.max_positions = int(self.config.get('max_positions', 10))
        self.position_size_pct = float(self.config.get('position_size_pct', 0.10))
        self.stop_loss_pct = float(self.config.get('stop_loss_pct', 0.12))
        self.breakout_lookback = int(self.config.get('breakout_lookback', 55))
        self.trend_ma_period = int(self.config.get('trend_ma_period', 50))
        self.vol_ma_period = int(self.config.get('vol_ma_period', 20))
        self.volume_ratio_threshold = float(self.config.get('volume_ratio_threshold', 1.2))
        self.exit_donchian_days = int(self.config.get('exit_donchian_days', 20))
        self.use_ma60_loss_exit = bool(self.config.get('use_ma60_loss_exit', True))
        self.one_way_cost_rate = float(self.config.get('one_way_cost_rate', 0.0))
        self.use_regime_filter = bool(self.config.get('use_regime_filter', False))
        self.regime_benchmarks: List[str] = list(self.config.get('regime_benchmarks', ['HSI.HK', 'SPY.US']))
        self.regime_mode = str(self.config.get('regime_mode', 'all')).lower()
        self.regime_ma_days = int(self.config.get('regime_ma_days', 200))
        self.vol_target_annual = float(self.config.get('vol_target_annual', 0.0))
        self.vol_lookback = int(self.config.get('vol_lookback', 20))
        self.vol_scale_min = float(self.config.get('vol_scale_min', 0.35))
        self.vol_scale_max = float(self.config.get('vol_scale_max', 2.5))
        self.trailing_activation_pct = float(self.config.get('trailing_activation_pct', 0.0))
        self.trailing_stop_pct = float(self.config.get('trailing_stop_pct', 0.0))
        ss = self.config.get('symbols_subset')
        self.symbols_subset: Optional[Set[str]] = set(ss) if ss else None

        self.initial_capital = float(self.config.get('initial_capital', 100000))
        self.cash = self.initial_capital
        self.positions: Dict = {}
        self.trades: List[dict] = []
        self.daily_values: List = []
        self._verbose = True

    def _min_buy_notional(self, symbol: str) -> float:
        return 800.0 if symbol.endswith('.US') else 5000.0

    def _regime_ok(self) -> bool:
        if not self.use_regime_filter:
            return True
        conds = []
        for sym in self.regime_benchmarks:
            if sym not in self.dm._all_data:
                continue
            conds.append(self.dm.is_regime_bull(sym, self.regime_ma_days))
        if not conds:
            return True
        return all(conds) if self.regime_mode == 'all' else any(conds)

    def _ann_vol_symbol(self, symbol: str) -> float:
        need = max(self.vol_lookback + 2, 5)
        df = self.dm.get_history(symbol, lookback_days=need)
        if df is None or len(df) < self.vol_lookback:
            return 0.25
        c = df['close'].astype(float).values
        if len(c) < 2:
            return 0.25
        lr = np.diff(np.log(np.clip(c, 1e-12, None)))
        if len(lr) < 5:
            return 0.25
        sig = float(np.std(lr, ddof=1) * np.sqrt(252.0))
        return max(sig, 1e-6)

    def _vol_scale(self, symbol: str) -> float:
        if self.vol_target_annual <= 0:
            return 1.0
        av = self._ann_vol_symbol(symbol)
        raw = self.vol_target_annual / av
        return float(np.clip(raw, self.vol_scale_min, self.vol_scale_max))

    def _shares_to_buy(self, symbol: str, buy_amount: float, price: float) -> int:
        if price <= 0:
            return 0
        if symbol.endswith('.US'):
            return max(1, int(buy_amount / price))
        lot = int(buy_amount / price / 100) * 100
        if lot <= 0:
            lot = int(buy_amount / price)
        return max(0, lot)

    @property
    def total_value(self) -> float:
        pv = 0.0
        for sym, pos in self.positions.items():
            px = self.dm.get_latest_price(sym)
            if px:
                pv += pos['shares'] * px
        return self.cash + pv

    def _ind(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.dm.calculate_indicators(
            symbol,
            breakout_lookback=self.breakout_lookback,
            trend_ma_period=self.trend_ma_period,
            vol_ma_period=self.vol_ma_period,
            exit_donchian_days=self.exit_donchian_days,
        )

    def run(
        self,
        start_date: date,
        end_date: date,
        benchmark_data: Optional[pd.DataFrame] = None,
        verbose: bool = True,
        compare_indices: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> dict:
        self._verbose = verbose
        warmup = max(self.breakout_lookback, self.trend_ma_period, self.exit_donchian_days, 60) + 5

        all_dates = self.dm.get_all_trading_dates()
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]

        if verbose:
            print('\n' + '=' * 60)
            print('双市场日K趋势突破')
            print('=' * 60)
            print(f'初始资金: {self.initial_capital:,.0f}  区间: {start_date} ~ {end_date}')
            print(
                f'参数: 突破{self.breakout_lookback}日高 | 趋势MA{self.trend_ma_period} | '
                f'量比≥{self.volume_ratio_threshold} | 止损{self.stop_loss_pct:.0%} | '
                f'唐奇安出场{self.exit_donchian_days}日低'
            )
            extra = []
            if self.use_regime_filter:
                extra.append(
                    f'大盘过滤 {self.regime_mode}({",".join(self.regime_benchmarks)},{self.regime_ma_days}MA)'
                )
            if self.vol_target_annual > 0:
                extra.append(f'波动目标{self.vol_target_annual:.0%}年化({self.vol_lookback}日)')
            if self.trailing_activation_pct > 0 and self.trailing_stop_pct > 0:
                extra.append(
                    f'移动止盈 浮盈≥{self.trailing_activation_pct:.0%}后回撤{self.trailing_stop_pct:.0%}清仓'
                )
            if extra:
                print('  ' + ' | '.join(extra))
            print(f'交易日数: {len(trading_dates)}  预热跳过: {warmup} 天')

        for i, current_date in enumerate(trading_dates):
            self.dm.set_current_date(current_date)
            if i < warmup:
                continue
            self._update_positions()
            self._check_sell_signals(current_date)
            pool = self.dm.get_tradable_pool(symbols_subset=self.symbols_subset)
            self._check_buy_signals(current_date, pool)
            self.daily_values.append((str(current_date), self.total_value))

        return self._generate_report(benchmark_data, verbose, compare_indices)

    def _update_positions(self) -> None:
        for sym in list(self.positions.keys()):
            px = self.dm.get_latest_price(sym)
            if px:
                self.positions[sym]['current_price'] = px
                pk = self.positions[sym].get('peak_close', px)
                self.positions[sym]['peak_close'] = max(pk, px)

    def _check_buy_signals(self, current_date: date, pool: List[str]) -> None:
        if len(self.positions) >= self.max_positions:
            return
        if not self._regime_ok():
            return
        for symbol in pool:
            if symbol in self.positions or len(self.positions) >= self.max_positions:
                continue
            df = self._ind(symbol)
            if df is None or len(df) < 2:
                continue
            row = df.iloc[-1]
            if not row.get('is_breakout', False):
                continue
            vr = row.get('volume_ratio', 0) or 0
            if vr < self.volume_ratio_threshold:
                continue
            if not row.get('trend_ok', False):
                continue
            avg_to = row.get('avg_turnover_20d', 0)
            lo, _ = self.dm._turn_bounds(symbol)
            if pd.isna(avg_to) or float(avg_to) < lo:
                continue
            px = float(row['close'])
            vs = self._vol_scale(symbol)
            self._execute_buy(
                current_date,
                symbol,
                px,
                f'突破{self.breakout_lookback}日高,量比{vr:.2f},>MA{self.trend_ma_period}',
                size_scale=vs,
            )

    def _check_sell_signals(self, current_date: date) -> None:
        for symbol in list(self.positions.keys()):
            df = self._ind(symbol)
            if df is None or len(df) < 1:
                continue
            row = df.iloc[-1]
            pos = self.positions[symbol]
            price = float(row['close'])
            buy_price = pos['buy_price']
            pnl_pct = price / buy_price - 1.0

            if pnl_pct < -self.stop_loss_pct:
                self._execute_sell(current_date, symbol, price, 1.0, f'止损 {pnl_pct:.1%}')
                continue

            if (
                self.trailing_activation_pct > 0
                and self.trailing_stop_pct > 0
                and pnl_pct >= self.trailing_activation_pct
            ):
                peak = float(pos.get('peak_close', price))
                if peak > 0 and (peak - price) / peak >= self.trailing_stop_pct:
                    self._execute_sell(
                        current_date,
                        symbol,
                        price,
                        1.0,
                        f'移动止盈 峰值回撤{(peak - price) / peak:.1%}',
                    )
                    continue

            low_exit = row.get('low_exit_level')
            if low_exit is not None and pd.notna(low_exit) and price < float(low_exit):
                self._execute_sell(
                    current_date,
                    symbol,
                    price,
                    1.0,
                    f'跌破{self.exit_donchian_days}日低 盈亏{pnl_pct:+.1%}',
                )
                continue

            if self.use_ma60_loss_exit:
                ma60 = row.get('ma60', 0)
                if ma60 and ma60 > 0 and price < float(ma60) and pnl_pct < 0:
                    self._execute_sell(current_date, symbol, price, 1.0, '跌破MA60且亏')

    def _execute_buy(
        self,
        current_date: date,
        symbol: str,
        price: float,
        reason: str,
        size_scale: float = 1.0,
    ) -> None:
        max_amt = self.total_value * self.position_size_pct * float(size_scale)
        buy_amt = min(self.cash * 0.92, max_amt)
        min_b = self._min_buy_notional(symbol)
        if buy_amt < min_b:
            return
        shares = self._shares_to_buy(symbol, buy_amt, price)
        if shares <= 0:
            return
        fee = self.one_way_cost_rate
        cost = shares * price * (1.0 + fee)
        if cost > self.cash:
            return
        self.cash -= cost
        self.positions[symbol] = {
            'shares': shares,
            'buy_price': price,
            'buy_date': str(current_date),
            'current_price': price,
            'peak_close': price,
        }
        self.trades.append(
            {
                'date': str(current_date),
                'action': 'BUY',
                'symbol': symbol,
                'price': price,
                'shares': shares,
                'reason': reason,
            }
        )
        nav = self.total_value
        mv = float(shares) * float(price)
        wp = 100.0 * mv / nav if nav > 0 else 0.0
        self.trades[-1]['weight_pct'] = wp
        self.trades[-1]['nav'] = nav
        self.trades[-1]['pnl_amount'] = None
        self.trades[-1]['realized_pnl_pct'] = None

    def _execute_sell(self, current_date: date, symbol: str, price: float, ratio: float, reason: str) -> None:
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        sell_shares = int(pos['shares'] * ratio)
        if sell_shares <= 0:
            return
        fee = self.one_way_cost_rate
        nav_before = self.total_value
        sell_mv = float(sell_shares) * float(price)
        sw = 100.0 * sell_mv / nav_before if nav_before > 0 else 0.0
        self.cash += sell_shares * price * (1.0 - fee)
        pnl_pct = (price / pos['buy_price'] - 1.0) * 100
        buy_px = float(pos['buy_price'])
        buy_cost = float(sell_shares) * buy_px * (1.0 + fee)
        sell_proceeds = float(sell_shares) * float(price) * (1.0 - fee)
        pnl_amt = sell_proceeds - buy_cost
        rpct = 100.0 * pnl_amt / buy_cost if buy_cost > 0 else 0.0
        self.trades.append(
            {
                'date': str(current_date),
                'action': 'SELL',
                'symbol': symbol,
                'price': price,
                'shares': sell_shares,
                'reason': f'{reason}, 价差盈亏:{pnl_pct:+.1f}%',
                'weight_pct': sw,
                'nav': nav_before,
                'pnl_amount': pnl_amt,
                'realized_pnl_pct': rpct,
            }
        )
        pos['shares'] -= sell_shares
        if pos['shares'] <= 0:
            del self.positions[symbol]

    def _calculate_yearly_returns(self, benchmark_data: Optional[pd.DataFrame]) -> dict:
        yearly_data: Dict[int, list] = {}
        for d, v in self.daily_values:
            d_date = datetime.strptime(d, '%Y-%m-%d').date() if isinstance(d, str) else d
            yearly_data.setdefault(d_date.year, []).append((d_date, v))
        out = {}
        for year, data in yearly_data.items():
            if len(data) < 2:
                continue
            s0, s1 = data[0][1], data[-1][1]
            strat_ret = (s1 / s0 - 1) * 100
            bench_ret = 0.0
            if benchmark_data is not None and len(benchmark_data) > 1:
                b = benchmark_data.copy()
                b.index = pd.to_datetime(b.index)
                sub = b[(b.index >= pd.Timestamp(data[0][0])) & (b.index <= pd.Timestamp(data[-1][0]))]
                if len(sub) > 1:
                    bench_ret = (sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100
            out[year] = {'strategy': strat_ret, 'benchmark': bench_ret}
        return out

    def _generate_report(
        self,
        benchmark_data: Optional[pd.DataFrame],
        verbose: bool,
        compare_indices: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> dict:
        initial = self.initial_capital
        final = self.total_value
        total_return = (final / initial - 1) * 100
        values = [v for _, v in self.daily_values]
        max_dd = 0.0
        peak = values[0] if values else initial
        for v in values:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak if peak else 0)
        days = len(self.daily_values)
        annual = ((final / initial) ** (252 / max(days, 1)) - 1) * 100
        dr = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                dr.append(values[i] / values[i - 1] - 1.0)
        arr = np.array(dr, dtype=float)
        sharpe = (
            float(np.sqrt(252) * np.mean(arr) / np.std(arr, ddof=1))
            if len(arr) > 1 and np.std(arr, ddof=1) > 1e-12
            else 0.0
        )
        neg = arr[arr < 0]
        dstd = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
        sortino = float(np.sqrt(252) * np.mean(arr) / dstd) if dstd > 1e-12 else 0.0

        yearly = self._calculate_yearly_returns(benchmark_data)
        bench_ret = ex = None
        if benchmark_data is not None and self.daily_values:
            b = benchmark_data.copy()
            b.index = pd.to_datetime(b.index)
            s0 = pd.to_datetime(self.daily_values[0][0])
            s1 = pd.to_datetime(self.daily_values[-1][0])
            sub = b[(b.index >= s0) & (b.index <= s1)]
            if len(sub) > 1:
                bench_ret = (sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100
                ex = total_return - bench_ret

        buys = [t for t in self.trades if t['action'] == 'BUY']
        sells = [t for t in self.trades if t['action'] == 'SELL']

        if verbose:
            print('\n' + '=' * 60 + '\n回测结果\n' + '=' * 60)
            print(f'\n【策略收益】\n  总收益率: {total_return:+.2f}%\n  年化收益: {annual:+.2f}%')
            print(f'  最大回撤: {max_dd:.2%}\n  年化Sharpe(Rf=0): {sharpe:.3f}\n  年化Sortino(Rf=0): {sortino:.3f}')
            if yearly:
                print('\n【年度收益】策略 vs 等权(恒指+SPY)归一基准')
                print('  （首年若预热结束较晚，该年收益为「年内已有净值区间的首尾」非完整自然年）')
                py = 0
                for y in sorted(yearly.keys()):
                    r = yearly[y]
                    ok = '✓' if r['strategy'] > 0 else '✗'
                    if r['strategy'] > 0:
                        py += 1
                    print(f'  {y}: 策略{r["strategy"]:+.1f}% | 基准{r["benchmark"]:+.1f}% {ok}')
                print(f'  盈利年份: {py}/{len(yearly)}')
            if bench_ret is not None:
                print(f'\n【全样本基准】等权恒指+SPY: {bench_ret:+.2f}%  超额: {ex:+.2f}%')
            if compare_indices and self.daily_values:
                s0 = pd.to_datetime(self.daily_values[0][0])
                s1 = pd.to_datetime(self.daily_values[-1][0])
                print('\n【同期港股指数】回测区间首尾收盘，买入持有（与上表策略区间一致）')
                idx_meta: Dict[str, float] = {}
                idx_excess: Dict[str, float] = {}
                for name, idf in compare_indices.items():
                    if idf is None or getattr(idf, 'empty', True):
                        print(f'  {name}: （无数据）')
                        continue
                    ir = buy_hold_return_pct(idf, s0, s1)
                    if ir is None:
                        print(f'  {name}: （区间内数据不足）')
                        continue
                    idx_meta[name] = ir
                    ex_i = total_return - ir
                    idx_excess[name] = ex_i
                    print(f'  {name}: 指数区间 {ir:+.2f}%  ｜ 策略超额 {ex_i:+.2f}%（= 策略总收益 {total_return:+.2f}% − 指数）')
                if not idx_meta:
                    print('  （恒指/恒生科技数据均未就绪）')
                elif len(idx_excess) >= 2:
                    names = list(idx_excess.keys())
                    # 典型为「恒生指数」「恒生科技」：并排对比两指数上的超额
                    print('\n【超额收益对比】相对恒指 vs 相对恒生科技')
                    print(f'  策略全区间总收益: {total_return:+.2f}%（与上方「总收益率」一致）')
                    for n in names:
                        ir = idx_meta.get(n)
                        ex = idx_excess[n]
                        ir_s = f'{ir:+.2f}%' if ir is not None else '—'
                        print(f'  · {n}: 指数同期 {ir_s} → 超额 {ex:+.2f}%')
                    hsi_n = next((k for k in idx_excess if '恒生指数' in k and '科技' not in k), None)
                    hst_n = next((k for k in idx_excess if '科技' in k), None)
                    if hsi_n and hst_n and hsi_n in idx_meta and hst_n in idx_meta:
                        d_ex = idx_excess[hsi_n] - idx_excess[hst_n]
                        d_idx = idx_meta[hsi_n] - idx_meta[hst_n]
                        print(
                            f'  说明：两指数同期涨跌差 {d_idx:+.2f}%（恒指−科技）；'
                            f'策略相对两指数的超额之差 {d_ex:+.2f}%（= 相对恒指超额 − 相对科技超额）。'
                            f'恒指跌得更多时，同一策略收益下「相对恒指超额」往往高于「相对科技超额」。'
                        )
            print(f'\n【交易】买入{len(buys)} 卖出{len(sells)}  期末持仓{len(self.positions)}只')
            if self.trades:
                cmap = load_hk_cn_name_map(HK_CN_NAMES_CSV)
                cmap = enrich_cn_map_for_trades(cmap, self.trades)
                print_trades_with_names(self.trades, cmap, self.config)
                self._trade_print_done = True

        idx_returns: Dict[str, float] = {}
        if compare_indices and self.daily_values:
            s0 = pd.to_datetime(self.daily_values[0][0])
            s1 = pd.to_datetime(self.daily_values[-1][0])
            for name, idf in compare_indices.items():
                if idf is None or getattr(idf, 'empty', True):
                    continue
                ir = buy_hold_return_pct(idf, s0, s1)
                if ir is not None:
                    idx_returns[name] = ir

        excess_vs_indices: Optional[Dict[str, float]] = None
        if idx_returns:
            excess_vs_indices = {n: total_return - ir for n, ir in idx_returns.items()}

        return {
            'total_return': total_return,
            'annual_return': annual,
            'max_drawdown': max_dd * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'trade_count': len(buys) + len(sells),
            'benchmark_return': bench_ret,
            'excess_return': ex,
            'yearly_returns': yearly,
            'index_buy_hold_returns': idx_returns or None,
            'excess_vs_indices': excess_vs_indices,
        }


def load_us_etf(symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    from hk_stock_api import fetch_daily_bars

    pause = float(os.getenv('LONGPORT_REQUEST_PAUSE', '0.15'))
    df = fetch_daily_bars(symbol, start_date, end_date, log_cache=False)
    time.sleep(pause)
    return df if df is not None and len(df) > 0 else None


def buy_hold_return_pct(
    df: Optional[pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Optional[float]:
    """区间首尾收盘价涨跌幅（%），用于指数买入持有对比。"""
    if df is None or getattr(df, 'empty', True):
        return None
    b = df.copy()
    b.index = pd.to_datetime(b.index)
    sub = b[(b.index >= start_ts) & (b.index <= end_ts)]
    if len(sub) < 2:
        return None
    return float((sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100)


def load_hstech_data(start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """恒生科技指数：先试指数代码，再试跟踪 ETF。"""
    from hk_stock_api import fetch_daily_bars

    pause = float(os.getenv('LONGPORT_REQUEST_PAUSE', '0.15'))
    for sym in ('HSTECH.HK', '03067.HK'):
        try:
            df = fetch_daily_bars(sym, start_date, end_date, log_cache=False)
            time.sleep(pause)
            if df is not None and len(df) > 0:
                print(f'[回测] 恒生科技基准: {sym}，{len(df)} 条', flush=True)
                return df
        except Exception as e:
            print(f'[回测] 加载 {sym} 失败: {e}', flush=True)
    print('[回测] 未能加载恒生科技基准（HSTECH.HK / 03067.HK）', flush=True)
    return None


def build_blended_benchmark(hsi: Optional[pd.DataFrame], spy: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if hsi is None or spy is None or hsi.empty or spy.empty:
        return None
    a = hsi.copy()
    b = spy.copy()
    a.index = pd.to_datetime(a.index)
    b.index = pd.to_datetime(b.index)
    merged = pd.merge(
        a[['close']].rename(columns={'close': 'h'}),
        b[['close']].rename(columns={'close': 's'}),
        left_index=True,
        right_index=True,
        how='inner',
    )
    if len(merged) < 50:
        return None
    merged['h_n'] = merged['h'] / merged['h'].iloc[0] * 100.0
    merged['s_n'] = merged['s'] / merged['s'].iloc[0] * 100.0
    merged['close'] = (merged['h_n'] + merged['s_n']) / 2.0
    return merged[['close']]


def load_hsi_data(start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    from hk_stock_api import fetch_daily_bars

    try:
        df = fetch_daily_bars('HSI.HK', start_date, end_date, log_cache=False)
        if df is not None and len(df) > 0:
            print(f'[回测] 恒生指数 HSI.HK: {len(df)} 条', flush=True)
            return df
    except Exception as e:
        print(f'[回测] 加载 HSI.HK 失败: {e}', flush=True)
    try:
        df = fetch_daily_bars('02800.HK', start_date, end_date, log_cache=False)
        if df is not None and len(df) > 0:
            print(f'[回测] 恒生基准改用盈富 02800.HK: {len(df)} 条', flush=True)
            return df
    except Exception as e:
        print(f'[回测] 加载 02800.HK 失败: {e}', flush=True)
    return None


def load_universe_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = 'symbol' if 'symbol' in df.columns else '代码'
    out = []
    for raw in df[col].astype(str):
        s = raw.strip()
        if not s:
            continue
        if '.' not in s:
            out.append(f'{s.zfill(5)}.HK')
        else:
            out.append(s.upper() if s.endswith(('.HK', '.US')) else s)
    return list(dict.fromkeys(out))


def enrich_cn_map_for_trades(
    base: Dict[str, str],
    trades: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    为成交里的代码补充中文简称：优先 HK_CN_NAMES_CSV（若配置）；缺省用 Longport static_info（需凭证）。
    设 TREND_RESOLVE_NAMES=0 可关闭 API 补全。
    """
    syms = list(dict.fromkeys(str(t.get('symbol', '')) for t in trades if t.get('symbol')))
    miss = [s for s in syms if s and not (base.get(s) or '').strip()]
    if not miss:
        return base
    if os.getenv('TREND_RESOLVE_NAMES', '1').lower() in ('0', 'false', 'no'):
        return base
    try:
        from hk_stock_api import fetch_static_display_names

        extra = fetch_static_display_names(miss)
    except Exception:
        return base
    out = dict(base)
    for k, v in extra.items():
        if v and k not in out:
            out[k] = v
    return out


def load_hk_cn_name_map(path: str) -> Dict[str, str]:
    """港股代码 -> 中文名称（CSV 需含「代码」「中文名称」列；path 为空或不存在则返回空）。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except (OSError, UnicodeDecodeError):
        try:
            df = pd.read_csv(path)
        except OSError:
            return {}
    if '代码' not in df.columns or '中文名称' not in df.columns:
        return {}
    out: Dict[str, str] = {}
    codes = df['代码'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    for sym, raw in zip(codes, df['中文名称'].astype(str)):
        name = raw.strip()
        if not name or name.lower() == 'nan':
            continue
        key = f'{sym}.HK'
        if key not in out:
            out[key] = name
    return out


def _sort_trades_for_display(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按标的代码聚合：同代码内按日期、买卖顺序排列。"""
    ord_act = {'BUY': 0, 'SELL': 1}
    return sorted(
        trades,
        key=lambda t: (
            str(t.get('symbol', '')),
            str(t.get('date', '')),
            ord_act.get(str(t.get('action', '')), 9),
        ),
    )


def _trade_label_name(sym: str, cn_map: Dict[str, str]) -> str:
    """控制台「名称」列：中文名 + 代码，便于辨认。"""
    cn = (cn_map.get(sym, '') or '').strip()
    if cn:
        return f'{cn} ({sym})'
    return sym


def _fmt_trade_lines(trades: List[Dict[str, Any]], cn_map: Dict[str, str]) -> List[str]:
    """控制台用：仅名称、时间、方向、总金额、盈亏（按代码排序）。"""
    ordered = _sort_trades_for_display(trades)
    lines: List[str] = []
    w = 110
    lines.append('')
    lines.append('=' * w)
    lines.append('【成交明细】按代码分组；卖出盈亏为费后实现额，括号内为相对买入成本%')
    lines.append(f'  {"#":>4}  {"名称":40} {"交易日":12} {"方向":6} {"总金额":>14} {"盈亏":>18}')
    lines.append('-' * w)
    for i, t in enumerate(ordered, start=1):
        sym = str(t.get('symbol', ''))
        label = _trade_label_name(sym, cn_map)
        if len(label) > 40:
            label = label[:37] + '...'
        sh = int(t.get('shares', 0))
        px = float(t.get('price', 0.0))
        amt = sh * px
        act = str(t.get('action', ''))
        dt = str(t.get('date', ''))
        side = '买入' if act == 'BUY' else '卖出' if act == 'SELL' else act
        pamt = t.get('pnl_amount')
        rp = t.get('realized_pnl_pct')
        if pamt is not None and rp is not None:
            pnl_s = f'{float(pamt):+,.2f} ({float(rp):+.2f}%)'
        else:
            pnl_s = '—'
        lines.append(
            f'  {i:4d}  {label:40} {dt:12} {side:6} {amt:14,.2f} {pnl_s:>18}'
        )
    lines.append('=' * w)
    return lines


def print_trades_with_names(
    trades: List[Dict[str, Any]],
    cn_map: Dict[str, str],
    config: Dict[str, Any],
) -> None:
    _ = config  # 保留参数以兼容调用方
    if not trades:
        print('\n【成交明细】本区间无成交记录。')
        return
    for line in _fmt_trade_lines(trades, cn_map):
        print(line, flush=True)


def write_trades_csv(
    trades: List[Dict[str, Any]],
    cn_map: Dict[str, str],
    path: str,
) -> None:
    """成交明细 CSV：与控制台一致的精简列（按代码排序）。"""
    import csv as _csv

    ordered = _sort_trades_for_display(trades)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = _csv.writer(f)
        w.writerow(
            [
                '序号',
                '名称_显示',
                '代码',
                '交易日',
                '方向',
                '总金额_股数x价',
                '盈亏_费后',
                '盈亏_pct_相对成本',
            ]
        )
        for i, t in enumerate(ordered, start=1):
            sym = str(t.get('symbol', ''))
            label = _trade_label_name(sym, cn_map)
            sh = int(t.get('shares', 0))
            px = float(t.get('price', 0.0))
            amt = sh * px
            act = str(t.get('action', ''))
            side = '买入' if act == 'BUY' else '卖出' if act == 'SELL' else act
            pamt = t.get('pnl_amount')
            rp = t.get('realized_pnl_pct')
            w.writerow(
                [
                    i,
                    label,
                    sym,
                    t.get('date', ''),
                    side,
                    f'{amt:.2f}',
                    f'{float(pamt):.4f}' if pamt is not None else '',
                    f'{float(rp):.4f}' if rp is not None else '',
                ]
            )


def maybe_emit_trade_log(eng: 'DualBreakoutEngine', config: Dict[str, Any]) -> None:
    """环境变量：BACKTEST_TRADE_LOG=1 打印明细；BACKTEST_TRADES_CSV=路径 写入 CSV。"""
    want_print = os.environ.get('BACKTEST_TRADE_LOG', '').lower() in ('1', 'true', 'yes')
    csv_path = (os.environ.get('BACKTEST_TRADES_CSV') or '').strip()
    if not want_print and not csv_path:
        return
    cmap = enrich_cn_map_for_trades(load_hk_cn_name_map(HK_CN_NAMES_CSV), eng.trades)
    if want_print and not getattr(eng, '_trade_print_done', False):
        print_trades_with_names(eng.trades, cmap, config)
    if csv_path:
        write_trades_csv(eng.trades, cmap, csv_path)
        print(f'[回测] 成交明细 CSV: {csv_path}（共 {len(eng.trades)} 笔）', flush=True)


def load_trained_strategy_param_overrides() -> Dict[str, Any]:
    """读取 train_params 写出的 JSON；不存在或解析失败则返回空 dict。"""
    path = STRATEGY_PARAMS_JSON
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if 'params' in data and isinstance(data['params'], dict):
        return dict(data['params'])
    return {}


def engine_config(symbols: List[str]) -> dict:
    cfg = {
        'initial_capital': INITIAL_CAPITAL,
        'max_positions': MAX_POSITIONS,
        'position_size_pct': POSITION_SIZE_PCT,
        'stop_loss_pct': STOP_LOSS_PCT,
        'breakout_lookback': BREAKOUT_LOOKBACK,
        'trend_ma_period': TREND_MA_PERIOD,
        'vol_ma_period': VOL_MA_PERIOD,
        'volume_ratio_threshold': VOLUME_RATIO_THRESHOLD,
        'exit_donchian_days': EXIT_DONCHIAN_DAYS,
        'use_ma60_loss_exit': USE_MA60_LOSS_EXIT,
        'one_way_cost_rate': ONE_WAY_COST_RATE,
        'use_regime_filter': USE_REGIME_FILTER,
        'regime_benchmarks': REGIME_BENCHMARKS,
        'regime_mode': REGIME_MODE,
        'regime_ma_days': REGIME_MA_DAYS,
        'vol_target_annual': VOL_TARGET_ANNUAL,
        'vol_lookback': VOL_LOOKBACK,
        'vol_scale_min': VOL_SCALE_MIN,
        'vol_scale_max': VOL_SCALE_MAX,
        'trailing_activation_pct': TRAILING_ACTIVATION_PCT,
        'trailing_stop_pct': TRAILING_STOP_PCT,
        'symbols_subset': set(symbols),
    }
    ov = load_trained_strategy_param_overrides()
    if ov:
        for k, v in ov.items():
            if k in cfg and k != 'symbols_subset':
                cfg[k] = v
    cfg['symbols_subset'] = set(symbols)
    return cfg


def main() -> None:
    end_date = BACKTEST_END if BACKTEST_END is not None else date.today()
    start_bt = BACKTEST_START
    data_start = start_bt - timedelta(days=DATA_WARMUP_DAYS_BEFORE_START)

    if UNIVERSE_MODE == 'hsi_hstech':
        from hk_universe import build_hsi_hstech_universe

        try:
            symbols, _desc = build_hsi_hstech_universe(
                hsi_csv=HSI_CONSTITUENTS_CSV,
                hstech_csv=HSTECH_CONSTITUENTS_CSV,
                hsi_example=HSI_CONSTITUENTS_EXAMPLE,
                hstech_example=HSTECH_CONSTITUENTS_EXAMPLE,
            )
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        if not symbols:
            print('候选池为空：请检查恒指/恒生科技成分 CSV', file=sys.stderr)
            sys.exit(1)
    else:
        uni = UNIVERSE_CSV.strip()
        if not uni:
            for cand in ('dual_universe.csv', 'dual_universe.example.csv'):
                if os.path.exists(cand):
                    uni = cand
                    break
        if not uni or not os.path.exists(uni):
            print('未找到股票池 CSV：请设置 UNIVERSE_CSV 或放置 dual_universe.csv', file=sys.stderr)
            sys.exit(1)
        symbols = load_universe_csv(uni)
        print(f'候选池: CSV {uni}（{len(symbols)} 只）', flush=True)

    dm = DualMarketDataManager()
    load_syms = list(dict.fromkeys(symbols + ['HSI.HK', 'SPY.US', 'HSTECH.HK']))
    print(
        f'[回测] 加载 {len(load_syms)} 个标的日线（缓存在 data_cache/daily/，详见 README）',
        flush=True,
    )
    print(
        f'[回测] 请求区间：{data_start} ~ {end_date}（起点早于回测日约 {DATA_WARMUP_DAYS_BEFORE_START} 自然日，供指标预热；'
        f'不产生 {start_bt} 以前的交易）。',
        flush=True,
    )
    dm.load_stock_data(load_syms, data_start, end_date)

    if symbols:
        s0 = symbols[0]
        raw = dm._all_data.get(s0)
        if raw is not None and len(raw) > 0:
            t0, t1 = raw.index.min().date(), raw.index.max().date()
            print(
                f'[回测] 示例 {s0} 缓存 K 线：{t0} ~ {t1} 共 {len(raw)} 根（可能早于请求起点，以缓存为准；'
                f'策略决策仅从 {start_bt} 起）。'
            )
            if t0 > start_bt:
                print(
                    f'注意: 最早数据晚于回测起点 {start_bt}，净值与年度收益从 {t0.year} 年附近才有可比性。'
                )

    hsi = dm._all_data.get('HSI.HK')
    spy = dm._all_data.get('SPY.US')
    if hsi is None or getattr(hsi, 'empty', True):
        print('[回测] 恒指未载入，尝试单独拉取…', flush=True)
        hsi = load_hsi_data(data_start, end_date)
    if spy is None or getattr(spy, 'empty', True):
        print('[回测] SPY 未载入，尝试单独拉取…', flush=True)
        spy = load_us_etf('SPY.US', data_start, end_date)

    hstech = dm._all_data.get('HSTECH.HK')
    if hstech is None or getattr(hstech, 'empty', True):
        print('[回测] 恒生科技未载入，尝试单独拉取…', flush=True)
        hstech = load_hstech_data(data_start, end_date)

    blend = build_blended_benchmark(hsi, spy)

    compare_indices: Dict[str, pd.DataFrame] = {}
    if hsi is not None and not getattr(hsi, 'empty', True):
        compare_indices['恒生指数'] = hsi
    if hstech is not None and not getattr(hstech, 'empty', True):
        compare_indices['恒生科技'] = hstech

    if load_trained_strategy_param_overrides():
        print(
            f'[回测] 已加载 {STRATEGY_PARAMS_JSON} 中的训练参数（覆盖 backtest 顶部常量）。',
            flush=True,
        )
    print(
        f'[回测] 数据就绪。**交易与净值统计区间**：{start_bt} ~ {end_date}（仅此段计入回测结果）。',
        flush=True,
    )
    cfg = engine_config(symbols)
    eng = DualBreakoutEngine(dm, cfg)
    eng.run(
        start_bt,
        end_date,
        benchmark_data=blend,
        verbose=True,
        compare_indices=compare_indices if compare_indices else None,
    )
    maybe_emit_trade_log(eng, cfg)


if __name__ == '__main__':
    main()
