#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股 + 美股 日K 纯技术面趋势突破回测（无基本面）

【未来函数检查】
- 回测日 T 仅使用「截止 T-1」的日线：`get_history` 过滤 index.date <= T-1。
- 突破：`is_breakout = close > high_nd.shift(1)`，其中 `high_nd` 为滚动最高价；
  在任意一日，与之比较的是「前一日的 N 日最高价」，不包含当日 high，无前视。
- 唐奇安出场：`low_exit_level = rolling_min(low).shift(1)`，同样不包含当日 low。
- 信号与成交：信号基于 T-1 收盘；成交价用该收盘价，等价于「在 T 日按昨收成交」的简化，
  实盘中通常更差（滑点、开盘跳空），故回测往往偏乐观。

【效果为何可能「看起来很好」】
- 股票池很小或偏大盘时，夏普/回撤会失真；扩大标的与加入成本后通常会弱化。
- 年度收益若某自然年只有「较晚才开始记录的净值」，该年数字是「年内片段」而非整年开盘起算。

说明：不保证长期有效；需样本外、多区间与手续费再验证。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from trend_breakout_v2 import HistoricalDataManager, load_hsi_data


def load_us_etf(symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    from hk_stock_api import HKStockAPI

    api = HKStockAPI()
    pause = float(os.getenv('LONGPORT_REQUEST_PAUSE', '0.15'))
    df = api.get_daily_data(symbol, start_date, end_date)
    time.sleep(pause)
    return df if df is not None and len(df) > 0 else None


def build_blended_benchmark(hsi: Optional[pd.DataFrame], spy: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """等权归一化：两只指数从各自起点调到 100，再取平均收盘价作为组合基准。"""
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


class DualMarketDataManager(HistoricalDataManager):
    """在父类基础上：按 .HK / .US 使用不同成交额区间（仍无未来函数）。"""

    def __init__(
        self,
        hk_turnover: tuple = (5e6, 50e9),
        us_turnover: tuple = (2e6, 50e9),
        min_history_days: int = 80,
    ):
        super().__init__()
        self._hk_lo, self._hk_hi = hk_turnover
        self._us_lo, self._us_hi = us_turnover
        self._min_history_days = min_history_days

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
            if symbol.endswith('.HK') is False and symbol.endswith('.US') is False:
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
    def __init__(self, dm: DualMarketDataManager, config: Optional[dict] = None):
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
    ) -> dict:
        self._verbose = verbose
        warmup = max(self.breakout_lookback, self.trend_ma_period, self.exit_donchian_days, 60) + 5

        all_dates = self.dm.get_all_trading_dates()
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]

        if verbose:
            print('\n' + '=' * 60)
            print('双市场日K趋势突破（纯技术面）')
            print('=' * 60)
            print(f'初始资金: {self.initial_capital:,.0f}  区间: {start_date} ~ {end_date}')
            print(
                f'参数: 突破{self.breakout_lookback}日高 | 趋势MA{self.trend_ma_period} | '
                f'量比≥{self.volume_ratio_threshold} | 止损{self.stop_loss_pct:.0%} | '
                f'唐奇安出场{self.exit_donchian_days}日低'
            )
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

        return self._generate_report(benchmark_data, verbose)

    def _update_positions(self) -> None:
        for sym in list(self.positions.keys()):
            px = self.dm.get_latest_price(sym)
            if px:
                self.positions[sym]['current_price'] = px

    def _check_buy_signals(self, current_date: date, pool: List[str]) -> None:
        if len(self.positions) >= self.max_positions:
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
            self._execute_buy(
                current_date,
                symbol,
                px,
                f'突破{self.breakout_lookback}日高,量比{vr:.2f},>MA{self.trend_ma_period}',
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

    def _execute_buy(self, current_date: date, symbol: str, price: float, reason: str) -> None:
        max_amt = self.total_value * self.position_size_pct
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

    def _execute_sell(self, current_date: date, symbol: str, price: float, ratio: float, reason: str) -> None:
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        sell_shares = int(pos['shares'] * ratio)
        if sell_shares <= 0:
            return
        fee = self.one_way_cost_rate
        self.cash += sell_shares * price * (1.0 - fee)
        pnl_pct = (price / pos['buy_price'] - 1.0) * 100
        self.trades.append(
            {
                'date': str(current_date),
                'action': 'SELL',
                'symbol': symbol,
                'price': price,
                'shares': sell_shares,
                'reason': f'{reason}, 盈亏:{pnl_pct:+.1f}%',
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

    def _generate_report(self, benchmark_data: Optional[pd.DataFrame], verbose: bool) -> dict:
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
        sharpe = float(np.sqrt(252) * np.mean(arr) / np.std(arr, ddof=1)) if len(arr) > 1 and np.std(arr, ddof=1) > 1e-12 else 0.0

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
            print(f'  最大回撤: {max_dd:.2%}\n  年化Sharpe(Rf=0): {sharpe:.3f}')
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
            print(f'\n【交易】买入{len(buys)} 卖出{len(sells)}  期末持仓{len(self.positions)}只')

        return {
            'total_return': total_return,
            'annual_return': annual,
            'max_drawdown': max_dd * 100,
            'sharpe_ratio': sharpe,
            'trade_count': len(buys) + len(sells),
            'benchmark_return': bench_ret,
            'excess_return': ex,
            'yearly_returns': yearly,
        }


def load_dual_universe(path: str) -> List[str]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description='港股+美股日K技术面趋势突破回测')
    ap.add_argument(
        '--universe',
        default='',
        help='CSV 含 symbol 列；默认依次尝试 dual_universe.csv、dual_universe.example.csv',
    )
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default='')
    ap.add_argument('--max-symbols', type=int, default=0, help='0=全部')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    uni = args.universe
    if not uni:
        for cand in ('dual_universe.csv', 'dual_universe.example.csv'):
            if os.path.exists(cand):
                uni = cand
                break
    if not uni or not os.path.exists(uni):
        print('未找到股票池 CSV，请使用 --universe 或放置 dual_universe.csv', file=sys.stderr)
        sys.exit(1)
    print(f'使用股票池: {uni}')

    symbols = load_dual_universe(uni)
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    end_date = date.today() if not args.end else datetime.strptime(args.end, '%Y-%m-%d').date()
    start_bt = datetime.strptime(args.start, '%Y-%m-%d').date()
    data_start = start_bt - timedelta(days=400)

    dm = DualMarketDataManager()
    load_syms = list(dict.fromkeys(symbols + ['HSI.HK', 'SPY.US']))
    print(f'加载 {len(load_syms)} 个代码日线 …')
    dm.load_stock_data(load_syms, data_start, end_date)

    if symbols:
        s0 = symbols[0]
        raw = dm._all_data.get(s0)
        if raw is not None and len(raw) > 0:
            t0, t1 = raw.index.min().date(), raw.index.max().date()
            print(f'行情覆盖（示例 {s0}）: {t0} ~ {t1} 共 {len(raw)} 根')
            if t0 > start_bt:
                print(
                    f'注意: 最早数据晚于回测起点 {start_bt}，'
                    f'净值与「年度收益」实际从 {t0.year} 年附近才有意义。'
                    f'已启用 hk_stock_api 分段拉取以尽量延长历史。'
                )

    hsi = load_hsi_data(data_start, end_date)
    spy = load_us_etf('SPY.US', data_start, end_date)
    blend = build_blended_benchmark(hsi, spy)

    cfg = {
        'initial_capital': 100000,
        'max_positions': 10,
        'position_size_pct': 0.10,
        'stop_loss_pct': 0.12,
        'breakout_lookback': 55,
        'trend_ma_period': 50,
        'volume_ratio_threshold': 1.2,
        'exit_donchian_days': 20,
        'use_ma60_loss_exit': True,
        'symbols_subset': set(symbols),
    }
    eng = DualBreakoutEngine(dm, cfg)
    eng.run(start_bt, end_date, benchmark_data=blend, verbose=not args.quiet)

    print('\n【说明】此为探索性回测；「长期有效」需跨时段、样本外与加入成本后再论证。')


if __name__ == '__main__':
    main()
