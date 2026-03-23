#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粗粒度参数扫描：数据只加载一次，批量回测，按夏普排序。

用法:
  python param_sweep.py
  python param_sweep.py --quick          # 约 12 组，冒烟
  python param_sweep.py --skip-ipo       # 构建候选池时不跑新股 static_info 扫描

结果写入 param_sweep_results.csv，并打印相对基准夏普的提升。

性能：对网格内每种（突破窗口×唐奇安日）预计算全历史指标并按日切片，
避免每日对全池重复 rolling；数据仍与逐日 `calculate_indicators` 一致。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# 复用 backtest 的配置与引擎
import backtest as bt
from hk_universe import build_hsi_hstech_ipo_universe


def _build_symbols(skip_ipo: bool) -> List[str]:
    if bt.UNIVERSE_MODE != 'hsi_hstech_ipo':
        print('param_sweep 当前仅针对 UNIVERSE_MODE=hsi_hstech_ipo，请在 backtest.py 中设置', file=sys.stderr)
        sys.exit(1)
    include_ipo = bt.INCLUDE_IPO_UNIVERSE and not skip_ipo
    symbols, _ = build_hsi_hstech_ipo_universe(
        hsi_csv=bt.HSI_CONSTITUENTS_CSV,
        hstech_csv=bt.HSTECH_CONSTITUENTS_CSV,
        hsi_example=bt.HSI_CONSTITUENTS_EXAMPLE,
        hstech_example=bt.HSTECH_CONSTITUENTS_EXAMPLE,
        hk_all_csv=bt.HK_ALL_STOCKS_CSV,
        include_ipo=include_ipo,
        ipo_max_age_days=bt.IPO_LISTING_MAX_AGE_DAYS,
    )
    if not symbols:
        sys.exit('候选池为空')
    return symbols


def _load_dm_and_benchmark(symbols: List[str]) -> Tuple[Any, Any]:
    end_date = bt.BACKTEST_END if bt.BACKTEST_END is not None else date.today()
    start_bt = bt.BACKTEST_START
    data_start = start_bt - timedelta(days=bt.DATA_WARMUP_DAYS_BEFORE_START)

    dm = bt.DualMarketDataManager()
    load_syms = list(dict.fromkeys(symbols + ['HSI.HK', 'SPY.US']))
    print(f'[扫描] 加载日线 {len(load_syms)} 个标的…', flush=True)
    n = dm.load_stock_data(load_syms, data_start, end_date)
    if n == 0:
        sys.exit('未加载到任何标的日线')

    hsi = dm._all_data.get('HSI.HK')
    spy = dm._all_data.get('SPY.US')
    if hsi is None or getattr(hsi, 'empty', True):
        hsi = bt.load_hsi_data(data_start, end_date)
    if spy is None or getattr(spy, 'empty', True):
        spy = bt.load_us_etf('SPY.US', data_start, end_date)

    blend = bt.build_blended_benchmark(hsi, spy)
    print('[扫描] 数据就绪。', flush=True)
    return dm, blend


def _make_cfg(
    symbols: List[str],
    *,
    breakout_lookback: int,
    stop_loss_pct: float,
    volume_ratio_threshold: float,
    use_regime_filter: bool,
    vol_target_annual: float,
    exit_donchian_days: int | None = None,
) -> dict:
    base = bt.engine_config(symbols)
    base['breakout_lookback'] = breakout_lookback
    base['stop_loss_pct'] = stop_loss_pct
    base['volume_ratio_threshold'] = volume_ratio_threshold
    base['use_regime_filter'] = use_regime_filter
    base['vol_target_annual'] = vol_target_annual
    if exit_donchian_days is not None:
        base['exit_donchian_days'] = exit_donchian_days
    return base


class IndCacheEngine(bt.DualBreakoutEngine):
    """用预计算的全历史指标表按日切片，避免每日对全池重复 rolling（扫描可快两个数量级）。"""

    def __init__(self, dm: Any, config: dict, ind_cache: Dict[Tuple[str, int, int], pd.DataFrame]) -> None:
        super().__init__(dm, config)
        self._ind_cache = ind_cache

    def _ind(self, symbol: str):
        key = (symbol, self.breakout_lookback, self.exit_donchian_days)
        full = self._ind_cache.get(key)
        if full is None:
            return super()._ind(symbol)
        if self.dm._current_date is None:
            return None
        cutoff = self.dm._current_date - timedelta(days=1)
        d = pd.DatetimeIndex(full.index).date
        sub = full.loc[d <= cutoff]
        if sub is None or len(sub) < 2:
            return None
        return sub


def _precompute_indicators(
    dm: Any,
    symbols: List[str],
    breakouts: Set[int],
    exits: Set[int],
) -> Dict[Tuple[str, int, int], pd.DataFrame]:
    end_date = bt.BACKTEST_END if bt.BACKTEST_END is not None else date.today()
    dm.set_current_date(end_date + timedelta(days=1))
    tp = int(bt.TREND_MA_PERIOD)
    vp = int(bt.VOL_MA_PERIOD)
    cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    for sym in symbols:
        for brk in breakouts:
            for ex in exits:
                df = dm.calculate_indicators(
                    sym,
                    breakout_lookback=brk,
                    trend_ma_period=tp,
                    vol_ma_period=vp,
                    exit_donchian_days=ex,
                )
                if df is not None and len(df) >= 2:
                    cache[(sym, brk, ex)] = df
    dm._current_date = None
    return cache


def _brk_exit_sets(grid: List[Dict[str, Any]], baseline_cfg: dict) -> Tuple[Set[int], Set[int]]:
    brk = {g['breakout_lookback'] for g in grid}
    ex = {g['exit_donchian_days'] for g in grid}
    brk.add(int(baseline_cfg['breakout_lookback']))
    ex.add(int(baseline_cfg['exit_donchian_days']))
    return brk, ex


def _run_one(
    dm: Any,
    blend: Any,
    symbols: List[str],
    start_bt: date,
    end_date: date,
    cfg: dict,
    ind_cache: Dict[Tuple[str, int, int], pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    if ind_cache is not None:
        eng = IndCacheEngine(dm, cfg, ind_cache)
    else:
        eng = bt.DualBreakoutEngine(dm, cfg)
    return eng.run(start_bt, end_date, benchmark_data=blend, verbose=False)


def _grid(quick: bool) -> List[Dict[str, Any]]:
    """粗网格；quick 时 12 组（3×2×2×固定）。"""
    if quick:
        breakouts = [45, 55, 65]
        stops = [0.10, 0.12]
        vols = [1.0, 1.2]
        regimes = [True]
        vtargets = [0.15]
        exits = [20]
    else:
        # 粗网格 48 组：3×2×2×2×2（止损两档、量比两档、大盘开关、波动目标开关），唐奇安固定 20
        breakouts = [45, 55, 65]
        stops = [0.08, 0.12]
        vols = [1.0, 1.2]
        regimes = [True, False]
        vtargets = [0.0, 0.15]
        exits = [20]

    rows = []
    for b, s, v, rg, vt, ex in itertools.product(breakouts, stops, vols, regimes, vtargets, exits):
        rows.append(
            {
                'breakout_lookback': b,
                'stop_loss_pct': s,
                'volume_ratio_threshold': v,
                'use_regime_filter': rg,
                'vol_target_annual': vt,
                'exit_donchian_days': ex,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='粗粒度参数扫描（夏普）')
    ap.add_argument('--quick', action='store_true', help='约 12 组组合（冒烟）')
    ap.add_argument(
        '--skip-ipo',
        action='store_true',
        help='构建候选池时不扫描新股（与 backtest 中 INCLUDE_IPO_UNIVERSE=False 等效）',
    )
    ap.add_argument(
        '--out',
        default='param_sweep_results.csv',
        help='结果 CSV 路径',
    )
    args = ap.parse_args()

    skip_ipo = args.skip_ipo or os.getenv('PARAM_SWEEP_SKIP_IPO', '').lower() in ('1', 'true', 'yes')

    grid = _grid(args.quick)
    print(f'[扫描] 组合数: {len(grid)}（quick={args.quick}）', flush=True)

    symbols = _build_symbols(skip_ipo=skip_ipo)
    print(f'[扫描] 候选池 {len(symbols)} 只', flush=True)

    dm, blend = _load_dm_and_benchmark(symbols)
    end_date = bt.BACKTEST_END if bt.BACKTEST_END is not None else date.today()
    start_bt = bt.BACKTEST_START

    # 基准：当前 backtest.py 顶层默认
    baseline_cfg = bt.engine_config(symbols)
    brk_set, ex_set = _brk_exit_sets(grid, baseline_cfg)
    print(
        f'[扫描] 预计算指标缓存（突破×唐奇安 = {len(brk_set)}×{len(ex_set)} 档 × {len(symbols)} 只）…',
        flush=True,
    )
    ind_cache = _precompute_indicators(dm, symbols, brk_set, ex_set)
    print(f'[扫描] 指标缓存条目 {len(ind_cache)}（键: 标的+突破日+唐奇安日）', flush=True)

    try:
        base_r = _run_one(dm, blend, symbols, start_bt, end_date, baseline_cfg, ind_cache)
        base_sharpe = float(base_r.get('sharpe_ratio', 0.0))
    except Exception as e:
        print(f'[扫描] 基准回测失败: {e}', flush=True)
        base_sharpe = 0.0

    print(f'[扫描] 基准（当前 backtest 默认）Sharpe: {base_sharpe:.4f}', flush=True)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, g in enumerate(grid, start=1):
        cfg = _make_cfg(
            symbols,
            breakout_lookback=g['breakout_lookback'],
            stop_loss_pct=g['stop_loss_pct'],
            volume_ratio_threshold=g['volume_ratio_threshold'],
            use_regime_filter=g['use_regime_filter'],
            vol_target_annual=g['vol_target_annual'],
            exit_donchian_days=g['exit_donchian_days'],
        )
        print(
            f'[扫描] 开始 {i}/{len(grid)} 回测 brk={g["breakout_lookback"]} stop={g["stop_loss_pct"]} '
            f'vr={g["volume_ratio_threshold"]} …',
            flush=True,
        )
        try:
            r = _run_one(dm, blend, symbols, start_bt, end_date, cfg, ind_cache)
            sh = float(r.get('sharpe_ratio', 0.0))
            ann = float(r.get('annual_return', 0.0))
            mdd = float(r.get('max_drawdown', 0.0))
            tot = float(r.get('total_return', 0.0))
            tc = int(r.get('trade_count', 0))
        except Exception as e:
            print(f'[扫描] {i}/{len(grid)} 失败: {e}', flush=True)
            continue

        d_sh = sh - base_sharpe
        results.append(
            {
                **g,
                'sharpe_ratio': sh,
                'd_sharpe_vs_baseline': d_sh,
                'annual_return': ann,
                'max_drawdown': mdd,
                'total_return': tot,
                'trade_count': tc,
            }
        )
        print(f'[扫描] 进度 {i}/{len(grid)} Sharpe={sh:.4f} Δ={d_sh:+.4f}', flush=True)

    dt = time.time() - t0
    print(f'[扫描] 耗时 {dt:.1f}s', flush=True)

    if not results:
        print('无有效结果', file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    # 写 CSV
    keys = list(results[0].keys())
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f'[扫描] 已写入 {args.out}', flush=True)

    # 控制台：提升最大的若干组
    print('\n' + '=' * 72)
    print('按夏普降序 TOP 12（相对基准 ΔSharpe）')
    print('=' * 72)
    hdr = f"{'#':>3} {'Sharpe':>8} {'ΔSharpe':>9} {'年化%':>9} {'回撤%':>8} {'交易':>6} 参数摘要"
    print(hdr)
    print('-' * len(hdr))
    for i, row in enumerate(results[:12], start=1):
        summ = (
            f"brk={row['breakout_lookback']} stop={row['stop_loss_pct']:.2f} "
            f"vr={row['volume_ratio_threshold']:.1f} reg={row['use_regime_filter']} "
            f"vt={row['vol_target_annual']:.2f} exit={row['exit_donchian_days']}"
        )
        print(
            f"{i:3d} {row['sharpe_ratio']:8.4f} {row['d_sharpe_vs_baseline']:+9.4f} "
            f"{row['annual_return']:+9.2f} {row['max_drawdown']:8.2f} {row['trade_count']:6d}  {summ}"
        )

    # 单参数敏感度：各维度取该维度下最佳夏普
    print('\n' + '=' * 72)
    print('粗看：各维度单独取「该维度内最优夏普」对应的参数值')
    print('=' * 72)
    for dim in (
        'breakout_lookback',
        'stop_loss_pct',
        'volume_ratio_threshold',
        'use_regime_filter',
        'vol_target_annual',
        'exit_donchian_days',
    ):
        best_val = None
        best_sh = -1e9
        for val in sorted(set(r[dim] for r in results), key=lambda x: (str(type(x)), x)):
            sub = [r for r in results if r[dim] == val]
            if not sub:
                continue
            mx = max(sub, key=lambda x: x['sharpe_ratio'])
            if mx['sharpe_ratio'] > best_sh:
                best_sh = mx['sharpe_ratio']
                best_val = val
        print(f'  {dim}: 当前网格内较优取值 ≈ {best_val!r}（该值下最高 Sharpe {best_sh:.4f}）')


if __name__ == '__main__':
    main()
