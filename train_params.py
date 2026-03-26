#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本（与回测推理分离）：在训练期网格搜索，样本外验证；最优参数写入 JSON，
之后可多次运行 `backtest.py` 加载该 JSON 做推理，无需重复训练。

**数据与回测一致**：对称信号模型 — **日 K** 突破/趋势/量能入场，**同窗口通道下破或趋势空头**出场；**60m** 镜像（均线、上下破、首根阴阳）；**无止损/移动止盈**。成交价 **第二根 60m 开盘价**。

**默认 20 组**（5×2×2：通道宽×量比×60m 均线）；训练 CSV：`--out`；样本外：`*_oos_best.csv`；
策略参数 JSON：默认 `trained_strategy_params.json`（与 backtest.STRATEGY_PARAMS_JSON 一致）。

用法:
  python train_params.py                  # 默认 20 组 + 样本外 + 写 JSON
  python train_params.py --quick          # 少量冒烟
  python train_params.py --exp-v2         # 扩展：趋势MA×量均线×通道宽（18 组）
  python train_params.py --exp-v3         # 扩展：波动目标×60m 参数（24 组）
  python train_params.py --exp-hourly     # 扩展网格：日K×60m 联合（48 组）
  python train_params.py --full           # 约 48 组粗网格（可选）
  python train_params.py --refine-regime-off   # 约 36 组：通道×量比×60m均线
  python train_params.py --no-save-strategy-json   # 不写 JSON
  python train_params.py --train-start 2024-01-01 --train-end 2025-12-31 \\
      --test-start 2022-01-01 --test-end 2023-12-31 --out-strategy-json exp.json

训练期见本文件 `TRAIN_*`（可用 `--train-start`/`--train-end` 覆盖）；样本外默认自 `DEFAULT_OOS_TEST_START`
起至今日（或 `BACKTEST_END`），避免与训练段重叠；可用 `--test-start`/`--test-end` 覆盖。

性能：对网格内每种（突破窗口×趋势MA×量均线）预计算全历史指标并按日切片；
数据与逐日 `calculate_indicators` 一致（出场与入场共用通道宽 `breakout_lookback`）。

环境变量 `TREND_TRAIN_MAX_SYMBOLS`：设为正整数时仅取候选池前 N 只，用于加快粗网格（正式选参请不设）。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# 复用 backtest 的配置与引擎
import backtest as bt
from hk_universe import build_hsi_hstech_universe

# 训练期：须落在 60m 可拉取区间内（见 hk_stock_api 默认 TREND_HOURLY_MIN_DATE=2024-04-01）
TRAIN_START = date(2024, 4, 1)
TRAIN_END = date(2025, 8, 31)
# 未指定 --test-start 时的样本外默认起点（与训练段不重叠）
DEFAULT_OOS_TEST_START = date(2025, 9, 1)


def _save_trained_strategy_json(
    path: str,
    best_row: Dict[str, Any],
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    train_sharpe: float,
    test_sharpe: float,
) -> None:
    params: Dict[str, Any] = {
        'breakout_lookback': int(best_row['breakout_lookback']),
        'volume_ratio_threshold': float(best_row['volume_ratio_threshold']),
        'use_regime_filter': bool(best_row['use_regime_filter']),
        'vol_target_annual': float(best_row.get('vol_target_annual', 0.0)),
        'hourly_ma_period': int(best_row.get('hourly_ma_period', bt.HOURLY_MA_PERIOD)),
        'hourly_breakout_bars': int(best_row.get('hourly_breakout_bars', bt.HOURLY_BREAKOUT_BARS)),
        'use_hourly_first_bar_bullish': bool(
            best_row.get('use_hourly_first_bar_bullish', bt.USE_HOURLY_FIRST_BAR_BULLISH)
        ),
        'relax_hourly_when_incomplete': bool(
            best_row.get('relax_hourly_when_incomplete', bt.RELAX_HOURLY_WHEN_INCOMPLETE)
        ),
    }
    for k in ('trend_ma_period', 'vol_ma_period'):
        if k in best_row and best_row[k] is not None:
            params[k] = int(best_row[k])
    payload = {
        'version': 3,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'train_period': {'start': str(train_start), 'end': str(train_end)},
        'test_period': {'start': str(test_start), 'end': str(test_end)},
        'metrics': {
            'train_sharpe_best': train_sharpe,
            'test_sharpe_oos': test_sharpe,
        },
        'params': params,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_symbols() -> List[str]:
    if bt.UNIVERSE_MODE != 'hsi_hstech':
        print('train_params 当前仅针对 UNIVERSE_MODE=hsi_hstech，请在 backtest.py 中设置', file=sys.stderr)
        sys.exit(1)
    symbols, _ = build_hsi_hstech_universe(
        hsi_csv=bt.HSI_CONSTITUENTS_CSV,
        hstech_csv=bt.HSTECH_CONSTITUENTS_CSV,
        hsi_example=bt.HSI_CONSTITUENTS_EXAMPLE,
        hstech_example=bt.HSTECH_CONSTITUENTS_EXAMPLE,
    )
    if not symbols:
        sys.exit('候选池为空')
    cap = os.environ.get('TREND_TRAIN_MAX_SYMBOLS', '').strip()
    if cap.isdigit():
        n = int(cap)
        if 0 < n < len(symbols):
            symbols = symbols[:n]
            print(
                f'[扫描] TREND_TRAIN_MAX_SYMBOLS={n}：仅取候选池前 {n} 只（加快网格；全池请不设该变量）',
                flush=True,
            )
    return symbols


def _load_dm_and_benchmark(
    symbols: List[str],
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> Tuple[Any, Any, date]:
    """一次载入：覆盖训练/测试两段，取最早起点前 warmup、最晚终点为 load_end。"""
    earliest = min(train_start, test_start)
    te = test_end if test_end is not None else date.today()
    load_end = max(train_end, te)
    data_start = earliest - timedelta(days=bt.DATA_WARMUP_DAYS_BEFORE_START)
    anchor = bt.strategy_anchor_date()
    hourly_load_start = max(data_start, anchor)

    dm = bt.DualMarketDataManager()
    load_syms = list(dict.fromkeys(symbols + ['HSI.HK', 'SPY.US', 'HSTECH.HK']))
    print(
        f'[扫描] 日线请求：{data_start} ~ {load_end}（较最早评估窗口早约 '
        f'{bt.DATA_WARMUP_DAYS_BEFORE_START} 自然日预热；训练/测试切片仍按各自起止）。',
        flush=True,
    )
    print(f'[扫描] 加载日线 {len(load_syms)} 个标的…', flush=True)
    n = dm.load_stock_data(load_syms, data_start, load_end)
    if n == 0:
        sys.exit('未加载到任何标的日线')

    sym_trade = [s for s in symbols if s.endswith('.HK') or s.endswith('.US')]
    if sym_trade:
        print(
            f'[扫描] 60m 请求：{hourly_load_start} ~ {load_end}（锚点 {anchor}，与 hk_stock_api 一致）。',
            flush=True,
        )
        print(f'[扫描] 加载 60m K {len(sym_trade)} 个标的…', flush=True)
        dm.load_hourly_data(sym_trade, hourly_load_start, load_end)

    hsi = dm._all_data.get('HSI.HK')
    spy = dm._all_data.get('SPY.US')
    if hsi is None or getattr(hsi, 'empty', True):
        hsi = bt.load_hsi_data(data_start, load_end)
    if spy is None or getattr(spy, 'empty', True):
        spy = bt.load_us_etf('SPY.US', data_start, load_end)

    blend = bt.build_blended_benchmark(hsi, spy)
    print('[扫描] 数据就绪。', flush=True)
    return dm, blend, load_end


def _make_cfg_from_row(symbols: List[str], row: Dict[str, Any]) -> dict:
    """从网格行合并到 engine_config；未出现的键沿用 backtest 默认 + 已加载 JSON。"""
    base = bt.engine_config(symbols)
    if 'breakout_lookback' in row:
        base['breakout_lookback'] = int(row['breakout_lookback'])
    if 'volume_ratio_threshold' in row:
        base['volume_ratio_threshold'] = float(row['volume_ratio_threshold'])
    if 'use_regime_filter' in row:
        base['use_regime_filter'] = bool(row['use_regime_filter'])
    if 'vol_target_annual' in row:
        base['vol_target_annual'] = float(row['vol_target_annual'])
    if 'trend_ma_period' in row and row['trend_ma_period'] is not None:
        base['trend_ma_period'] = int(row['trend_ma_period'])
    if 'vol_ma_period' in row and row['vol_ma_period'] is not None:
        base['vol_ma_period'] = int(row['vol_ma_period'])
    if 'hourly_ma_period' in row and row['hourly_ma_period'] is not None:
        base['hourly_ma_period'] = int(row['hourly_ma_period'])
    if 'hourly_breakout_bars' in row and row['hourly_breakout_bars'] is not None:
        base['hourly_breakout_bars'] = int(row['hourly_breakout_bars'])
    if 'use_hourly_first_bar_bullish' in row and row['use_hourly_first_bar_bullish'] is not None:
        base['use_hourly_first_bar_bullish'] = bool(row['use_hourly_first_bar_bullish'])
    if 'relax_hourly_when_incomplete' in row and row['relax_hourly_when_incomplete'] is not None:
        base['relax_hourly_when_incomplete'] = bool(row['relax_hourly_when_incomplete'])
    return base


IndCacheKey = Tuple[str, int, int, int]


class IndCacheEngine(bt.DualBreakoutEngine):
    """用预计算的全历史指标表按日切片，避免每日对全池重复 rolling（扫描可快两个数量级）。"""

    def __init__(self, dm: Any, config: dict, ind_cache: Dict[IndCacheKey, pd.DataFrame]) -> None:
        super().__init__(dm, config)
        self._ind_cache = ind_cache

    def _ind(self, symbol: str):
        key: IndCacheKey = (
            symbol,
            self.breakout_lookback,
            self.trend_ma_period,
            self.vol_ma_period,
        )
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
    trend_periods: Set[int],
    vol_periods: Set[int],
    *,
    load_end: date,
) -> Dict[IndCacheKey, pd.DataFrame]:
    dm.set_current_date(load_end + timedelta(days=1))
    cache: Dict[IndCacheKey, pd.DataFrame] = {}
    for sym in symbols:
        for brk in breakouts:
            for tp in trend_periods:
                for vp in vol_periods:
                    df = dm.calculate_indicators(
                        sym,
                        breakout_lookback=brk,
                        trend_ma_period=tp,
                        vol_ma_period=vp,
                    )
                    if df is not None and len(df) >= 2:
                        cache[(sym, brk, tp, vp)] = df
    dm._current_date = None
    return cache


def _collect_cache_dims(grid: List[Dict[str, Any]], baseline_cfg: dict) -> Tuple[Set[int], Set[int], Set[int]]:
    btp = int(baseline_cfg['trend_ma_period'])
    bvp = int(baseline_cfg['vol_ma_period'])
    brk: Set[int] = set()
    tp: Set[int] = set()
    vp: Set[int] = set()
    for g in grid:
        brk.add(int(g['breakout_lookback']))
        tp.add(int(g.get('trend_ma_period', btp)))
        vp.add(int(g.get('vol_ma_period', bvp)))
    brk.add(int(baseline_cfg['breakout_lookback']))
    tp.add(btp)
    vp.add(bvp)
    return brk, tp, vp


def _run_one(
    dm: Any,
    blend: Any,
    symbols: List[str],
    start_bt: date,
    end_date: date,
    cfg: dict,
    ind_cache: Dict[IndCacheKey, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    if ind_cache is not None:
        eng = IndCacheEngine(dm, cfg, ind_cache)
    else:
        eng = bt.DualBreakoutEngine(dm, cfg)
    return eng.run(start_bt, end_date, benchmark_data=blend, verbose=False)


def _grid_default_train() -> List[Dict[str, Any]]:
    """默认 20 组：5×2×2（通道宽 × 量比 × 60m 均线）；关大盘、关波动缩放；hourly_breakout=0。"""
    breakouts = [40, 44, 48, 52, 55]
    vols = [1.0, 1.2]
    hourly_ma_opts = [0, 12]
    rows = []
    for b, v, hm in itertools.product(breakouts, vols, hourly_ma_opts):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': v,
                'use_regime_filter': False,
                'vol_target_annual': 0.0,
                'hourly_ma_period': hm,
                'hourly_breakout_bars': 0,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _grid_quick_smoke() -> List[Dict[str, Any]]:
    """冒烟 3 组（对称信号）。"""
    return [
        {
            'breakout_lookback': 45,
            'volume_ratio_threshold': 1.2,
            'use_regime_filter': False,
            'vol_target_annual': 0.0,
            'hourly_ma_period': 12,
            'hourly_breakout_bars': 0,
            'use_hourly_first_bar_bullish': True,
        },
        {
            'breakout_lookback': 52,
            'volume_ratio_threshold': 1.2,
            'use_regime_filter': False,
            'vol_target_annual': 0.0,
            'hourly_ma_period': 0,
            'hourly_breakout_bars': 0,
            'use_hourly_first_bar_bullish': True,
        },
        {
            'breakout_lookback': 48,
            'volume_ratio_threshold': 1.2,
            'use_regime_filter': False,
            'vol_target_annual': 0.0,
            'hourly_ma_period': 12,
            'hourly_breakout_bars': 12,
            'use_hourly_first_bar_bullish': True,
        },
    ]


def _grid_full() -> List[Dict[str, Any]]:
    """可选粗网格 48 组：通道×量比×大盘×波动×60m均线。"""
    breakouts = [45, 55, 65]
    vols = [1.0, 1.2]
    regimes = [True, False]
    vtargets = [0.0, 0.15]
    hmas = [0, 12]

    rows = []
    for b, v, rg, vt, hm in itertools.product(breakouts, vols, regimes, vtargets, hmas):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': v,
                'use_regime_filter': rg,
                'vol_target_annual': vt,
                'hourly_ma_period': hm,
                'hourly_breakout_bars': 0,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _grid_refine_regime_off() -> List[Dict[str, Any]]:
    """固定关大盘：通道宽 × 量比 × 60m 均线（6×3×2=36）。"""
    breakouts = [40, 42, 45, 48, 50, 52]
    vols = [1.0, 1.1, 1.2]
    hmas = [0, 12]

    rows = []
    for b, v, hm in itertools.product(breakouts, vols, hmas):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': v,
                'use_regime_filter': False,
                'vol_target_annual': 0.0,
                'hourly_ma_period': hm,
                'hourly_breakout_bars': 0,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _grid_exp_hourly() -> List[Dict[str, Any]]:
    """日K×60m 联合扩展：通道×量比 × 小时均线 × 小时上下破，共 48 组。"""
    breakouts = [48, 55]
    vols = [1.0, 1.2]
    hmas = [0, 8, 12, 20]
    hbrk = [0, 10, 20]
    rows: List[Dict[str, Any]] = []
    for b, v, hm, hb in itertools.product(breakouts, vols, hmas, hbrk):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': v,
                'use_regime_filter': False,
                'vol_target_annual': 0.0,
                'hourly_ma_period': hm,
                'hourly_breakout_bars': hb,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _grid_exp_v2() -> List[Dict[str, Any]]:
    """扩展：通道宽 × 趋势MA × 量均线；关大盘、关波动缩放。共 18 组。"""
    breakouts = [48, 55, 65]
    trend_ma = [45, 50, 60]
    vol_ma = [15, 20]
    rows: List[Dict[str, Any]] = []
    for b, tp, vp in itertools.product(breakouts, trend_ma, vol_ma):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': 1.2,
                'use_regime_filter': False,
                'vol_target_annual': 0.0,
                'trend_ma_period': tp,
                'vol_ma_period': vp,
                'hourly_ma_period': 12,
                'hourly_breakout_bars': 0,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _grid_exp_v3() -> List[Dict[str, Any]]:
    """扩展：波动目标 × 60m 均线 × 60m 上下破。共 24 组。"""
    breakouts = [48, 52]
    vtargets = [0.0, 0.15]
    hmas = [0, 12, 20]
    hbrk = [0, 12]
    rows: List[Dict[str, Any]] = []
    for b, vt, hm, hb in itertools.product(breakouts, vtargets, hmas, hbrk):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': 1.2,
                'use_regime_filter': False,
                'vol_target_annual': vt,
                'trend_ma_period': 50,
                'vol_ma_period': 20,
                'hourly_ma_period': hm,
                'hourly_breakout_bars': hb,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _grid_exp_multitarget() -> List[Dict[str, Any]]:
    """多目标粗扫：偏少交易（抬高量比/通道）× 大盘过滤 × 60m；3×3×2×3×2=108 组。用于约束年化/笔数/胜率。"""
    breakouts = [55, 65, 75]
    vols = [1.15, 1.3, 1.45]
    regimes = [False, True]
    hmas = [0, 12, 20]
    hbrk = [0, 12]
    rows: List[Dict[str, Any]] = []
    for b, v, rg, hm, hb in itertools.product(breakouts, vols, regimes, hmas, hbrk):
        rows.append(
            {
                'breakout_lookback': b,
                'volume_ratio_threshold': v,
                'use_regime_filter': rg,
                'vol_target_annual': 0.0,
                'hourly_ma_period': hm,
                'hourly_breakout_bars': hb,
                'use_hourly_first_bar_bullish': True,
            }
        )
    return rows


def _multitarget_score(row: Dict[str, Any]) -> float:
    """在满足硬约束时综合 Sharpe / 胜率 / 年化；不满足为负大数。"""
    ann = float(row.get('annual_return', 0.0))
    tpy = float(row.get('trades_per_year', 0.0))
    wr = float(row.get('win_rate', 0.0))
    sh = float(row.get('sharpe_ratio', 0.0))
    if ann < 20.0:
        return -1e9
    if tpy < 20.0 or tpy > 50.0:
        return -1e9
    return sh * 3.0 + wr * 0.08 + ann * 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description='粗粒度参数扫描（夏普）')
    ap.add_argument('--quick', action='store_true', help='3 组冒烟')
    ap.add_argument(
        '--full',
        action='store_true',
        help='约 48 组粗网格（可选；默认 20 组对称信号）',
    )
    ap.add_argument(
        '--refine-regime-off',
        action='store_true',
        help='约 36 组细网格（brk/stop 局部；与 --quick/--full 互斥）',
    )
    ap.add_argument(
        '--exp-v2',
        action='store_true',
        help='扩展实验 v2：趋势MA×量均线×通道（18 组；与上列模式互斥）',
    )
    ap.add_argument(
        '--exp-v3',
        action='store_true',
        help='扩展实验 v3：波动目标×60m 参数（24 组；与上列模式互斥）',
    )
    ap.add_argument(
        '--exp-hourly',
        action='store_true',
        help='扩展实验：日K×60m 联合（48 组；与上列模式互斥）',
    )
    ap.add_argument(
        '--exp-multitarget',
        action='store_true',
        help='多目标粗扫：约 54 组（抬高量比/通道 + 大盘×60m），结果含胜率与年化笔数',
    )
    ap.add_argument(
        '--out',
        default='param_sweep_results.csv',
        help='训练期网格结果 CSV 路径',
    )
    ap.add_argument(
        '--out-strategy-json',
        default=None,
        metavar='PATH',
        help='最优策略参数 JSON（默认与 backtest.STRATEGY_PARAMS_JSON 相同）',
    )
    ap.add_argument(
        '--no-save-strategy-json',
        action='store_true',
        help='不写入 trained_strategy_params.json',
    )
    ap.add_argument('--train-start', type=str, default=None, metavar='YYYY-MM-DD', help='训练区间起点（覆盖默认 TRAIN_START）')
    ap.add_argument('--train-end', type=str, default=None, metavar='YYYY-MM-DD', help='训练区间终点（覆盖默认 TRAIN_END）')
    ap.add_argument(
        '--test-start',
        type=str,
        default=None,
        metavar='YYYY-MM-DD',
        help='样本外区间起点（默认 DEFAULT_OOS_TEST_START，与训练段错开）',
    )
    ap.add_argument('--test-end', type=str, default=None, metavar='YYYY-MM-DD', help='样本外区间终点（覆盖默认 BACKTEST_*）')
    args = ap.parse_args()

    mode_flags = sum(
        bool(x)
        for x in (
            args.quick,
            args.full,
            args.refine_regime_off,
            args.exp_v2,
            args.exp_v3,
            args.exp_hourly,
            args.exp_multitarget,
        )
    )
    if mode_flags > 1:
        print(
            '请只选其一：--quick / --full / --refine-regime-off / --exp-v2 / --exp-v3 / '
            '--exp-hourly / --exp-multitarget',
            file=sys.stderr,
        )
        sys.exit(2)

    if args.refine_regime_off:
        grid = _grid_refine_regime_off()
        print(f'[扫描] 组合数: {len(grid)}（模式=refine-regime-off）', flush=True)
    elif args.full:
        grid = _grid_full()
        print(f'[扫描] 组合数: {len(grid)}（模式=full）', flush=True)
    elif args.exp_v2:
        grid = _grid_exp_v2()
        print(f'[扫描] 组合数: {len(grid)}（模式=exp-v2 趋势MA×量均线）', flush=True)
    elif args.exp_v3:
        grid = _grid_exp_v3()
        print(f'[扫描] 组合数: {len(grid)}（模式=exp-v3 波动×60m）', flush=True)
    elif args.exp_hourly:
        grid = _grid_exp_hourly()
        print(f'[扫描] 组合数: {len(grid)}（模式=exp-hourly 日K×60m）', flush=True)
    elif args.exp_multitarget:
        grid = _grid_exp_multitarget()
        print(f'[扫描] 组合数: {len(grid)}（模式=exp-multitarget 年化/笔数/胜率）', flush=True)
    elif args.quick:
        grid = _grid_quick_smoke()
        print(f'[扫描] 组合数: {len(grid)}（模式=quick 冒烟）', flush=True)
    else:
        grid = _grid_default_train()
        print(f'[扫描] 组合数: {len(grid)}（模式=default 训练 20 组 日K×60m）', flush=True)

    symbols = _build_symbols()
    print(f'[扫描] 候选池 {len(symbols)} 只', flush=True)

    def _pd(s: str | None, default: date) -> date:
        if not s:
            return default
        return date.fromisoformat(s.strip())

    train_start = _pd(args.train_start, TRAIN_START)
    train_end = _pd(args.train_end, TRAIN_END)
    test_start = _pd(args.test_start, DEFAULT_OOS_TEST_START)
    test_end = _pd(args.test_end, bt.BACKTEST_END if bt.BACKTEST_END is not None else date.today())
    print(
        f'[扫描] 训练期 {train_start} ~ {train_end} ｜ 测试期 {test_start} ~ {test_end}（样本外）',
        flush=True,
    )

    dm, blend, load_end = _load_dm_and_benchmark(symbols, train_start, train_end, test_start, test_end)

    # 基准：当前 backtest.py 顶层默认
    baseline_cfg = bt.engine_config(symbols)
    brk_set, tp_set, vp_set = _collect_cache_dims(grid, baseline_cfg)
    print(
        f'[扫描] 预计算指标缓存（突破×趋势MA×量均线 = '
        f'{len(brk_set)}×{len(tp_set)}×{len(vp_set)} 档 × {len(symbols)} 只）…',
        flush=True,
    )
    ind_cache = _precompute_indicators(
        dm, symbols, brk_set, tp_set, vp_set, load_end=load_end
    )
    print(
        f'[扫描] 指标缓存条目 {len(ind_cache)}（键: 标的+突破+趋势MA+量均线）',
        flush=True,
    )

    try:
        base_train = _run_one(dm, blend, symbols, train_start, train_end, baseline_cfg, ind_cache)
        base_sharpe = float(base_train.get('sharpe_ratio', 0.0))
    except Exception as e:
        print(f'[扫描] 训练期基准回测失败: {e}', flush=True)
        base_sharpe = 0.0

    print(f'[扫描] 训练期基准（当前 backtest 默认参数）Sharpe: {base_sharpe:.4f}', flush=True)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, g in enumerate(grid, start=1):
        cfg = _make_cfg_from_row(symbols, g)
        tma = g.get('trend_ma_period', cfg.get('trend_ma_period', bt.TREND_MA_PERIOD))
        hm = g.get('hourly_ma_period', cfg.get('hourly_ma_period', bt.HOURLY_MA_PERIOD))
        hb = g.get('hourly_breakout_bars', cfg.get('hourly_breakout_bars', bt.HOURLY_BREAKOUT_BARS))
        print(
            f'[扫描] 开始 {i}/{len(grid)} 回测 brk={g["breakout_lookback"]} '
            f'vr={g["volume_ratio_threshold"]} reg={g["use_regime_filter"]} vt={g["vol_target_annual"]} '
            f'trendMA={tma} hMA={hm} hBrk={hb} …',
            flush=True,
        )
        try:
            r = _run_one(dm, blend, symbols, train_start, train_end, cfg, ind_cache)
            sh = float(r.get('sharpe_ratio', 0.0))
            ann = float(r.get('annual_return', 0.0))
            mdd = float(r.get('max_drawdown', 0.0))
            tot = float(r.get('total_return', 0.0))
            tc = int(r.get('trade_count', 0))
            tpy = float(r.get('trades_per_year', 0.0))
            wr = float(r.get('win_rate', 0.0))
            rtc = int(r.get('round_trip_count', 0))
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
                'round_trip_count': rtc,
                'trades_per_year': tpy,
                'win_rate': wr,
            }
        )
        print(
            f'[扫描] 进度 {i}/{len(grid)} Sharpe={sh:.4f} Δ={d_sh:+.4f} '
            f'年化={ann:+.1f}% 笔/年={tpy:.1f} 胜率={wr:.1f}%',
            flush=True,
        )

    dt = time.time() - t0
    print(f'[扫描] 耗时 {dt:.1f}s', flush=True)

    if not results:
        print('无有效结果', file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    # 写 CSV（合并列名，避免不同网格行键集不一致）
    keys = list(dict.fromkeys(k for row in results for k in row))
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f'[扫描] 已写入 {args.out}', flush=True)

    # 控制台：提升最大的若干组
    print('\n' + '=' * 72)
    print('训练期：按夏普降序 TOP 12（相对训练期基准 ΔSharpe）')
    print('=' * 72)
    hdr = f"{'#':>3} {'Sharpe':>8} {'ΔSharpe':>9} {'年化%':>9} {'回撤%':>8} {'交易':>6} 参数摘要"
    print(hdr)
    print('-' * len(hdr))
    for i, row in enumerate(results[:12], start=1):
        summ = (
            f"brk={row['breakout_lookback']} "
            f"vr={row['volume_ratio_threshold']:.1f} reg={row['use_regime_filter']} "
            f"vt={row['vol_target_annual']:.2f}"
        )
        if 'trend_ma_period' in row:
            summ += f" tMA={row['trend_ma_period']}"
        if 'hourly_ma_period' in row:
            summ += f" hMA={row['hourly_ma_period']}"
        if 'hourly_breakout_bars' in row:
            summ += f" hBrk={row['hourly_breakout_bars']}"
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
        'volume_ratio_threshold',
        'use_regime_filter',
        'vol_target_annual',
        'trend_ma_period',
        'vol_ma_period',
        'hourly_ma_period',
        'hourly_breakout_bars',
        'use_hourly_first_bar_bullish',
    ):
        if not any(dim in r for r in results):
            print(f'  {dim}: 当前网格未包含该维度，跳过')
            continue
        best_val = None
        best_sh = -1e9
        for val in sorted(set(r[dim] for r in results if dim in r), key=lambda x: (str(type(x)), x)):
            sub = [r for r in results if r.get(dim) == val]
            if not sub:
                continue
            mx = max(sub, key=lambda x: x['sharpe_ratio'])
            if mx['sharpe_ratio'] > best_sh:
                best_sh = mx['sharpe_ratio']
                best_val = val
        print(f'  {dim}: 当前网格内较优取值 ≈ {best_val!r}（该值下最高 Sharpe {best_sh:.4f}）')

    feasible = [r for r in results if _multitarget_score(r) > -1e8]
    feasible.sort(key=_multitarget_score, reverse=True)

    print('\n' + '=' * 72)
    print('训练期：多目标约束（年化≥20%，20≤完整交易/年≤50）综合分 TOP 10')
    print('=' * 72)
    if feasible:
        h2 = f"{'#':>3} {'综合分':>10} {'Sharpe':>8} {'年化%':>9} {'笔/年':>8} {'胜率%':>8} {'回撤%':>8} 摘要"
        print(h2)
        print('-' * len(h2))
        for i, row in enumerate(feasible[:10], start=1):
            sc = _multitarget_score(row)
            summ = (
                f"brk={row['breakout_lookback']} vr={row['volume_ratio_threshold']:.2f} "
                f"reg={row['use_regime_filter']}"
            )
            if 'hourly_ma_period' in row:
                summ += f" hMA={row['hourly_ma_period']}"
            if 'hourly_breakout_bars' in row:
                summ += f" hBrk={row['hourly_breakout_bars']}"
            print(
                f"{i:3d} {sc:10.2f} {row['sharpe_ratio']:8.4f} "
                f"{row['annual_return']:+9.2f} {row['trades_per_year']:8.1f} "
                f"{row['win_rate']:8.1f} {row['max_drawdown']:8.2f}  {summ}"
            )
    else:
        print('  当前网格内无同时满足「年化≥20% 且 20≤笔/年≤50」的组合。')
        relaxed: List[Dict[str, Any]] = []
        for r in results:
            ann = float(r.get('annual_return', 0.0))
            tpy = float(r.get('trades_per_year', 0.0))
            if ann < 15.0 or tpy < 12.0 or tpy > 80.0:
                continue
            sh = float(r.get('sharpe_ratio', 0.0))
            wr = float(r.get('win_rate', 0.0))
            r['_rel'] = sh * 3.0 + wr * 0.08 + ann * 0.05
            relaxed.append(r)
        relaxed.sort(key=lambda x: float(x.get('_rel', 0.0)), reverse=True)
        print('  放宽参考（年化≥15%，12≤笔/年≤80）按启发分 TOP 8：')
        for i, row in enumerate(relaxed[:8], start=1):
            summ = f"brk={row['breakout_lookback']} vr={row['volume_ratio_threshold']:.2f}"
            print(
                f"  {i}. 启发分={row['_rel']:.2f} Sharpe={row['sharpe_ratio']:.4f} "
                f"年化={row['annual_return']:+.1f}% 笔/年={row['trades_per_year']:.1f} "
                f"胜率={row['win_rate']:.1f}%  {summ}"
            )

    # 训练期最优参数 → 测试期样本外（--exp-multitarget 时优先用约束内综合最优）
    best_row = results[0]
    best_pick = 'train_sharpe_max'
    if args.exp_multitarget and feasible:
        best_row = feasible[0]
        best_pick = 'multitarget_feasible'
    best_cfg = _make_cfg_from_row(symbols, best_row)
    try:
        oos = _run_one(dm, blend, symbols, test_start, test_end, best_cfg, ind_cache)
        base_test = _run_one(dm, blend, symbols, test_start, test_end, baseline_cfg, ind_cache)
        oos_sh = float(oos.get('sharpe_ratio', 0.0))
        base_test_sh = float(base_test.get('sharpe_ratio', 0.0))
    except Exception as e:
        print(f'[扫描] 样本外回测失败: {e}', flush=True)
        oos = {}
        base_test = {}
        oos_sh = 0.0
        base_test_sh = 0.0

    print('\n' + '=' * 72)
    print(f'样本外（测试期）：选用「{best_pick}」对应参数')
    print('=' * 72)
    extra = ''
    if 'trend_ma_period' in best_row:
        extra += f" trendMA={best_row['trend_ma_period']}"
    if best_row.get('hourly_ma_period') is not None:
        extra += f" hMA={best_row['hourly_ma_period']}"
    if best_row.get('hourly_breakout_bars') is not None:
        extra += f" hBrk={best_row['hourly_breakout_bars']}"
    print(
        f"  训练期最优 Sharpe: {best_row['sharpe_ratio']:.4f}  "
        f"brk={best_row['breakout_lookback']} "
        f"vr={best_row['volume_ratio_threshold']:.2f} reg={best_row['use_regime_filter']} "
        f"vt={best_row['vol_target_annual']:.2f}{extra}"
    )
    print(f'  测试期 {test_start} ~ {test_end} 同参数 Sharpe: {oos_sh:.4f}')
    if oos:
        print(
            f'  测试期年化%: {float(oos.get("annual_return", 0.0)):+.2f}  '
            f'回撤%: {float(oos.get("max_drawdown", 0.0)):.2f}  '
            f'买卖笔数: {int(oos.get("trade_count", 0))}  '
            f'笔/年≈{float(oos.get("trades_per_year", 0.0)):.1f}  '
            f'胜率≈{float(oos.get("win_rate", 0.0)):.1f}%'
        )
    print(f'  测试期默认基准参数 Sharpe: {base_test_sh:.4f}（对照）')
    print('=' * 72)

    oos_path = args.out.replace('.csv', '_oos_best.csv') if args.out.endswith('.csv') else args.out + '_oos_best.csv'
    oos_row: Dict[str, Any] = {
        'period': 'test_oos',
        'train_start': str(train_start),
        'train_end': str(train_end),
        'test_start': str(test_start),
        'test_end': str(test_end),
        'train_sharpe_best': best_row['sharpe_ratio'],
        'test_sharpe_same_params': oos_sh,
        'test_sharpe_baseline': base_test_sh,
        'test_annual_return': float(oos.get('annual_return', 0.0)) if oos else 0.0,
        'test_max_drawdown': float(oos.get('max_drawdown', 0.0)) if oos else 0.0,
        'test_trade_count': int(oos.get('trade_count', 0)) if oos else 0,
        'sharpe_ratio_train': best_row['sharpe_ratio'],
    }
    for k in (
        'breakout_lookback',
        'volume_ratio_threshold',
        'use_regime_filter',
        'vol_target_annual',
        'trend_ma_period',
        'vol_ma_period',
        'hourly_ma_period',
        'hourly_breakout_bars',
        'use_hourly_first_bar_bullish',
    ):
        oos_row[f'best_{k}'] = best_row[k] if k in best_row else ''
    with open(oos_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(oos_row.keys()))
        w.writeheader()
        w.writerow(oos_row)
    print(f'[扫描] 样本外摘要已写入 {oos_path}', flush=True)

    json_path = args.out_strategy_json or bt.STRATEGY_PARAMS_JSON
    if not args.no_save_strategy_json:
        try:
            _save_trained_strategy_json(
                json_path,
                best_row,
                train_start,
                train_end,
                test_start,
                test_end,
                float(best_row['sharpe_ratio']),
                oos_sh,
            )
            print(f'[扫描] 已写入 {json_path}（运行 backtest.py 时将自动加载，可多次推理）', flush=True)
        except OSError as e:
            print(f'[扫描] 写入策略 JSON 失败: {e}', flush=True)


if __name__ == '__main__':
    main()
