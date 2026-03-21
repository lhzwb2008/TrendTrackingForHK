#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略验证：Walk-forward（样本内选参 → 样本外评估）、交易成本与大盘 regime 消融。
无 CSV / API 时使用合成市场（前半熊后半牛）以保证本脚本可独立跑通。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trend_breakout_v2 import (
    BacktestEngine,
    HistoricalDataManager,
    load_hsi_data,
)

# 与 optimize_params 一致，略增两组仓位相关以便观察维度
PARAM_GRID: List[Dict[str, Any]] = [
    {'stop_loss_pct': 0.30, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': 'SL30_L120_V1.5'},
    {'stop_loss_pct': 0.35, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': 'SL35_L120_V1.5'},
    {'stop_loss_pct': 0.40, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': 'SL40_L120_V1.5'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 60, 'volume_ratio_threshold': 1.5, 'name': 'SL25_L60_V1.5'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 90, 'volume_ratio_threshold': 1.5, 'name': 'SL25_L90_V1.5'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 180, 'volume_ratio_threshold': 1.5, 'name': 'SL25_L180_V1.5'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 2.0, 'name': 'SL25_L120_V2'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 2.5, 'name': 'SL25_L120_V2.5'},
    {'stop_loss_pct': 0.35, 'breakout_lookback': 90, 'volume_ratio_threshold': 2.0, 'name': 'SL35_L90_V2'},
    {'stop_loss_pct': 0.40, 'breakout_lookback': 60, 'volume_ratio_threshold': 2.0, 'name': '组合B'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'max_positions': 5, 'name': 'L120_最多5只'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'position_size_pct': 0.10, 'name': 'L120_仓位10%'},
]

BASE_CONFIG: Dict[str, Any] = {
    'initial_capital': 100000,
    'max_positions': 8,
    'position_size_pct': 0.15,
}

MIN_TRADES_FOR_SELECTION = 6
# 港股双边约 0.2%～0.3% 量级常见，这里取偏保守单边 0.15%
DEFAULT_ONE_WAY_COST = 0.0015

# Walk-forward 用较小网格以控制耗时（全样本回测仍较慢）
WF_PARAM_GRID: List[Dict[str, Any]] = [
    {'stop_loss_pct': 0.35, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': 'SL35_L120_V1.5'},
    {'stop_loss_pct': 0.40, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': 'SL40_L120_V1.5'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 60, 'volume_ratio_threshold': 1.5, 'name': 'SL25_L60_V1.5'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 2.0, 'name': 'SL25_L120_V2'},
    {'stop_loss_pct': 0.40, 'breakout_lookback': 60, 'volume_ratio_threshold': 2.0, 'name': '组合B'},
    {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'max_positions': 5, 'name': 'L120_最多5只'},
]


def _selection_score(result: Dict[str, Any]) -> float:
    """样本内选参：交易过少时仍可用 Sharpe 比较，但加惩罚避免空交易占优。"""
    sr = float(result.get('sharpe_ratio') or 0.0)
    if result.get('trade_count', 0) < MIN_TRADES_FOR_SELECTION:
        return sr - 5.0
    return sr


def _positive_year_ratio(yearly: Dict[int, Any]) -> str:
    if not yearly:
        return '0/0'
    pos = sum(1 for _, d in yearly.items() if d.get('strategy', 0) > 0)
    return f"{pos}/{len(yearly)}"


def _run_engine(
    dm: HistoricalDataManager,
    start: date,
    end: date,
    hsi: Optional[pd.DataFrame],
    param_row: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {**BASE_CONFIG, **param_row}
    for k in ('name',):
        cfg.pop(k, None)
    if extra:
        cfg.update(extra)
    eng = BacktestEngine(dm, cfg)
    return eng.run(start, end, hsi, verbose=False)


def _best_params_on_train(
    dm: HistoricalDataManager,
    train_start: date,
    train_end: date,
    hsi: Optional[pd.DataFrame],
    extra: Optional[Dict[str, Any]] = None,
    grid: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], float]:
    best_row: Optional[Dict[str, Any]] = None
    best_score = float('-inf')
    use_grid = grid if grid is not None else PARAM_GRID
    for row in use_grid:
        r = _run_engine(dm, train_start, train_end, hsi, row, extra)
        s = _selection_score(r)
        if best_row is None or s > best_score:
            best_score = s
            best_row = row
    if best_row is None:
        raise ValueError('参数网格为空')
    return best_row, best_score


def build_synthetic_dm(
    n_days: int = 680,
    n_stocks: int = 12,
    seed: int = 42,
) -> Tuple[HistoricalDataManager, pd.DataFrame, date, date]:
    """
    合成市场：恒指前半段偏熊、后半段偏牛；个股收益与恒指相关 + 特质波动。
    用于在无 API 时验证流程与机制（regime / 成本）方向性影响。
    """
    rng = np.random.default_rng(seed)
    end = date(2024, 12, 31)
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=n_days)

    bench_ret = np.full(n_days, 0.0002)
    bench_ret[: n_days // 2] = -0.00022
    bench_ret += rng.normal(0, 0.009, n_days)
    bench_close = 20000.0 * np.cumprod(1.0 + bench_ret)

    hsi_df = pd.DataFrame(
        {
            'open': bench_close * (1 + rng.normal(0, 0.002, n_days)),
            'high': bench_close * (1 + np.abs(rng.normal(0, 0.006, n_days))),
            'low': bench_close * (1 - np.abs(rng.normal(0, 0.006, n_days))),
            'close': bench_close,
            'volume': (2e9 / bench_close).astype(np.int64),
        },
        index=dates,
    )

    dm = HistoricalDataManager()
    dm.register_symbol_frame('HSI.HK', hsi_df)

    for i in range(n_stocks):
        sym = f"{10001 + i:05d}.HK"
        beta = float(rng.uniform(0.55, 1.15))
        noise = rng.normal(0, 0.014, n_days)
        # 偶发放量上冲，制造可交易突破形态
        spike = rng.random(n_days) < 0.03
        noise = noise + spike * rng.uniform(0.01, 0.04, n_days)
        r = beta * bench_ret + noise
        close = 30.0 * np.cumprod(1.0 + r)
        vol = np.maximum(300000, (12_000_000.0 / close).astype(np.float64))
        vol = vol * (1.0 + spike.astype(float) * rng.uniform(0.5, 2.0, n_days))
        vol = vol.astype(np.int64)
        odf = pd.DataFrame(
            {
                'open': close * (1 + rng.normal(0, 0.003, n_days)),
                'high': np.maximum(close * 1.01, close * (1 + np.abs(rng.normal(0, 0.01, n_days)))),
                'low': np.minimum(close * 0.99, close * (1 - np.abs(rng.normal(0, 0.01, n_days)))),
                'close': close,
                'volume': vol,
            },
            index=dates,
        )
        dm.register_symbol_frame(sym, odf)

    # 约 230 根 K 后恒指可有 MA200；与最大 lookback(180) 兼顾
    start_bt = dates[230].date()
    end_bt = dates[-1].date()
    return dm, hsi_df, start_bt, end_bt


def try_prepare_live_dm(
    symbols_cap: int = 200,
    data_start: date = date(2019, 6, 1),
    end_d: Optional[date] = None,
) -> Optional[Tuple[HistoricalDataManager, pd.DataFrame, str]]:
    csv_path = 'hk_all_stocks.csv'
    if not os.path.exists(csv_path):
        return None
    end_d = end_d or date.today()
    try:
        from hk_stock_api import HKStockAPI  # noqa: F401
    except Exception:
        return None

    df = pd.read_csv(csv_path)
    df['code'] = df['代码'].astype(str).str.zfill(5)
    df = df[~df['code'].str.match(r'^07[0-9]{3}$')]
    df = df[~df['code'].str.match(r'^028[0-9]{2}$')]
    symbols = [f"{c}.HK" for c in df['code'].tolist()[:symbols_cap]]

    dm = HistoricalDataManager()
    n = dm.load_stock_data(symbols, data_start, end_d)
    if n < 10:
        return None
    dm.load_stock_data(['HSI.HK'], data_start, end_d)
    hsi = load_hsi_data(data_start, end_d)
    if hsi is None or len(hsi) < 300:
        return None
    return dm, hsi, csv_path


def run_walk_forward(
    dm: HistoricalDataManager,
    hsi: pd.DataFrame,
    folds: List[Tuple[date, date, date, date]],
    label: str,
) -> None:
    print(f"\n{'='*60}\nWalk-forward: {label}\n{'='*60}")
    baseline_row = next(p for p in PARAM_GRID if p.get('name') == '组合B')

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        best_row, tr_score = _best_params_on_train(
            dm, tr_s, tr_e, hsi, None, grid=WF_PARAM_GRID
        )
        r_is = _run_engine(dm, tr_s, tr_e, hsi, best_row, None)
        r_oos = _run_engine(dm, te_s, te_e, hsi, best_row, None)
        r_base_oos = _run_engine(dm, te_s, te_e, hsi, baseline_row, None)

        print(f"\n--- Fold {fi} 训练 {tr_s} ~ {tr_e} | 测试 {te_s} ~ {te_e} ---")
        print(
            f"  样本内({best_row['name']}): Sharpe={r_is.get('sharpe_ratio', 0):.3f} "
            f"收益={r_is['total_return']:+.1f}% 交易{r_is.get('trade_count', 0)}笔 "
            f"盈利年{_positive_year_ratio(r_is.get('yearly_returns', {}))}"
        )
        print(
            f"  样本外(同一参数):   Sharpe={r_oos.get('sharpe_ratio', 0):.3f} "
            f"收益={r_oos['total_return']:+.1f}% 交易{r_oos.get('trade_count', 0)}笔 "
            f"盈利年{_positive_year_ratio(r_oos.get('yearly_returns', {}))}"
        )
        print(
            f"  样本外(固定组合B):  Sharpe={r_base_oos.get('sharpe_ratio', 0):.3f} "
            f"收益={r_base_oos['total_return']:+.1f}% 交易{r_base_oos.get('trade_count', 0)}笔 "
            f"盈利年{_positive_year_ratio(r_base_oos.get('yearly_returns', {}))}"
        )
        print(
            f"  [选参] 训练期排序分={tr_score:.3f}（交易<{MIN_TRADES_FOR_SELECTION}笔时 Sharpe-5）"
            f" 原始训练Sharpe={r_is.get('sharpe_ratio', 0):.3f}"
        )


def run_ablation(
    dm: HistoricalDataManager,
    hsi: pd.DataFrame,
    start: date,
    end: date,
    label: str,
) -> None:
    print(f"\n{'='*60}\n消融实验（全样本）: {label}\n{'='*60}")
    baseline = next(p for p in PARAM_GRID if p.get('name') == '组合B')
    scenarios = [
        ('基准:组合B', baseline, {}),
        ('+恒指>MA200过滤', baseline, {'use_regime_filter': True, 'regime_benchmark': 'HSI.HK', 'regime_ma_days': 200}),
        (f'+单边成本{DEFAULT_ONE_WAY_COST*100:.2f}%', baseline, {'one_way_cost_rate': DEFAULT_ONE_WAY_COST}),
        (
            '+过滤+成本',
            baseline,
            {
                'use_regime_filter': True,
                'regime_benchmark': 'HSI.HK',
                'regime_ma_days': 200,
                'one_way_cost_rate': DEFAULT_ONE_WAY_COST,
            },
        ),
    ]
    print(f"{'场景':<22} {'Sharpe':>8} {'总收益%':>10} {'最大回撤%':>10} {'盈利年':>8}")
    print("-" * 62)
    for name, prow, extra in scenarios:
        r = _run_engine(dm, start, end, hsi, prow, extra)
        print(
            f"{name:<22} {r.get('sharpe_ratio', 0):8.3f} {r['total_return']:>+9.1f}% "
            f"{r['max_drawdown']:>9.1f}% {_positive_year_ratio(r.get('yearly_returns', {})):>8}"
        )


def main() -> None:
    live = try_prepare_live_dm()
    if live is not None:
        dm, hsi, csv_path = live
        print(f"使用真实数据 ({csv_path} + API)")
        folds = [
            (date(2020, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2024, 12, 31)),
            (date(2020, 1, 1), date(2023, 12, 31), date(2024, 1, 1), date.today()),
        ]
        run_walk_forward(dm, hsi, folds, label='真实行情')
        run_ablation(dm, hsi, date(2020, 1, 1), date.today(), label='真实行情')
        return

    print("未检测到可用真实数据，使用合成市场（熊→牛分段）完成同等验证流程。")
    dm, hsi_df, start_bt, end_bt = build_synthetic_dm()
    all_dates = [d for d in dm.get_all_trading_dates() if start_bt <= d <= end_bt]
    n = len(all_dates)
    mid, q3 = max(2, n // 2), max(3, n * 3 // 4)
    folds = [
        (all_dates[0], all_dates[mid - 1], all_dates[mid], all_dates[q3 - 1]),
        (
            all_dates[0],
            all_dates[max(1, (mid + q3) // 2) - 1],
            all_dates[max(1, (mid + q3) // 2)],
            end_bt,
        ),
    ]
    run_walk_forward(dm, hsi_df, folds, label='合成行情')
    run_ablation(dm, hsi_df, start_bt, end_bt, label='合成行情')


if __name__ == '__main__':
    main()
    sys.exit(0)
