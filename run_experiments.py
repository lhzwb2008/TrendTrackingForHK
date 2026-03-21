#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照实验：同一批行情只拉取一次，通过 symbols_subset 切换股票子集与策略参数。

用法:
  .venv/bin/python run_experiments.py
  .venv/bin/python run_experiments.py --max-symbols 120   # 网络不稳时减少拉取数量
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import Any, Dict, List, Set

import pandas as pd

from trend_breakout_v2 import BacktestEngine, HistoricalDataManager, load_hsi_data

BASE_B = {
    'initial_capital': 100000,
    'max_positions': 8,
    'position_size_pct': 0.15,
    'stop_loss_pct': 0.40,
    'breakout_lookback': 60,
    'volume_ratio_threshold': 2.0,
}


def _load_universe_df() -> pd.DataFrame:
    path = (
        'universe_hk_small_quality.csv'
        if os.path.exists('universe_hk_small_quality.csv')
        else 'hk_all_stocks.csv'
    )
    if not os.path.exists(path):
        print('未找到 universe_hk_small_quality.csv 或 hk_all_stocks.csv', file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df['code'] = df['代码'].astype(str).str.zfill(5)
    df = df[~df['code'].str.match(r'^07[0-9]{3}$')]
    df = df[~df['code'].str.match(r'^028[0-9]{2}$')]
    return df, path


def _codes_to_symbols(codes: List[str]) -> List[str]:
    return [f'{c}.HK' for c in codes]


def main() -> None:
    ap = argparse.ArgumentParser(description='多组回测对照实验')
    ap.add_argument(
        '--max-symbols',
        type=int,
        default=0,
        help='最多加载多少只港股（0=不限制，与 CSV 一致）',
    )
    args = ap.parse_args()

    df, path = _load_universe_df()
    if args.max_symbols:
        df = df.iloc[: args.max_symbols].copy()
    # 全量代码（保序）
    all_codes = list(dict.fromkeys(df['code'].tolist()))
    all_syms = _codes_to_symbols(all_codes)

    # 子集定义
    mcap_ordered_codes = all_codes  # CSV 已是 build_universe 按市值升序
    subset_mcap_200: Set[str] = set(_codes_to_symbols(mcap_ordered_codes[:200]))
    subset_mcap_all: Set[str] = set(all_syms)

    df_turn = df.copy()
    if 'last_turnover' in df_turn.columns:
        df_turn = df_turn.sort_values('last_turnover', ascending=False, na_position='last')
        subset_liq_200 = set(_codes_to_symbols(df_turn['code'].tolist()[:200]))
    else:
        subset_liq_200 = subset_mcap_200

    load_list = list(dict.fromkeys(all_syms + ['HSI.HK']))

    print('=' * 72)
    print('实验批次：单次加载行情，多组子集 + 参数')
    cap = f'（--max-symbols 截断至 {args.max_symbols}）' if args.max_symbols else ''
    print(f'股票池文件: {path} | 本批加载 {len(all_syms)} 只 + HSI.HK{cap}')
    print('=' * 72)

    dm = HistoricalDataManager()
    data_start = date(2019, 6, 1)
    end_date = date.today()
    print(f'\n加载日线 ({data_start} ~ {end_date})，共 {len(load_list)} 个代码 …')
    n = dm.load_stock_data(load_list, data_start, end_date)
    print(f'成功加载 {n}/{len(load_list)}')

    hsi = load_hsi_data(data_start, end_date)
    if hsi is None:
        print('基准加载失败，实验继续但无超额对比', file=sys.stderr)

    bt0, bt1 = date(2020, 1, 1), end_date

    experiments: List[Tuple[str, Dict[str, Any]]] = [
        ('E1_小市值前200_组合B', {**BASE_B, 'symbols_subset': subset_mcap_200}),
        ('E2_质量池全量_组合B', {**BASE_B, 'symbols_subset': subset_mcap_all}),
        ('E3_按最近成交额Top200_组合B', {**BASE_B, 'symbols_subset': subset_liq_200}),
        (
            'E4_小市值200+恒指MA200过滤',
            {
                **BASE_B,
                'symbols_subset': subset_mcap_200,
                'use_regime_filter': True,
                'regime_benchmark': 'HSI.HK',
                'regime_ma_days': 200,
            },
        ),
        (
            'E5_小市值200_紧止损90日',
            {
                **BASE_B,
                'symbols_subset': subset_mcap_200,
                'stop_loss_pct': 0.25,
                'breakout_lookback': 90,
                'volume_ratio_threshold': 1.5,
            },
        ),
        (
            'E6_小市值200_少仓位',
            {
                **BASE_B,
                'symbols_subset': subset_mcap_200,
                'max_positions': 5,
                'position_size_pct': 0.12,
            },
        ),
    ]

    rows = []
    print(f"\n{'实验':<32} {'总收益%':>9} {'年化%':>8} {'回撤%':>8} {'Sharpe':>7} {'超额%':>9} {'交易':>6}")
    print('-' * 92)

    for name, cfg in experiments:
        eng = BacktestEngine(dm, cfg)
        r = eng.run(bt0, bt1, hsi, verbose=False)
        ex = r.get('excess_return')
        exs = f'{ex:+.1f}' if ex is not None else '  n/a'
        print(
            f'{name:<32} {r["total_return"]:>+8.1f}% {r["annual_return"]:>+7.1f}% '
            f'{r["max_drawdown"]:>7.1f}% {r.get("sharpe_ratio", 0):>7.3f} {exs:>9} {r.get("trade_count", 0):>6}'
        )
        rows.append({'name': name, **{k: r[k] for k in ('total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'trade_count')}, 'excess_return': ex})

    out = 'experiments_last_run.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('-' * 92)
    print(f'已保存摘要: {out}')


if __name__ == '__main__':
    main()
