#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""60 分钟 K 线本地缓存（Longport Period.Min_60），索引为 datetime。"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from daily_cache import Progress, normalize_df_index


def hourly_cache_path(symbol: str) -> str:
    root = os.getenv('TREND_HOURLY_CACHE_DIR', os.path.join(os.getcwd(), 'data_cache', 'hourly_60'))
    os.makedirs(root, exist_ok=True)
    safe = symbol.replace('.', '_')
    return os.path.join(root, f'{safe}.csv')


def _load_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return None
    try:
        d = pd.read_csv(path, index_col=0, parse_dates=True)
        return normalize_df_index(d)
    except Exception:
        return None


def _save_cache(path: str, df: pd.DataFrame) -> None:
    out = normalize_df_index(df)
    tmp = path + '.tmp'
    out.to_csv(tmp, date_format='%Y-%m-%d %H:%M:%S')
    os.replace(tmp, path)


def _idx_date(ts) -> date:
    return pd.Timestamp(ts).date()


def merge_hourly_cache(
    symbol: str,
    start_date: date,
    end_date: date,
    fetch_range: Callable[[date, date], pd.DataFrame],
    log_cache: bool = True,
    progress: Progress = None,
) -> Tuple[pd.DataFrame, str]:
    """合并本地缓存与增量请求，返回 [start_date, end_date] 内所有小时 K。"""
    path = hourly_cache_path(symbol)
    cached = _load_cache(path)

    def slice_req(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        out = normalize_df_index(df)
        idx = out.index
        dv = np.array([pd.Timestamp(x).date() for x in idx])
        sd, ed = start_date, end_date
        mask = (dv >= sd) & (dv <= ed)
        return out.loc[mask].copy()

    if cached is None or len(cached) == 0:
        df = fetch_range(start_date, end_date)
        if df is not None and len(df) > 0:
            _save_cache(path, df)
        n = len(df) if df is not None else 0
        msg = f'全量拉取 {n} 根(60m)'
        if log_cache:
            _print_line(symbol, msg, progress)
        return slice_req(df if df is not None else pd.DataFrame()), msg

    cmin = _idx_date(cached.index.min())
    cmax = _idx_date(cached.index.max())
    merged = cached.copy()
    parts = []

    if start_date < cmin:
        older = fetch_range(start_date, cmin - timedelta(days=1))
        if older is not None and len(older) > 0:
            merged = pd.concat([older, merged]).sort_index()
            merged = merged[~merged.index.duplicated(keep='first')]
            parts.append(f'向前补 {len(older)} 根')

    if end_date > cmax:
        ns = cmax + timedelta(days=1)
        if ns <= end_date:
            newer = fetch_range(ns, end_date)
            if newer is not None and len(newer) > 0:
                merged = pd.concat([merged, newer]).sort_index()
                merged = merged[~merged.index.duplicated(keep='first')]
                parts.append(f'增量 {len(newer)} 根')

    if parts:
        _save_cache(path, merged)

    n = len(merged)
    if parts:
        msg = '缓存命中，' + '，'.join(parts) + f' → 合计 {n} 根(60m)'
    else:
        msg = f'缓存命中 {n} 根(60m)（未请求网络）'

    if log_cache:
        _print_line(symbol, msg, progress)
    return slice_req(merged), msg


def _print_line(symbol: str, msg: str, progress: Progress) -> None:
    pre = ''
    if progress:
        pre = f'{progress[0]}/{progress[1]} '
    print(f'[60m] {pre}{symbol} — {msg}', flush=True)
