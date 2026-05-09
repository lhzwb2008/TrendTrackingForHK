#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Longport 分钟级 K 线接口（带本地 parquet 缓存）。

回溯能力：分钟数据上限约 2 年（早于此抛 301600 out of minute kline begin date）。
单次接口上限 1000 根 bar，本模块按 last_bar_date+1 推进多次合并拉取。

主要导出：
    fetch_intraday_bars(symbol, start, end, period_label='5min') -> pd.DataFrame
    PERIOD_MAP                                                   # 标签 → Period
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from longport.openapi import AdjustType, Period

from longport_api import get_api_singleton

logger = logging.getLogger(__name__)

PERIOD_MAP = {
    "1min":  Period.Min_1,
    "5min":  Period.Min_5,
    "15min": Period.Min_15,
    "30min": Period.Min_30,
    "60min": Period.Min_60,
}

PERIOD_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "60min": 60}

API_BAR_LIMIT = 1000


def _ts_to_utc_naive(ts) -> pd.Timestamp:
    """Longport 返回的分钟 K 时间戳是 tz-naive 的 **HKT (UTC+8)**。
    本函数统一转为 tz-naive UTC，与 daily 缓存一致；筛 RTH 时再转 ET。"""
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("Asia/Hong_Kong")
    return t.tz_convert("UTC").tz_localize(None)


def intraday_cache_path(symbol: str, period_label: str) -> str:
    root = os.getenv(
        "TREND_INTRADAY_CACHE_DIR",
        os.path.join(os.getcwd(), "data_cache", "intraday"),
    )
    sub = os.path.join(root, period_label)
    os.makedirs(sub, exist_ok=True)
    safe = symbol.replace(".", "_")
    return os.path.join(sub, f"{safe}.parquet")


def _load_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df.sort_index()
    except Exception as e:
        logger.warning(f"读取缓存失败 {path}: {e}")
        return None


def _save_cache(path: str, df: pd.DataFrame) -> None:
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.index.name = "ts"
    tmp = path + ".tmp"
    out.to_parquet(tmp)
    os.replace(tmp, path)


def _fetch_range_api(symbol: str, period: Period,
                     start_d: date, end_d: date) -> pd.DataFrame:
    """单次区间拉取（反向分页）。

    Longport 在 [start, end] 内若 bar 数超过 API_BAR_LIMIT，仅返回**最新**的
    1000 根。本函数检测到满载时，把 end 向前推到首根 bar 的前一天再次拉取，
    直到首根日期回到 start 或返回空。
    """
    api = get_api_singleton()
    pause = float(os.getenv("LONGPORT_REQUEST_PAUSE", "0.15"))
    rows = []
    cur_end = end_d
    safety = 500  # 防止死循环
    while cur_end >= start_d and safety > 0:
        safety -= 1
        try:
            candles = api.quote_ctx.history_candlesticks_by_date(
                symbol, period, AdjustType.ForwardAdjust, start_d, cur_end,
            )
        except Exception as e:
            msg = str(e)
            if "out of minute kline begin date" in msg or "301600" in msg:
                # 起点早于平台支持范围
                break
            raise
        if not candles:
            break
        for c in candles:
            rows.append({
                "ts": _ts_to_utc_naive(c.timestamp),
                "open": float(c.open), "high": float(c.high),
                "low": float(c.low), "close": float(c.close),
                "volume": int(c.volume), "turnover": float(c.turnover),
            })
        first_ts = candles[0].timestamp
        first_d = first_ts.date() if hasattr(first_ts, "date") else date.fromtimestamp(int(first_ts))
        # API 返回的 first_d 是 HKT 日期，转成 UTC 日期可能差 1 天；保守起见多退 1 天
        if len(candles) < API_BAR_LIMIT:
            break
        new_end = first_d - timedelta(days=1)
        if new_end >= cur_end:
            break  # 无推进
        cur_end = new_end
        time.sleep(pause)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_intraday_bars(
    symbol: str,
    start_date: date,
    end_date: date,
    period_label: str = "5min",
    log_cache: bool = True,
) -> pd.DataFrame:
    """拉取 [start, end] 的分钟级 K（含两端日期），命中缓存时不调 API。"""
    if period_label not in PERIOD_MAP:
        raise ValueError(f"不支持的周期 {period_label}，可选: {list(PERIOD_MAP)}")
    period = PERIOD_MAP[period_label]
    path = intraday_cache_path(symbol, period_label)
    cached = _load_cache(path)

    def slice_req(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        t0 = pd.Timestamp(start_date)
        t1 = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return df.loc[(df.index >= t0) & (df.index < t1)].copy()

    if cached is None or len(cached) == 0:
        df = _fetch_range_api(symbol, period, start_date, end_date)
        if len(df) > 0:
            _save_cache(path, df)
        if log_cache:
            print(f"[分钟] {symbol} ({period_label}) 全量拉取 {len(df)} 根", flush=True)
        return slice_req(df)

    cmin = cached.index.min().date()
    cmax = cached.index.max().date()
    merged = cached
    parts = []

    if start_date < cmin:
        older = _fetch_range_api(symbol, period, start_date, cmin - timedelta(days=1))
        if len(older) > 0:
            merged = pd.concat([older, merged]).sort_index()
            merged = merged[~merged.index.duplicated(keep="first")]
            parts.append(f"前补 {len(older)} 根")

    if end_date > cmax:
        ns = cmax + timedelta(days=1)
        if ns <= end_date:
            newer = _fetch_range_api(symbol, period, ns, end_date)
            if len(newer) > 0:
                merged = pd.concat([merged, newer]).sort_index()
                merged = merged[~merged.index.duplicated(keep="first")]
                parts.append(f"增量 {len(newer)} 根")

    if parts:
        _save_cache(path, merged)

    if log_cache:
        if parts:
            print(f"[分钟] {symbol} ({period_label}) 缓存命中, "
                  f"{', '.join(parts)} → 合计 {len(merged)} 根", flush=True)
        else:
            print(f"[分钟] {symbol} ({period_label}) 缓存命中 {len(merged)} 根",
                  flush=True)
    return slice_req(merged)


# ---------------- 时区与 RTH 工具 ----------------

ET_TZ = "America/New_York"

def to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """tz-naive UTC → tz-aware ET（自动处理 DST）。"""
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(ET_TZ)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留美股 RTH (09:30-16:00 ET) 的 bar。"""
    if df is None or len(df) == 0:
        return df
    et = to_et(df.index)
    secs = et.hour * 3600 + et.minute * 60 + et.second
    rth_open = 9 * 3600 + 30 * 60     # 09:30:00
    rth_close = 16 * 3600              # 16:00:00
    mask = (secs >= rth_open) & (secs < rth_close)
    return df.loc[mask].copy()


def parse_decision_time(s: str) -> tuple[int, int]:
    """'15:50' → (15, 50)"""
    h, m = s.split(":")
    return int(h), int(m)


# ---------------- CLI ----------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python intraday_api.py SYMBOL [period=5min] [start=YYYY-MM-DD] [end=YYYY-MM-DD]")
        sys.exit(1)
    sym = sys.argv[1]
    pl = sys.argv[2] if len(sys.argv) >= 3 else "5min"
    s = (datetime.strptime(sys.argv[3], "%Y-%m-%d").date()
         if len(sys.argv) >= 4 else date.today() - timedelta(days=30))
    e = (datetime.strptime(sys.argv[4], "%Y-%m-%d").date()
         if len(sys.argv) >= 5 else date.today())
    df = fetch_intraday_bars(sym, s, e, pl)
    print(f"\n{sym} ({pl})  {s}~{e}: {len(df)} 根（全量）")
    if len(df):
        print(df.head(3))
        print("...")
        print(df.tail(3))
        rth = filter_rth(df)
        print(f"\nRTH only: {len(rth)} 根")
        print(rth.tail(3))
