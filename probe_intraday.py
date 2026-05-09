#!/usr/bin/env python3
"""探测 Longport 分钟 K 线最早可回溯日期，以及单次拉取上限。"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

from longport.openapi import AdjustType, Period

from longport_api import get_api_singleton

PERIODS = [
    ("1min",  Period.Min_1),
    ("5min",  Period.Min_5),
    ("15min", Period.Min_15),
    ("30min", Period.Min_30),
    ("60min", Period.Min_60),
]


def probe_period(api, symbol: str, label: str, period) -> None:
    print(f"\n=== {symbol} @ {label} ===")
    today = date.today()

    # 1) 试拉最近 5 个自然日 → 看单日 bar 数 + 最近一根的时间戳
    try:
        c = api.quote_ctx.history_candlesticks_by_date(
            symbol, period, AdjustType.ForwardAdjust,
            today - timedelta(days=7), today,
        )
        print(f"  最近 7 日: {len(c)} 根 bar")
        if c:
            print(f"    首根: {c[0].timestamp}  收盘={float(c[0].close):.2f}")
            print(f"    末根: {c[-1].timestamp}  收盘={float(c[-1].close):.2f}")
    except Exception as e:
        print(f"  [失败] 最近 7 日: {e}")
        return

    # 2) 二分查找最早可回溯日期（粗粒度按年）
    earliest_known = today
    for years_back in (1, 2, 3, 5, 7, 10):
        probe_start = today - timedelta(days=365 * years_back)
        probe_end = probe_start + timedelta(days=7)
        try:
            c = api.quote_ctx.history_candlesticks_by_date(
                symbol, period, AdjustType.ForwardAdjust,
                probe_start, probe_end,
            )
            n = len(c) if c else 0
            print(f"  {years_back}年前 ({probe_start} ~ {probe_end}): {n} 根 bar")
            if n > 0:
                earliest_known = probe_start
            else:
                print(f"    → 该时段无数据，可能已超出回溯上限")
                break
        except Exception as e:
            print(f"    [失败] {e}")
            break
    print(f"  → 实测可回溯到至少 {earliest_known}")

    # 3) 测试单次拉取上限：拉一段连续区间看返回多少根
    try:
        long_start = today - timedelta(days=60)
        c = api.quote_ctx.history_candlesticks_by_date(
            symbol, period, AdjustType.ForwardAdjust,
            long_start, today,
        )
        print(f"  最近 60 日单次拉取: {len(c)} 根 (估上限~1000)")
    except Exception as e:
        print(f"  [失败] 60 日拉取: {e}")


def main():
    symbols = sys.argv[1:] or ["AAPL.US"]
    api = get_api_singleton()
    for sym in symbols:
        for label, period in PERIODS:
            probe_period(api, sym, label, period)


if __name__ == "__main__":
    main()
