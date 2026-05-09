#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 DECISION_TIME_ET，对比不同决策时点的 INTRADAY 回测效果。

数据加载与 backtest.py 共用缓存，只重复跑回测主循环。
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest import (
    Config, INTRA_START, BACKTEST_END,
    build_intraday_enhanced_panel, build_panel,
    load_all_data, load_all_intraday, run_backtest, summarize,
)
from longport_api import fetch_daily_bars
from nas100_universe import get_universe


# 候选决策时点；5min K 边界对齐
DECISION_TIMES: List[str] = [
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "15:30",
    "15:50",
    "15:55",
]


def main():
    end = date.today() if BACKTEST_END == "today" else date.fromisoformat(BACKTEST_END)
    start = date.fromisoformat(INTRA_START)

    base_cfg = Config(start=start, end=end, mode="intraday", verbose_trades=False)

    print(f"[加载] 日线 {start} ~ {end}")
    syms = get_universe()
    data = load_all_data(syms, start, end)

    print(f"[加载] 分钟 {base_cfg.intraday_period}")
    intraday = load_all_intraday(list(data.keys()), start, end, base_cfg.intraday_period)

    spy = fetch_daily_bars("SPY.US", start - timedelta(days=400), end, log_cache=False)
    regime = (spy["close"] > spy["close"].rolling(200).mean())

    qqq = fetch_daily_bars("QQQ.US", start - timedelta(days=10), end, log_cache=False)
    qqq_close = qqq.loc[
        (qqq.index >= pd.Timestamp(start)) & (qqq.index <= pd.Timestamp(end)), "close"
    ]

    print("\n" + "=" * 96)
    print(f"  DECISION_TIME 扫描（INTRADAY 模式，区间 {start} ~ {end}）")
    print("=" * 96)
    print(f"  {'决策时点':<10}{'收益%':>10}{'CAGR%':>10}{'Sharpe':>9}"
          f"{'最大回撤%':>12}{'Calmar':>9}{'交易笔数':>10}{'成本%':>9}")
    print("-" * 96)

    rows = []
    for t in DECISION_TIMES:
        cfg = replace(base_cfg, decision_time_et=t)
        enhanced = build_intraday_enhanced_panel(
            data, intraday, cfg.intraday_period, t,
        )
        panel = build_panel(enhanced)
        result = run_backtest(panel, cfg, regime_series=regime)
        s = summarize(result, cfg, qqq_close)

        ret = float(s["累计收益"].split("%")[0])
        cagr = float(s["年化收益(CAGR)"].rstrip("%"))
        sh = float(s["Sharpe"])
        mdd = float(s["最大回撤"].rstrip("%"))
        cal = float(s["Calmar"])
        n = int(s["总交易笔数"])
        cost_pct = float(s["总交易成本"].split("(")[-1].rstrip("%)"))

        rows.append((t, ret, cagr, sh, mdd, cal, n, cost_pct))
        print(f"  {t:<10}{ret:>+10.2f}{cagr:>+10.2f}{sh:>+9.2f}"
              f"{mdd:>+12.2f}{cal:>+9.2f}{n:>10d}{cost_pct:>+9.2f}")

    print("=" * 96)

    # 突出最优
    best_sharpe = max(rows, key=lambda r: r[3])
    best_cagr = max(rows, key=lambda r: r[2])
    best_calmar = max(rows, key=lambda r: r[5])
    print(f"\n  Sharpe 最优 → {best_sharpe[0]}  (Sharpe={best_sharpe[3]:.2f}, "
          f"CAGR={best_sharpe[2]:+.2f}%)")
    print(f"  CAGR   最优 → {best_cagr[0]}  (CAGR={best_cagr[2]:+.2f}%, "
          f"Sharpe={best_cagr[3]:.2f})")
    print(f"  Calmar 最优 → {best_calmar[0]}  (Calmar={best_calmar[5]:.2f})")


if __name__ == "__main__":
    main()
