#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 / 验证两阶段回测（walk-forward）

流程：
  1. 加载全期数据（日线 + 分钟）一次
  2. 训练期：在小参数网格上跑回测，按 Sharpe 排序选最优参数
  3. 验证期：用最优参数 vs 默认参数各跑一次，对比样本外表现

切分：
  TRAIN: TRAIN_START ~ TRAIN_END   （选参数）
  VALID: VALID_START ~ VALID_END   （评估泛化）

运行：
  python walkforward.py
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import replace
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd

from backtest import (
    Config, build_panel, load_all_data, load_all_intraday,
    merge_daily_with_intraday, run_backtest, summarize,
    summarize_intraday_per_day,
)
from longport_api import fetch_daily_bars
from nas100_universe import get_universe

# ============================================================================
#                              切分与网格
# ============================================================================

TRAIN_START = "2024-05-08"
TRAIN_END   = "2025-09-30"     # ~17 个月

VALID_START = "2025-10-01"
VALID_END   = "2026-05-07"     # ~7 个月

# 参数网格（保持小一点防过拟合）
GRID: Dict[str, List] = {
    "mom_weight":      [0.5, 0.7, 0.8],   # 动量/反转权重
    "hysteresis_mult": [3.0, 4.0, 5.0],   # 信号退出滞后
    "k_long":          [6, 8, 10],        # 持仓数
}
# 共 3*3*3 = 27 组，缓存命中下整轮 ~5-10 min

TOP_N_REPORT = 5  # 训练期 Top-N 参数组也打印


# ============================================================================

def _parse(s: str) -> date:
    return date.fromisoformat(s)


def _metrics(summary: dict) -> dict:
    """从 summarize 输出抽取数值型指标。"""
    return {
        "CAGR":   float(summary["年化收益(CAGR)"].rstrip("%")),
        "Sharpe": float(summary["Sharpe"]),
        "MDD":    float(summary["最大回撤"].rstrip("%")),
        "Calmar": float(summary["Calmar"]),
        "Trades": int(summary["总交易笔数"]),
        "WinRate": float(summary["胜率"].rstrip("%")),
    }


def _print_row(prefix: str, m: dict):
    print(f"  {prefix:<40s} "
          f"Sharpe={m['Sharpe']:>5.2f}  "
          f"CAGR={m['CAGR']:>+7.2f}%  "
          f"MDD={m['MDD']:>+7.2f}%  "
          f"Calmar={m['Calmar']:>5.2f}  "
          f"Trades={m['Trades']:>4d}  "
          f"Win={m['WinRate']:>4.1f}%")


def main():
    full_start = _parse(TRAIN_START)
    full_end   = _parse(VALID_END)

    base_cfg = Config(start=full_start, end=full_end, verbose_trades=False)

    print("=" * 70)
    print("  Walk-Forward 训练/验证")
    print("=" * 70)
    print(f"  TRAIN: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  VALID: {VALID_START} ~ {VALID_END}")
    print(f"  网格规模: {len(list(itertools.product(*GRID.values())))} 组")
    print(f"  分钟周期: {base_cfg.intraday_period}  决策时点: {base_cfg.decision_time_et} ET")

    # ---------- 1) 加载全期数据（仅一次） ----------
    syms = get_universe()
    print(f"\n[数据] 加载 {len(syms)} 只 NAS100 日线...")
    data = load_all_data(syms, base_cfg.start, base_cfg.end)

    print(f"\n[数据] 加载分钟级数据 ({base_cfg.intraday_period})...")
    intraday = load_all_intraday(list(data.keys()), base_cfg.start, base_cfg.end,
                                  base_cfg.intraday_period)

    print(f"\n[数据] 聚合分钟数据到决策时点截面...")
    enhanced: Dict[str, pd.DataFrame] = {}
    for sym, dfd in data.items():
        intra_df = intraday.get(sym, pd.DataFrame())
        intra_summary = summarize_intraday_per_day(
            intra_df, base_cfg.intraday_period, base_cfg.decision_time_et,
        )
        enhanced[sym] = merge_daily_with_intraday(dfd, intra_summary)
    panel = build_panel(enhanced)

    spy_df = fetch_daily_bars("SPY.US", base_cfg.start - timedelta(days=400),
                              base_cfg.end, log_cache=False)
    regime_series = (spy_df["close"] > spy_df["close"].rolling(200).mean())

    # ---------- 2) 训练期网格搜索 ----------
    print("\n" + "=" * 70)
    print("  训练期网格搜索（按 Sharpe 排序）")
    print("=" * 70)

    train_cfg_proto = replace(base_cfg,
                               start=_parse(TRAIN_START), end=_parse(TRAIN_END))
    keys = list(GRID.keys())
    grid_runs = []

    for vals in itertools.product(*[GRID[k] for k in keys]):
        params = dict(zip(keys, vals))
        cfg = replace(train_cfg_proto, **params)
        result = run_backtest(panel, cfg, regime_series=regime_series)
        summ = summarize(result, cfg)
        m = _metrics(summ)
        grid_runs.append((params, m))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        _print_row(label, m)

    # 默认参数（DEFAULT）也跑一次训练期作为对照
    default_train_cfg = train_cfg_proto
    result = run_backtest(panel, default_train_cfg, regime_series=regime_series)
    default_train_m = _metrics(summarize(result, default_train_cfg))
    print(f"\n  [DEFAULT 训练期]")
    _print_row("default", default_train_m)

    # 排序、打印 Top-N
    grid_runs.sort(key=lambda x: x[1]["Sharpe"], reverse=True)
    print("\n----- 训练期 Top {} -----".format(TOP_N_REPORT))
    for params, m in grid_runs[:TOP_N_REPORT]:
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        _print_row(label, m)

    best_params, best_m = grid_runs[0]
    print(f"\n  → 训练期最优: {best_params}")

    # ---------- 3) 验证期评估 ----------
    print("\n" + "=" * 70)
    print("  验证期（样本外）")
    print("=" * 70)

    valid_cfg_default = replace(base_cfg,
                                start=_parse(VALID_START), end=_parse(VALID_END))
    valid_cfg_best = replace(valid_cfg_default, **best_params)

    res_default = run_backtest(panel, valid_cfg_default, regime_series=regime_series)
    res_best    = run_backtest(panel, valid_cfg_best,    regime_series=regime_series)

    valid_default_m = _metrics(summarize(res_default, valid_cfg_default))
    valid_best_m    = _metrics(summarize(res_best,    valid_cfg_best))

    print()
    _print_row("DEFAULT 参数 (验证期)", valid_default_m)
    _print_row("训练期最优 → 验证期",    valid_best_m)

    # ---------- 4) 总结表 ----------
    print("\n" + "=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  训练期 (DEFAULT):  Sharpe={default_train_m['Sharpe']:.2f}  "
          f"CAGR={default_train_m['CAGR']:+.2f}%  MDD={default_train_m['MDD']:+.2f}%")
    print(f"  训练期 (BEST):     Sharpe={best_m['Sharpe']:.2f}  "
          f"CAGR={best_m['CAGR']:+.2f}%  MDD={best_m['MDD']:+.2f}%   "
          f"params={best_params}")
    print(f"  验证期 (DEFAULT):  Sharpe={valid_default_m['Sharpe']:.2f}  "
          f"CAGR={valid_default_m['CAGR']:+.2f}%  MDD={valid_default_m['MDD']:+.2f}%")
    print(f"  验证期 (BEST):     Sharpe={valid_best_m['Sharpe']:.2f}  "
          f"CAGR={valid_best_m['CAGR']:+.2f}%  MDD={valid_best_m['MDD']:+.2f}%")

    # 诊断
    print()
    deg_best   = best_m["Sharpe"] - valid_best_m["Sharpe"]
    deg_def    = default_train_m["Sharpe"] - valid_default_m["Sharpe"]

    if deg_best > 0.5:
        print(f"  ⚠ BEST 参数 Sharpe 训练→验证 衰减 {deg_best:+.2f}，疑似过拟合训练期")
    elif deg_best > 0.2:
        print(f"  ⓘ BEST 参数 Sharpe 训练→验证 衰减 {deg_best:+.2f}，泛化尚可但需谨慎")
    elif deg_best > -0.5:
        print(f"  ✓ BEST 参数 Sharpe 训练→验证 衰减仅 {deg_best:+.2f}，参数较稳定")
    else:
        print(f"  ⚠ BEST 参数 Sharpe 训练→验证 提升 {-deg_best:+.2f}，"
              f"两段市场状态差异大；验证期可能是 lucky regime，"
              f"不要直接外推 CAGR")

    # 默认参数若在验证期碾压 BEST，提示：网格选参没意义，行情更重要
    if valid_default_m["Sharpe"] > valid_best_m["Sharpe"] + 0.3:
        print(f"  ⚠ DEFAULT 参数验证期 Sharpe ({valid_default_m['Sharpe']:.2f}) "
              f"显著高于 BEST ({valid_best_m['Sharpe']:.2f})，"
              f"说明训练期网格选参在验证期失效；建议保留 DEFAULT")
    elif abs(valid_default_m["Sharpe"] - valid_best_m["Sharpe"]) < 0.2:
        print(f"  ⓘ DEFAULT 与 BEST 验证期表现接近，保留 DEFAULT 即可")


if __name__ == "__main__":
    main()
