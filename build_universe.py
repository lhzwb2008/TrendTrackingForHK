#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从候选列表构建「小盘 + 质量 + 流动性」股票池（港股 / 美股），依赖 Longport OpenAPI。

质量（可调整）：主板普通股、每股净资产>0、TTM 每股收益>0、PB/PE 上限（避免极端投机盘）。
流动性：默认用最近一个交易日的成交额作粗筛；加 --deep-liquidity 则用近 N 日均成交额（更准、更慢）。

示例：
  .venv/bin/python build_universe.py --market hk
  .venv/bin/python build_universe.py --market hk --deep-liquidity --max-symbols 300
  .venv/bin/python build_universe.py --market us --input us_tickers.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Longport（延迟导入，便于 --help 在无凭证环境可用）


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_candidates_hk(path: str) -> List[str]:
    df = pd.read_csv(path)
    if '代码' not in df.columns:
        raise ValueError(f'{path} 需包含列「代码」')
    df['code'] = df['代码'].astype(str).str.zfill(5)
    df = df[~df['code'].str.match(r'^07[0-9]{3}$')]
    df = df[~df['code'].str.match(r'^028[0-9]{2}$')]
    return [f"{c}.HK" for c in df['code'].tolist()]


def _load_candidates_us(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = '代码' if '代码' in df.columns else ('symbol' if 'symbol' in df.columns else None)
    if col is None:
        raise ValueError(f'{path} 需包含列「代码」或「symbol」')
    out = []
    for raw in df[col].astype(str):
        s = raw.strip().upper()
        if not s:
            continue
        if '.' in s:
            out.append(s if s.endswith('.US') else s)
        else:
            out.append(f'{s}.US')
    return list(dict.fromkeys(out))


def _batch_static(ctx, symbols: List[str], batch: int) -> Dict[str, Any]:
    merged = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        rows = ctx.static_info(chunk)
        for r in rows:
            merged[r.symbol] = r
        time.sleep(float(os.getenv('LONGPORT_BATCH_PAUSE', '0.12')))
    return merged


def _batch_calc(ctx, symbols: List[str], batch: int) -> Dict[str, Any]:
    from longport.openapi import CalcIndex

    idxs = [
        CalcIndex.TotalMarketValue,
        CalcIndex.PeTtmRatio,
        CalcIndex.PbRatio,
        CalcIndex.Turnover,
    ]
    merged = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        rows = ctx.calc_indexes(chunk, idxs)
        for r in rows:
            merged[r.symbol] = r
        time.sleep(float(os.getenv('LONGPORT_BATCH_PAUSE', '0.12')))
    return merged


def _passes_quality_board(st, market: str) -> bool:
    b = st.board
    name = str(b)
    if market == 'hk':
        return 'HKEquity' in name and 'Warrant' not in name and 'PreIPO' not in name
    if market == 'us':
        return 'USMain' in name or 'USNSDQ' in name
    return True


def _deep_avg_turnover(api, symbol: str, days: int) -> Optional[float]:
    end = date.today()
    start = end - timedelta(days=days + 30)
    df = api.get_daily_data(symbol, start, end)
    if df is None or df.empty or len(df) < 5:
        return None
    tail = df.tail(days)
    if tail.empty:
        return None
    return float(tail['turnover'].mean())


def run() -> int:
    parser = argparse.ArgumentParser(description='构建小盘+质量+流动性股票池')
    parser.add_argument('--market', choices=['hk', 'us'], default='hk')
    parser.add_argument(
        '--input',
        default='hk_all_stocks.csv',
        help='港股默认 hk_all_stocks.csv；美股为含代码/symbol 的 CSV',
    )
    parser.add_argument('--output', default='')
    parser.add_argument('--min-mcap', type=float, default=5e8, help='最小总市值（港币或美元，与标的币种一致）')
    parser.add_argument('--max-mcap', type=float, default=8e10, help='最大总市值')
    parser.add_argument('--min-turnover', type=float, default=2e7, help='最近交易日成交额下限（粗筛）')
    parser.add_argument('--max-pe-ttm', type=float, default=80.0, help='TTM 市盈率上限；<=0 表示不限制')
    parser.add_argument('--max-pb', type=float, default=10.0, help='市净率上限；<=0 表示不限制')
    parser.add_argument('--require-positive-eps-ttm', action='store_true', default=True)
    parser.add_argument('--no-require-positive-eps-ttm', action='store_false', dest='require_positive_eps_ttm')
    parser.add_argument('--batch', type=int, default=150)
    parser.add_argument('--max-symbols', type=int, default=0, help='最多处理候选数，0 表示全部')
    parser.add_argument('--deep-liquidity', action='store_true', help='对初筛结果拉日线算近N日均成交额')
    parser.add_argument('--avg-turnover-days', type=int, default=20)
    parser.add_argument('--min-avg-turnover', type=float, default=3e7, help='deep 模式下近N日均成交额下限')
    args = parser.parse_args()

    out_path = args.output or (
        'universe_hk_small_quality.csv' if args.market == 'hk' else 'universe_us_small_quality.csv'
    )

    if args.market == 'hk':
        if not os.path.exists(args.input):
            print(f'未找到 {args.input}', file=sys.stderr)
            return 1
        symbols = _load_candidates_hk(args.input)
    else:
        if not os.path.exists(args.input):
            print(f'美股请提供 --input，例如含 ticker 列的 CSV', file=sys.stderr)
            return 1
        symbols = _load_candidates_us(args.input)

    if args.max_symbols and len(symbols) > args.max_symbols:
        symbols = symbols[: args.max_symbols]

    from longport.openapi import Config, QuoteContext

    try:
        cfg = Config.from_env()
        ctx = QuoteContext(cfg)
    except Exception as e:
        print(f'Longport 初始化失败: {e}', file=sys.stderr)
        return 1

    print(f'候选 {len(symbols)} 只，拉取 static_info / calc_indexes …')
    static_map = _batch_static(ctx, symbols, args.batch)
    calc_map = _batch_calc(ctx, symbols, args.batch)

    rows_out: List[Dict[str, Any]] = []
    for sym in symbols:
        st = static_map.get(sym)
        ca = calc_map.get(sym)
        if st is None or ca is None:
            continue
        if not _passes_quality_board(st, args.market):
            continue

        mcap = _to_float(ca.total_market_value)
        pe = _to_float(ca.pe_ttm_ratio)
        pb = _to_float(ca.pb_ratio)
        last_to = _to_float(ca.turnover)
        eps_ttm = _to_float(st.eps_ttm)
        bps = _to_float(st.bps)

        if mcap is None or mcap < args.min_mcap or mcap > args.max_mcap:
            continue
        if last_to is None or last_to < args.min_turnover:
            continue
        if bps is None or bps <= 0:
            continue
        if args.require_positive_eps_ttm and (eps_ttm is None or eps_ttm <= 0):
            continue
        if args.max_pe_ttm > 0 and pe is not None and (pe <= 0 or pe > args.max_pe_ttm):
            continue
        if args.max_pb > 0 and pb is not None and pb > args.max_pb:
            continue

        code = sym.split('.')[0]
        rows_out.append(
            {
                '代码': code,
                'symbol': sym,
                'name_cn': st.name_cn,
                'name_en': st.name_en,
                'total_market_value': mcap,
                'pe_ttm': pe,
                'pb': pb,
                'eps_ttm': eps_ttm,
                'bps': bps,
                'last_turnover': last_to,
                'avg_turnover_20d': None,
            }
        )

    print(f'初筛通过: {len(rows_out)} 只')

    if args.deep_liquidity and rows_out:
        from hk_stock_api import HKStockAPI

        api = HKStockAPI()
        kept: List[Dict[str, Any]] = []
        for i, r in enumerate(rows_out):
            sym = r['symbol']
            avg = _deep_avg_turnover(api, sym, args.avg_turnover_days)
            r['avg_turnover_20d'] = avg
            if avg is not None and avg >= args.min_avg_turnover:
                kept.append(r)
            if (i + 1) % 20 == 0:
                print(f'  深度流动性 {i+1}/{len(rows_out)}')
            time.sleep(0.12)
        rows_out = kept
        print(f'深度流动性后: {len(rows_out)} 只')

    df = pd.DataFrame(rows_out)
    if df.empty:
        print('无标的通过筛选，请放宽参数。', file=sys.stderr)
        return 2

    df = df.sort_values('total_market_value', ascending=True)
    df.to_csv(out_path, index=False)
    print(f'已写入 {out_path} ，共 {len(df)} 行')
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
