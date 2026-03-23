#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股候选池：恒指 / 恒生科技成分（CSV） + 近一年上市新股（Longport static_info.listing_date）。

成分股名单需定期从恒生指数公司维护；本模块不负责自动抓取官网。
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import pandas as pd


def _resolve_path(primary: str, example: str) -> Optional[str]:
    if primary and os.path.exists(primary):
        return primary
    if example and os.path.exists(example):
        print(
            f'[候选池] 未找到 {primary}，改用样例 {example}（正式回测请复制为正式文件名并更新成分）',
            flush=True,
        )
        return example
    return None


def load_symbol_column_csv(path: str) -> List[str]:
    """读取含 symbol 或「代码」列的 CSV，统一为 00000.HK 形式。"""
    df = pd.read_csv(path)
    col = 'symbol' if 'symbol' in df.columns else '代码'
    out: List[str] = []
    for raw in df[col].astype(str):
        s = raw.strip()
        if not s or s.lower() == 'nan':
            continue
        if '.' in s:
            out.append(s.upper() if s.endswith('.HK') else s)
        else:
            out.append(f'{s.zfill(5)}.HK')
    return list(dict.fromkeys(out))


def iter_hk_numeric_codes(hk_all_csv: str) -> List[str]:
    """从港股全表 CSV 提取 5 位代码列表（去重）。"""
    if not os.path.exists(hk_all_csv):
        return []
    df = pd.read_csv(hk_all_csv)
    if '代码' not in df.columns:
        return []
    codes = (
        df['代码']
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.replace(r'\D', '', regex=True)
    )
    raw = [c.zfill(5)[-5:] for c in codes if c and len(c.zfill(5)[-5:]) == 5]
    return list(dict.fromkeys(raw))


def fetch_ipo_symbols_recent(
    hk_all_csv: str,
    max_age_days: int,
    *,
    batch_pause: float = 0.2,
) -> List[str]:
    """
    对全表代码批量请求 static_info，保留 listing_date 在 max_age_days 内的 .HK。
    需已配置 Longport；约 len(codes)/500 次请求。
    """
    codes = iter_hk_numeric_codes(hk_all_csv)
    if not codes:
        return []

    from hk_stock_api import get_api_singleton

    api = get_api_singleton()
    today = date.today()
    cutoff = today - timedelta(days=max_age_days)
    out: List[str] = []
    nbatch = (len(codes) + 499) // 500
    total_infos = 0
    with_ld = 0
    print(
        f'[候选池] 新股扫描: {len(codes)} 只代码，约 {nbatch} 次 static_info 请求（上市日 ≥ {cutoff}）…',
        flush=True,
    )

    for i in range(0, len(codes), 500):
        chunk = codes[i : i + 500]
        syms = [f'{c}.HK' for c in chunk]
        try:
            resp = api._call_with_retry(api.quote_ctx.static_info, syms)
        except Exception as e:
            print(f'[候选池] static_info 批次失败 ({i // 500 + 1}/{nbatch}): {e}', flush=True)
            time.sleep(batch_pause)
            continue

        infos: List = []
        if resp is None:
            pass
        elif hasattr(resp, 'secu_static_info'):
            infos = list(resp.secu_static_info)
        elif isinstance(resp, (list, tuple)):
            infos = list(resp)

        for info in infos:
            total_infos += 1
            sym = getattr(info, 'symbol', None)
            ld = (
                getattr(info, 'listing_date', None)
                or getattr(info, 'listingDate', None)
                or ''
            )
            if str(ld).strip():
                with_ld += 1
            if not sym or not str(ld).strip():
                continue
            try:
                d = datetime.strptime(str(ld)[:10], '%Y-%m-%d').date()
            except ValueError:
                continue
            if d >= cutoff:
                out.append(sym)

        time.sleep(batch_pause)

    out = list(dict.fromkeys(out))
    print(f'[候选池] 新股（近 {max_age_days} 天上市）: {len(out)} 只', flush=True)
    if len(out) == 0 and total_infos > 0:
        if with_ld == 0:
            print(
                '[候选池] 提示：static_info 中几乎无 listing_date 字段，无法按上市日筛新股（与行情权限/SDK 有关，可设 INCLUDE_IPO_UNIVERSE=False 跳过扫描）',
                flush=True,
            )
        else:
            print(
                f'[候选池] 提示：有 {with_ld} 条含上市日，但均早于 {cutoff}（该窗口内无新股或需缩短回测窗口再试）',
                flush=True,
            )
    return out


def build_hsi_hstech_ipo_universe(
    *,
    hsi_csv: str,
    hstech_csv: str,
    hsi_example: str,
    hstech_example: str,
    hk_all_csv: str,
    include_ipo: bool,
    ipo_max_age_days: int,
) -> Tuple[List[str], str]:
    """
    合并：恒指成分 ∪ 恒生科技成分 ∪（可选）近一年新股。
    返回 (symbols, 说明字符串)。
    """
    hp = _resolve_path(hsi_csv, hsi_example)
    tp = _resolve_path(hstech_csv, hstech_example)
    if not hp and not tp and not include_ipo:
        raise FileNotFoundError(
            '未找到恒指/恒生科技成分 CSV，且未开启新股。请放置 data/hsi_constituents.csv 等，见 README。'
        )

    parts: List[str] = []
    hsi_syms: List[str] = []
    hst_syms: List[str] = []

    if not hp and not tp and include_ipo:
        if not os.path.exists(hk_all_csv):
            raise FileNotFoundError(f'仅新股模式需要 {hk_all_csv}')
        ipo_syms = fetch_ipo_symbols_recent(hk_all_csv, ipo_max_age_days)
        desc = f'仅新股（listing≤{ipo_max_age_days}天）{len(ipo_syms)} 只'
        print(f'[候选池] {desc}', flush=True)
        return ipo_syms, desc

    if hp:
        hsi_syms = load_symbol_column_csv(hp)
        parts.append(f'HSI成分 {len(hsi_syms)}（{os.path.basename(hp)}）')
    if tp:
        hst_syms = load_symbol_column_csv(tp)
        parts.append(f'HSTECH成分 {len(hst_syms)}（{os.path.basename(tp)}）')

    merged = list(dict.fromkeys(hsi_syms + hst_syms))

    ipo_syms: List[str] = []
    if include_ipo:
        if not os.path.exists(hk_all_csv):
            print(f'[候选池] 跳过新股：未找到 {hk_all_csv}', flush=True)
        else:
            ipo_syms = fetch_ipo_symbols_recent(hk_all_csv, ipo_max_age_days)
            parts.append(f'新股 {len(ipo_syms)}（listing≤{ipo_max_age_days}天）')

    all_syms = list(dict.fromkeys(merged + ipo_syms))
    desc = ' ∪ '.join(parts) if parts else '空'
    print(f'[候选池] 合并去重后共 {len(all_syms)} 只：{desc}', flush=True)
    return all_syms, desc


def merge_with_us_watchlist(hk_symbols: Sequence[str], us_from_csv: Optional[List[str]]) -> List[str]:
    """若你另有美股自选，可拼在港股候选后（去重）。"""
    if not us_from_csv:
        return list(hk_symbols)
    return list(dict.fromkeys(list(hk_symbols) + list(us_from_csv)))
