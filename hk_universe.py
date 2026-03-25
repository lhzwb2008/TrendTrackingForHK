#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股候选池：恒指成分 ∪ 恒生科技成分（仅依赖成分 CSV）。

成分名单需定期从恒生指数公司维护；本模块不负责自动抓取官网。
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

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


def build_hsi_hstech_universe(
    *,
    hsi_csv: str,
    hstech_csv: str,
    hsi_example: str,
    hstech_example: str,
) -> Tuple[List[str], str]:
    """
    恒指成分 ∪ 恒生科技成分（去重）。
    返回 (symbols, 说明字符串)。
    """
    hp = _resolve_path(hsi_csv, hsi_example)
    tp = _resolve_path(hstech_csv, hstech_example)
    if not hp and not tp:
        raise FileNotFoundError(
            '未找到恒指/恒生科技成分 CSV。请放置 data/hsi_constituents.csv 与 data/hstech_constituents.csv（或对应 *.example.csv），见 README。'
        )

    parts: List[str] = []
    hsi_syms: List[str] = []
    hst_syms: List[str] = []

    if hp:
        hsi_syms = load_symbol_column_csv(hp)
        parts.append(f'HSI成分 {len(hsi_syms)}（{os.path.basename(hp)}）')
    if tp:
        hst_syms = load_symbol_column_csv(tp)
        parts.append(f'HSTECH成分 {len(hst_syms)}（{os.path.basename(tp)}）')

    merged = list(dict.fromkeys(hsi_syms + hst_syms))
    desc = ' ∪ '.join(parts) if parts else '空'
    print(f'[候选池] 合并去重后共 {len(merged)} 只：{desc}', flush=True)
    return merged, desc
