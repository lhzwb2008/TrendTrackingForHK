#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NAS100 成分股清单（静态快照，2025 年中口径）。

为了快速验证策略，这里使用一份静态清单而非 point-in-time 名单；
后续若需消除幸存者偏差，再接入历史变更日。
所有代码用 Longport 美股口径加 .US 后缀。
"""

NAS100_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA",
    "COST", "NFLX", "TMUS", "ASML", "CSCO", "AZN", "AMD", "PEP", "LIN",
    "ADBE", "INTU", "ISRG", "QCOM", "TXN", "BKNG", "AMGN", "PDD", "AMAT",
    "CMCSA", "ARM", "PANW", "GILD", "HON", "ADP", "MU", "VRTX", "ADI",
    "MELI", "LRCX", "SBUX", "INTC", "MDLZ", "PYPL", "REGN", "KLAC", "CTAS",
    "CDNS", "MAR", "SNPS", "CRWD", "ABNB", "FTNT", "ORLY", "CEG", "MNST",
    "ADSK", "DASH", "WDAY", "CHTR", "PCAR", "ROP", "PAYX", "NXPI", "AEP",
    "ROST", "FAST", "BKR", "MCHP", "KDP", "ODFL", "CPRT", "EXC", "VRSK",
    "CTSH", "DDOG", "GEHC", "EA", "FANG", "KHC", "AZO", "LULU", "XEL",
    "IDXX", "CSGP", "TTWO", "ANSS", "TEAM", "ZS", "ON", "CDW", "BIIB",
    "MDB", "DXCM", "MRVL", "WBD", "ILMN", "TTD", "WBA", "SIRI", "GFS",
    "ENPH", "LCID",
]

def get_universe():
    """返回带 .US 后缀的 Longport 代码列表。"""
    return [f"{s}.US" for s in NAS100_SYMBOLS]


if __name__ == "__main__":
    syms = get_universe()
    print(f"NAS100 universe: {len(syms)} symbols")
    print(syms[:10], "...")
