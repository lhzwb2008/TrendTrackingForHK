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

# 中文名映射（含常见简称；若官方与俗称不一致，优先选大众认知度更高的）
NAS100_NAMES_CN = {
    "AAPL":  "苹果",          "MSFT":  "微软",         "NVDA":  "英伟达",
    "AMZN":  "亚马逊",        "META":  "Meta",         "GOOGL": "谷歌-A",
    "GOOG":  "谷歌-C",        "AVGO":  "博通",         "TSLA":  "特斯拉",
    "COST":  "好市多",        "NFLX":  "奈飞",         "TMUS":  "T-Mobile",
    "ASML":  "阿斯麦",        "CSCO":  "思科",         "AZN":   "阿斯利康",
    "AMD":   "AMD",          "PEP":   "百事",         "LIN":   "林德气体",
    "ADBE":  "Adobe",        "INTU":  "财捷",         "ISRG":  "直觉外科",
    "QCOM":  "高通",          "TXN":   "德州仪器",     "BKNG":  "Booking",
    "AMGN":  "安进",          "PDD":   "拼多多",       "AMAT":  "应用材料",
    "CMCSA": "康卡斯特",      "ARM":   "ARM",         "PANW":  "派拓网络",
    "GILD":  "吉利德",        "HON":   "霍尼韦尔",     "ADP":   "ADP",
    "MU":    "美光",          "VRTX":  "福泰制药",     "ADI":   "亚德诺",
    "MELI":  "MercadoLibre", "LRCX":  "泛林集团",     "SBUX":  "星巴克",
    "INTC":  "英特尔",        "MDLZ":  "亿滋",         "PYPL":  "PayPal",
    "REGN":  "再生元",        "KLAC":  "科磊",         "CTAS":  "信达思",
    "CDNS":  "铿腾电子",      "MAR":   "万豪",         "SNPS":  "新思科技",
    "CRWD":  "CrowdStrike",  "ABNB":  "爱彼迎",       "FTNT":  "飞塔",
    "ORLY":  "奥莱利汽配",    "CEG":   "星座能源",     "MNST":  "怪兽饮料",
    "ADSK":  "欧特克",        "DASH":  "DoorDash",    "WDAY":  "Workday",
    "CHTR":  "特许通讯",      "PCAR":  "帕卡",         "ROP":   "罗珀",
    "PAYX":  "Paychex",      "NXPI":  "恩智浦",       "AEP":   "美电力",
    "ROST":  "罗斯百货",      "FAST":  "Fastenal",    "BKR":   "贝克休斯",
    "MCHP":  "微芯",          "KDP":   "Keurig",      "ODFL":  "ODFL",
    "CPRT":  "Copart",       "EXC":   "爱克斯龙",     "VRSK":  "Verisk",
    "CTSH":  "高知特",        "DDOG":  "Datadog",     "GEHC":  "GE医疗",
    "EA":    "艺电",          "FANG":  "钻石能源",     "KHC":   "卡夫亨氏",
    "AZO":   "汽车地带",      "LULU":  "露露乐蒙",     "XEL":   "Xcel能源",
    "IDXX":  "IDEXX",        "CSGP":  "CoStar",      "TTWO":  "Take-Two",
    "ANSS":  "ANSYS",        "TEAM":  "Atlassian",   "ZS":    "Zscaler",
    "ON":    "安森美",        "CDW":   "CDW",         "BIIB":  "渤健",
    "MDB":   "MongoDB",      "DXCM":  "DexCom",      "MRVL":  "迈威尔",
    "WBD":   "华纳兄弟探索",  "ILMN":  "因美纳",       "TTD":   "TTD",
    "WBA":   "沃尔格林",      "SIRI":  "天狼星广播",   "GFS":   "格芯",
    "ENPH":  "Enphase",      "LCID":  "Lucid汽车",
}


def get_universe():
    """返回带 .US 后缀的 Longport 代码列表。"""
    return [f"{s}.US" for s in NAS100_SYMBOLS]


def get_name_cn(symbol: str) -> str:
    """返回中文名；接受 'AAPL' 或 'AAPL.US' 形式。找不到则返回 ticker 本身。"""
    base = symbol.split(".")[0].upper()
    return NAS100_NAMES_CN.get(base, base)


def label(symbol: str) -> str:
    """格式化为 'AAPL(苹果)' 用于交易日志。"""
    base = symbol.split(".")[0].upper()
    cn = NAS100_NAMES_CN.get(base)
    return f"{base}({cn})" if cn and cn != base else base


if __name__ == "__main__":
    syms = get_universe()
    print(f"NAS100 universe: {len(syms)} symbols")
    print(f"中文名覆盖: {sum(1 for s in NAS100_SYMBOLS if s in NAS100_NAMES_CN)}/{len(NAS100_SYMBOLS)}")
    print([label(s) for s in syms[:10]])
