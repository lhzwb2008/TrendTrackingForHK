#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股票池清单（静态快照）。

默认使用 NAS100 ∪ S&P 500 的合集（约 518 只），相比纯 NAS100 显著提升
横截面 alpha：DAILY 模式 Sharpe 1.12 → 1.51、CAGR 25.6% → 37.5%。

提供：
  get_universe()        默认 universe（合集）→ 主线策略使用
  get_nas100_universe() 仅 NAS100 子集，用于做对照实验
  get_sp500_universe()  仅 S&P 500 成分

为了快速验证策略，这里使用静态清单而非 point-in-time 名单；
后续若需消除幸存者偏差，再接入历史变更日。
所有代码用 Longport 美股口径加 .US 后缀；类股代号统一用 dash（如 GOOG / GOOGL）。
注意：BF-B、BRK-B 等带 dash 的类股代号 Longport 暂不支持，已从 SP500_SYMBOLS 预先剔除。
"""
from __future__ import annotations

# ---------- NAS100 成分（2025 年中静态快照，101 只） ----------
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

# ---------- S&P 500 成分（来自 Wikipedia 静态快照） ----------
SP500_SYMBOLS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRO",
    "BSX", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CASY", "CAT", "CB",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW",
    "CHTR", "CI", "CIEN", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COHR", "COIN", "COO", "COP", "COR", "COST",
    "CPAY", "CPB", "CPRT", "CPT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP",
    "CSX", "CTAS", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH",
    "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL",
    "ELV", "EME", "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES",
    "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS",
    "FISV", "FITB", "FIX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
    "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON", "HOOD", "HPE", "HPQ",
    "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR",
    "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY",
    "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB",
    "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII",
    "LIN", "LITE", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS",
    "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO",
    "MDLZ", "MDT", "MET", "META", "MGM", "MKC", "MLM", "MMM", "MNST", "MO",
    "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH", "MS", "MSCI", "MSFT", "MSI",
    "MTB", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI",
    "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR",
    "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY",
    "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG",
    "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR",
    "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSKY", "PSX", "PTC",
    "PWR", "PYPL", "Q", "QCOM", "RCL", "REG", "REGN", "RF", "RJF", "RL",
    "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SATS", "SBAC",
    "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNDK", "SNPS", "SO",
    "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW",
    "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH",
    "TEL", "TER", "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP",
    "UPS", "URI", "USB", "V", "VEEV", "VICI", "VLO", "VLTO", "VMC", "VRSK",
    "VRSN", "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD",
    "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM",
    "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH",
    "ZBRA", "ZTS",
]

# ---------- 默认 universe = NAS100 ∪ S&P500（去重排序，~518 只） ----------
DEFAULT_SYMBOLS = sorted(set(NAS100_SYMBOLS) | set(SP500_SYMBOLS))


# ---------- 中文名映射（仅 NAS100 主流标的；其他默认显示英文 ticker） ----------
NAMES_CN = {
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
    # SP500 中常见的额外补充
    "JPM":   "摩根大通",      "V":     "Visa",        "MA":    "万事达",
    "JNJ":   "强生",          "WMT":   "沃尔玛",      "PG":    "宝洁",
    "XOM":   "埃克森美孚",    "CVX":   "雪佛龙",      "BAC":   "美国银行",
    "BRK-B": "伯克希尔-B",    "UNH":   "联合健康",    "HD":    "家得宝",
    "DIS":   "迪士尼",        "KO":    "可口可乐",    "MCD":   "麦当劳",
    "ORCL":  "甲骨文",        "CRM":   "Salesforce", "IBM":   "IBM",
    "PFE":   "辉瑞",          "MRK":   "默沙东",      "LLY":   "礼来",
    "BA":    "波音",          "GE":    "通用电气",    "F":     "福特",
    "GM":    "通用汽车",      "UBER":  "Uber",       "PLTR":  "Palantir",
    "COIN":  "Coinbase",     "HOOD":  "Robinhood",   "SMCI":  "超微电脑",
    "DELL":  "戴尔",          "HPQ":   "惠普",        "HPE":   "慧与",
    "T":     "AT&T",         "VZ":    "Verizon",     "WFC":   "富国银行",
    "GS":    "高盛",          "MS":    "摩根士丹利",  "C":     "花旗",
}


def get_universe():
    """默认 universe（NAS100 ∪ SP500，~518 只），返回带 .US 后缀的 Longport 代码列表。"""
    return [f"{s}.US" for s in DEFAULT_SYMBOLS]


def get_nas100_universe():
    """仅 NAS100 子集（101 只），用于对照实验。"""
    return [f"{s}.US" for s in NAS100_SYMBOLS]


def get_sp500_universe():
    """仅 S&P 500 子集（502 只）。"""
    return [f"{s}.US" for s in SP500_SYMBOLS]


def get_name_cn(symbol: str) -> str:
    """返回中文名；接受 'AAPL' 或 'AAPL.US' 形式。找不到则返回 ticker 本身。"""
    base = symbol.split(".")[0].upper()
    return NAMES_CN.get(base, base)


def label(symbol: str) -> str:
    """格式化为 'AAPL(苹果)' 用于交易日志；无中文名时直接返回 'AAPL'。"""
    base = symbol.split(".")[0].upper()
    cn = NAMES_CN.get(base)
    return f"{base}({cn})" if cn and cn != base else base


if __name__ == "__main__":
    print(f"DEFAULT (NAS100 ∪ SP500): {len(DEFAULT_SYMBOLS)} 只")
    print(f"NAS100 子集: {len(NAS100_SYMBOLS)} 只")
    print(f"SP500  子集: {len(SP500_SYMBOLS)} 只")
    print(f"中文名覆盖: {len(NAMES_CN)} 只")
    print("示例:", [label(s) for s in get_universe()[:8]])
