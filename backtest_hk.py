#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股版 cross-sectional rank 选股回测
=====================================

复用 `backtest.py` 的全部信号 / 引擎 / 评估代码，仅替换：

  1) Universe   : 恒生指数 ∪ 恒生科技指数（~85 只静态快照）
  2) 基准/Regime: 2800.HK（盈富基金，跟踪 HSI）替代 SPY/QQQ
  3) 成本模型   : 港股印花税 0.1% (双向) + 交易费 ~0.012% + 平台费
                  折算为 ~15 bps/侧 塞入 slippage；关闭美股专属 SEC/TAF
  4) 超参微调   : K_LONG=6（池小）、MIN_DOLLAR_VOLUME=2e7 港币、
                  其余主参数沿用美股 baseline 后做小幅扫描

只跑 DAILY 模式（与 US 的 DAILY 同口径：当日 close 信号 + close 成交，
跨牛熊压力测试参考；港股分钟数据 Longport 同样有 ~2 年上限，结构相同，
本文件先聚焦长周期稳健性研究）。

运行:
    python backtest_hk.py
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import backtest as bt
from longport_api import fetch_daily_bars


# ============================================================================
#                              港股池 (静态快照)
# ============================================================================
# 恒生指数 (HSI, ~82 只) ∪ 恒生科技指数 (HSTECH, 30 只) 去重后 ~85 只
# 数据日期: 2025 年中静态名单，不做 point-in-time 还原（与美股版一致，存在
# 幸存者偏差）。代码统一为 4 位港股代号 + .HK 后缀。

HSI_SYMBOLS = [
    "0001", "0002", "0003", "0005", "0006", "0011", "0012", "0016", "0017",
    "0027", "0066", "0083", "0101", "0151", "0175", "0177", "0241", "0267",
    "0285", "0288", "0291", "0316", "0322", "0386", "0388", "0669", "0688",
    "0700", "0762", "0823", "0857", "0868", "0883", "0939", "0941", "0960",
    "0968", "0981", "0992", "1038", "1044", "1093", "1109", "1113", "1177",
    "1209", "1211", "1299", "1378", "1398", "1810", "1876", "1928", "1929",
    "1972", "2015", "2018", "2020", "2269", "2313", "2318", "2319", "2331",
    "2382", "2388", "2628", "2688", "2899", "3690", "3692", "3968", "3988",
    "6098", "6862", "9618", "9633", "9888", "9961", "9988", "9999",
]

HSTECH_SYMBOLS = [
    "0700", "9988", "3690", "9618", "9999", "1810", "9888", "9961", "1024",
    "0981", "1347", "0992", "0285", "2382", "6618", "0241", "6060", "1833",
    "2015", "9866", "9868", "0772", "1357", "0780", "6088", "6160", "0268",
]

DEFAULT_HK_SYMBOLS = sorted(set(HSI_SYMBOLS) | set(HSTECH_SYMBOLS))


# 中文名（覆盖主流权重股，方便日志阅读；其他默认显示代码）
NAMES_CN_HK = {
    "0001": "长和",        "0002": "中电控股",    "0003": "香港中华煤气",
    "0005": "汇丰控股",    "0006": "电能实业",    "0011": "恒生银行",
    "0012": "恒基地产",    "0016": "新鸿基地产",  "0017": "新世界发展",
    "0027": "银河娱乐",    "0066": "港铁",        "0083": "信和置业",
    "0101": "恒隆地产",    "0151": "中国旺旺",    "0175": "吉利汽车",
    "0241": "阿里健康",    "0267": "中信股份",    "0268": "金蝶国际",
    "0285": "比亚迪电子",  "0288": "万洲国际",    "0291": "华润啤酒",
    "0316": "东方海外",    "0386": "中石化",      "0388": "港交所",
    "0669": "创科实业",    "0688": "中国海外",    "0700": "腾讯",
    "0762": "中国联通",    "0772": "阅文集团",    "0780": "同程旅行",
    "0823": "领展房产",    "0857": "中石油",      "0868": "信义玻璃",
    "0883": "中海油",      "0939": "建设银行",    "0941": "中国移动",
    "0960": "龙湖集团",    "0968": "信义光能",    "0981": "中芯国际",
    "0992": "联想集团",    "1024": "快手",        "1038": "长江基建",
    "1044": "恒安国际",    "1093": "石药集团",    "1109": "华润置地",
    "1113": "长实集团",    "1177": "中国生物制药","1209": "华润万象",
    "1211": "比亚迪",      "1299": "友邦保险",    "1347": "华虹半导体",
    "1357": "美图",        "1378": "中国宏桥",    "1398": "工商银行",
    "1810": "小米集团",    "1833": "平安好医生",  "1876": "百威亚太",
    "1928": "金沙中国",    "1929": "周大福",      "1972": "太古地产",
    "2015": "理想汽车",    "2018": "瑞声科技",    "2020": "安踏体育",
    "2269": "药明生物",    "2313": "申洲国际",    "2318": "中国平安",
    "2319": "蒙牛乳业",    "2331": "李宁",        "2382": "舜宇光学",
    "2388": "中银香港",    "2628": "中国人寿",    "2688": "新奥能源",
    "2899": "紫金矿业",    "3690": "美团",        "3692": "翰森制药",
    "3968": "招商银行",    "3988": "中国银行",    "6060": "众安在线",
    "6088": "FIT Hon Teng","6098": "碧桂园服务",  "6160": "百济神州",
    "6618": "京东健康",    "6862": "海底捞",      "9618": "京东集团",
    "9633": "农夫山泉",    "9866": "蔚来",        "9868": "小鹏汽车",
    "9888": "百度",        "9961": "携程集团",    "9988": "阿里巴巴",
    "9999": "网易",
}


def get_hk_universe() -> List[str]:
    return [f"{s}.HK" for s in DEFAULT_HK_SYMBOLS]


def hk_label(symbol: str) -> str:
    base = symbol.split(".")[0]
    cn = NAMES_CN_HK.get(base)
    return f"{base}({cn})" if cn else base


# Monkey-patch backtest.sym_label so 日志里港股显示中文名
bt.sym_label = hk_label


# ============================================================================
#                              港股专属配置
# ============================================================================
# 港股交易成本（双向收取的项放在「滑点」里粗略折算）：
#   印花税:        0.10%  (买卖双方各征)            -> 10 bps/侧
#   交易费/AFRC等: ~0.0117%                          -> 1.2 bps/侧
#   港交所交易系统使用费/CCASS:  ~0.005%             -> 0.5 bps/侧
#   Longport HK 平台费: 港币 15 起 / 0.03% 取大     -> ~3 bps/侧 (中等单)
#   实际滑点:      ~5 bps/侧
#   合计 ≈ 19~20 bps/侧 round-trip ~40 bps
#
# 这里把所有按金额比例的成本统一塞进 SLIPPAGE_BPS=20，并关闭美股专属的
# 按股计费 / SEC / TAF（设 0），简化但保守地反映港股摩擦成本。

HK_DAILY_START = "2018-01-01"      # 港股长周期回测起点（含 2018-2020 熊市）
HK_BACKTEST_END = "today"

HK_BENCHMARK = "2800.HK"           # 盈富基金，跟踪 HSI；同时用作 200DMA regime

HK_OVERRIDES = dict(
    starting_capital=1_000_000,    # 100 万港币
    k_long=6,                      # 池子 ~85 只，K=6 ≈ US 的 K=10/516
    k_short=0,
    long_weight_frac=1.0,
    gross_leverage=1.0,
    hysteresis_mult=4.0,
    stop_loss_pct=0.05,
    stop_loss_atr_mult=1.5,
    max_hold_days=80,
    min_dollar_volume=2e7,         # 2,000 万港币 20D 平均成交额下限
    mom_weight=0.8,
    bias_weight=0.3,
    regime_filter=True,
    vol_target_annual=0.20,
    # ---- 成本：折算为 20 bps/侧，关闭美股按股计费 ----
    enable_costs=True,
    platform_fee_per_share=0.0,
    platform_fee_min=0.0,
    sec_fee_rate=0.0,
    taf_per_share=0.0,
    taf_max_per_order=0.0,
    slippage_bps=20.0,
    mode="daily",
)


def make_hk_cfg(start: date, end: date, **overrides) -> bt.Config:
    params = dict(HK_OVERRIDES)
    params.update(overrides)
    return bt.Config(start=start, end=end, **params)


# ============================================================================
#                              主流程
# ============================================================================

def _slice_close(df: pd.DataFrame, start: date, end: date) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    s = df.loc[(df.index >= pd.Timestamp(start)) &
               (df.index <= pd.Timestamp(end)), "close"]
    return s if len(s) else None


def _yearly_table(result: bt.BacktestResult, label: str = "HK") -> None:
    yearly = bt._yearly_stats(result)
    print(f"\n  ----- {label} 逐年表现 -----")
    print(f"  {'年份':<6}{'收益%':>10}{'最大回撤%':>14}{'Sharpe':>10}{'平仓笔数':>12}")
    print("  " + "-" * 52)
    for y in sorted(yearly.keys()):
        r = yearly[y]
        print(f"  {y:<6}{r['ret']:>+10.2f}{r['mdd']:>+14.2f}"
              f"{r['sharpe']:>10.2f}{r['trades']:>12d}")


def run_one(label: str, panel, cfg, regime_series, bench_close):
    print(f"\n========== 回测: {label} | {cfg.start} ~ {cfg.end} | "
          f"K={cfg.k_long} mom_w={cfg.mom_weight} bias_w={cfg.bias_weight} "
          f"hyst={cfg.hysteresis_mult} sl_atr={cfg.stop_loss_atr_mult} "
          f"max_hold={cfg.max_hold_days} ==========")
    res = bt.run_backtest(panel, cfg, regime_series=regime_series)
    summary = bt.summarize(res, cfg, bench_close)
    # 把基准标签从 QQQ 改写为 HSI(2800.HK)
    for k in list(summary.keys()):
        if k.startswith("基准(QQQ)"):
            summary[k.replace("(QQQ)", "(HSI 2800.HK)")] = summary.pop(k)
    bt.print_summary(summary, title=label)
    _yearly_table(res, label)
    return res, summary


def main():
    end_date = date.today() if HK_BACKTEST_END == "today" else date.fromisoformat(HK_BACKTEST_END)
    start_date = date.fromisoformat(HK_DAILY_START)

    print("=" * 64)
    print("  港股 cross-sectional rank 策略回测 (DAILY)")
    print("=" * 64)
    print(f"  区间        {start_date} ~ {end_date}")
    print(f"  Universe    HSI ∪ HSTECH (~{len(DEFAULT_HK_SYMBOLS)} 只)")
    print(f"  基准/Regime {HK_BENCHMARK} (盈富基金)")
    print(f"  起始本金    HK${HK_OVERRIDES['starting_capital']:,}")
    print(f"  成本模型    20 bps/侧 (含印花税/交易费/平台费)")

    syms = get_hk_universe()
    print(f"\n[数据] 加载 {len(syms)} 只港股日线...")
    data = bt.load_all_data(syms, start_date, end_date)

    # 流动性 / 数据完整性二次过滤：剔除没有任何分钟数据 / 历史太短的标的
    keep = {s: df for s, df in data.items() if len(df) >= 200}
    dropped = sorted(set(data.keys()) - set(keep.keys()))
    if dropped:
        print(f"[过滤] 剔除历史 < 200 根的标的: {dropped}")
    data = keep
    print(f"[数据] 最终 panel = {len(data)} 只")

    print("[数据] 构造 DAILY panel ...")
    panel = bt.build_daily_panel(data)

    # ---- 基准 + regime ----
    print(f"[regime] 拉取基准 {HK_BENCHMARK} 用于 200DMA 过滤 ...")
    try:
        bench_df = fetch_daily_bars(HK_BENCHMARK,
                                     start_date - timedelta(days=400),
                                     end_date, log_cache=False)
    except Exception as e:
        print(f"[警告] 基准 {HK_BENCHMARK} 拉取失败: {e}")
        bench_df = pd.DataFrame()

    if bench_df is None or len(bench_df) == 0 or "close" not in bench_df.columns:
        # 基准缺失（API 配额）→ fallback：universe 等权指数仅用作 benchmark 展示，
        # regime 过滤直接禁用（universe 等权与 HSI 走势差异大，做 regime 不可靠）
        print("[regime] 基准缺失 → 禁用 regime 过滤；benchmark 用 universe 等权指数展示")
        closes = pd.concat({s: df["close"].pct_change()
                            for s, df in data.items() if "close" in df},
                           axis=1, sort=True)
        eq_idx = (closes.mean(axis=1).fillna(0) + 1).cumprod() * 100
        bench_close = eq_idx
        bench_df = pd.DataFrame({"close": bench_close})
        regime_series = None
        # 同时关闭 baseline 的 regime_filter
        HK_OVERRIDES["regime_filter"] = False
    else:
        bench_close = bench_df["close"]
        regime_series = (bench_close > bench_close.rolling(200).mean())
        up_days = int(regime_series.sum())
        print(f"[regime] 基准 > 200DMA: {up_days}/{len(regime_series)} 日上行")

    bench_slice = _slice_close(bench_df, start_date, end_date)

    # ===================== 1) Baseline =====================
    cfg_base = make_hk_cfg(start_date, end_date)
    res_base, sum_base = run_one("HK Baseline (K=6, mom=0.8/bias=0.3, hyst=4)",
                                  panel, cfg_base, regime_series, bench_slice)

    # ===================== 2) 关键超参对比扫描 =====================
    # 港股池小、且大型权重股集中（腾讯/美团/小米/比亚迪/友邦），动量信号更
    # 容易被 mega-cap 主导。这里轮换几组关键参数，验证 baseline 的稳健性。
    variants = [
        ("K=4 更集中",        dict(k_long=4)),
        ("K=8 更分散",        dict(k_long=8)),
        ("纯动量 mom=1.0",    dict(mom_weight=1.0, bias_weight=0.0)),
        ("均衡 mom=0.6",      dict(mom_weight=0.6, bias_weight=0.2)),
        ("无 regime",         dict(regime_filter=False)),
        ("止损更紧 ATR=1.0",  dict(stop_loss_atr_mult=1.0)),
        ("止损更松 ATR=2.5",  dict(stop_loss_atr_mult=2.5)),
        ("Hold 最长 150",     dict(max_hold_days=150)),
        ("无成本(对照)",      dict(enable_costs=False)),
    ]

    print("\n" + "=" * 76)
    print("  超参扫描结果（与 Baseline 对比）")
    print("=" * 76)
    rows = []
    base_metrics = (sum_base["累计收益"], sum_base["年化收益(CAGR)"],
                    sum_base["Sharpe"], sum_base["最大回撤"])
    rows.append(("Baseline", *base_metrics, sum_base["总交易笔数"]))

    for name, ovr in variants:
        cfg_v = make_hk_cfg(start_date, end_date, **ovr)
        res_v = bt.run_backtest(panel, cfg_v, regime_series=regime_series)
        s_v = bt.summarize(res_v, cfg_v, None)
        rows.append((name, s_v["累计收益"], s_v["年化收益(CAGR)"],
                      s_v["Sharpe"], s_v["最大回撤"], s_v["总交易笔数"]))

    print(f"  {'变体':<22}{'累计':>20}{'CAGR':>10}{'Sharpe':>9}"
          f"{'MDD':>11}{'笔数':>8}")
    print("  " + "-" * 80)
    for r in rows:
        # 取累计收益的百分比部分
        cum = r[1].split("  ")[0] if isinstance(r[1], str) else r[1]
        print(f"  {r[0]:<22}{cum:>20}{r[2]:>10}{r[3]:>9}{r[4]:>11}{r[5]:>8}")
    print("=" * 76)

    # ===================== 3) Top 盈亏 =====================
    bt.print_top_trades(res_base.trades)

    # ===================== 4) vs HSI 累计对比 =====================
    if bench_slice is not None and len(bench_slice) > 1:
        bcum = bench_slice.iloc[-1] / bench_slice.iloc[0] - 1
        scum = res_base.equity.iloc[-1] / cfg_base.starting_capital - 1
        years = len(res_base.daily_returns) / 252
        bcagr = (1 + bcum) ** (1 / years) - 1 if years > 0 else 0
        print(f"\n[对比] 期间 HSI(2800.HK) 累计 {bcum*100:+.2f}% / CAGR {bcagr*100:.2f}%")
        print(f"[对比] 策略 (Baseline)      累计 {scum*100:+.2f}% / "
              f"CAGR {((1+scum)**(1/years)-1)*100:.2f}%  "
              f"Sharpe {sum_base['Sharpe']}")


if __name__ == "__main__":
    main()
