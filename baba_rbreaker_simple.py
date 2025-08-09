#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股BABA R-Breaker日内交易策略 - 简化版（无图表）
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from longport.openapi import QuoteContext, Config, Period, AdjustType
import time
import os
import pickle
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """交易记录"""
    datetime: datetime
    symbol: str
    action: str  # BUY/SELL
    price: float
    quantity: int
    amount: float
    reason: str
    pnl: float = 0.0
    pnl_percent: float = 0.0
    hold_minutes: int = 0

class RBreakerStrategy:
    """R-Breaker日内交易策略"""
    
    def __init__(self):
        """初始化策略"""
        self.config = Config.from_env()
        self.quote_ctx = QuoteContext(self.config)
        
        # R-Breaker策略参数（优化后的参数）
        self.f1 = 0.35  # 突破买入系数（提高阈值减少假突破）
        self.f2 = 0.15  # 观察卖出系数（提高阈值）
        self.f3 = 0.25  # 反转卖出系数（适中阈值）
        self.f4 = 0.15  # 观察买入系数（提高阈值）
        self.f5 = 0.25  # 反转买入系数（适中阈值）
        
        # 交易参数
        self.max_position_size = 1000  # 最大持仓数量
        self.stop_loss_percent = 0.015  # 止损1.5%（更严格）
        self.max_hold_minutes = 180    # 最大持仓时间3小时（更短）
        self.min_price_move = 0.10     # 最小价格变动阈值
        self.cooldown_minutes = 5      # 交易冷却时间5分钟
        self.initial_capital = 100000  # 初始资金10万美元
        
        # 回测数据
        self.trades: List[Trade] = []
        self.position = 0  # 当前持仓
        self.position_price = 0.0  # 持仓成本
        self.position_time = None  # 开仓时间
        self.last_trade_time = None  # 上次交易时间
        self.daily_stats = {}  # 每日统计
        
        # 数据缓存
        self.cache_dir = "stock_data_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_minute_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取分钟级K线数据（分批获取）"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_minute_{start_date}_{end_date}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=1):  # 缓存1小时有效
                logger.info(f"从缓存加载 {symbol} 分钟数据")
                return pd.read_pickle(cache_file)
        
        logger.info(f"分批获取 {symbol} 分钟级K线数据: {start_date} 到 {end_date}")
        
        all_data = []
        current_date = start_date
        batch_days = 5  # 每次获取5天的数据
        
        while current_date <= end_date:
            batch_end_date = min(current_date + timedelta(days=batch_days-1), end_date)
            logger.info(f"获取批次数据: {current_date} 到 {batch_end_date}")
            
            try:
                # 获取分钟级数据
                candles = self.quote_ctx.history_candlesticks_by_date(
                    symbol,
                    Period.Min_1,  # 1分钟K线
                    AdjustType.ForwardAdjust,
                    current_date,
                    batch_end_date
                )
                
                if candles:
                    logger.info(f"批次 {current_date}-{batch_end_date}: 获取到 {len(candles)} 条数据")
                    
                    for candle in candles:
                        all_data.append({
                            'datetime': candle.timestamp,
                            'open': float(candle.open),
                            'high': float(candle.high),
                            'low': float(candle.low),
                            'close': float(candle.close),
                            'volume': int(candle.volume),
                            'turnover': float(candle.turnover)
                        })
                else:
                    logger.warning(f"批次 {current_date}-{batch_end_date}: API返回空数据")
                
                # 添加延迟避免API限制
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"获取批次数据失败 {current_date}-{batch_end_date}: {e}")
            
            current_date = batch_end_date + timedelta(days=1)
        
        if not all_data:
            logger.error(f"{symbol}: 所有批次都返回空数据")
            return pd.DataFrame()
        
        logger.info(f"{symbol}: 总共获取到 {len(all_data)} 条分钟数据")
        
        df = pd.DataFrame(all_data)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # 去重（可能有重叠数据）
        df = df[~df.index.duplicated(keep='first')]
        
        # 过滤交易时间（美股交易时间：9:30-16:00 EST）
        df = self.filter_trading_hours(df)
        
        # 保存缓存
        df.to_pickle(cache_file)
        logger.info(f"数据已缓存到 {cache_file}，最终数据量: {len(df)} 条")
        
        return df
    
    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤美股交易时间"""
        if df.empty:
            return df
        
        # 转换为美东时间并过滤交易时间
        df_filtered = df.copy()
        
        # 简单过滤：保留工作日的数据
        df_filtered = df_filtered[df_filtered.index.weekday < 5]
        
        return df_filtered
    
    def calculate_rbreaker_levels(self, prev_high: float, prev_low: float, prev_close: float) -> Dict[str, float]:
        """计算R-Breaker的六个价位"""
        # 计算枢轴点
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # 计算六个关键价位
        levels = {
            'bbreak': prev_high + self.f1 * (prev_close - prev_low),      # 突破买入价
            'ssetup': pivot + self.f2 * (prev_high - prev_low),           # 观察卖出价
            'senter': (1 + self.f3) * pivot - self.f3 * prev_low,        # 反转卖出价
            'benter': (1 + self.f5) * pivot - self.f5 * prev_high,       # 反转买入价
            'bsetup': pivot - self.f4 * (prev_high - prev_low),           # 观察买入价
            'sbreak': prev_low - self.f1 * (prev_high - prev_close)       # 突破卖出价
        }
        
        return levels
    
    def get_daily_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """从分钟数据计算每日OHLC"""
        if df.empty:
            return pd.DataFrame()
        
        # 按日期分组计算OHLC
        daily_data = df.groupby(df.index.date).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum'
        })
        
        return daily_data
    
    def check_trading_signal(self, current_price: float, levels: Dict[str, float], 
                           current_time: datetime) -> Tuple[str, str]:
        """检查交易信号"""
        signal = "HOLD"
        reason = ""
        
        # 检查交易冷却时间
        if self.last_trade_time:
            minutes_since_last_trade = (current_time - self.last_trade_time).total_seconds() / 60
            if minutes_since_last_trade < self.cooldown_minutes:
                return "HOLD", "冷却时间"
        
        # 如果有持仓，检查平仓信号
        if self.position != 0:
            # 检查止损
            if self.position > 0:  # 多头持仓
                if current_price <= self.position_price * (1 - self.stop_loss_percent):
                    return "SELL", "止损"
                # 检查反转卖出
                if current_price >= levels['senter'] and abs(current_price - levels['senter']) >= self.min_price_move:
                    return "SELL", "反转卖出"
            else:  # 空头持仓
                if current_price >= self.position_price * (1 + self.stop_loss_percent):
                    return "BUY", "止损"
                # 检查反转买入
                if current_price <= levels['benter'] and abs(levels['benter'] - current_price) >= self.min_price_move:
                    return "BUY", "反转买入"
            
            # 检查最大持仓时间
            if self.position_time and (current_time - self.position_time).total_seconds() / 60 >= self.max_hold_minutes:
                if self.position > 0:
                    return "SELL", "超时平仓"
                else:
                    return "BUY", "超时平仓"
        
        # 如果没有持仓，检查开仓信号
        else:
            # 突破买入
            if current_price > levels['bbreak'] and abs(current_price - levels['bbreak']) >= self.min_price_move:
                return "BUY", "突破买入"
            # 突破卖出
            elif current_price < levels['sbreak'] and abs(levels['sbreak'] - current_price) >= self.min_price_move:
                return "SELL", "突破卖出"
        
        return signal, reason
    
    def execute_trade(self, signal: str, price: float, current_time: datetime, reason: str):
        """执行交易"""
        if signal == "HOLD":
            return
        
        quantity = 0
        amount = 0
        pnl = 0
        pnl_percent = 0
        hold_minutes = 0
        
        if signal == "BUY":
            if self.position <= 0:  # 开多仓或平空仓
                if self.position < 0:  # 平空仓
                    quantity = abs(self.position)
                    amount = quantity * price
                    pnl = (self.position_price - price) * quantity
                    pnl_percent = pnl / (self.position_price * quantity) * 100
                    if self.position_time:
                        hold_minutes = int((current_time - self.position_time).total_seconds() / 60)
                    self.position = 0
                else:  # 开多仓
                    quantity = self.max_position_size
                    amount = quantity * price
                    self.position = quantity
                    self.position_price = price
                    self.position_time = current_time
        
        elif signal == "SELL":
            if self.position >= 0:  # 平多仓或开空仓
                if self.position > 0:  # 平多仓
                    quantity = self.position
                    amount = quantity * price
                    pnl = (price - self.position_price) * quantity
                    pnl_percent = pnl / (self.position_price * quantity) * 100
                    if self.position_time:
                        hold_minutes = int((current_time - self.position_time).total_seconds() / 60)
                    self.position = 0
                else:  # 开空仓
                    quantity = self.max_position_size
                    amount = quantity * price
                    self.position = -quantity
                    self.position_price = price
                    self.position_time = current_time
        
        # 记录交易
        trade = Trade(
            datetime=current_time,
            symbol="BABA.US",
            action=signal,
            price=price,
            quantity=quantity,
            amount=amount,
            reason=reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_minutes=hold_minutes
        )
        
        self.trades.append(trade)
        self.last_trade_time = current_time  # 更新最后交易时间
        logger.info(f"{current_time}: {signal} {quantity}股 @{price:.2f} - {reason} (PnL: {pnl:.2f})")
    
    def run_backtest(self, symbol: str, start_date: date, end_date: date) -> Dict:
        """运行回测"""
        print(f"开始回测 {symbol}: {start_date} 到 {end_date}")
        
        # 获取分钟级数据
        minute_data = self.get_minute_data(symbol, start_date, end_date)
        if minute_data.empty:
            logger.error("无法获取数据，回测终止")
            return {}
        
        # 打印数据的日期范围
        logger.info(f"分钟数据日期范围: {minute_data.index.min()} 到 {minute_data.index.max()}")
        logger.info(f"分钟数据包含的日期: {sorted(set(minute_data.index.date))}")
        
        # 获取日线数据用于计算R-Breaker水平
        daily_data = self.get_daily_ohlc(minute_data)
        
        logger.info(f"数据准备完成: {len(minute_data)} 条分钟数据, {len(daily_data)} 个交易日")
        
        # 重置状态
        self.trades = []
        self.position = 0
        self.position_price = 0.0
        self.position_time = None
        
        # 打印所有可用的交易日期
        logger.info(f"可用的交易日期: {list(daily_data.index)}")
        
        # 按日期进行回测
        for current_date in daily_data.index[1:]:  # 从第二天开始，因为需要前一天的数据
            prev_date = daily_data.index[daily_data.index.get_loc(current_date) - 1]
            
            # 获取前一日的OHLC
            prev_high = daily_data.loc[prev_date, 'high']
            prev_low = daily_data.loc[prev_date, 'low']
            prev_close = daily_data.loc[prev_date, 'close']
            
            # 计算R-Breaker水平
            levels = self.calculate_rbreaker_levels(prev_high, prev_low, prev_close)
            
            logger.info(f"\n{current_date} R-Breaker水平:")
            for level_name, level_value in levels.items():
                logger.info(f"  {level_name}: {level_value:.2f}")
            
            # 获取当日分钟数据
            day_minute_data = minute_data[minute_data.index.date == current_date]
            
            if day_minute_data.empty:
                logger.info(f"{current_date}: 没有分钟数据")
                continue
            
            logger.info(f"{current_date}: 有 {len(day_minute_data)} 条分钟数据")
            
            # 遍历当日每分钟数据
            signal_count = 0
            for current_time, row in day_minute_data.iterrows():
                current_price = row['close']
                
                # 检查交易信号
                signal, reason = self.check_trading_signal(current_price, levels, current_time)
                
                # 执行交易
                if signal != "HOLD":
                    self.execute_trade(signal, current_price, current_time, reason)
                    signal_count += 1
            
            print(f"{current_date}: 当日产生 {signal_count} 个交易信号")
        
        # 如果最后还有持仓，强制平仓
        if self.position != 0:
            last_price = minute_data.iloc[-1]['close']
            last_time = minute_data.index[-1]
            if self.position > 0:
                self.execute_trade("SELL", last_price, last_time, "强制平仓")
            else:
                self.execute_trade("BUY", last_price, last_time, "强制平仓")
        
        # 生成回测报告
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """生成详细回测报告"""
        if not self.trades:
            return {"error": "没有交易记录"}
        
        # 基础统计指标
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        break_even_trades = len([t for t in self.trades if t.pnl == 0])
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = sum(t.pnl_percent for t in self.trades if t.pnl != 0)
        
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if profitable_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_profit * profitable_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        max_profit = max([t.pnl for t in self.trades]) if self.trades else 0
        max_loss = min([t.pnl for t in self.trades]) if self.trades else 0
        
        avg_hold_time = np.mean([t.hold_minutes for t in self.trades if t.hold_minutes > 0]) if self.trades else 0
        
        # 交易类型分析
        trade_types = {}
        for trade in self.trades:
            reason = trade.reason
            if reason not in trade_types:
                trade_types[reason] = {'count': 0, 'pnl': 0, 'wins': 0}
            trade_types[reason]['count'] += 1
            trade_types[reason]['pnl'] += trade.pnl
            if trade.pnl > 0:
                trade_types[reason]['wins'] += 1
        
        # 计算最大回撤
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        for trade in self.trades:
            cumulative_pnl += trade.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 每日统计
        daily_stats = {}
        for trade in self.trades:
            trade_date = trade.datetime.date()
            if trade_date not in daily_stats:
                daily_stats[trade_date] = {'trades': 0, 'pnl': 0, 'wins': 0}
            daily_stats[trade_date]['trades'] += 1
            daily_stats[trade_date]['pnl'] += trade.pnl
            if trade.pnl > 0:
                daily_stats[trade_date]['wins'] += 1
        
        # 风险指标和夏普比率计算
        daily_pnl = [stats['pnl'] for stats in daily_stats.values()]
        sharpe_ratio = 0
        annual_return = 0
        annual_volatility = 0
        
        if len(daily_pnl) > 1 and self.initial_capital > 0:
            # 计算每日收益率（百分比）
            daily_returns = [pnl / self.initial_capital for pnl in daily_pnl]
            
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns, ddof=1)  # 样本标准差
            
            # 年化收益率和波动率（假设252个交易日）
            annual_return = avg_daily_return * 252 * 100  # 转换为百分比
            annual_volatility = std_daily_return * np.sqrt(252) * 100  # 转换为百分比
            
            # 夏普比率（假设无风险利率为3%）
            risk_free_rate = 0.03
            sharpe_ratio = (annual_return/100 - risk_free_rate) / (annual_volatility/100) if annual_volatility != 0 else 0
        
        # 持仓收益对比分析
        buy_hold_return = 0
        buy_hold_pnl = 0
        strategy_vs_hold = 0
        alpha = 0
        
        # 如果有交易记录，计算买入持有策略收益
        if self.trades:
            # 使用第一笔交易的价格作为买入价，最后一笔交易的价格作为卖出价
            first_price = self.trades[0].price
            last_price = self.trades[-1].price
            buy_hold_return = ((last_price - first_price) / first_price) * 100
            buy_hold_pnl = (last_price - first_price) * self.max_position_size  # 假设买入最大持仓数量
            
            strategy_vs_hold = total_return - buy_hold_return
            alpha = strategy_vs_hold  # 超额收益
        
        return {
            "基础统计": {
                "总交易次数": total_trades,
                "盈利交易": profitable_trades,
                "亏损交易": losing_trades,
                "平局交易": break_even_trades,
                "胜率": f"{win_rate:.2f}%",
                "总盈亏": f"{total_pnl:.2f}",
                "总收益率": f"{total_return:.2f}%",
                "平均盈利": f"{avg_profit:.2f}",
                "平均亏损": f"{avg_loss:.2f}",
                "盈亏比": f"{profit_factor:.2f}",
                "最大盈利": f"{max_profit:.2f}",
                "最大亏损": f"{max_loss:.2f}",
                "平均持仓时间": f"{avg_hold_time:.1f}分钟"
            },
            "风险指标": {
                "最大回撤": f"{max_drawdown:.2f}",
                "夏普比率": f"{sharpe_ratio:.3f}",
                "年化收益率": f"{annual_return:.2f}%",
                "年化波动率": f"{annual_volatility:.2f}%",
                "交易天数": len(daily_stats),
                "平均每日交易": f"{total_trades/len(daily_stats):.1f}" if daily_stats else "0"
            },
            "收益对比": {
                "策略收益率": f"{total_return:.2f}%",
                "买入持有收益率": f"{buy_hold_return:.2f}%",
                "超额收益(Alpha)": f"{alpha:.2f}%",
                "策略盈亏": f"{total_pnl:.2f}",
                "持仓盈亏": f"{buy_hold_pnl:.2f}"
            },
            "交易类型分析": trade_types,
            "每日统计": daily_stats
        }
    
    def print_report(self, results: Dict):
        """打印策略统计报告"""
        print("\n" + "="*60)
        print("         BABA R-Breaker策略统计报告")
        print("="*60)
        
        # 打印基础统计
        print("\n📊 基础统计:")
        print("-"*40)
        for key, value in results["基础统计"].items():
            print(f"{key:12}: {value}")
        
        # 打印风险指标
        print("\n⚠️  风险指标:")
        print("-"*40)
        for key, value in results["风险指标"].items():
            print(f"{key:12}: {value}")
        
        # 打印收益对比
        print("\n💰 收益对比:")
        print("-"*40)
        for key, value in results["收益对比"].items():
            print(f"{key:12}: {value}")
        
        # 打印交易类型分析
        print("\n📈 交易类型分析:")
        print("-"*60)
        print(f"{'类型':15} {'次数':8} {'总盈亏':10} {'胜率':8}")
        print("-"*60)
        for reason, stats in results["交易类型分析"].items():
            win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f"{reason:15} {stats['count']:8} {stats['pnl']:10.2f} {win_rate:7.1f}%")
        
        # 打印每日统计（前10天）
        print("\n📅 每日统计 (前10天):")
        print("-"*50)
        print(f"{'日期':12} {'交易次数':8} {'盈亏':10} {'胜率':8}")
        print("-"*50)
        daily_items = list(results["每日统计"].items())[:10]
        for date, stats in daily_items:
            win_rate = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            print(f"{str(date):12} {stats['trades']:8} {stats['pnl']:10.2f} {win_rate:7.1f}%")
        
        if len(results["每日统计"]) > 10:
            print(f"... 还有 {len(results['每日统计']) - 10} 天数据")
        
        print(f"\n📋 总交易次数: {len(self.trades)} 笔")

def main():
    """主函数"""
    strategy = RBreakerStrategy()
    
    # 回测参数
    symbol = "QQQ.US"  # 阿里巴巴美股代码（longport格式）
    end_date = date.today()
    start_date = end_date - timedelta(days=600)  # 回测最近365天（1年）
    
    print(f"开始BABA R-Breaker策略回测")
    print(f"回测期间: {start_date} 到 {end_date}")
    print(f"策略参数:")
    print(f"  突破系数: {strategy.f1}")
    print(f"  观察系数: {strategy.f2}, {strategy.f4}")
    print(f"  反转系数: {strategy.f3}, {strategy.f5}")
    print(f"  止损比例: {strategy.stop_loss_percent*100}%")
    print(f"  最大持仓时间: {strategy.max_hold_minutes}分钟")
    
    # 运行回测
    results = strategy.run_backtest(symbol, start_date, end_date)
    
    if results:
        # 打印报告
        strategy.print_report(results)
    else:
        print("回测失败，请检查数据获取")

if __name__ == "__main__":
    main()