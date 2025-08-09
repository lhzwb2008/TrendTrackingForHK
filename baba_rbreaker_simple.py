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
    commission: float = 0.0  # 交易费用

@dataclass
class StrategyConfig:
    """策略配置参数"""
    # 交易标的设置
    symbol: str = "QQQ.US"  # 交易标的代码
    
    # R-Breaker策略参数
    f1: float = 0.5   # 突破买入系数（提高阈值减少假突破）
    f2: float = 0.15  # 观察卖出系数
    f3: float = 0.4   # 反转卖出系数（提高阈值减少频繁交易）
    f4: float = 0.15  # 观察买入系数
    f5: float = 0.3   # 反转买入系数（适度提高）
    
    # 交易控制参数
    initial_capital: float = 100000  # 初始资金10万美元
    stop_loss_percent: float = 0.02   # 止损2%（放宽一点减少止损频率）
    max_hold_minutes: int = 300       # 最大持仓时间5小时（延长持仓）
    min_price_move: float = 0.25      # 最小价格变动阈值（提高过滤噪音）
    cooldown_minutes: int = 30        # 交易冷却时间30分钟（大幅延长）
    
    # 费率设置
    commission_per_share: float = 0.01  # 每股交易费用0.01美元
    
    # 回测设置
    backtest_days: int = 600  # 回测天数
    
    # 连接设置
    max_retries: int = 3  # API连接最大重试次数
    retry_delay: float = 2.0  # 重试间隔秒数
    
    def print_config(self):
        """打印配置参数"""
        print(f"交易标的: {self.symbol}")
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"R-Breaker参数: f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4}, f5={self.f5}")
        print(f"止损比例: {self.stop_loss_percent*100:.1f}%")
        print(f"最大持仓时间: {self.max_hold_minutes}分钟")
        print(f"最小价格变动: ${self.min_price_move}")
        print(f"交易冷却时间: {self.cooldown_minutes}分钟")
        print(f"每股手续费: ${self.commission_per_share}")
        print(f"回测天数: {self.backtest_days}天")
        print(f"最大重试次数: {self.max_retries}次")
        print(f"重试间隔: {self.retry_delay}秒")

class RBreakerStrategy:
    """R-Breaker日内交易策略"""
    
    def __init__(self, config: StrategyConfig = None):
        """初始化策略"""
        self.longport_config = Config.from_env()
        self.quote_ctx = QuoteContext(self.longport_config)
        
        # 策略配置
        self.config = config if config else StrategyConfig()
        
        # 资金管理
        self.current_capital = self.config.initial_capital  # 当前可用资金
        self.total_commission = 0.0  # 总交易费用
        
        # 回测数据
        self.trades: List[Trade] = []
        self.position = 0  # 当前持仓数量（正数为多头，负数为空头）
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
            
            # 重试机制
            success = False
            for retry in range(self.config.max_retries):
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
                        success = True
                        break
                    else:
                        logger.warning(f"批次 {current_date}-{batch_end_date}: API返回空数据")
                        success = True
                        break
                    
                except Exception as e:
                    logger.error(f"获取批次数据失败 {current_date}-{batch_end_date} (重试 {retry+1}/{self.config.max_retries}): {e}")
                    if retry < self.config.max_retries - 1:
                        logger.info(f"等待 {self.config.retry_delay} 秒后重试...")
                        time.sleep(self.config.retry_delay)
                    else:
                        logger.error(f"批次 {current_date}-{batch_end_date}: 重试次数已用完，跳过此批次")
            
            if success:
                # 添加延迟避免API限制
                time.sleep(0.5)
            
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
            'bbreak': prev_high + self.config.f1 * (prev_close - prev_low),      # 突破买入价
            'ssetup': pivot + self.config.f2 * (prev_high - prev_low),           # 观察卖出价
            'senter': (1 + self.config.f3) * pivot - self.config.f3 * prev_low,        # 反转卖出价
            'benter': (1 + self.config.f5) * pivot - self.config.f5 * prev_high,       # 反转买入价
            'bsetup': pivot - self.config.f4 * (prev_high - prev_low),           # 观察买入价
            'sbreak': prev_low - self.config.f1 * (prev_high - prev_close)       # 突破卖出价
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
            if minutes_since_last_trade < self.config.cooldown_minutes:
                return "HOLD", "冷却时间"
        
        # 如果有持仓，检查平仓信号
        if self.position != 0:
            # 检查止损
            if self.position > 0:  # 多头持仓
                if current_price <= self.position_price * (1 - self.config.stop_loss_percent):
                    return "SELL", "止损"
                # 检查反转卖出
                if current_price >= levels['senter'] and abs(current_price - levels['senter']) >= self.config.min_price_move:
                    return "SELL", "反转卖出"
            else:  # 空头持仓
                if current_price >= self.position_price * (1 + self.config.stop_loss_percent):
                    return "BUY", "止损"
                # 检查反转买入
                if current_price <= levels['benter'] and abs(levels['benter'] - current_price) >= self.config.min_price_move:
                    return "BUY", "反转买入"
            
            # 检查最大持仓时间
            if self.position_time and (current_time - self.position_time).total_seconds() / 60 >= self.config.max_hold_minutes:
                if self.position > 0:
                    return "SELL", "超时平仓"
                else:
                    return "BUY", "超时平仓"
        
        # 如果没有持仓，检查开仓信号
        else:
            # 突破买入
            if current_price > levels['bbreak'] and abs(current_price - levels['bbreak']) >= self.config.min_price_move:
                return "BUY", "突破买入"
            # 突破卖出
            elif current_price < levels['sbreak'] and abs(levels['sbreak'] - current_price) >= self.config.min_price_move:
                return "SELL", "突破卖出"
        
        return signal, reason
    
    def execute_trade(self, signal: str, price: float, current_time: datetime, reason: str):
        """执行交易（全仓交易）"""
        if signal == "HOLD":
            return
        
        quantity = 0
        amount = 0
        pnl = 0.0  # 开仓时pnl为0
        pnl_percent = 0.0  # 开仓时pnl_percent为0
        hold_minutes = 0  # 开仓时hold_minutes为0
        commission = 0
        
        if signal == "BUY":
            if self.position <= 0:  # 开多仓或平空仓
                if self.position < 0:  # 平空仓
                    quantity = abs(self.position)
                    amount = quantity * price
                    commission = quantity * self.config.commission_per_share
                    pnl = (self.position_price - price) * quantity - commission
                    pnl_percent = pnl / (self.position_price * quantity) * 100
                    if self.position_time:
                        hold_minutes = int((current_time - self.position_time).total_seconds() / 60)
                    
                    # 更新资金：平空仓后资金 = 当前资金 + 原保证金 + 盈亏
                    original_margin = self.position_price * quantity
                    self.current_capital += original_margin + pnl
                    self.position = 0
                    self.position_price = 0.0
                    self.position_time = None
                else:  # 开多仓（全仓）
                    # 计算能买入的最大股数（考虑手续费）
                    max_quantity = int(self.current_capital / (price + self.config.commission_per_share))
                    if max_quantity > 0:
                        quantity = max_quantity
                        amount = quantity * price
                        commission = quantity * self.config.commission_per_share
                        total_cost = amount + commission
                        
                        self.current_capital -= total_cost
                        self.position = quantity
                        self.position_price = price
                        self.position_time = current_time
                    else:
                        return  # 资金不足，不执行交易
        
        elif signal == "SELL":
            if self.position >= 0:  # 平多仓或开空仓
                if self.position > 0:  # 平多仓
                    quantity = self.position
                    amount = quantity * price
                    commission = quantity * self.config.commission_per_share
                    pnl = (price - self.position_price) * quantity - commission
                    pnl_percent = pnl / (self.position_price * quantity) * 100
                    if self.position_time:
                        hold_minutes = int((current_time - self.position_time).total_seconds() / 60)
                    
                    # 更新资金：平多仓后资金 = 当前资金 + 卖出金额 - 手续费
                    self.current_capital += amount - commission
                    self.position = 0
                    self.position_price = 0.0
                    self.position_time = None
                else:  # 开空仓（全仓）
                    max_quantity = int(self.current_capital / (price + self.config.commission_per_share))
                    if max_quantity > 0:
                        quantity = max_quantity
                        amount = quantity * price
                        commission = quantity * self.config.commission_per_share
                        total_margin = amount + commission
                        
                        # 做空：冻结保证金和手续费
                        self.current_capital -= total_margin
                        self.position = -quantity
                        self.position_price = price
                        self.position_time = current_time
                    else:
                        return  # 资金不足，不执行交易
        
        # 更新总手续费
        self.total_commission += commission
        
        # 记录交易
        trade = Trade(
            datetime=current_time,
            symbol=self.config.symbol,
            action=signal,
            price=price,
            quantity=quantity,
            amount=amount,
            reason=reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_minutes=hold_minutes,
            commission=commission
        )
        
        self.trades.append(trade)
        self.last_trade_time = current_time  # 更新最后交易时间
        logger.info(f"{current_time}: {signal} {quantity}股 @{price:.2f} - {reason} (PnL: {pnl:.2f}, 手续费: {commission:.2f}, 可用资金: {self.current_capital:.2f})")
    
    def run_backtest(self, symbol: str, start_date: date, end_date: date) -> Dict:
        """运行回测"""
        print(f"开始回测 {symbol}: {start_date} 到 {end_date}")
        
        # 获取分钟级数据
        minute_data = self.get_minute_data(symbol, start_date, end_date)
        if minute_data.empty:
            logger.error("无法获取数据，回测终止")
            return {}
        
        # 获取日线数据用于计算R-Breaker水平
        daily_data = self.get_daily_ohlc(minute_data)
        
        # 重置状态
        self.trades = []
        self.position = 0
        self.position_price = 0.0
        self.position_time = None
        
        # 按日期进行回测
        for current_date in daily_data.index[1:]:  # 从第二天开始，因为需要前一天的数据
            prev_date = daily_data.index[daily_data.index.get_loc(current_date) - 1]
            
            # 获取前一日的OHLC
            prev_high = daily_data.loc[prev_date, 'high']
            prev_low = daily_data.loc[prev_date, 'low']
            prev_close = daily_data.loc[prev_date, 'close']
            
            # 计算R-Breaker水平
            levels = self.calculate_rbreaker_levels(prev_high, prev_low, prev_close)
            
            # 获取当日分钟数据
            day_minute_data = minute_data[minute_data.index.date == current_date]
            
            if day_minute_data.empty:
                continue
            
            # 遍历当日每分钟数据
            for current_time, row in day_minute_data.iterrows():
                current_price = row['close']
                
                # 检查交易信号
                signal, reason = self.check_trading_signal(current_price, levels, current_time)
                
                # 执行交易
                if signal != "HOLD":
                    self.execute_trade(signal, current_price, current_time, reason)
        
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
        
        # 多空统计分析
        long_trades = 0  # 做多交易次数
        short_trades = 0  # 做空交易次数
        long_pnl = 0  # 做多总盈亏
        short_pnl = 0  # 做空总盈亏
        long_wins = 0  # 做多盈利次数
        short_wins = 0  # 做空盈利次数
        
        for trade in self.trades:
            if trade.action == "BUY":
                long_trades += 1
                long_pnl += trade.pnl
                if trade.pnl > 0:
                    long_wins += 1
            elif trade.action == "SELL":
                short_trades += 1
                short_pnl += trade.pnl
                if trade.pnl > 0:
                    short_wins += 1
        
        # 计算多空比例和胜率
        long_ratio = (long_trades / total_trades * 100) if total_trades > 0 else 0
        short_ratio = (short_trades / total_trades * 100) if total_trades > 0 else 0
        long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
        short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
        
        # 计算平均盈亏
        avg_long_pnl = long_pnl / long_trades if long_trades > 0 else 0
        avg_short_pnl = short_pnl / short_trades if short_trades > 0 else 0
        
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
        
        if len(daily_pnl) > 1 and self.config.initial_capital > 0:
            # 计算每日收益率（百分比）
            daily_returns = [pnl / self.config.initial_capital for pnl in daily_pnl]
            
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
            # 计算买入持有策略的股数（使用初始资金全仓买入）
            shares_bought = int(self.config.initial_capital / first_price)
            buy_hold_pnl = (last_price - first_price) * shares_bought
            
            strategy_vs_hold = total_return - buy_hold_return
            alpha = strategy_vs_hold  # 超额收益
        
        # 计算费率统计
        total_commission = sum(t.commission for t in self.trades)
        commission_percent = (total_commission / self.config.initial_capital) * 100 if self.config.initial_capital > 0 else 0
        
        # 计算净收益（扣除费率后）
        net_pnl = total_pnl - total_commission
        net_return = (net_pnl / self.config.initial_capital) * 100 if self.config.initial_capital > 0 else 0
        
        # 最终资金
        final_capital = self.current_capital
        capital_return = ((final_capital - self.config.initial_capital) / self.config.initial_capital) * 100 if self.config.initial_capital > 0 else 0
        
        return {
            "资金管理": {
                "初始资金": f"{self.config.initial_capital:.2f}",
                "最终资金": f"{final_capital:.2f}",
                "资金收益率": f"{capital_return:.2f}%",
                "总交易费用": f"{total_commission:.2f}",
                "费率占比": f"{commission_percent:.3f}%",
                "净盈亏": f"{net_pnl:.2f}",
                "净收益率": f"{net_return:.2f}%"
            },
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
            "多空统计": {
                "做多交易次数": long_trades,
                "做空交易次数": short_trades,
                "做多比例": f"{long_ratio:.2f}%",
                "做空比例": f"{short_ratio:.2f}%",
                "做多总盈亏": f"{long_pnl:.2f}",
                "做空总盈亏": f"{short_pnl:.2f}",
                "做多胜率": f"{long_win_rate:.2f}%",
                "做空胜率": f"{short_win_rate:.2f}%",
                "做多平均盈亏": f"{avg_long_pnl:.2f}",
                "做空平均盈亏": f"{avg_short_pnl:.2f}"
            },
            "每日统计": daily_stats
        }
    
    def print_report(self, results: Dict):
        """打印策略统计报告"""
        print("\n" + "="*60)
        print("         BABA R-Breaker策略统计报告")
        print("="*60)
        
        # 首先显示最终资金状况
        print("\n💰 最终资金状况:")
        print("-"*40)
        for key, value in results["资金管理"].items():
            print(f"{key:12}: {value}")
        
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
        
        # 打印多空统计
        print("\n📊 多空统计:")
        print("-"*40)
        for key, value in results["多空统计"].items():
            print(f"{key:12}: {value}")
        
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
        
        # 单独打印总手续费
        print("\n💰 手续费统计:")
        print("=" * 50)
        print(f"总交易次数: {len(self.trades)}")
        print(f"总手续费消耗: ${self.total_commission:.2f}")
        print(f"平均每笔手续费: ${self.total_commission/len(self.trades):.2f}" if len(self.trades) > 0 else "平均每笔手续费: $0.00")

def main():
    """主函数"""
    # 创建策略配置
    config = StrategyConfig()
    
    # 打印配置参数
    print("🔧 策略配置参数:")
    print("="*50)
    config.print_config()
    print("="*50)
    
    # 创建策略实例
    strategy = RBreakerStrategy(config)
    
    # 回测参数
    symbol = config.symbol  # 使用配置中的交易标的
    end_date = date.today()
    start_date = end_date - timedelta(days=config.backtest_days)  # 使用配置中的回测天数
    
    print(f"开始 {symbol} R-Breaker策略回测")
    print(f"回测期间: {start_date} 到 {end_date}")
    print(f"策略参数:")
    print(f"  突破系数: {config.f1}")
    print(f"  观察系数: {config.f2}, {config.f4}")
    print(f"  反转系数: {config.f3}, {config.f5}")
    print(f"  止损比例: {config.stop_loss_percent*100}%")
    print(f"  最大持仓时间: {config.max_hold_minutes}分钟")
    print(f"  最大重试次数: {config.max_retries}")
    print(f"  重试间隔: {config.retry_delay}秒")
    
    # 运行回测
    results = strategy.run_backtest(symbol, start_date, end_date)
    
    if results:
        # 打印报告
        strategy.print_report(results)
    else:
        print("回测失败，请检查数据获取")

if __name__ == "__main__":
    main()