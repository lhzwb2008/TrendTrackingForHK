#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内反弹策略
专门针对9988阿里巴巴的日内大跌后反弹机会
策略逻辑：日内大跌后判断反弹时机买入，设置止盈止损，收盘前清仓
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, time
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from longport.openapi import QuoteContext, Config, Period, AdjustType
import time as time_module
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志（优化为一行显示）
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntradayTrade:
    """日内交易记录"""
    date: date
    symbol: str
    entry_time: str  # 进场时间
    exit_time: str   # 出场时间
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    exit_reason: str  # 止盈/止损/收盘清仓
    max_profit: float  # 最大盈利
    max_loss: float    # 最大亏损
    hold_minutes: int  # 持仓分钟数

class IntradayReversalStrategy:
    """日内反弹策略"""
    
    def __init__(self):
        """初始化策略"""
        self.config = Config.from_env()
        self.quote_ctx = QuoteContext(self.config)
        
        # 资金管理
        self.initial_capital = 100000     # 初始资金10万港币
        self.current_capital = self.initial_capital
        self.use_full_position = True     # 全仓操作
        
        # 目标股票
        self.target_symbol = "9988.HK"  # 阿里巴巴
        
        # 回测时间配置
        self.default_start_date = date(2024, 1, 1)
        self.default_end_date = date(2025, 1, 1)
        
        # 策略参数
        self.min_drop_percent = 0.03      # 最小跌幅3%触发关注
        self.reversal_confirm_percent = 0.005  # 反弹确认0.5%
        self.stop_loss_percent = 0.02     # 止损2%
        self.take_profit_percent = 0.05   # 止盈5%
        
        # 交易成本
        self.commission_rate = 0.0025     # 佣金费率0.25%
        self.stamp_duty_rate = 0.001      # 印花税0.1%（仅卖出）
        self.min_commission = 3.0         # 最低佣金3港币
        
        # 风险控制
        self.max_position_ratio = 0.95   # 最大仓位比例95%
        
        # 时间控制
        self.market_open = time(9, 30)    # 开盘时间
        self.market_close = time(16, 0)   # 收盘时间
        self.force_close_time = time(15, 45)  # 强制平仓时间
        self.min_hold_minutes = 5         # 最小持仓时间5分钟
        
        # 成交量确认
        self.min_volume_surge = 1.5       # 最小成交量放大1.5倍
        
        # 交易记录
        self.trades: List[IntradayTrade] = []
        self.current_position = None
        self.daily_stats = []
        
    def get_intraday_data(self, symbol: str, target_date: date) -> pd.DataFrame:
        """获取指定日期的真实分钟级数据"""
        try:
            # 直接获取分钟级历史数据
            candles = self.quote_ctx.history_candlesticks_by_date(
                symbol,
                Period.Min_1,  # 1分钟级别
                AdjustType.ForwardAdjust,
                target_date,
                target_date
            )
            
            if not candles:
                logger.warning(f"无法获取{symbol}在{target_date}的分钟级数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            data = []
            for candle in candles:
                # 处理时间戳（可能是datetime对象或时间戳）
                if isinstance(candle.timestamp, datetime):
                    timestamp = candle.timestamp
                else:
                    timestamp = datetime.fromtimestamp(candle.timestamp)
                
                # 只保留交易时间内的数据（9:30-16:00，排除12:00-13:00午休）
                if not self._is_trading_time(timestamp.time()):
                    continue
                    
                data.append({
                    'datetime': timestamp,
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': int(candle.volume),
                    'turnover': float(candle.turnover)
                })
            
            if not data:
                logger.warning(f"{symbol}在{target_date}没有交易时间内的数据")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            
            # 计算技术指标
            df = self.calculate_intraday_indicators(df)
            
            logger.info(f"成功获取{symbol}在{target_date}的{len(df)}条分钟级数据")
            return df
            
        except Exception as e:
            logger.error(f"获取{symbol}在{target_date}的数据失败: {e}")
            return pd.DataFrame()
    
    def _is_trading_time(self, time_obj: time) -> bool:
        """判断是否为交易时间"""
        # 上午：9:30-12:00
        morning_start = time(9, 30)
        morning_end = time(12, 0)
        
        # 下午：13:00-16:00
        afternoon_start = time(13, 0)
        afternoon_end = time(16, 0)
        
        return (morning_start <= time_obj < morning_end) or (afternoon_start <= time_obj <= afternoon_end)
    

    

    

    
    def calculate_intraday_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算日内技术指标（避免未来函数）"""
        if len(df) < 10:
            return df
        
        # 短期移动平均（使用shift确保不使用当前值）
        df['ma5'] = df['close'].shift(1).rolling(5, min_periods=5).mean()
        df['ma10'] = df['close'].shift(1).rolling(10, min_periods=10).mean()
        df['ma20'] = df['close'].shift(1).rolling(20, min_periods=20).mean()
        
        # 成交量指标（使用历史数据）
        df['volume_ma10'] = df['volume'].shift(1).rolling(10, min_periods=10).mean()
        df['volume_surge'] = df['volume'] / df['volume_ma10']
        
        # 价格变化（基于前一分钟）
        df['price_change'] = df['close'].pct_change()
        df['price_change_5min'] = df['close'].pct_change(5)
        
        # 从开盘的累计涨跌幅（使用当日开盘价）
        first_price = df['open'].iloc[0]  # 使用开盘价而不是第一个收盘价
        df['cumulative_return'] = (df['close'] / first_price - 1)
        
        # 振幅（基于前一分钟收盘价）
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        return df
    
    def check_drop_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        """检查大跌信号（避免未来函数）"""
        if current_idx < 20:  # 需要足够的历史数据
            return False, 0.0
        
        current_data = df.iloc[current_idx]
        
        # 检查从开盘的累计跌幅（使用当前已知价格）
        cumulative_drop = current_data['cumulative_return']
        
        # 大跌条件：累计跌幅超过阈值
        if cumulative_drop <= -self.min_drop_percent:
            # 确认成交量放大（使用历史平均成交量比较）
            if pd.notna(current_data['volume_surge']) and current_data['volume_surge'] >= self.min_volume_surge:
                return True, abs(cumulative_drop)
        
        return False, 0.0
    
    def check_reversal_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """检查反弹信号（避免未来函数）"""
        if current_idx < 20:  # 需要更多历史数据确保MA计算有效
            return False, ""
        
        current_data = df.iloc[current_idx]
        prev_data = df.iloc[current_idx - 1]
        
        # 反弹确认条件
        conditions = []
        
        # 1. 价格开始回升
        if current_data['price_change'] > self.reversal_confirm_percent:
            conditions.append("价格回升")
        
        # 2. 突破短期均线（确保MA值有效）
        if (pd.notna(current_data['ma5']) and pd.notna(prev_data['ma5']) and
            current_data['close'] > current_data['ma5'] and 
            prev_data['close'] <= prev_data['ma5']):
            conditions.append("突破MA5")
        
        # 3. 成交量配合（确保volume_surge有效）
        if pd.notna(current_data['volume_surge']) and current_data['volume_surge'] > 1.2:
            conditions.append("成交量配合")
        
        # 4. 技术面改善（连续2分钟上涨）
        if (current_idx >= 2 and 
            current_data['price_change'] > 0 and 
            df.iloc[current_idx - 1]['price_change'] > 0):
            conditions.append("连续上涨")
        
        # 至少满足2个条件
        if len(conditions) >= 2:
            return True, "; ".join(conditions)
        
        return False, ""
    
    def calculate_trading_cost(self, price: float, quantity: int, is_buy: bool) -> float:
        """计算交易成本"""
        trade_value = price * quantity
        
        # 佣金（买卖都有）
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 印花税（仅卖出）
        stamp_duty = trade_value * self.stamp_duty_rate if not is_buy else 0
        
        return commission + stamp_duty
    
    def calculate_position_size(self, price: float) -> int:
        """计算仓位大小 - 考虑交易成本"""
        if self.use_full_position:
            # 预留交易成本，不能真正全仓
            available_capital = self.current_capital * self.max_position_ratio
            
            # 估算买入成本
            estimated_shares = int(available_capital / price)
            estimated_shares = (estimated_shares // 100) * 100  # 港股100股整数倍
            
            # 计算实际交易成本
            if estimated_shares > 0:
                buy_cost = self.calculate_trading_cost(price, estimated_shares, True)
                total_cost = price * estimated_shares + buy_cost
                
                # 确保有足够资金
                if total_cost <= self.current_capital:
                    return estimated_shares
                else:
                    # 重新计算，减少股数
                    max_shares = int((self.current_capital - buy_cost) / price)
                    return (max_shares // 100) * 100
        else:
            # 固定仓位
            max_shares = int(50000 / price)
            return (max_shares // 100) * 100
        
        return 0
    
    def simulate_trading_day(self, target_date: date) -> Dict:
        """模拟单日交易"""
        # 获取日内数据
        df = self.get_intraday_data(self.target_symbol, target_date)
        if df.empty:
            return {'date': target_date, 'trades': 0, 'pnl': 0, 'capital': self.current_capital}
        
        # 交易状态
        position = None
        daily_trades = []
        looking_for_drop = True
        drop_detected = False
        
        # 遍历分钟数据
        for i in range(len(df)):
            current_time = df.index[i]
            current_data = df.iloc[i]
            current_price = current_data['close']
            
            # 检查是否在交易时间内
            if not (self.market_open <= current_time.time() <= self.market_close):
                continue
            
            # 强制平仓时间
            if (current_time.time() >= self.force_close_time and position):
                # 平仓
                exit_price = current_price
                sell_cost = self.calculate_trading_cost(exit_price, position['quantity'], False)
                pnl = (exit_price - position['entry_price']) * position['quantity'] - position['buy_cost'] - sell_cost
                pnl_percent = pnl / (position['entry_price'] * position['quantity']) * 100
                
                trade = IntradayTrade(
                    date=target_date,
                    symbol=self.target_symbol,
                    entry_time=position['entry_time'],
                    exit_time=current_time.strftime('%H:%M'),
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    quantity=position['quantity'],
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    exit_reason="收盘清仓",
                    max_profit=position['max_profit'],
                    max_loss=position['max_loss'],
                    hold_minutes=position['hold_minutes']
                )
                
                daily_trades.append(trade)
                # 更新资金（卖出 - 交易成本）
                self.current_capital += exit_price * position['quantity'] - sell_cost
                
                # 打印详细卖出信息
                logger.info(f"🔴 收盘清仓 {current_time.strftime('%H:%M')} | 价格:{exit_price:.2f} | 数量:{position['quantity']:,}股 | 金额:{exit_price*position['quantity']:,.0f} | 成本:{sell_cost:.2f} | 盈亏:{pnl:+.0f} ({pnl_percent:+.2f}%) | 总资金:{self.current_capital:,.0f}")
                
                position = None
                break
            
            # 无持仓时寻找机会
            if not position:
                # 检查大跌信号
                if looking_for_drop:
                    is_drop, drop_magnitude = self.check_drop_signal(df, i)
                    if is_drop:
                        drop_detected = True
                        looking_for_drop = False
                        logger.info(f"{current_time.strftime('%H:%M')} 检测到大跌: {drop_magnitude:.2%}")
                
                # 在大跌后寻找反弹机会
                if drop_detected:
                    is_reversal, reversal_reason = self.check_reversal_signal(df, i)
                    if is_reversal:
                        # 买入
                        entry_price = current_price
                        quantity = self.calculate_position_size(entry_price)
                        
                        if quantity > 0:
                            # 计算买入成本
                            buy_cost = self.calculate_trading_cost(entry_price, quantity, True)
                            total_cost = entry_price * quantity + buy_cost
                            
                            if total_cost <= self.current_capital:
                                position = {
                                    'entry_time': current_time.strftime('%H:%M'),
                                    'entry_datetime': current_time,  # 记录完整的买入时间
                                    'entry_price': entry_price,
                                    'quantity': quantity,
                                    'buy_cost': buy_cost,
                                    'max_profit': 0,
                                    'max_loss': 0,
                                    'hold_minutes': 0
                                }
                                
                                # 更新资金（买入 + 交易成本）
                                self.current_capital -= total_cost
                                
                                # 打印详细买入信息
                                logger.info(f"🟢 买入信号 {current_time.strftime('%H:%M')} | 价格:{entry_price:.2f} | 数量:{quantity:,}股 | 金额:{entry_price*quantity:,.0f} | 成本:{buy_cost:.2f} | 原因:{reversal_reason} | 剩余资金:{self.current_capital:,.0f}")
                                
                                drop_detected = False  # 重置状态
            
            # 有持仓时检查出场信号
            else:
                # 计算实际持仓时间（分钟）
                hold_time_delta = current_time - position['entry_datetime']
                actual_hold_minutes = hold_time_delta.total_seconds() / 60
                position['hold_minutes'] = int(actual_hold_minutes)
                
                # 检查最小持仓时间
                if actual_hold_minutes < self.min_hold_minutes:
                    continue  # 未达到最小持仓时间，跳过出场检查
                
                # 更新最大盈亏（考虑交易成本）
                sell_cost = self.calculate_trading_cost(current_price, position['quantity'], False)
                current_pnl = (current_price - position['entry_price']) * position['quantity'] - position['buy_cost'] - sell_cost
                position['max_profit'] = max(position['max_profit'], current_pnl)
                position['max_loss'] = min(position['max_loss'], current_pnl)
                
                # 检查止盈
                profit_percent = (current_price / position['entry_price'] - 1)
                if profit_percent >= self.take_profit_percent:
                    exit_price = current_price
                    sell_cost = self.calculate_trading_cost(exit_price, position['quantity'], False)
                    pnl = (exit_price - position['entry_price']) * position['quantity'] - position['buy_cost'] - sell_cost
                    pnl_percent = pnl / (position['entry_price'] * position['quantity']) * 100
                    
                    trade = IntradayTrade(
                        date=target_date,
                        symbol=self.target_symbol,
                        entry_time=position['entry_time'],
                        exit_time=current_time.strftime('%H:%M'),
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        exit_reason="止盈",
                        max_profit=position['max_profit'],
                        max_loss=position['max_loss'],
                        hold_minutes=position['hold_minutes']
                    )
                    
                    daily_trades.append(trade)
                    # 更新资金（卖出 - 交易成本）
                    self.current_capital += exit_price * position['quantity'] - sell_cost
                    
                    # 打印详细卖出信息
                    logger.info(f"🟢 止盈出场 {current_time.strftime('%H:%M')} | 价格:{exit_price:.2f} | 数量:{position['quantity']:,}股 | 金额:{exit_price*position['quantity']:,.0f} | 成本:{sell_cost:.2f} | 盈亏:{pnl:+.0f} ({pnl_percent:+.2f}%) | 总资金:{self.current_capital:,.0f}")
                    
                    position = None
                    looking_for_drop = True  # 重新寻找机会
                
                # 检查止损
                elif profit_percent <= -self.stop_loss_percent:
                    exit_price = current_price
                    sell_cost = self.calculate_trading_cost(exit_price, position['quantity'], False)
                    pnl = (exit_price - position['entry_price']) * position['quantity'] - position['buy_cost'] - sell_cost
                    pnl_percent = pnl / (position['entry_price'] * position['quantity']) * 100
                    
                    trade = IntradayTrade(
                        date=target_date,
                        symbol=self.target_symbol,
                        entry_time=position['entry_time'],
                        exit_time=current_time.strftime('%H:%M'),
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        exit_reason="止损",
                        max_profit=position['max_profit'],
                        max_loss=position['max_loss'],
                        hold_minutes=position['hold_minutes']
                    )
                    
                    daily_trades.append(trade)
                    # 更新资金（卖出 - 交易成本）
                    self.current_capital += exit_price * position['quantity'] - sell_cost
                    
                    # 打印详细卖出信息
                    logger.info(f"🔴 止损出场 {current_time.strftime('%H:%M')} | 价格:{exit_price:.2f} | 数量:{position['quantity']:,}股 | 金额:{exit_price*position['quantity']:,.0f} | 成本:{sell_cost:.2f} | 盈亏:{pnl:+.0f} ({pnl_percent:+.2f}%) | 总资金:{self.current_capital:,.0f}")
                    
                    position = None
                    looking_for_drop = True  # 重新寻找机会
        
        # 统计当日结果
        daily_pnl = sum(trade.pnl for trade in daily_trades)
        
        # 保存交易记录
        self.trades.extend(daily_trades)
        
        return {
            'date': target_date,
            'trades': len(daily_trades),
            'pnl': daily_pnl,
            'capital': self.current_capital,
            'details': daily_trades
        }
    
    def run_backtest(self, start_date: date = None, end_date: date = None) -> Dict:
        """运行回测"""
        # 使用默认时间范围
        if start_date is None:
            start_date = self.default_start_date
        if end_date is None:
            end_date = self.default_end_date
            
        logger.info(f"开始回测: {start_date} 到 {end_date}, 初始资金: {self.initial_capital:,.0f}港币")
        
        # 重置状态
        self.trades = []
        self.daily_stats = []
        self.current_capital = self.initial_capital
        
        # 获取交易日列表
        trading_days = self.get_trading_days(start_date, end_date)
        
        total_pnl = 0
        trading_days_count = 0
        
        for trading_day in trading_days:
            try:
                daily_result = self.simulate_trading_day(trading_day)
                self.daily_stats.append(daily_result)
                
                total_pnl += daily_result['pnl']
                trading_days_count += 1
                
                # 优化日志显示：一天的交易显示在一行
                if daily_result['trades'] > 0:
                    trades_info = []
                    for trade in daily_result['details']:
                        trades_info.append(f"{trade.entry_time}-{trade.exit_time}({trade.exit_reason}:{trade.pnl:+.0f})")
                    logger.info(f"{trading_day}: {daily_result['trades']}笔 [{', '.join(trades_info)}] 日盈亏:{daily_result['pnl']:+.0f} 总资金:{daily_result['capital']:,.0f}")
                
            except Exception as e:
                logger.error(f"{trading_day} 交易模拟失败: {e}")
                continue
        
        # 生成回测报告
        final_capital = self.current_capital
        results = self.generate_backtest_report(total_pnl, trading_days_count)
        results['initial_capital'] = self.initial_capital
        results['final_capital'] = final_capital
        
        return results
    
    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """获取交易日列表（简化版，实际应该排除节假日）"""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # 排除周末
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def generate_backtest_report(self, total_pnl: float, trading_days: int) -> Dict:
        """生成详细回测报告"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_pnl_per_trade': 0,
                'max_profit': 0,
                'max_loss': 0,
                'total_return_percent': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_loss_ratio': 0
            }
        
        # 基本统计
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # 最大盈亏
        max_profit = max(t.pnl for t in self.trades) if self.trades else 0
        max_loss = min(t.pnl for t in self.trades) if self.trades else 0
        
        # 计算收益率
        total_return_percent = (total_pnl / self.initial_capital) * 100
        
        # 计算平均盈亏
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # 计算盈亏比
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 计算最大连续盈利/亏损
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # 计算最大回撤
        cumulative_capital = self.initial_capital
        peak_capital = self.initial_capital
        max_drawdown = 0
        daily_returns = []
        
        for trade in self.trades:
            cumulative_capital += trade.pnl
            if cumulative_capital > peak_capital:
                peak_capital = cumulative_capital
            drawdown = (peak_capital - cumulative_capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
            daily_returns.append(trade.pnl / self.initial_capital)
        
        # 计算夏普比率（简化版本，假设无风险利率为0）
        if daily_returns and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0
        
        # 计算交易日比例
        trading_day_ratio = len(set(t.date for t in self.trades)) / trading_days * 100 if trading_days > 0 else 0
        
        # 按退出原因统计
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += trade.pnl
        
        # 平均持仓时间
        avg_hold_minutes = np.mean([t.hold_minutes for t in self.trades]) if self.trades else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_hold_minutes': avg_hold_minutes,
            'exit_reasons': exit_reasons,
            'trading_days': trading_days,
            'total_return_percent': total_return_percent,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown * 100,  # 转换为百分比
            'sharpe_ratio': sharpe_ratio,
            'trading_day_ratio': trading_day_ratio
        }
    
    def print_detailed_report(self, results: Dict):
        """打印详细报告"""
        print("\n" + "="*80)
        print("           日内反弹策略回测报告")
        print("="*80)
        
        print(f"\n💰 资金统计:")
        print(f"   初始资金: {results['initial_capital']:,.0f} 港币")
        print(f"   最终资金: {results['final_capital']:,.0f} 港币")
        print(f"   总盈亏: {results['total_pnl']:+,.0f} 港币")
        print(f"   总收益率: {results['total_return_percent']:+.2f}%")
        
        print(f"\n📊 基本统计:")
        print(f"   总交易次数: {results['total_trades']}")
        print(f"   胜率: {results['win_rate']:.1f}% ({results['winning_trades']}/{results['total_trades']})")
        print(f"   平均每笔盈亏: {results['avg_pnl_per_trade']:+,.0f} 港币")
        print(f"   平均盈利: {results['avg_win']:+,.0f} 港币")
        print(f"   平均亏损: {results['avg_loss']:+,.0f} 港币")
        print(f"   盈亏比: {results['profit_loss_ratio']:.2f}")
        print(f"   平均持仓时间: {results['avg_hold_minutes']:.1f} 分钟")
        
        print(f"\n📈 风险指标:")
        print(f"   最大回撤: {results['max_drawdown']:.2f}%")
        print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"   最大单笔盈利: {results['max_profit']:+,.0f} 港币")
        print(f"   最大单笔亏损: {results['max_loss']:+,.0f} 港币")
        print(f"   最大连续盈利: {results['max_consecutive_wins']} 笔")
        print(f"   最大连续亏损: {results['max_consecutive_losses']} 笔")
        
        print(f"\n📅 交易统计:")
        print(f"   交易天数: {results['trading_days']} 天")
        print(f"   交易日比例: {results['trading_day_ratio']:.1f}%")
        
        print(f"\n🚪 退出原因统计:")
        for reason, stats in results['exit_reasons'].items():
            avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
            print(f"   {reason}: {stats['count']} 笔, 总盈亏: {stats['pnl']:+,.0f}, 平均: {avg_pnl:+,.0f}")
        
        if self.trades:
            print(f"\n📋 最近5笔交易:")
            for trade in self.trades[-5:]:
                print(f"   {trade.date} {trade.entry_time}-{trade.exit_time}: "
                      f"{trade.pnl:+.0f} ({trade.pnl_percent:+.2f}%) - {trade.exit_reason}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    strategy = IntradayReversalStrategy()
    
    # 设置回测期间
    start_date = date(2024, 1, 1)
    end_date = date(2025, 3, 31)  # 3个月回测
    
    print(f"🚀 开始日内反弹策略回测")
    print(f"📅 回测期间: {start_date} 至 {end_date}")
    print(f"🎯 目标股票: {strategy.target_symbol} (阿里巴巴)")
    print(f"💰 初始资金: {strategy.initial_capital:,} 港币 (全仓操作)")
    print(f"📉 大跌阈值: {strategy.min_drop_percent:.1%}")
    print(f"📈 反弹确认: {strategy.reversal_confirm_percent:.1%}")
    print(f"🛑 止损: {strategy.stop_loss_percent:.1%}")
    print(f"🎯 止盈: {strategy.take_profit_percent:.1%}")
    
    # 运行回测
    results = strategy.run_backtest(start_date, end_date)
    
    # 打印报告
    strategy.print_detailed_report(results)
    
    # 保存交易记录
    if strategy.trades:
        trades_df = pd.DataFrame([
            {
                'date': t.date,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'exit_reason': t.exit_reason,
                'hold_minutes': t.hold_minutes
            }
            for t in strategy.trades
        ])
        
        trades_df.to_csv('intraday_reversal_trades.csv', index=False)
        print(f"\n💾 交易记录已保存到 intraday_reversal_trades.csv")

if __name__ == "__main__":
    main()