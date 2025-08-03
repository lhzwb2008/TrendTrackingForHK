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
class DailyTrade:
    """日级别交易记录"""
    entry_date: date
    exit_date: date
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    exit_reason: str  # 止盈/止损/时间止损
    max_profit: float  # 最大盈利
    max_loss: float    # 最大亏损
    hold_days: int     # 持仓天数

class DailyReversalStrategy:
    """日级别暴跌反弹策略（支持多日持仓）"""
    
    def __init__(self):
        """初始化策略"""
        # ========================================
        # 📊 策略配置中心 - 所有参数集中管理
        # ========================================
        
        # 🎯 目标股票配置
        self.target_symbol = "9988.HK"  # 阿里巴巴
        
        # 📅 回测时间配置
        self.backtest_start_date = date(2024, 1, 1)   # 回测开始日期
        self.backtest_end_date = date(2024, 12, 31)   # 回测结束日期
        
        # 💰 资金管理配置
        self.initial_capital = 100000     # 初始资金10万港币
        self.use_full_position = True     # 全仓操作
        self.max_position_ratio = 0.95   # 最大仓位比例95%
        
        # 📈 核心策略参数（最优配置）
        self.min_drop_percent = 0.05      # 最小回撤5%触发关注（从20日高点）
        self.severe_drop_percent = 0.07   # 严重回撤7%（更强信号，从20日高点）
        self.stop_loss_percent = 0.08     # 止损8%（放宽止损，避免周末跳空被误杀）
        self.take_profit_percent = 0.20   # 止盈20%（提高目标，适合持仓过周）
        
        # ⏰ 持仓时间控制
        self.max_hold_days = 21           # 最大持仓天数3周（支持持仓过周）
        self.min_hold_days = 2            # 最小持仓天数2天（避免过于频繁交易）
        self.weekend_hold_enabled = True  # 允许持仓过周末
        
        # 📊 成交量确认参数
        self.min_volume_surge = 1.5       # 最小成交量放大1.5倍
        self.severe_volume_surge = 2.0    # 严重回撤时成交量放大2倍
        
        # 🛡️ 风险控制参数
        self.trailing_stop_enabled = True # 启用移动止损
        self.trailing_stop_percent = 0.06 # 移动止损6%（从最高点回撤）
        
        # 💸 交易成本配置
        self.commission_rate = 0.0025     # 佣金费率0.25%
        self.stamp_duty_rate = 0.001      # 印花税0.1%（仅卖出）
        self.min_commission = 3.0         # 最低佣金3港币
        
        # ========================================
        # 系统初始化（无需修改）
        # ========================================
        self.config = Config.from_env()
        self.quote_ctx = QuoteContext(self.config)
        self.current_capital = self.initial_capital
        self.default_start_date = self.backtest_start_date
        self.default_end_date = self.backtest_end_date
        
        # 交易记录
        self.trades: List[DailyTrade] = []
        self.current_position = None
        self.daily_stats = []
        
    def get_daily_data(self, symbol: str, start_date: date, end_date: date, max_retries: int = 3) -> pd.DataFrame:
        """获取指定日期范围的日线数据（带重试机制）"""
        for attempt in range(max_retries):
            try:
                # 获取日线历史数据
                candles = self.quote_ctx.history_candlesticks_by_date(
                    symbol,
                    Period.Day,  # 日线级别
                    AdjustType.ForwardAdjust,
                    start_date,
                    end_date
                )
                
                if not candles:
                    if attempt < max_retries - 1:
                        logger.warning(f"第{attempt + 1}次尝试：无法获取{symbol}的日线数据，将重试...")
                        time_module.sleep(1)  # 等待1秒后重试
                        continue
                    else:
                        logger.debug(f"无法获取{symbol}的日线数据")
                        return pd.DataFrame()
                
                # 成功获取数据，跳出重试循环
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"第{attempt + 1}次尝试获取{symbol}的日线数据失败: {e}，将重试...")
                    time_module.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    logger.debug(f"获取{symbol}的日线数据失败: {e}")
                    return pd.DataFrame()
        
        # 转换为DataFrame
        data = []
        for candle in candles:
            # 处理时间戳（可能是datetime对象或时间戳）
            if isinstance(candle.timestamp, datetime):
                timestamp = candle.timestamp
            else:
                timestamp = datetime.fromtimestamp(candle.timestamp)
                
            data.append({
                'date': timestamp.date(),
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': int(candle.volume),
                'turnover': float(candle.turnover)
            })
        
        if not data:
            logger.warning(f"{symbol}没有日线数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # 计算技术指标
        df = self.calculate_daily_indicators(df)
        
        logger.info(f"成功获取{symbol}的{len(df)}条日线数据")
        return df
    

    

    

    
    def calculate_daily_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算日线技术指标（避免未来函数）"""
        if len(df) < 20:
            return df
        
        # 移动平均（使用shift确保不使用当前值）
        df['ma5'] = df['close'].shift(1).rolling(5, min_periods=5).mean()
        df['ma10'] = df['close'].shift(1).rolling(10, min_periods=10).mean()
        df['ma20'] = df['close'].shift(1).rolling(20, min_periods=20).mean()
        
        # 成交量指标（使用历史数据）
        df['volume_ma10'] = df['volume'].shift(1).rolling(10, min_periods=10).mean()
        df['volume_surge'] = df['volume'] / df['volume_ma10']
        
        # 价格变化（基于前一日）
        df['price_change'] = df['close'].pct_change()
        df['price_change_3d'] = df['close'].pct_change(3)
        
        # 日内振幅
        df['daily_amplitude'] = (df['high'] - df['low']) / df['open']
        
        # 相对前期高点的回撤
        df['high_20d'] = df['high'].shift(1).rolling(20, min_periods=20).max()
        df['drawdown_from_high'] = (df['close'] / df['high_20d'] - 1)
        
        return df
    
    def check_drop_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        """检查暴跌信号（基于最高点回撤，避免未来函数）"""
        if current_idx < 20:  # 需要足够的历史数据
            return False, 0.0
        
        current_data = df.iloc[current_idx]
        
        # 检查从最高点的回撤幅度
        drawdown = current_data['drawdown_from_high']
        
        # 暴跌条件：从最高点回撤超过阈值
        if pd.notna(drawdown) and drawdown <= -self.min_drop_percent:
            # 根据回撤程度确定所需的成交量放大倍数
            required_volume_surge = self.min_volume_surge
            if drawdown <= -self.severe_drop_percent:
                required_volume_surge = self.severe_volume_surge
                logger.info(f"检测到严重回撤 {abs(drawdown):.2%}（从20日高点），要求成交量放大{required_volume_surge}倍")
            
            # 确认成交量放大（使用历史平均成交量比较）
            if pd.notna(current_data['volume_surge']) and current_data['volume_surge'] >= required_volume_surge:
                return True, abs(drawdown)
            else:
                logger.debug(f"回撤{abs(drawdown):.2%}但成交量放大不足：{current_data.get('volume_surge', 0):.1f}倍 < {required_volume_surge}倍")
        
        return False, 0.0
    
    def check_reversal_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """检查反弹信号（日线级别，避免未来函数）"""
        if current_idx < 20:  # 需要更多历史数据确保MA计算有效
            return False, ""
        
        current_data = df.iloc[current_idx]
        prev_data = df.iloc[current_idx - 1]
        
        # 反弹确认条件（日线级别，适度放宽以增加交易机会）
        conditions = []
        
        # 1. 当日反弹幅度确认（降低门槛）
        if current_data['price_change'] > 0.01:  # 当日上涨超过1%
            conditions.append("反弹")
        elif current_data['price_change'] > 0.005:  # 当日上涨超过0.5%
            conditions.append("微反弹")
        
        # 2. 突破短期均线或接近均线
        if pd.notna(current_data['ma5']):
            if current_data['close'] > current_data['ma5']:
                conditions.append("突破MA5")
            elif current_data['close'] > current_data['ma5'] * 0.98:  # 接近MA5（2%以内）
                conditions.append("接近MA5")
        
        # 3. 成交量配合（降低要求）
        if pd.notna(current_data['volume_surge']):
            if current_data['volume_surge'] > 1.2:  # 降低至1.2倍
                conditions.append("成交量配合")
            elif current_data['volume_surge'] > 1.0:  # 成交量正常
                conditions.append("成交量正常")
        
        # 4. 相对高点回撤后反弹（放宽条件）
        if (pd.notna(current_data['drawdown_from_high']) and 
            current_data['drawdown_from_high'] < -0.05 and  # 相对20日高点回撤超过5%
            current_data['price_change'] > 0.005):  # 当日反弹超过0.5%
            conditions.append("回撤反弹")
        
        # 5. 日内振幅较大（降低门槛）
        if current_data['daily_amplitude'] > 0.03:  # 日内振幅超过3%
            conditions.append("高振幅")
        
        # 6. 连续下跌后的反弹（新增条件）
        if (current_idx >= 2 and 
            df.iloc[current_idx-1]['price_change'] < 0 and 
            df.iloc[current_idx-2]['price_change'] < 0 and
            current_data['price_change'] > 0):
            conditions.append("连跌反弹")
        
        # 至少满足2个条件（保持质量控制）
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
    
    def process_trading_day(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """处理单个交易日的逻辑"""
        current_date = df.index[current_idx]
        current_data = df.iloc[current_idx]
        current_price = current_data['close']
        
        result = {
            'date': current_date,
            'action': 'hold',
            'signal_type': '',
            'price': current_price,
            'trade': None
        }
        
        # 如果没有持仓，检查买入信号
        if not self.current_position:
            # 检查暴跌信号
            drop_detected, drop_percent = self.check_drop_signal(df, current_idx)
            if drop_detected:
                # 检查反弹信号（可以是同一天或后续几天）
                reversal_detected, reversal_reason = self.check_reversal_signal(df, current_idx)
                if reversal_detected:
                    # 买入
                    quantity = self.calculate_position_size(current_price)
                    buy_cost = self.calculate_trading_cost(current_price, quantity, True)
                    
                    if self.current_capital >= current_price * quantity + buy_cost:
                        self.current_position = {
                            'entry_date': current_date,
                            'entry_price': current_price,
                            'quantity': quantity,
                            'buy_cost': buy_cost,
                            'max_profit': 0,
                            'max_loss': 0,
                            'hold_days': 0,
                            'highest_price': current_price,  # 记录最高价格（用于移动止损）
                            'trailing_stop_price': 0         # 移动止损价格
                        }
                        
                        # 更新资金
                        self.current_capital -= current_price * quantity + buy_cost
                        
                        result.update({
                            'action': 'buy',
                            'signal_type': f"回撤{drop_percent:.1%} + {reversal_reason}",
                            'quantity': quantity,
                            'cost': buy_cost
                        })
                        
                        logger.info(f"🟢 买入 {current_date} | 价格:{current_price:.2f} | 数量:{quantity:,}股 | 金额:{current_price*quantity:,.0f} | 成本:{buy_cost:.2f} | 信号:{result['signal_type']} | 剩余资金:{self.current_capital:,.0f}")
        
        # 如果有持仓，检查卖出信号
        else:
            # 更新持仓天数
            self.current_position['hold_days'] = (current_date - self.current_position['entry_date']).days
            
            # 计算当前盈亏
            sell_cost = self.calculate_trading_cost(current_price, self.current_position['quantity'], False)
            current_pnl = (current_price - self.current_position['entry_price']) * self.current_position['quantity'] - self.current_position['buy_cost'] - sell_cost
            current_pnl_percent = current_pnl / (self.current_position['entry_price'] * self.current_position['quantity']) * 100
            
            # 更新最大盈亏和最高价格
            self.current_position['max_profit'] = max(self.current_position['max_profit'], current_pnl)
            self.current_position['max_loss'] = min(self.current_position['max_loss'], current_pnl)
            
            # 更新最高价格和移动止损价格
            if current_price > self.current_position['highest_price']:
                self.current_position['highest_price'] = current_price
                if self.trailing_stop_enabled:
                    self.current_position['trailing_stop_price'] = current_price * (1 - self.trailing_stop_percent)
            
            # 检查卖出条件
            should_sell = False
            exit_reason = ""
            
            # 最小持仓天数检查（避免过于频繁交易）
            if self.current_position['hold_days'] < self.min_hold_days:
                # 在最小持仓期内，只有严重止损才卖出
                if current_pnl_percent <= -self.stop_loss_percent * 100 * 1.5:  # 严重止损阈值
                    should_sell = True
                    exit_reason = "严重止损"
            else:
                # 超过最小持仓期后，正常止盈止损逻辑
                
                # 止盈
                if current_pnl_percent >= self.take_profit_percent * 100:
                    should_sell = True
                    exit_reason = "止盈"
                
                # 移动止损（优先级高于固定止损）
                elif self.trailing_stop_enabled and self.current_position['trailing_stop_price'] > 0 and current_price <= self.current_position['trailing_stop_price']:
                    should_sell = True
                    exit_reason = "移动止损"
                
                # 固定止损
                elif current_pnl_percent <= -self.stop_loss_percent * 100:
                    should_sell = True
                    exit_reason = "止损"
                
                # 时间止损（超过最大持仓天数）
                elif self.current_position['hold_days'] >= self.max_hold_days:
                    should_sell = True
                    exit_reason = "时间止损"
            
            # 执行卖出
            if should_sell:
                trade = DailyTrade(
                    entry_date=self.current_position['entry_date'],
                    exit_date=current_date,
                    symbol=self.target_symbol,
                    entry_price=self.current_position['entry_price'],
                    exit_price=current_price,
                    quantity=self.current_position['quantity'],
                    pnl=current_pnl,
                    pnl_percent=current_pnl_percent,
                    exit_reason=exit_reason,
                    max_profit=self.current_position['max_profit'],
                    max_loss=self.current_position['max_loss'],
                    hold_days=self.current_position['hold_days']
                )
                
                # 更新资金
                self.current_capital += current_price * self.current_position['quantity'] - sell_cost
                
                result.update({
                    'action': 'sell',
                    'trade': trade,
                    'exit_reason': exit_reason
                })
                
                logger.info(f"🔴 卖出 {current_date} | 价格:{current_price:.2f} | 数量:{self.current_position['quantity']:,}股 | 金额:{current_price*self.current_position['quantity']:,.0f} | 成本:{sell_cost:.2f} | 盈亏:{current_pnl:+.0f} ({current_pnl_percent:+.2f}%) | 原因:{exit_reason} | 总资金:{self.current_capital:,.0f}")
                
                # 清空持仓
                self.current_position = None
        
        return result
    
    def run_backtest(self, start_date: date = None, end_date: date = None) -> Dict:
        """运行日级别回测"""
        if start_date is None:
            start_date = self.default_start_date
        if end_date is None:
            end_date = self.default_end_date
        
        logger.info(f"开始日级别暴跌反弹策略回测: {self.target_symbol} ({start_date} 到 {end_date})")
        logger.info(f"初始资金: {self.current_capital:,.0f} 港币")
        
        # 获取整个回测期间的日线数据
        df = self.get_daily_data(self.target_symbol, start_date, end_date)
        if df.empty:
            logger.error("无法获取历史数据")
            return {}
        
        # 重置交易状态
        self.trades = []
        self.current_position = None
        self.current_capital = self.initial_capital
        
        # 遍历每个交易日
        for i in range(len(df)):
            current_date = df.index[i]
            
            # 处理当日交易逻辑
            day_result = self.process_trading_day(df, i)
            
            # 如果有交易，记录到trades列表
            if day_result['action'] == 'sell' and day_result['trade']:
                self.trades.append(day_result['trade'])
        
        # 如果回测结束时还有持仓，强制平仓
        if self.current_position:
            final_date = df.index[-1]
            final_price = df.iloc[-1]['close']
            sell_cost = self.calculate_trading_cost(final_price, self.current_position['quantity'], False)
            final_pnl = (final_price - self.current_position['entry_price']) * self.current_position['quantity'] - self.current_position['buy_cost'] - sell_cost
            final_pnl_percent = final_pnl / (self.current_position['entry_price'] * self.current_position['quantity']) * 100
            
            final_trade = DailyTrade(
                entry_date=self.current_position['entry_date'],
                exit_date=final_date,
                symbol=self.target_symbol,
                entry_price=self.current_position['entry_price'],
                exit_price=final_price,
                quantity=self.current_position['quantity'],
                pnl=final_pnl,
                pnl_percent=final_pnl_percent,
                exit_reason="回测结束",
                max_profit=self.current_position['max_profit'],
                max_loss=self.current_position['max_loss'],
                hold_days=(final_date - self.current_position['entry_date']).days
            )
            
            self.trades.append(final_trade)
            self.current_capital += final_price * self.current_position['quantity'] - sell_cost
            self.current_position = None
        
        # 计算总体结果
        total_pnl = self.current_capital - self.initial_capital
        trading_days = len(df)
        
        # 生成回测报告
        results = self.generate_backtest_report(total_pnl, trading_days)
        results['initial_capital'] = self.initial_capital
        results['final_capital'] = self.current_capital
        
        return results
    

    
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
                'profit_loss_ratio': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_hold_days': 0,
                'exit_reasons': {},
                'trading_days': trading_days,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'trading_day_ratio': 0
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
        trading_day_ratio = len(set(t.entry_date for t in self.trades)) / trading_days * 100 if trading_days > 0 else 0
        
        # 按退出原因统计
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += trade.pnl
        
        # 平均持仓时间
        avg_hold_days = np.mean([t.hold_days for t in self.trades]) if self.trades else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_hold_days': avg_hold_days,
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
        print("           日级别反弹策略回测报告")
        print("="*80)
        
        print(f"\n💰 资金统计:")
        print(f"   初始资金: {results.get('initial_capital', 0):,.0f} 港币")
        print(f"   最终资金: {results.get('final_capital', 0):,.0f} 港币")
        print(f"   总盈亏: {results.get('total_pnl', 0):+,.0f} 港币")
        print(f"   总收益率: {results.get('total_return_percent', 0):+.2f}%")
        
        print(f"\n📊 基本统计:")
        print(f"   总交易次数: {results.get('total_trades', 0)}")
        print(f"   胜率: {results.get('win_rate', 0):.1f}% ({results.get('winning_trades', 0)}/{results.get('total_trades', 0)})")
        print(f"   平均每笔盈亏: {results.get('avg_pnl_per_trade', 0):+,.0f} 港币")
        print(f"   平均盈利: {results.get('avg_win', 0):+,.0f} 港币")
        print(f"   平均亏损: {results.get('avg_loss', 0):+,.0f} 港币")
        print(f"   盈亏比: {results.get('profit_loss_ratio', 0):.2f}")
        print(f"   平均持仓时间: {results.get('avg_hold_days', 0):.1f} 天")
        
        print(f"\n📈 风险指标:")
        print(f"   最大回撤: {results.get('max_drawdown', 0):.2f}%")
        print(f"   夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        print(f"   最大单笔盈利: {results.get('max_profit', 0):+,.0f} 港币")
        print(f"   最大单笔亏损: {results.get('max_loss', 0):+,.0f} 港币")
        print(f"   最大连续盈利: {results.get('max_consecutive_wins', 0)} 笔")
        print(f"   最大连续亏损: {results.get('max_consecutive_losses', 0)} 笔")
        
        print(f"\n📅 交易统计:")
        print(f"   交易天数: {results.get('trading_days', 0)} 天")
        print(f"   交易日比例: {results.get('trading_day_ratio', 0):.1f}%")
        
        print(f"\n🚪 退出原因统计:")
        for reason, stats in results.get('exit_reasons', {}).items():
            if isinstance(stats, dict):
                avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
                print(f"   {reason}: {stats['count']} 笔, 总盈亏: {stats['pnl']:+,.0f}, 平均: {avg_pnl:+,.0f}")
            else:
                print(f"   {reason}: {stats} 笔")
        
        if self.trades:
            print(f"\n📋 最近5笔交易:")
            for trade in self.trades[-5:]:
                print(f"   {trade.entry_date} -> {trade.exit_date}: "
                      f"{trade.pnl:+.0f} ({trade.pnl_percent:+.2f}%) - {trade.exit_reason}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    strategy = DailyReversalStrategy()
    
    print(f"🚀 开始日级别暴跌反弹策略回测（支持持仓过周）")
    print(f"📅 回测期间: {strategy.backtest_start_date} 至 {strategy.backtest_end_date}")
    print(f"🎯 目标股票: {strategy.target_symbol} (阿里巴巴)")
    print(f"💰 初始资金: {strategy.initial_capital:,} 港币 (全仓操作)")
    print(f"📊 策略参数（最优配置）:")
    print(f"   回撤阈值: {strategy.min_drop_percent:.1%} (严重回撤: {strategy.severe_drop_percent:.1%})")
    print(f"   止损/止盈: {strategy.stop_loss_percent:.1%} / {strategy.take_profit_percent:.1%}")
    print(f"   成交量确认: {strategy.min_volume_surge}倍 / 严重回撤{strategy.severe_volume_surge}倍")
    print(f"   持仓天数: {strategy.min_hold_days}-{strategy.max_hold_days}天")
    print(f"   移动止损: {'启用' if strategy.trailing_stop_enabled else '禁用'} ({strategy.trailing_stop_percent:.1%})")
    print(f"   周末持仓: {'支持' if strategy.weekend_hold_enabled else '不支持'}")
    
    # 运行回测（使用策略对象中配置的日期）
    results = strategy.run_backtest(strategy.backtest_start_date, strategy.backtest_end_date)
    
    # 打印报告
    strategy.print_detailed_report(results)
    
    # 保存交易记录
    if strategy.trades:
        trades_df = pd.DataFrame([
            {
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'exit_reason': t.exit_reason,
                'hold_days': t.hold_days
            }
            for t in strategy.trades
        ])
        
        trades_df.to_csv('daily_reversal_trades.csv', index=False)
        print(f"\n💾 交易记录已保存到 daily_reversal_trades.csv")

if __name__ == "__main__":
    main()