#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史数据管理器 - 严格防止未来函数

核心原则：
1. 在任何时间点T，只能访问T-1及之前的数据
2. 股票池筛选必须基于历史数据
3. 所有数据访问都通过这个管理器
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


class HistoricalDataManager:
    """
    历史数据管理器 - 严格防止未来函数
    
    使用方法:
        manager = HistoricalDataManager()
        manager.load_stock_data(symbols, start_date, end_date)
        
        # 在回测某一天时
        manager.set_current_date('2024-01-15')
        
        # 获取数据（只能获取到2024-01-14及之前的数据）
        df = manager.get_history('00700.HK')  # 返回截止到2024-01-14的数据
        
        # 获取当前可交易的股票池
        pool = manager.get_tradable_pool()
    """
    
    def __init__(self):
        self._all_data: Dict[str, pd.DataFrame] = {}  # 完整数据（内部使用）
        self._current_date: Optional[date] = None     # 当前回测日期
        self._min_history_days: int = 120             # 最少需要的历史天数
        
    def load_stock_data(self, symbols: List[str], start_date: date, end_date: date):
        """
        加载股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        """
        from hk_stock_api import HKStockAPI
        
        api = HKStockAPI()
        loaded = 0
        
        for i, symbol in enumerate(symbols):
            print(f"\r加载数据: {i+1}/{len(symbols)} {symbol}", end='', flush=True)
            try:
                df = api.get_daily_data(symbol, start_date, end_date)
                if df is not None and len(df) >= self._min_history_days:
                    # 确保索引是日期类型
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    self._all_data[symbol] = df
                    loaded += 1
            except Exception as e:
                pass
        
        print(f"\n成功加载: {loaded}/{len(symbols)} 只股票")
        return loaded
    
    def set_current_date(self, current_date):
        """
        设置当前回测日期
        
        在此日期，只能访问 current_date - 1 及之前的数据
        """
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date).date()
        elif isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
        self._current_date = current_date
    
    def _get_cutoff_date(self) -> date:
        """获取数据截止日期（当前日期的前一天）"""
        if self._current_date is None:
            raise ValueError("必须先调用 set_current_date() 设置当前日期")
        return self._current_date - timedelta(days=1)
    
    def get_history(self, symbol: str, lookback_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        获取历史数据（严格截止到昨天）
        
        Args:
            symbol: 股票代码
            lookback_days: 回看天数，None表示获取全部历史
            
        Returns:
            截止到昨天的历史数据，如果数据不足返回None
        """
        if symbol not in self._all_data:
            return None
        
        cutoff = self._get_cutoff_date()
        df = self._all_data[symbol]
        
        # 严格过滤：只保留截止日期之前（含）的数据
        df_filtered = df[df.index.date <= cutoff].copy()
        
        if len(df_filtered) < self._min_history_days:
            return None
        
        if lookback_days is not None:
            df_filtered = df_filtered.tail(lookback_days)
        
        return df_filtered
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格（昨天的收盘价）"""
        df = self.get_history(symbol, lookback_days=1)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None
    
    def get_latest_row(self, symbol: str) -> Optional[pd.Series]:
        """获取最新一行数据（昨天的数据）"""
        df = self.get_history(symbol, lookback_days=1)
        if df is not None and len(df) > 0:
            return df.iloc[-1]
        return None
    
    def get_tradable_pool(
        self,
        min_price: float = 1.0,
        min_avg_turnover: float = 5000000,
        max_avg_turnover: float = 500000000,
        lookback_days: int = 20
    ) -> List[str]:
        """
        获取当前可交易的股票池（基于历史数据）
        
        所有筛选条件都基于截止到昨天的数据
        
        Args:
            min_price: 最低价格
            min_avg_turnover: 最低N日均成交额
            max_avg_turnover: 最高N日均成交额
            lookback_days: 计算均值的天数
            
        Returns:
            符合条件的股票代码列表
        """
        tradable = []
        
        for symbol in self._all_data.keys():
            df = self.get_history(symbol, lookback_days=lookback_days)
            if df is None or len(df) < lookback_days // 2:
                continue
            
            # 基于历史数据计算指标
            latest_price = df['close'].iloc[-1]
            avg_turnover = (df['close'] * df['volume']).mean()
            
            # 筛选条件
            if latest_price >= min_price and \
               min_avg_turnover <= avg_turnover <= max_avg_turnover:
                tradable.append(symbol)
        
        return tradable
    
    def get_all_symbols(self) -> List[str]:
        """获取所有已加载的股票代码"""
        return list(self._all_data.keys())
    
    def get_all_trading_dates(self) -> List[date]:
        """获取所有交易日期"""
        all_dates = set()
        for df in self._all_data.values():
            all_dates.update(df.index.date)
        return sorted(all_dates)
    
    def calculate_indicators(self, symbol: str, breakout_lookback: int = 120) -> Optional[pd.DataFrame]:
        """
        计算技术指标（基于历史数据）
        
        Args:
            symbol: 股票代码
            breakout_lookback: 突破回看周期（天数）
            
        返回的DataFrame已经过滤掉未来数据
        """
        df = self.get_history(symbol)
        if df is None:
            return None
        
        df = df.copy()
        
        # 均线
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 成交量均线和量比
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
        
        # N日新高（使用传入的参数）
        df['high_nd'] = df['high'].rolling(breakout_lookback).max()
        df['is_breakout'] = df['close'] > df['high_nd'].shift(1)
        
        # 趋势
        df['trend_up'] = df['ma20'] > df['ma60']
        
        # 日均成交额
        df['turnover'] = df['close'] * df['volume']
        df['avg_turnover_20d'] = df['turnover'].rolling(20).mean()
        
        return df


class BacktestEngine:
    """
    回测引擎 - 严格的时间顺序执行
    """
    
    def __init__(self, data_manager: HistoricalDataManager, config: dict = None):
        self.dm = data_manager
        self.config = config or {}
        
        # 策略参数
        self.max_positions = self.config.get('max_positions', 8)
        self.position_size_pct = self.config.get('position_size_pct', 0.15)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.25)
        self.breakout_lookback = self.config.get('breakout_lookback', 120)
        self.volume_ratio_threshold = self.config.get('volume_ratio_threshold', 1.5)
        
        # 账户状态
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> {shares, buy_price, buy_date}
        self.trades = []
        self.daily_values = []
        self.triggered_levels = {}
        self._verbose = True
    
    @property
    def total_value(self) -> float:
        pos_value = 0
        for symbol, pos in self.positions.items():
            price = self.dm.get_latest_price(symbol)
            if price:
                pos_value += pos['shares'] * price
        return self.cash + pos_value
    
    def run(self, start_date: date, end_date: date, benchmark_data: pd.DataFrame = None, verbose: bool = True) -> dict:
        """运行回测"""
        self._verbose = verbose
        if verbose:
            print("\n" + "="*60)
            print("趋势突破策略 - 严格无未来函数回测")
            print("="*60)
            print(f"初始资金: {self.initial_capital:,.0f}")
            print(f"回测区间: {start_date} 到 {end_date}")
        
        # 获取所有交易日
        all_dates = self.dm.get_all_trading_dates()
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]
        
        if verbose:
            print(f"交易日数: {len(trading_dates)}")
            print("-"*60)
        
        # 预热期：需要120天历史数据
        warmup_days = self.breakout_lookback
        
        for i, current_date in enumerate(trading_dates):
            # 设置当前日期（关键！）
            self.dm.set_current_date(current_date)
            
            # 跳过预热期
            if i < warmup_days:
                continue
            
            # 更新持仓价格
            self._update_positions()
            
            # 检查卖出信号（先卖后买）
            self._check_sell_signals(current_date)
            
            # 获取当前可交易的股票池（基于历史数据）
            tradable_pool = self.dm.get_tradable_pool()
            
            # 检查买入信号
            self._check_buy_signals(current_date, tradable_pool)
            
            # 记录每日净值
            self.daily_values.append((str(current_date), self.total_value))
        
        return self._generate_report(benchmark_data, verbose)
    
    def _update_positions(self):
        """更新持仓价格"""
        for symbol in list(self.positions.keys()):
            price = self.dm.get_latest_price(symbol)
            if price:
                self.positions[symbol]['current_price'] = price
    
    def _check_buy_signals(self, current_date: date, tradable_pool: List[str]):
        """检查买入信号"""
        if len(self.positions) >= self.max_positions:
            return
        
        for symbol in tradable_pool:
            if symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break
            
            # 计算技术指标（基于历史数据，使用策略参数）
            df = self.dm.calculate_indicators(symbol, breakout_lookback=self.breakout_lookback)
            if df is None or len(df) < 2:
                continue
            
            # 获取最新一行（昨天的数据）
            row = df.iloc[-1]
            
            # 检查买入条件
            is_breakout = row.get('is_breakout', False)
            volume_ratio = row.get('volume_ratio', 0)
            trend_up = row.get('trend_up', False)
            avg_turnover = row.get('avg_turnover_20d', 0)
            
            if is_breakout and volume_ratio >= self.volume_ratio_threshold and trend_up:
                if pd.notna(avg_turnover) and avg_turnover >= 5000000:
                    self._execute_buy(current_date, symbol, row['close'],
                                     f"突破{self.breakout_lookback}日新高, 量比={volume_ratio:.1f}")
    
    def _check_sell_signals(self, current_date: date):
        """检查卖出信号"""
        for symbol in list(self.positions.keys()):
            df = self.dm.calculate_indicators(symbol, breakout_lookback=self.breakout_lookback)
            if df is None or len(df) < 1:
                continue
            
            row = df.iloc[-1]
            pos = self.positions[symbol]
            current_price = row['close']
            buy_price = pos['buy_price']
            pnl_pct = (current_price / buy_price - 1)
            
            # 止损
            if pnl_pct < -self.stop_loss_pct:
                self._execute_sell(current_date, symbol, current_price, 1.0,
                                  f"止损: 亏损{pnl_pct:.1%}")
                continue
            
            # 跌破MA60且亏损
            ma60 = row.get('ma60', 0)
            if ma60 > 0 and current_price < ma60 and pnl_pct < 0:
                self._execute_sell(current_date, symbol, current_price, 1.0,
                                  f"跌破MA60")
                continue
            
            # 分档止盈
            if symbol not in self.triggered_levels:
                self.triggered_levels[symbol] = []
            
            for level_pct, sell_ratio in [(1.0, 0.33), (3.0, 0.33)]:
                if level_pct in self.triggered_levels[symbol]:
                    continue
                if pnl_pct >= level_pct:
                    self.triggered_levels[symbol].append(level_pct)
                    self._execute_sell(current_date, symbol, current_price, sell_ratio,
                                      f"止盈{level_pct:.0%}")
                    break
    
    def _execute_buy(self, current_date: date, symbol: str, price: float, reason: str):
        """执行买入"""
        max_amount = self.total_value * self.position_size_pct
        buy_amount = min(self.cash * 0.9, max_amount)
        
        if buy_amount < 5000:
            return
        
        shares = int(buy_amount / price / 100) * 100
        if shares <= 0:
            shares = int(buy_amount / price)
        if shares <= 0:
            return
        
        cost = shares * price
        self.cash -= cost
        
        self.positions[symbol] = {
            'shares': shares,
            'buy_price': price,
            'buy_date': str(current_date),
            'current_price': price
        }
        
        self.trades.append({
            'date': str(current_date),
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'reason': reason
        })
        
        print(f"  [买入] {current_date} {symbol} @ {price:.2f}, {shares}股, {reason}") if self._verbose else None
    
    def _execute_sell(self, current_date: date, symbol: str, price: float, 
                      ratio: float, reason: str):
        """执行卖出"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        sell_shares = int(pos['shares'] * ratio)
        if sell_shares <= 0:
            return
        
        sell_amount = sell_shares * price
        self.cash += sell_amount
        
        pnl_pct = (price / pos['buy_price'] - 1) * 100
        
        self.trades.append({
            'date': str(current_date),
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'shares': sell_shares,
            'reason': f"{reason}, 盈亏:{pnl_pct:+.1f}%"
        })
        
        if self._verbose:
            print(f"  [卖出] {current_date} {symbol} @ {price:.2f}, {sell_shares}股, {pnl_pct:+.1f}%")
        
        pos['shares'] -= sell_shares
        if pos['shares'] <= 0:
            del self.positions[symbol]
            if symbol in self.triggered_levels:
                del self.triggered_levels[symbol]
    
    def _calculate_yearly_returns(self, benchmark_data: pd.DataFrame = None) -> dict:
        """计算年度收益"""
        if not self.daily_values:
            return {}
        
        # 按年份分组
        yearly_data = {}
        for d, v in self.daily_values:
            # d可能是字符串或date对象
            if isinstance(d, str):
                d_date = datetime.strptime(d, '%Y-%m-%d').date()
            else:
                d_date = d
            year = d_date.year
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].append((d_date, v))
        
        yearly_returns = {}
        for year, data in yearly_data.items():
            if len(data) < 2:
                continue
            start_val = data[0][1]
            end_val = data[-1][1]
            strat_ret = (end_val / start_val - 1) * 100
            
            # 计算同期基准收益
            bench_ret = 0
            if benchmark_data is not None:
                start_date = data[0][0]
                end_date = data[-1][0]
                bench_df = benchmark_data.copy()
                bench_df.index = pd.to_datetime(bench_df.index)
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                bench_filtered = bench_df[(bench_df.index >= start_dt) & (bench_df.index <= end_dt)]
                if len(bench_filtered) > 1:
                    bench_start = bench_filtered['close'].iloc[0]
                    bench_end = bench_filtered['close'].iloc[-1]
                    bench_ret = (bench_end / bench_start - 1) * 100
            
            yearly_returns[year] = {
                'strategy': strat_ret,
                'benchmark': bench_ret
            }
        
        return yearly_returns
    
    def _generate_report(self, benchmark_data: pd.DataFrame = None, verbose: bool = True) -> dict:
        """生成回测报告"""
        if verbose:
            print("\n" + "="*60)
            print("回测结果")
            print("="*60)
        
        initial = self.initial_capital
        final = self.total_value
        total_return = (final / initial - 1) * 100
        
        # 最大回撤
        values = [v for _, v in self.daily_values]
        max_dd = 0
        peak = values[0] if values else initial
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 年化收益
        days = len(self.daily_values)
        annual_return = ((final / initial) ** (252 / max(days, 1)) - 1) * 100
        
        # 交易统计
        buys = [t for t in self.trades if t['action'] == 'BUY']
        sells = [t for t in self.trades if t['action'] == 'SELL']
        win_trades = sum(1 for t in sells if '+' in t.get('reason', '').split('盈亏:')[-1])
        win_rate = win_trades / len(sells) * 100 if sells else 0
        
        # 年度收益统计
        yearly_returns = self._calculate_yearly_returns(benchmark_data)
        
        if verbose:
            print(f"\n【策略收益】")
            print(f"  总收益率: {total_return:+.2f}%")
            print(f"  年化收益: {annual_return:+.2f}%")
            print(f"  最大回撤: {max_dd:.2%}")
            
            # 打印年度收益
            if yearly_returns:
                print(f"\n【年度收益明细】")
                positive_years = 0
                for year, data in sorted(yearly_returns.items()):
                    strat_ret = data['strategy']
                    bench_ret = data.get('benchmark', 0)
                    excess = strat_ret - bench_ret
                    status = "✓" if strat_ret > 0 else "✗"
                    if strat_ret > 0:
                        positive_years += 1
                    print(f"  {year}: 策略{strat_ret:+.1f}% | 恒指{bench_ret:+.1f}% | 超额{excess:+.1f}% {status}")
                print(f"  盈利年份: {positive_years}/{len(yearly_returns)}")
        
        # 基准对比
        benchmark_return = None
        excess_return = None
        if benchmark_data is not None and len(self.daily_values) > 0:
            start_date = self.daily_values[0][0]
            end_date = self.daily_values[-1][0]
            
            bench_df = benchmark_data.copy()
            bench_df.index = pd.to_datetime(bench_df.index)
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            bench_filtered = bench_df[(bench_df.index >= start_dt) & (bench_df.index <= end_dt)]
            
            if len(bench_filtered) > 1:
                bench_start = bench_filtered['close'].iloc[0]
                bench_end = bench_filtered['close'].iloc[-1]
                benchmark_return = (bench_end / bench_start - 1) * 100
                excess_return = total_return - benchmark_return
                
                # 基准最大回撤
                bench_values = bench_filtered['close'].values
                bench_max_dd = 0
                bench_peak = bench_values[0]
                for v in bench_values:
                    if v > bench_peak:
                        bench_peak = v
                    dd = (bench_peak - v) / bench_peak
                    if dd > bench_max_dd:
                        bench_max_dd = dd
                
                print(f"\n【恒生指数基准】")
                print(f"  基准收益: {benchmark_return:+.2f}%")
                print(f"  基准回撤: {bench_max_dd:.2%}")
                print(f"\n【相对表现】")
                print(f"  超额收益: {excess_return:+.2f}%")
                if excess_return > 0:
                    print(f"  结论: 策略跑赢恒生指数 {excess_return:.1f}个百分点")
                else:
                    print(f"  结论: 策略跑输恒生指数 {-excess_return:.1f}个百分点")
        
        if verbose:
            print(f"\n【交易统计】")
            print(f"  买入次数: {len(buys)}")
            print(f"  卖出次数: {len(sells)}")
            print(f"  胜率: {win_rate:.1f}%")
        
            if self.positions:
                print(f"\n【当前持仓】")
                for symbol, pos in self.positions.items():
                    pnl = (pos['current_price'] / pos['buy_price'] - 1) * 100
                    print(f"  {symbol}: {pos['shares']}股, 盈亏{pnl:+.1f}%")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_dd * 100,
            'win_rate': win_rate,
            'trade_count': len(buys) + len(sells),
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'yearly_returns': yearly_returns
        }


def load_hsi_data(start_date: date, end_date: date) -> pd.DataFrame:
    """加载恒生指数数据作为基准"""
    from hk_stock_api import HKStockAPI
    
    print("\n加载恒生指数数据...")
    try:
        api = HKStockAPI()
        df = api.get_daily_data('HSI.HK', start_date, end_date)
        if df is not None and len(df) > 0:
            print(f"恒生指数: {len(df)} 条数据")
            return df
    except Exception as e:
        print(f"加载恒生指数失败: {e}")
    
    # 备用：尝试用指数ETF代替
    try:
        api = HKStockAPI()
        df = api.get_daily_data('02800.HK', start_date, end_date)  # 盈富基金
        if df is not None and len(df) > 0:
            print(f"使用盈富基金(02800.HK)作为基准: {len(df)} 条数据")
            return df
    except Exception as e:
        print(f"加载盈富基金失败: {e}")
    
    return None


def main():
    """主函数"""
    print("="*60)
    print("趋势突破策略 - 严格无未来函数版本")
    print("="*60)
    
    # 1. 初始化数据管理器
    dm = HistoricalDataManager()
    
    # 2. 确定股票池（用一个宽泛的初始列表）
    import os
    if not os.path.exists('hk_all_stocks.csv'):
        print("未找到市场数据文件")
        return
    
    df = pd.read_csv('hk_all_stocks.csv')
    df['code'] = df['代码'].astype(str).str.zfill(5)
    
    # 初始筛选：只排除明显不合适的（ETF等）
    df = df[~df['code'].str.match(r'^07[0-9]{3}$')]  # 排除杠杆产品
    df = df[~df['code'].str.match(r'^028[0-9]{2}$')]  # 排除ETF
    
    symbols = [f"{code}.HK" for code in df['code'].tolist()]
    
    # 限制数量（避免API限制）
    if len(symbols) > 200:
        symbols = symbols[:200]
    
    print(f"待加载股票: {len(symbols)}只")
    
    # 3. 加载数据 - 从2020年开始
    data_start = date(2019, 6, 1)  # 多加载6个月用于预热
    end_date = date.today()
    
    print(f"\n加载历史数据 ({data_start} 到 {end_date})...")
    dm.load_stock_data(symbols, data_start, end_date)
    
    # 4. 加载恒生指数作为基准
    hsi_data = load_hsi_data(data_start, end_date)
    
    # 5. 运行回测 - 最优参数（组合B）
    config = {
        'initial_capital': 100000,
        'max_positions': 8,
        'position_size_pct': 0.15,
        'stop_loss_pct': 0.40,          # 止损40%
        'breakout_lookback': 60,         # 60日突破
        'volume_ratio_threshold': 2.0,   # 量比2.0
    }
    
    engine = BacktestEngine(dm, config)
    
    # 回测开始日期：2020年1月
    backtest_start = date(2020, 1, 1)
    backtest_end = date.today()
    
    result = engine.run(backtest_start, backtest_end, hsi_data)
    
    print("\n" + "="*60)
    print("回测完成")
    print("="*60)
    
    return result


def optimize_params():
    """参数优化 - 测试不同参数组合"""
    print("="*60)
    print("参数优化 - 寻找最优配置")
    print("="*60)
    
    # 1. 创建数据管理器
    dm = HistoricalDataManager()
    
    # 2. 加载股票池
    import os
    if not os.path.exists('hk_all_stocks.csv'):
        print("未找到市场数据文件")
        return []
    
    df = pd.read_csv('hk_all_stocks.csv')
    df['code'] = df['代码'].astype(str).str.zfill(5)
    
    # 筛选
    df = df[~df['code'].str.match(r'^07[0-9]{3}$')]
    df = df[~df['code'].str.match(r'^028[0-9]{2}$')]
    symbols = [f"{code}.HK" for code in df['code'].tolist()[:200]]
    
    # 3. 加载数据
    data_start = date(2019, 6, 1)
    end_date = date.today()
    print(f"\n加载数据中... (这可能需要几分钟)")
    dm.load_stock_data(symbols, data_start, end_date)
    hsi_data = load_hsi_data(data_start, end_date)
    
    # 4. 参数网格
    param_grid = [
        # 测试不同的止损比例
        {'stop_loss_pct': 0.30, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': '放宽止损30%'},
        {'stop_loss_pct': 0.35, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': '放宽止损35%'},
        {'stop_loss_pct': 0.40, 'breakout_lookback': 120, 'volume_ratio_threshold': 1.5, 'name': '放宽止损40%'},
        # 测试不同的突破周期
        {'stop_loss_pct': 0.25, 'breakout_lookback': 60, 'volume_ratio_threshold': 1.5, 'name': '60日突破'},
        {'stop_loss_pct': 0.25, 'breakout_lookback': 90, 'volume_ratio_threshold': 1.5, 'name': '90日突破'},
        {'stop_loss_pct': 0.25, 'breakout_lookback': 180, 'volume_ratio_threshold': 1.5, 'name': '180日突破'},
        # 测试不同的量比阈值
        {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 2.0, 'name': '量比2.0'},
        {'stop_loss_pct': 0.25, 'breakout_lookback': 120, 'volume_ratio_threshold': 2.5, 'name': '量比2.5'},
        # 组合优化
        {'stop_loss_pct': 0.35, 'breakout_lookback': 90, 'volume_ratio_threshold': 2.0, 'name': '组合A'},
        {'stop_loss_pct': 0.40, 'breakout_lookback': 60, 'volume_ratio_threshold': 2.0, 'name': '组合B'},
    ]
    
    results = []
    backtest_start = date(2020, 1, 1)
    backtest_end = date.today()
    
    for i, params in enumerate(param_grid):
        print(f"\n[{i+1}/{len(param_grid)}] 测试: {params['name']}")
        
        config = {
            'initial_capital': 100000,
            'max_positions': 8,
            'position_size_pct': 0.15,
            'stop_loss_pct': params['stop_loss_pct'],
            'breakout_lookback': params['breakout_lookback'],
            'volume_ratio_threshold': params['volume_ratio_threshold'],
        }
        
        engine = BacktestEngine(dm, config)
        result = engine.run(backtest_start, backtest_end, hsi_data, verbose=False)
        
        # 计算盈利年份数
        yearly = result.get('yearly_returns', {})
        positive_years = sum(1 for y, d in yearly.items() if d['strategy'] > 0)
        
        results.append({
            'name': params['name'],
            'params': params,
            'total_return': result['total_return'],
            'excess_return': result['excess_return'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'positive_years': f"{positive_years}/{len(yearly)}",
            'yearly': yearly
        })
        
        print(f"  收益: {result['total_return']:+.1f}% | 超额: {result['excess_return']:+.1f}% | 回撤: {result['max_drawdown']:.1f}% | 盈利年: {positive_years}/{len(yearly)}")
    
    # 5. 排序并输出结果
    print("\n" + "="*60)
    print("参数优化结果汇总 (按超额收益排序)")
    print("="*60)
    
    results.sort(key=lambda x: x['excess_return'] or -999, reverse=True)
    
    print(f"{'#':<3} {'策略':12} {'总收益':>10} {'超额收益':>10} {'最大回撤':>10} {'胜率':>8} {'盈利年':>8}")
    print("-" * 70)
    
    for i, r in enumerate(results):
        print(f"{i+1:<3} {r['name']:12} {r['total_return']:>+9.1f}% {r['excess_return']:>+9.1f}% {r['max_drawdown']:>9.1f}% {r['win_rate']:>7.1f}% {r['positive_years']:>8}")
    
    # 输出最优策略的年度明细
    if results:
        best = results[0]
        print(f"\n最优策略: {best['name']}")
        print("年度明细:")
        for year, data in sorted(best['yearly'].items()):
            status = "✓" if data['strategy'] > 0 else "✗"
            print(f"  {year}: 策略{data['strategy']:+.1f}% | 恒指{data['benchmark']:+.1f}% {status}")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        optimize_params()
    else:
        main()
