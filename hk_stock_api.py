#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股数据API工具
基于Longport OpenAPI的港股数据获取工具

使用方法:
    # 作为模块导入使用
    from hk_stock_api import HKStockAPI
    api = HKStockAPI()
    df = api.get_daily_data("00700.HK", start_date, end_date)
    
    # 或作为脚本直接运行
    python hk_stock_api.py <股票代码> [开始日期] [结束日期]
    python hk_stock_api.py 00700.HK
    python hk_stock_api.py 0100.HK 2024-01-01 2024-12-31
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from longport.openapi import QuoteContext, Config, Period, AdjustType
import time
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HKStockAPI:
    """港股数据API封装类"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        初始化港股数据API
        
        Args:
            max_retries: API调用失败时的最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 初始化 Longport（连接易超时，与「并发」无关；此处串行重试扩大间隔）
        init_retries = int(os.getenv('LONGPORT_INIT_RETRIES', '5'))
        init_base_delay = float(os.getenv('LONGPORT_INIT_RETRY_DELAY', '2.0'))
        last_err: Optional[Exception] = None
        for attempt in range(init_retries):
            try:
                self.config = Config.from_env()
                self.quote_ctx = QuoteContext(self.config)
                logger.info("港股数据API初始化成功")
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning(
                    "API初始化失败 (%s/%s): %s",
                    attempt + 1,
                    init_retries,
                    e,
                )
                if attempt < init_retries - 1:
                    time.sleep(init_base_delay * (attempt + 1))
        if last_err is not None:
            logger.error(f"API初始化失败: {last_err}")
            raise last_err
    
    def _call_with_retry(self, func, *args, **kwargs):
        """带重试机制的API调用"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API调用失败 ({attempt + 1}/{self.max_retries}): {e}，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"API调用失败，已重试{self.max_retries}次: {e}")
                    raise
        return None
    
    def get_daily_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjust: AdjustType = AdjustType.ForwardAdjust
    ) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            symbol: 股票代码，如 "00700.HK"
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型，默认前复权
            
        Returns:
            DataFrame包含以下列: open, high, low, close, volume, turnover
        """
        try:
            candles = self._call_with_retry(
                self.quote_ctx.history_candlesticks_by_date,
                symbol,
                Period.Day,
                adjust,
                start_date,
                end_date
            )
            
            if not candles:
                logger.warning(f"{symbol}: 未获取到数据")
                return pd.DataFrame()
            
            data = []
            for candle in candles:
                data.append({
                    'date': candle.timestamp.date() if isinstance(candle.timestamp, datetime) else datetime.fromtimestamp(candle.timestamp).date(),
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': int(candle.volume),
                    'turnover': float(candle.turnover)
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            if os.getenv('LONGPORT_VERBOSE_PER_SYMBOL', '').lower() in ('1', 'true', 'yes'):
                logger.info(f"{symbol}: 成功获取 {len(df)} 条日线数据")
            return df
            
        except Exception as e:
            logger.error(f"{symbol}: 获取日线数据失败 - {e}")
            return pd.DataFrame()
    
    def get_minute_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: Period = Period.Min_1,
        adjust: AdjustType = AdjustType.ForwardAdjust,
        batch_days: int = 5
    ) -> pd.DataFrame:
        """
        获取股票分钟级K线数据（分批获取避免API限制）
        
        Args:
            symbol: 股票代码，如 "00700.HK"
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期，默认1分钟
            adjust: 复权类型，默认前复权
            batch_days: 每批获取的天数
            
        Returns:
            DataFrame包含以下列: open, high, low, close, volume, turnover
        """
        logger.info(f"开始获取 {symbol} 分钟数据: {start_date} 到 {end_date}")
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            batch_end = min(current_date + timedelta(days=batch_days - 1), end_date)
            
            try:
                candles = self._call_with_retry(
                    self.quote_ctx.history_candlesticks_by_date,
                    symbol,
                    period,
                    adjust,
                    current_date,
                    batch_end
                )
                
                if candles:
                    for candle in candles:
                        all_data.append({
                            'datetime': candle.timestamp if isinstance(candle.timestamp, datetime) else datetime.fromtimestamp(candle.timestamp),
                            'open': float(candle.open),
                            'high': float(candle.high),
                            'low': float(candle.low),
                            'close': float(candle.close),
                            'volume': int(candle.volume),
                            'turnover': float(candle.turnover)
                        })
                    logger.debug(f"批次 {current_date}-{batch_end}: 获取 {len(candles)} 条数据")
                
                # 添加延迟避免API限制
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"批次 {current_date}-{batch_end} 获取失败: {e}")
            
            current_date = batch_end + timedelta(days=1)
        
        if not all_data:
            logger.warning(f"{symbol}: 未获取到分钟数据")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # 去重
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"{symbol}: 成功获取 {len(df)} 条分钟数据")
        return df
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码，如 "00700.HK"
            
        Returns:
            包含股票基本信息的字典，失败返回None
        """
        try:
            # 获取最近的价格和成交量数据
            candles = self._call_with_retry(
                self.quote_ctx.history_candlesticks_by_date,
                symbol,
                Period.Day,
                AdjustType.ForwardAdjust,
                date.today() - timedelta(days=15),
                date.today()
            )
            
            if not candles or len(candles) < 5:
                return None
            
            latest = candles[-1]
            price = float(latest.close)
            
            # 计算平均成交量和成交额
            recent_volumes = [int(c.volume) for c in candles[-10:]]
            recent_turnovers = [float(c.turnover) for c in candles[-10:]]
            
            avg_volume = np.mean(recent_volumes)
            avg_turnover = np.mean(recent_turnovers)
            
            # 估算市值
            if avg_volume > 0 and avg_turnover > 0:
                turnover_rate = 0.015  # 假设平均换手率1.5%
                estimated_shares = avg_volume / turnover_rate
                estimated_market_cap = estimated_shares * price
            else:
                estimated_market_cap = avg_turnover * 80
            
            return {
                'symbol': symbol,
                'price': price,
                'avg_volume': avg_volume,
                'avg_turnover': avg_turnover,
                'estimated_market_cap': estimated_market_cap,
                'data_points': len(candles)
            }
            
        except Exception as e:
            logger.error(f"{symbol}: 获取股票信息失败 - {e}")
            return None
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        获取股票实时行情
        
        Args:
            symbol: 股票代码，如 "00700.HK"
            
        Returns:
            包含实时行情的字典，失败返回None
        """
        try:
            quote = self._call_with_retry(self.quote_ctx.quote, symbol)
            if quote:
                return {
                    'symbol': symbol,
                    'last_done': float(quote.last_done) if quote.last_done else None,
                    'open': float(quote.open) if quote.open else None,
                    'high': float(quote.high) if quote.high else None,
                    'low': float(quote.low) if quote.low else None,
                    'volume': int(quote.volume) if quote.volume else None,
                    'turnover': float(quote.turnover) if quote.turnover else None,
                    'timestamp': quote.timestamp
                }
            return None
        except Exception as e:
            logger.error(f"{symbol}: 获取实时行情失败 - {e}")
            return None


def main():
    """命令行入口函数"""
    # 检查环境变量
    if not os.getenv('LONGPORT_APP_KEY') or not os.getenv('LONGPORT_ACCESS_TOKEN'):
        print("错误: 未找到Longport API凭证，请检查.env文件")
        sys.exit(1)
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("港股数据获取工具")
        print("=" * 60)
        print("用法: python hk_stock_api.py <股票代码> [开始日期] [结束日期]")
        print()
        print("示例:")
        print("  python hk_stock_api.py 00700.HK                    # 腾讯，默认过去2年")
        print("  python hk_stock_api.py 0100.HK 2024-01-01 2024-12-31  # 指定日期范围")
        print()
        print("作为模块导入使用:")
        print("  from hk_stock_api import HKStockAPI")
        print("  api = HKStockAPI()")
        print("  df = api.get_daily_data('00700.HK', start_date, end_date)")
        sys.exit(1)
    
    symbol = sys.argv[1]
    
    # 解析日期
    if len(sys.argv) >= 3:
        start_date = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()
    else:
        # 默认2年前
        start_date = date.today().replace(year=date.today().year - 2)
    
    if len(sys.argv) >= 4:
        end_date = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
    else:
        end_date = date.today()
    
    print("=" * 60)
    print(f"获取股票数据: {symbol}")
    print(f"时间范围: {start_date} 到 {end_date}")
    print("=" * 60)
    
    try:
        # 初始化API
        print("正在初始化API...")
        api = HKStockAPI()
        print("API初始化成功!")
        
        # 获取日线数据
        print(f"\n正在获取日线数据...")
        df = api.get_daily_data(symbol, start_date, end_date)
        
        if df.empty:
            print(f"\n警告: 未获取到 {symbol} 的数据")
            print("可能原因:")
            print("  - 股票代码错误")
            print("  - 该时间段内股票未上市")
            print("  - API连接问题")
            sys.exit(1)
        
        # 显示数据
        print(f"\n✅ 成功获取 {len(df)} 条数据\n")
        print("-" * 60)
        print("数据预览:")
        print("-" * 60)
        print(df.to_string())
        
        # 数据统计
        print("\n" + "-" * 60)
        print("数据统计:")
        print("-" * 60)
        print(f"  交易日数: {len(df)}")
        print(f"  开盘价范围: {df['open'].min():.2f} - {df['open'].max():.2f}")
        print(f"  最高价范围: {df['high'].min():.2f} - {df['high'].max():.2f}")
        print(f"  最低价范围: {df['low'].min():.2f} - {df['low'].max():.2f}")
        print(f"  收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"  总成交量: {df['volume'].sum():,}")
        print(f"  平均日成交量: {df['volume'].mean():,.0f}")
        
        # 计算涨跌幅
        if len(df) > 1:
            first_close = df['close'].iloc[0]
            last_close = df['close'].iloc[-1]
            change_pct = (last_close - first_close) / first_close * 100
            print(f"  期间涨跌幅: {change_pct:+.2f}%")
        
        # 保存到CSV
        csv_filename = f'{symbol.replace(".", "_")}_daily.csv'
        df.to_csv(csv_filename)
        print(f"\n💾 数据已保存到: {csv_filename}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
