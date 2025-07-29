#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股中盘股突破策略
专门针对10-1000亿港币市值股票的量价突破策略
重点识别中盘股的巨量突破机会，避免小盘股的过度投机和大盘股的流动性问题
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

# 配置日志 - 减少冗余输出
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """交易记录"""
    date: date
    symbol: str
    action: str  # BUY/SELL/SHORT/COVER
    price: float
    quantity: int
    amount: float
    reason: str
    position_type: str = "LONG"  # LONG/SHORT
    pnl: float = 0.0
    pnl_percent: float = 0.0
    hold_days: int = 0
    market_cap: float = 0.0  # 市值

class MidCapBreakoutStrategy:
    """港股中盘股突破策略 - 专注10-1000亿港币市值"""
    
    def __init__(self):
        """初始化策略"""
        self.config = Config.from_env()
        self.quote_ctx = QuoteContext(self.config)
        
        # 策略参数 - 针对大牛股优化
        self.volume_surge_threshold = 5  # 成交量暴增3.5倍（降低门槛捕捉早期信号）
        self.min_price_rise = 0.1  # 最低涨幅6%（提高标准）
        self.max_price_rise = 0.3  # 最高涨幅20%（允许追强势股）
        self.stop_loss_percent = 0.05  # 止损6%（给予更多空间）
        self.take_profit_percent = 0.50  # 止盈20%（提高目标捕捉大波段）
        self.max_positions = 3  # 集中持仓（提高单只收益）
        
        # 做空策略参数
        self.enable_short = False  # 启用做空（默认关闭）
        self.max_short_positions = 3  # 最大空头持仓数
        self.short_volume_surge_threshold = 4.0  # 做空成交量阈值（更严格）
        self.min_price_fall = 0.08  # 最低跌幅8%（做空信号）
        self.max_price_fall = 0.25  # 最高跌幅25%（避免追跌过度）
        self.short_stop_loss_percent = 0.08  # 做空止损8%
        self.short_take_profit_percent = 0.15  # 做空止盈15%
        
        # 扩大市值范围，重点关注中小盘成长股
        self.min_market_cap = 5_0000_0000    # 5亿港币下限（降低）
        self.max_market_cap = 500_0000_0000  # 500亿港币上限（降低）
        
        # 适度放宽流动性要求（捕捉新兴成长股）
        self.min_avg_volume = 100_000     # 日均成交量至少10万港币
        self.min_price = 1   # 最低价格0.5港币
        self.max_price = 1000.0 # 最高价格100港币
        
        # 港股代码范围（全市场覆盖）
        self.hk_stock_ranges = [
            (1, 3999),      # 主板股票 0001-3999
            (8001, 8999),   # 创业板 8001-8999  
            (9001, 9999),   # 特殊股票 9001-9999
        ]
        
        # 回测数据
        self.trades: List[Trade] = []
        self.positions: Dict[str, Dict] = {}  # 多头持仓
        self.short_positions: Dict[str, Dict] = {}  # 空头持仓
        self.daily_portfolio = []
        self.stock_universe = []  # 符合条件的股票池
        self.stock_names = {}  # 股票名称缓存
        
        # 数据缓存配置
        self.cache_dir = "data_cache"
        self.use_cache = True  # 是否使用缓存
        self.cache_days = 7    # 缓存有效期（天）
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
    def generate_stock_symbols(self) -> List[str]:
        """生成港股代码列表"""
        symbols = []
        for start, end in self.hk_stock_ranges:
            for i in range(start, end + 1):
                symbols.append(f"{i:04d}.HK")
        return symbols
    
    def get_cache_filename(self, data_type: str, symbol: str = None) -> str:
        """获取缓存文件名"""
        if symbol:
            return os.path.join(self.cache_dir, f"{data_type}_{symbol.replace('.', '_')}.pkl")
        else:
            return os.path.join(self.cache_dir, f"{data_type}.pkl")
    
    def is_cache_valid(self, cache_file: str) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(cache_file):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return (datetime.now() - file_time).days < self.cache_days
    
    def save_to_cache(self, data, cache_file: str):
        """保存数据到缓存"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def load_from_cache(self, cache_file: str):
        """从缓存加载数据"""
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None
    
    def get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        # 使用预设的常见股票名称映射
        common_names = {
            # 原有股票
            "0323.HK": "马鞍山钢铁",
            "0187.HK": "北京控股", 
            "0308.HK": "中国旅游集团",
            "0314.HK": "南京熊猫",
            "0553.HK": "南京熊猫电子",
            "1456.HK": "国联证券",
            "8017.HK": "猫眼娱乐",
            "2208.HK": "金风科技",
            "0460.HK": "四环医药",
            "0347.HK": "鞍钢股份",
            "0588.HK": "北京北辰实业",
            "1833.HK": "平安好医生",
            "0179.HK": "德昌电机",
            "0354.HK": "中国软件国际",
            "0336.HK": "华润燃气",
            "2068.HK": "中铝国际",
            "0467.HK": "联合能源",
            "1313.HK": "华润水泥",
            "0546.HK": "阜丰集团",
            "0293.HK": "国泰航空",
            "0165.HK": "中国光大控股",
            # 大牛股和潜力股
            "9992.HK": "泡泡玛特",
            "6993.HK": "老铺黄金",
            "2015.HK": "理想汽车",
            "9868.HK": "小鹏汽车",
            "9866.HK": "蔚来",
            "1024.HK": "快手",
            "3690.HK": "美团",
            "9618.HK": "京东集团",
            "9988.HK": "阿里巴巴",
            "0700.HK": "腾讯控股",
            "1810.HK": "小米集团",
            "2331.HK": "李宁",
            "6969.HK": "思摩尔国际",
            "1929.HK": "周大福",
            "2382.HK": "舜宇光学",
            "0285.HK": "比亚迪电子",
            "1211.HK": "比亚迪",
            "6862.HK": "海底捞",
            "9999.HK": "网易",
            "3888.HK": "金山软件",
            "0772.HK": "阅文集团",
            "1691.HK": "JS环球生活",
            "2013.HK": "微盟集团",
            "6060.HK": "众安在线",
            "6618.HK": "京东健康",
            "0241.HK": "阿里健康",
            "1801.HK": "信达生物",
            "6160.HK": "百济神州",
            "2269.HK": "药明生物",
            "3692.HK": "翰森制药",
            "1093.HK": "石药集团",
            "1177.HK": "中国生物制药",
            # 交易中出现的新股票
            "2268.HK": "澳至尊",
            "1508.HK": "中国再保险",
            "0517.HK": "中远海发",
            "0440.HK": "大昌行集团",
            "0598.HK": "中国外运",
            "0558.HK": "力劲科技",
            "0565.HK": "中国服饰控股",
            "0400.HK": "均安控股",
            "0272.HK": "瑞安建业",
            "0119.HK": "保利协鑫能源",
            "1033.HK": "五矿资源",
            "0376.HK": "博耳电力",
            "1658.HK": "邮储银行",
            "0107.HK": "四川成渝",
            "1372.HK": "恒腾网络",
            "0345.HK": "维他奶国际",
            "0357.HK": "美建集团",
            "0596.HK": "浪潮国际",
            "0270.HK": "粤海投资",
            "1558.HK": "伯爵珠宝",
            "1415.HK": "高伟电子",
            "0303.HK": "伟易达",
            "0136.HK": "恒腾网络",
            "0326.HK": "中国星集团",
            "0144.HK": "招商局港口",
            "0570.HK": "中国中药",
            "0038.HK": "第一拖拉机股份",
            "0317.HK": "中船防务",
            "0590.HK": "六福集团",
            "0218.HK": "申万宏源集团",
            "0512.HK": "远大医药",
            "0586.HK": "海螺创业",
            "0412.HK": "中国有色矿业",
            "0004.HK": "九龙仓集团",
            "0327.HK": "百富环球",
            "0363.HK": "上海实业控股",
            "0200.HK": "美丽华酒店",
            "1528.HK": "红星美凯龙",
            "0489.HK": "东风集团股份",
            "0434.HK": "博雅互动",
            "0568.HK": "山东墨龙"
        }
        
        return common_names.get(symbol, symbol)
    
    def format_stock_display(self, symbol: str) -> str:
        """格式化股票显示名称"""
        name = self.get_stock_name(symbol)
        if name != symbol:
            return f"{symbol}({name})"
        return symbol
    
    def get_stock_basic_info(self, symbol: str) -> Optional[Dict]:
        """获取股票基本信息"""
        try:
            # 获取最近的价格和成交量数据
            recent_data = self.quote_ctx.history_candlesticks_by_date(
                symbol,
                Period.Day,
                AdjustType.ForwardAdjust,
                date.today() - timedelta(days=15),  # 增加到15天，提高准确性
                date.today()
            )
            
            if not recent_data or len(recent_data) < 5:  # 至少5天数据
                return None
            
            latest_candle = recent_data[-1]
            price = float(latest_candle.close)
            
            # 价格筛选
            if price < self.min_price or price > self.max_price:
                return None
            
            # 计算平均成交量和成交额
            recent_volumes = [int(c.volume) for c in recent_data[-10:]]
            recent_turnovers = [float(c.turnover) for c in recent_data[-10:]]
            
            avg_volume = np.mean(recent_volumes)
            avg_turnover = np.mean(recent_turnovers)
            
            # 流动性筛选
            if avg_turnover < self.min_avg_volume:
                return None
            
            # 改进的市值估算方法
            # 方法1: 基于换手率估算（更准确）
            if avg_volume > 0 and avg_turnover > 0:
                avg_price = avg_turnover / avg_volume
                # 假设换手率在0.5%-3%之间，估算流通股本
                turnover_rate = 0.015  # 假设平均换手率1.5%
                estimated_shares = avg_volume / turnover_rate
                estimated_market_cap = estimated_shares * price
            else:
                # 方法2: 基于成交额倍数估算（备用）
                # 根据港股经验，日成交额通常是市值的0.5%-2%
                estimated_market_cap = avg_turnover * 80  # 假设日成交额为市值的1.25%
            
            # 市值筛选
            if (estimated_market_cap < self.min_market_cap or 
                estimated_market_cap > self.max_market_cap):
                return None
            
            # 计算波动率（用于后续筛选）
            prices = [float(c.close) for c in recent_data[-10:]]
            price_changes = [prices[i]/prices[i-1]-1 for i in range(1, len(prices))]
            volatility = np.std(price_changes) if len(price_changes) > 1 else 0
            
            return {
                'symbol': symbol,
                'price': price,
                'avg_volume': avg_volume,
                'avg_turnover': avg_turnover,
                'estimated_market_cap': estimated_market_cap,
                'volatility': volatility,
                'data_points': len(recent_data)
            }
            
        except Exception as e:
            # 股票不存在或停牌等，正常情况
            return None
    
    def build_stock_universe(self, max_stocks: int = 800) -> List[str]:  # 增加目标数量
        """构建股票池 - 支持缓存加速"""
        cache_file = self.get_cache_filename("stock_universe")
        
        # 尝试从缓存加载
        if self.use_cache and self.is_cache_valid(cache_file):
            print("从缓存加载股票池...")
            cached_data = self.load_from_cache(cache_file)
            if cached_data:
                print(f"缓存加载成功: {len(cached_data)} 只有效股票")
                return cached_data
        
        logger.info("正在快速扫描全港股市场...")
        
        all_symbols = self.generate_stock_symbols()
        valid_stocks = []
        
        # 优化扫描策略 - 优先扫描活跃股票
        processed = 0
        failed_count = 0
        
        print(f"开始扫描 {len(all_symbols)} 只港股，目标: {max_stocks} 只")
        print("🚀 使用优化扫描策略，优先检查活跃股票...")
        
        # 分批处理，减少API调用频率
        batch_size = 20
        for i in range(0, len(all_symbols), batch_size):
            batch_symbols = all_symbols[i:i+batch_size]
            
            for symbol in batch_symbols:
                try:
                    info = self.get_stock_basic_info(symbol)
                    if info:
                        valid_stocks.append(info)
                        if len(valid_stocks) % 50 == 0:  # 减少日志输出
                            print(f"✅ 已发现 {len(valid_stocks)} 只有效股票")
                    else:
                        failed_count += 1
                    
                    processed += 1
                    
                    # 达到目标数量就停止
                    if len(valid_stocks) >= max_stocks:
                        print(f"🎯 已达到目标数量 {max_stocks}，停止扫描")
                        break
                        
                except Exception as e:
                    failed_count += 1
                    continue
            
            # 批次间暂停
            if len(valid_stocks) < max_stocks:
                time.sleep(0.05)  # 减少延迟
                
                # 进度更新
                if processed % 200 == 0:
                    progress = processed / len(all_symbols) * 100
                    print(f"📊 扫描进度: {progress:.1f}% ({processed}/{len(all_symbols)}) - 有效: {len(valid_stocks)}, 无效: {failed_count}")
            
            if len(valid_stocks) >= max_stocks:
                break
        
        # 按市值排序，优先选择合适市值的股票
        valid_stocks.sort(key=lambda x: x['estimated_market_cap'])
        selected_symbols = [stock['symbol'] for stock in valid_stocks]
        
        print(f"🏁 全市场扫描完成: 发现 {len(selected_symbols)} 只有效股票")
        if valid_stocks:
            min_cap = valid_stocks[0]['estimated_market_cap']/1e8
            max_cap = valid_stocks[-1]['estimated_market_cap']/1e8
            print(f"💰 市值范围: {min_cap:.2f}亿 - {max_cap:.1f}亿港币")
        
        # 保存到缓存
        if self.use_cache:
            self.save_to_cache(selected_symbols, cache_file)
            print(f"💾 股票池已保存到缓存，下次运行将直接加载")
        
        return selected_symbols
    
    def get_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取股票历史数据 - 支持缓存"""
        cache_file = self.get_cache_filename(f"data_{start_date}_{end_date}", symbol)
        
        # 尝试从缓存加载
        if self.use_cache and self.is_cache_valid(cache_file):
            cached_data = self.load_from_cache(cache_file)
            if cached_data is not None and not cached_data.empty:
                logger.debug(f"{symbol}: 从缓存加载数据 ({len(cached_data)} 行)")
                return cached_data
        
        try:
            candles = self.quote_ctx.history_candlesticks_by_date(
                symbol,
                Period.Day,
                AdjustType.ForwardAdjust,
                start_date,
                end_date
            )
            
            if not candles:
                logger.debug(f"{symbol}: API返回空数据")
                return pd.DataFrame()
            
            logger.debug(f"{symbol}: API返回 {len(candles)} 条数据")
            
            data = []
            for candle in candles:
                data.append({
                    'date': candle.timestamp.date(),
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
            
            logger.debug(f"{symbol}: DataFrame有 {len(df)} 行数据")
            
            # 计算技术指标
            if len(df) > 0:
                df = self.calculate_indicators(df)
                logger.debug(f"{symbol}: 计算指标后有 {len(df)} 行数据")
                
                # 保存到缓存
                if self.use_cache:
                    self.save_to_cache(df, cache_file)
            
            return df
            
        except Exception as e:
            logger.debug(f"{symbol}: 获取数据异常 - {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df.empty or len(df) < 5:  # 进一步降低要求
            return df
        
        # 移动平均线（短期为主）
        df['ma3'] = df['close'].rolling(3).mean()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        
        # 成交量指标
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        df['turnover_ma5'] = df['turnover'].rolling(5).mean()
        df['turnover_ma10'] = df['turnover'].rolling(10).mean()
        
        # 成交量暴增比率
        df['volume_surge'] = df['volume'] / df['volume_ma10']
        df['turnover_surge'] = df['turnover'] / df['turnover_ma10']
        
        # 价格变化
        df['price_change'] = df['close'].pct_change()
        df['price_change_3d'] = df['close'].pct_change(3)
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        # 相对强弱（简化版RSI）
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def check_breakout_signal(self, symbol: str, df: pd.DataFrame, current_date: date) -> Tuple[bool, str, float]:
        """检查突破信号 - 针对大牛股优化"""
        if current_date not in df.index or len(df.loc[:current_date]) < 10:  # 降低历史数据要求
            return False, "", 0.0
        
        current_data = df.loc[:current_date]
        latest = current_data.iloc[-1]
        
        # 核心条件1: 成交量放大（捕捉资金流入）
        volume_surge = latest['volume_surge']
        turnover_surge = latest['turnover_surge']
        
        if pd.isna(volume_surge) or volume_surge < self.volume_surge_threshold:
            return False, "", 0.0
        
        # 核心条件2: 价格强势上涨
        price_change = latest['price_change']
        if (pd.isna(price_change) or 
            price_change < self.min_price_rise or 
            price_change > self.max_price_rise):
            return False, "", 0.0
        
        # 核心条件3: 振幅显示活跃度
        amplitude = latest['amplitude']
        if pd.isna(amplitude) or amplitude < 0.06:  # 至少6%振幅
            return False, "", 0.0
        
        # 核心条件4: 突破关键价位
        if pd.isna(latest['ma5']) or latest['close'] <= latest['ma5']:
            return False, "", 0.0  # 必须在5日线上方
        
        # 确认条件 - 捕捉强势特征
        confirmation_score = 0.0
        reasons = []
        
        # 趋势确认（重要）
        if not pd.isna(latest['ma10']):
            if latest['close'] > latest['ma5'] > latest['ma10']:
                confirmation_score += 0.5
                reasons.append("趋势向上")
            elif latest['ma5'] > latest['ma10']:
                confirmation_score += 0.3
                reasons.append("短期走强")
        
        # 连续上涨动能（关键）
        if not pd.isna(latest['price_change_3d']):
            if latest['price_change_3d'] > 0.15:  # 3日涨幅超15%
                confirmation_score += 0.5
                reasons.append("超强动能")
            elif latest['price_change_3d'] > 0.08:  # 3日涨幅超8%
                confirmation_score += 0.3
                reasons.append("强势动能")
        
        # 连涨天数（牛股特征）
        if len(current_data) >= 5:
            recent_changes = current_data['price_change'].tail(5)
            up_days = sum(1 for change in recent_changes if change > 0.02)
            if up_days >= 4:  # 5天中至少4天上涨
                confirmation_score += 0.4
                reasons.append("连续上涨")
            elif up_days >= 3:
                confirmation_score += 0.2
                reasons.append("多日上涨")
        
        # 成交额暴增（资金关注）
        if not pd.isna(turnover_surge) and turnover_surge > self.volume_surge_threshold * 1.2:
            confirmation_score += 0.4
            reasons.append("资金涌入")
        
        # 创新高确认（牛股必备）
        if len(current_data) >= 20:
            recent_high = current_data['high'].tail(20).max()
            if latest['close'] >= recent_high * 0.98:  # 接近或创20日新高
                confirmation_score += 0.4
                reasons.append("创新高")
        
        # 板块轮动确认（通过成交量分布）
        if len(current_data) >= 10:
            recent_volumes = current_data['volume'].tail(10)
            volume_std = recent_volumes.std()
            volume_mean = recent_volumes.mean()
            if volume_std / volume_mean > 0.5:  # 成交量波动大说明有资金博弈
                confirmation_score += 0.2
                reasons.append("资金活跃")
        
        # RSI动能确认
        if not pd.isna(latest['rsi']):
            if 50 < latest['rsi'] < 80:  # 强势区间
                confirmation_score += 0.3
                reasons.append("RSI强势")
            elif latest['rsi'] >= 80:  # 超买但不淘汰（牛股可以持续超买）
                confirmation_score += 0.1
                reasons.append("超强势")
        
        # 信号强度计算
        signal_strength = min(1.0, 
                             volume_surge / 7.0 +       # 成交量权重
                             price_change * 5 +         # 涨幅权重
                             amplitude * 2 +            # 振幅权重
                             confirmation_score)        # 确认权重
        
        # 降低阈值，增加捕捉机会
        if signal_strength > 0.65:
            reason = f"突破信号: 量增{volume_surge:.1f}倍,涨{price_change*100:.1f}%,振幅{amplitude*100:.1f}%"
            if reasons:
                reason += f" ({','.join(reasons)})"
            return True, reason, signal_strength
        
        return False, "", 0.0
    
    def check_short_signal(self, symbol: str, df: pd.DataFrame, current_date: date) -> Tuple[bool, str, float]:
        """检查做空信号 - 识别下跌趋势股票"""
        if not self.enable_short or current_date not in df.index or len(df.loc[:current_date]) < 10:
            return False, "", 0.0
        
        current_data = df.loc[:current_date]
        latest = current_data.iloc[-1]
        
        # 核心条件1: 成交量放大 + 价格大跌
        volume_surge = latest['volume_surge']
        turnover_surge = latest['turnover_surge']
        if pd.isna(volume_surge) or volume_surge < self.short_volume_surge_threshold:
            return False, "", 0.0
        
        # 核心条件2: 价格大幅下跌
        price_change = latest['price_change']
        if (pd.isna(price_change) or 
            price_change > -self.min_price_fall or 
            price_change < -self.max_price_fall):
            return False, "", 0.0
        
        # 核心条件3: 振幅显示恐慌性抛售
        amplitude = latest['amplitude']
        if pd.isna(amplitude) or amplitude < 0.08:  # 至少8%振幅
            return False, "", 0.0
        
        # 核心条件4: 跌破关键支撑位
        if pd.isna(latest['ma5']) or latest['close'] >= latest['ma5']:
            return False, "", 0.0  # 必须跌破5日线
        
        # 确认条件计分
        confirmation_score = 0.0
        reasons = []
        
        # 跌破多条均线（趋势确认）
        if not pd.isna(latest['ma10']) and latest['close'] < latest['ma10']:
            confirmation_score += 0.2
            reasons.append("跌破10日线")
            
        if not pd.isna(latest['ma3']) and latest['close'] < latest['ma3']:
            confirmation_score += 0.1
            reasons.append("跌破3日线")
        
        # 成交量趋势确认
        if not pd.isna(latest['volume_ma5']):
            recent_volume_trend = latest['volume'] / latest['volume_ma5']
            if recent_volume_trend > 2.0:  # 成交量暴增
                confirmation_score += 0.2
                reasons.append("恐慌性抛售")
        
        # RSI超卖但仍在下跌（空头力量强）
        if not pd.isna(latest['rsi']):
            if latest['rsi'] < 30:  # 超卖区间
                confirmation_score += 0.1
                reasons.append("RSI超卖")
            elif 30 <= latest['rsi'] < 50:  # 弱势区间
                confirmation_score += 0.2
                reasons.append("RSI弱势")
        
        # 价格位置确认（在高位下跌更有效）
        if len(current_data) >= 20:
            recent_high = current_data['high'].tail(20).max()
            price_position = latest['close'] / recent_high
            if price_position > 0.8:  # 从高位下跌
                confirmation_score += 0.2
                reasons.append("高位下跌")
            elif price_position > 0.6:
                confirmation_score += 0.1
                reasons.append("中高位下跌")
        
        # 连续下跌确认
        if len(current_data) >= 3:
            recent_changes = current_data['price_change'].tail(3)
            negative_days = sum(1 for change in recent_changes if change < -0.02)
            if negative_days >= 2:
                confirmation_score += 0.1
                reasons.append("持续下跌")
        
        # 计算信号强度
        signal_strength = confirmation_score
        
        if signal_strength > 0.5:  # 做空信号阈值
            reason = f"做空信号: 量增{volume_surge:.1f}倍,跌{abs(price_change)*100:.1f}%,振幅{amplitude*100:.1f}%"
            if reasons:
                reason += f" ({','.join(reasons)})"
            return True, reason, signal_strength
        
        return False, "", 0.0
    
    def check_exit_signal(self, symbol: str, df: pd.DataFrame, current_date: date, 
                         entry_price: float, entry_date: date) -> Tuple[bool, str]:
        """检查退出信号 - 更积极的退出策略"""
        if current_date not in df.index:
            return False, ""
        
        current_data = df.loc[:current_date]
        latest = current_data.iloc[-1]
        current_price = latest['close']
        hold_days = (current_date - entry_date).days
        current_gain = (current_price - entry_price) / entry_price
        
        # 严格止损（无条件执行）
        stop_loss_price = entry_price * (1 - self.stop_loss_percent)
        if current_price <= stop_loss_price:
            pnl_pct = current_gain * 100
            return True, f"触发止损，亏损{abs(pnl_pct):.1f}%"
        
        # 固定止盈（快速获利了结）
        take_profit_price = entry_price * (1 + self.take_profit_percent)
        if current_price >= take_profit_price:
            pnl_pct = current_gain * 100
            return True, f"触发止盈，盈利{pnl_pct:.1f}%"
        
        # 大牛股的动态止盈策略
        # 第1-3天：快速锁定部分利润
        if 1 <= hold_days <= 3 and current_gain >= 0.15:
            if latest['price_change'] < -0.08:  # 大幅回调才考虑止盈
                return True, f"短期获利了结，盈利{current_gain*100:.1f}%"
        
        # 第4-10天：给予更多上涨空间
        if 4 <= hold_days <= 10 and current_gain >= 0.10:
            # 只有出现明显转弱信号才退出
            if latest['price_change'] < -0.10:  # 单日大跌10%
                return True, f"中期止盈，盈利{current_gain*100:.1f}%"
            elif len(current_data) >= 3:
                recent_changes = current_data['price_change'].tail(3)
                if all(change < -0.05 for change in recent_changes):  # 连续3天跌5%
                    return True, f"趋势转弱，盈利{current_gain*100:.1f}%"
        
        # 第11天以上：长期持有逻辑
        if hold_days >= 11 and current_gain >= 0.05:
            # 使用移动止盈策略
            if len(current_data) >= 10:
                recent_high = current_data['high'].tail(10).max()
                drawdown = (recent_high - current_price) / recent_high
                if drawdown > 0.15:  # 从10日高点回撤15%
                    return True, f"移动止盈，盈利{current_gain*100:.1f}%"
        
        # 强化技术面退出条件
        if len(current_data) >= 3:
            # 成交量萎缩且价格下跌（资金撤离）
            if (not pd.isna(latest['volume_surge']) and latest['volume_surge'] < 0.7 and
                latest['price_change'] < -0.04):
                return True, "量价背离，资金撤离"
            
            # 跌破5日均线且放量下跌（趋势转弱）
            if (not pd.isna(latest['ma5']) and current_price < latest['ma5'] and
                not pd.isna(latest['volume_surge']) and latest['volume_surge'] > 1.5 and
                latest['price_change'] < -0.04):
                return True, "跌破均线，趋势转弱"
            
            # 连续2天阴线（快速止损）
            if len(current_data) >= 2:
                recent_changes = current_data['price_change'].tail(2)
                if all(change < -0.02 for change in recent_changes):  # 连续2天跌超2%
                    return True, "连续阴线，及时止损"
        
        # 盈利状态下的风险控制（大牛股可以容忍更高估值）
        if current_gain > 0.10:  # 盈利10%以上才考虑
            # 高位滞涨（5天横盘才走）
            if hold_days >= 5:
                recent_changes = current_data['price_change'].tail(5)
                if all(abs(change) < 0.02 for change in recent_changes):  # 连续5天涨跌幅小于2%
                    return True, f"高位滞涨，锁定盈利{current_gain*100:.1f}%"
            
            # RSI过热退出（大牛股可以持续超买）
            if not pd.isna(latest['rsi']) and latest['rsi'] > 90:  # 提高到90
                return True, f"RSI极度过热，锁定盈利{current_gain*100:.1f}%"
        
        # 亏损状态下的止损加强
        if current_gain < -0.02:  # 亏损2%以上
            # 成交量放大下跌（可能有坏消息）
            if (not pd.isna(latest['volume_surge']) and latest['volume_surge'] > 2.0 and
                latest['price_change'] < -0.03):
                return True, "放量下跌，及时止损"
            
            # 连续下跌（趋势恶化）
            if len(current_data) >= 3:
                recent_changes = current_data['price_change'].tail(3)
                if sum(1 for change in recent_changes if change < -0.02) >= 2:  # 3天中2天跌超2%
                    return True, "持续下跌，止损离场"
        
        # 时间止损（给大牛股更多时间）
        if hold_days >= 20:  # 延长到20天
            if current_gain < -0.03:  # 亏损3%以上才时间止损
                return True, f"时间止损，持仓{hold_days}天"
            elif hold_days >= 30 and current_gain < 0.05:  # 30天后盈利不足5%
                return True, f"收益不佳，持仓{hold_days}天"
        
        # 新增：市场环境恶化退出
        if len(current_data) >= 3:
            recent_changes = current_data['price_change'].tail(3)
            if sum(1 for change in recent_changes if change < -0.04) >= 2:  # 3天中2天大跌
                return True, "市场环境恶化，谨慎退出"
        
        return False, ""
    
    def check_short_exit_signal(self, symbol: str, df: pd.DataFrame, current_date: date, 
                               entry_price: float, entry_date: date) -> Tuple[bool, str]:
        """检查做空平仓信号"""
        if current_date not in df.index:
            return False, ""
        
        current_data = df.loc[:current_date]
        latest = current_data.iloc[-1]
        current_price = latest['close']
        
        # 计算持仓天数和收益
        hold_days = (current_date - entry_date).days
        # 做空收益 = (开仓价 - 当前价) / 开仓价
        current_gain = (entry_price - current_price) / entry_price
        
        # 固定止损（价格上涨超过止损线）
        if current_gain < -self.short_stop_loss_percent:
            return True, f"做空止损，亏损{abs(current_gain)*100:.1f}%"
        
        # 固定止盈（价格下跌达到目标）
        if current_gain >= self.short_take_profit_percent:
            return True, f"做空止盈，盈利{current_gain*100:.1f}%"
        
        # 做空的动态平仓策略
        if hold_days >= 1:
            # 第1-3天：快速锁定利润
            if 1 <= hold_days <= 3 and current_gain >= 0.08:
                if latest['price_change'] > 0.06:  # 反弹超过6%
                    return True, f"短期平仓，盈利{current_gain*100:.1f}%"
            
            # 第4-10天：中期持有
            if 4 <= hold_days <= 10 and current_gain >= 0.05:
                if latest['price_change'] > 0.08:  # 单日大涨8%
                    return True, f"中期平仓，盈利{current_gain*100:.1f}%"
                elif len(current_data) >= 3:
                    recent_changes = current_data['price_change'].tail(3)
                    if all(change > 0.03 for change in recent_changes):  # 连续3天涨3%
                        return True, f"趋势反转，盈利{current_gain*100:.1f}%"
            
            # 第11天以上：长期持有逻辑
            if hold_days >= 11 and current_gain >= 0.03:
                if len(current_data) >= 10:
                    recent_low = current_data['low'].tail(10).min()
                    bounce = (current_price - recent_low) / recent_low
                    if bounce > 0.12:  # 从10日低点反弹12%
                        return True, f"反弹平仓，盈利{current_gain*100:.1f}%"
        
        # 技术面转强信号
        if current_gain > 0:  # 盈利状态
            # 突破关键阻力位
            if not pd.isna(latest['ma5']) and current_price > latest['ma5']:
                return True, f"突破均线，平仓保利{current_gain*100:.1f}%"
            
            # RSI从超卖反弹
            if not pd.isna(latest['rsi']) and latest['rsi'] > 50:
                return True, f"RSI转强，平仓保利{current_gain*100:.1f}%"
        
        # 亏损状态下的风险控制
        if current_gain < -0.02:  # 亏损2%以上
            # 成交量萎缩上涨（可能反弹开始）
            if (not pd.isna(latest['volume_surge']) and latest['volume_surge'] < 0.8 and
                latest['price_change'] > 0.03):
                return True, "缩量上涨，及时止损"
            
            # 连续上涨
            if len(current_data) >= 3:
                recent_changes = current_data['price_change'].tail(3)
                if sum(1 for change in recent_changes if change > 0.02) >= 2:  # 3天中2天涨超2%
                    return True, "持续上涨，止损离场"
        
        # 时间止损（给做空更短的时间窗口）
        if hold_days >= 15:  # 15天时间止损
            if current_gain < -0.02:  # 亏损2%以上
                return True, f"时间止损，持仓{hold_days}天"
            elif hold_days >= 20 and current_gain < 0.03:  # 20天后盈利不足3%
                return True, f"收益不佳，持仓{hold_days}天"
        
        # 市场环境转好
        if len(current_data) >= 3:
            recent_changes = current_data['price_change'].tail(3)
            if sum(1 for change in recent_changes if change > 0.03) >= 2:  # 3天中2天大涨
                return True, "市场转强，平仓离场"
        
        return False, ""
    
    def run_backtest(self, start_date: date, end_date: date, 
                    initial_capital: float = 100000) -> Dict:
        """运行回测"""
        logger.info(f"开始中盘股突破策略回测: {start_date} 至 {end_date}")
        
        # 构建股票池（全市场扫描）
        self.stock_universe = self.build_stock_universe(500)
        
        # 获取历史数据
        print("正在获取历史数据...")
        all_data = {}
        valid_count = 0
        failed_symbols = []
        
        for symbol in self.stock_universe:
            try:
                df = self.get_stock_data(symbol, start_date - timedelta(days=30), end_date)
                if not df.empty and len(df) >= 10:  # 降低数据长度要求
                    all_data[symbol] = df
                    valid_count += 1
                    if valid_count % 50 == 0:  # 减少输出频率
                        print(f"已获取 {valid_count} 只股票的历史数据")
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                failed_symbols.append(symbol)
        
        print(f"成功获取 {len(all_data)} 只股票的历史数据")
        if failed_symbols:
            logger.info(f"无法获取数据的股票: {failed_symbols[:10]}..." if len(failed_symbols) > 10 else f"无法获取数据的股票: {failed_symbols}")
        
        if not all_data:
            logger.error("无法获取历史数据")
            return {}
        
        # 初始化回测
        current_capital = initial_capital
        self.positions = {}
        self.short_positions = {}
        self.trades = []
        self.daily_portfolio = []
        
        # 按日期回测
        current_date = start_date
        trading_days = 0
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 工作日
                trading_days += 1
                
                # 更新持仓价格
                total_position_value = 0
                for symbol, position in list(self.positions.items()):
                    if symbol in all_data and current_date in all_data[symbol].index:
                        current_price = all_data[symbol].loc[current_date, 'close']
                        position['current_price'] = current_price
                        total_position_value += current_price * position['quantity']
                
                for symbol, position in list(self.short_positions.items()):
                    if symbol in all_data and current_date in all_data[symbol].index:
                        current_price = all_data[symbol].loc[current_date, 'close']
                        position['current_price'] = current_price
                        # 做空持仓价值 = 初始价值 + 浮动盈亏
                        short_pnl = (position['entry_price'] - current_price) * position['quantity']
                        total_position_value += position['entry_price'] * position['quantity'] + short_pnl
                
                # 检查卖出信号
                for symbol, position in list(self.positions.items()):
                    if symbol in all_data:
                        should_sell, reason = self.check_exit_signal(
                            symbol, all_data[symbol], current_date,
                            position['entry_price'], position['entry_date']
                        )
                        
                        if should_sell:
                            # 执行卖出
                            sell_price = position['current_price']
                            quantity = position['quantity']
                            sell_amount = sell_price * quantity
                            pnl = (sell_price - position['entry_price']) * quantity
                            pnl_percent = (sell_price - position['entry_price']) / position['entry_price'] * 100
                            hold_days = (current_date - position['entry_date']).days
                            
                            current_capital += sell_amount
                            
                            trade = Trade(
                                date=current_date,
                                symbol=symbol,
                                action='SELL',
                                price=sell_price,
                                quantity=quantity,
                                amount=sell_amount,
                                reason=reason,
                                position_type='LONG',
                                pnl=pnl,
                                pnl_percent=pnl_percent,
                                hold_days=hold_days,
                                market_cap=position.get('market_cap', 0)
                            )
                            self.trades.append(trade)
                            
                            del self.positions[symbol]
                            # 简化日志输出
                            print(f"卖出 {self.format_stock_display(symbol)}: {pnl_percent:+.1f}% ({hold_days}天) - {reason}")
                
                for symbol, position in list(self.short_positions.items()):
                    if symbol in all_data:
                        should_sell, reason = self.check_short_exit_signal(
                            symbol, all_data[symbol], current_date,
                            position['entry_price'], position['entry_date']
                        )
                        
                        if should_sell:
                            # 执行做空平仓
                            cover_price = position['current_price']
                            quantity = position['quantity']
                            cover_amount = cover_price * quantity
                            # 做空收益 = (开仓价 - 平仓价) * 数量
                            pnl = (position['entry_price'] - cover_price) * quantity
                            pnl_percent = (position['entry_price'] - cover_price) / position['entry_price'] * 100
                            hold_days = (current_date - position['entry_date']).days
                            
                            current_capital += pnl + position['entry_price'] * quantity  # 返还保证金 + 收益
                            
                            trade = Trade(
                                date=current_date,
                                symbol=symbol,
                                action='COVER',
                                price=cover_price,
                                quantity=quantity,
                                amount=cover_amount,
                                reason=reason,
                                position_type='SHORT',
                                pnl=pnl,
                                pnl_percent=pnl_percent,
                                hold_days=hold_days,
                                market_cap=position.get('market_cap', 0)
                            )
                            self.trades.append(trade)
                            
                            del self.short_positions[symbol]
                            # 简化日志输出
                            print(f"做空平仓 {self.format_stock_display(symbol)}: {pnl_percent:+.1f}% ({hold_days}天) - {reason}")
                
                # 检查买入信号
                if len(self.positions) < self.max_positions:
                    buy_candidates = []
                    
                    for symbol in all_data.keys():
                        if symbol not in self.positions:
                            should_buy, reason, strength = self.check_breakout_signal(
                                symbol, all_data[symbol], current_date
                            )
                            if should_buy:
                                price = all_data[symbol].loc[current_date, 'close']
                                buy_candidates.append((symbol, price, reason, strength))
                    
                    # 按信号强度排序，优先买入强信号
                    buy_candidates.sort(key=lambda x: x[3], reverse=True)
                    
                    # 执行买入（每天最多买3只小票）
                    for symbol, price, reason, strength in buy_candidates[:3]:
                        if len(self.positions) >= self.max_positions:
                            break
                        
                        # 高质量信号用更大仓位（每次8%资金）
                        position_size = current_capital * 0.08
                        quantity = int(position_size / price / 100) * 100  # 按手买入
                        
                        if quantity > 0:
                            buy_amount = price * quantity
                            current_capital -= buy_amount
                            
                            self.positions[symbol] = {
                                'entry_price': price,
                                'entry_date': current_date,
                                'quantity': quantity,
                                'current_price': price,
                                'market_cap': 0  # 简化处理
                            }
                            
                            trade = Trade(
                                date=current_date,
                                symbol=symbol,
                                action='BUY',
                                price=price,
                                quantity=quantity,
                                amount=buy_amount,
                                reason=reason,
                                position_type='LONG'
                            )
                            self.trades.append(trade)
                            
                            # 简化日志输出
                            print(f"买入 {self.format_stock_display(symbol)}: {quantity}股 @ {price:.2f} (强度:{strength:.2f}) - {reason}")
                
                # 检查做空信号
                if self.enable_short and len(self.short_positions) < self.max_short_positions:
                    for symbol in all_data.keys():
                        if symbol not in self.short_positions:
                            should_short, reason, strength = self.check_short_signal(
                                symbol, all_data[symbol], current_date
                            )
                            if should_short:
                                price = all_data[symbol].loc[current_date, 'close']
                                quantity = int(current_capital * 0.05 / price / 100) * 100 # 按手做空
                                if quantity > 0:
                                    short_amount = price * quantity
                                    current_capital -= short_amount
                                    
                                    self.short_positions[symbol] = {
                                        'entry_price': price,
                                        'entry_date': current_date,
                                        'quantity': quantity,
                                        'current_price': price,
                                        'market_cap': 0 # 简化处理
                                    }
                                    
                                    trade = Trade(
                                        date=current_date,
                                        symbol=symbol,
                                        action='SHORT',
                                        price=price,
                                        quantity=quantity,
                                        amount=short_amount,
                                        reason=reason,
                                        position_type='SHORT'
                                    )
                                    self.trades.append(trade)
                                    
                                    # 简化日志输出
                                    print(f"做空 {self.format_stock_display(symbol)}: {quantity}股 @ {price:.2f} (强度:{strength:.2f}) - {reason}")
                
                # 记录每日组合价值
                portfolio_value = current_capital + total_position_value
                self.daily_portfolio.append({
                    'date': current_date,
                    'capital': current_capital,
                    'positions_value': total_position_value,
                    'total_value': portfolio_value,
                    'positions_count': len(self.positions) + len(self.short_positions)
                })
                
                if trading_days % 30 == 0:
                    print(f"已完成 {trading_days} 个交易日，当前组合价值: {portfolio_value:,.0f}")
            
            current_date += timedelta(days=1)
        
        # 计算最终价值
        final_value = current_capital
        for symbol, position in self.positions.items():
            final_value += position['current_price'] * position['quantity']
        for symbol, position in self.short_positions.items():
            # 做空持仓价值 = 初始价值 + 浮动盈亏
            short_pnl = (position['entry_price'] - position['current_price']) * position['quantity']
            final_value += position['entry_price'] * position['quantity'] + short_pnl
        
        return self.generate_report(initial_capital, final_value, trading_days)
    
    def generate_report(self, initial_capital: float, final_value: float, trading_days: int) -> Dict:
        """生成回测报告"""
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # 交易统计
        buy_trades = [t for t in self.trades if t.action == 'BUY']
        sell_trades = [t for t in self.trades if t.action == 'SELL']
        short_trades = [t for t in self.trades if t.action == 'SHORT']
        cover_trades = [t for t in self.trades if t.action == 'COVER']
        
        # 合并所有平仓交易用于统计
        completed_trades = sell_trades + cover_trades
        
        if completed_trades:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100
            avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0
            avg_hold_days = np.mean([t.hold_days for t in completed_trades])
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # 大赢家统计
            big_winners = [t for t in winning_trades if t.pnl_percent > 20]
            big_losers = [t for t in losing_trades if t.pnl_percent < -10]
            
        else:
            win_rate = avg_win = avg_loss = avg_hold_days = profit_factor = 0
            big_winners = big_losers = []
        
        # 最大回撤
        portfolio_values = [p['total_value'] for p in self.daily_portfolio]
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return_percent': total_return,
            'annualized_return': total_return * 365 / trading_days,
            'max_drawdown_percent': max_drawdown,
            'total_trades': len(buy_trades) + len(short_trades),
            'completed_trades': len(completed_trades),
            'win_rate_percent': win_rate,
            'avg_win_percent': avg_win,
            'avg_loss_percent': avg_loss,
            'avg_hold_days': avg_hold_days,
            'profit_factor': profit_factor,
            'big_winners': len(big_winners),
            'big_losers': len(big_losers),
            'active_positions': len(self.positions) + len(self.short_positions),
            'trading_days': trading_days,
            'stock_universe_size': len(self.stock_universe)
        }
        
        self.print_report(results)
        return results
    
    def print_report(self, results: Dict):
        """打印回测报告"""
        print("\n" + "="*60)
        print("🎯 港股中盘股突破策略回测报告")
        print("="*60)
        
        print(f"\n💰 资金情况:")
        print(f"初始资金: {results['initial_capital']:,.0f} HKD")
        print(f"最终价值: {results['final_value']:,.0f} HKD")
        print(f"绝对收益: {results['final_value'] - results['initial_capital']:+,.0f} HKD")
        
        print(f"\n📈 收益指标:")
        print(f"总收益率: {results['total_return_percent']:+.2f}%")
        print(f"年化收益率: {results['annualized_return']:+.2f}%")
        print(f"最大回撤: {results['max_drawdown_percent']:.2f}%")
        
        print(f"\n📊 交易统计:")
        print(f"股票池规模: {results['stock_universe_size']} 只中盘股")
        print(f"交易天数: {results['trading_days']} 天")
        print(f"总交易次数: {results['total_trades']}")
        print(f"完成交易: {results['completed_trades']}")
        print(f"当前持仓: {results['active_positions']} 只")
        
        if results['completed_trades'] > 0:
            print(f"胜率: {results['win_rate_percent']:.1f}%")
            print(f"平均盈利: {results['avg_win_percent']:+.2f}%")
            print(f"平均亏损: {results['avg_loss_percent']:+.2f}%")
            print(f"盈亏比: {results['profit_factor']:.2f}")
            print(f"平均持仓: {results['avg_hold_days']:.1f} 天")
            print(f"大赢家(>20%): {results['big_winners']} 笔")
            print(f"大亏损(<-10%): {results['big_losers']} 笔")
        
        print(f"\n🏆 最佳交易:")
        winning_trades = [t for t in self.trades if t.action in ['SELL', 'COVER'] and t.pnl > 0]
        if winning_trades:
            best_trades = sorted(winning_trades, key=lambda x: x.pnl_percent, reverse=True)[:5]
            for trade in best_trades:
                trade_type = "多头" if trade.position_type == "LONG" else "空头"
                print(f"{trade.date} {self.format_stock_display(trade.symbol)} ({trade_type}): +{trade.pnl_percent:.1f}% "
                      f"({trade.hold_days}天) - {trade.reason}")
        
        print(f"\n💔 最差交易:")
        losing_trades = [t for t in self.trades if t.action in ['SELL', 'COVER'] and t.pnl < 0]
        if losing_trades:
            worst_trades = sorted(losing_trades, key=lambda x: x.pnl_percent)[:3]
            for trade in worst_trades:
                trade_type = "多头" if trade.position_type == "LONG" else "空头"
                print(f"{trade.date} {self.format_stock_display(trade.symbol)} ({trade_type}): {trade.pnl_percent:.1f}% "
                      f"({trade.hold_days}天) - {trade.reason}")
        
        # 当前持仓
        if self.positions:
            print(f"\n📋 当前多头持仓:")
            for symbol, pos in self.positions.items():
                pnl_pct = (pos['current_price'] - pos['entry_price']) / pos['entry_price'] * 100
                days = (date.today() - pos['entry_date']).days
                print(f"{self.format_stock_display(symbol)}: {pnl_pct:+.1f}% ({days}天)")
        if self.short_positions:
            print(f"\n📋 当前空头持仓:")
            for symbol, pos in self.short_positions.items():
                pnl_pct = (pos['entry_price'] - pos['current_price']) / pos['entry_price'] * 100
                days = (date.today() - pos['entry_date']).days
                print(f"{self.format_stock_display(symbol)}: {pnl_pct:+.1f}% ({days}天)")

def main():
    """主函数"""
    strategy = MidCapBreakoutStrategy()
    
    # 回测参数
    start_date = date(2020, 7, 1)
    end_date = date(2024, 7, 1)
    initial_capital = 100000
    
    print("🎯 港股中盘股多空策略")
    print(f"📅 回测期间: {start_date} 至 {end_date}")
    print(f"💰 初始资金: {initial_capital:,} HKD")
    print(f"🎲 策略特点: 中盘股多空并进，突破做多+暴跌做空")
    print(f"📏 市值范围: {strategy.min_market_cap/1e8:.1f}-{strategy.max_market_cap/1e8:.0f}亿港币")
    print(f"⚡ 多头成交量阈值: {strategy.volume_surge_threshold}倍暴增")
    if strategy.enable_short:
        print(f"⚡ 空头成交量阈值: {strategy.short_volume_surge_threshold}倍暴增")
        print(f"🎯 多头止盈止损: +{strategy.take_profit_percent*100:.0f}%/−{strategy.stop_loss_percent*100:.0f}%")
        print(f"🎯 空头止盈止损: +{strategy.short_take_profit_percent*100:.0f}%/−{strategy.short_stop_loss_percent*100:.0f}%")
        print(f"📊 最大持仓: 多头{strategy.max_positions}只 + 空头{strategy.max_short_positions}只")
    else:
        print(f"🚫 做空功能: 已关闭（纯多头策略）")
        print(f"🎯 止盈止损: +{strategy.take_profit_percent*100:.0f}%/−{strategy.stop_loss_percent*100:.0f}%")
        print(f"📊 最大持仓: {strategy.max_positions}只")
    
    # 运行回测 - 全市场扫描
    results = strategy.run_backtest(start_date, end_date, initial_capital)
    
    # 保存结果
    if strategy.trades:
        trades_df = pd.DataFrame([
            {
                'date': t.date,
                'symbol': t.symbol,
                'action': t.action,
                'position_type': t.position_type,
                'price': t.price,
                'quantity': t.quantity,
                'amount': t.amount,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'hold_days': t.hold_days,
                'reason': t.reason
            } for t in strategy.trades
        ])
        csv_filename = 'mid_cap_long_short_trades.csv' if strategy.enable_short else 'mid_cap_trades.csv'
        trades_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 交易记录已保存: {csv_filename}")

if __name__ == "__main__":
    main() 