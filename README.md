# 趋势突破回测

## 运行

- **港股 + 美股（推荐）**：`python dual_breakout_strategy.py`  
  股票池：`dual_universe.csv` 或 `dual_universe.example.csv`（`symbol` 列，如 `00700.HK`、`AAPL.US`）。
- **仅港股全表**：`python trend_breakout_v2.py`（需 `hk_all_stocks.csv`）  
- **参数网格**：`python trend_breakout_v2.py optimize`

环境变量：`.env` 配置 Longport；`LONGPORT_REQUEST_PAUSE` 控制拉数间隔。

## 未来函数与回测偏差

见 `dual_breakout_strategy.py` 文件头注释：数据截止 T-1、Donchian 使用 `shift(1)`；成交价按昨收简化，偏乐观；小股票池会抬高夏普。日线通过 `hk_stock_api.get_daily_data` **分段合并**以突破单次约 1000 根限制。

## 验证

`python strategy_validation.py`（无 CSV 时走合成数据）。
