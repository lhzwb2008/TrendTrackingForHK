# 港股趋势追踪系统

## 安装部署

### 1. 环境要求
- Python 3.8+

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 环境配置
创建 `.env` 文件并配置必要的 API 密钥：
```bash
cp .env.example .env
# 编辑 .env 文件，填入相应的 API 配置
```

## 运行命令

### 启动主程序
```bash
python small_cap_breakout.py
```

### 其他说明
- 程序会自动创建 `data_cache/` 目录用于缓存数据
- 交易记录会保存为 CSV 文件
- 确保网络连接正常，程序需要访问港股数据接口 