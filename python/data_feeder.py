import numpy as np
import pandas as pd
import yfinance as yf
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore') # 忽略 yfinance 的一些警告

current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.insert(0, build_path)

try:
    import finance_calc_core
    print("[系统] 成功加载 finance_calc_core 模块！")
except ImportError:
    print("[错误] 未找到编译后的模块，请确认 build 目录中存在 .so 文件")
    sys.exit(1)

def generate_mock_data(num_stocks=10, num_days=756):
    """生成仿真的股票数据（几何布朗运动），用于在断网时继续开发"""
    print(f"[*] 正在生成 {num_stocks} 只股票，过去 {num_days} 天的仿真数据...")
    np.random.seed(42)
    # 假设初始价格为 100，每天的波动率服从正态分布
    returns = np.random.normal(0.0005, 0.02, (num_stocks, num_days))
    prices = 100 * np.exp(np.cumsum(returns, axis=1))
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='B')
    stock_names = [f"MOCK_STOCK_{i}" for i in range(num_stocks)]
    
    # 转换为 C++ 引擎需要的连续内存矩阵
    data_matrix = np.ascontiguousarray(prices.astype(np.float64))
    return dates, stock_names, data_matrix

def fetch_real_data(tickers, period="3y"):
    """获取真实股票数据，带有自动容错降级机制"""
    print(f"[*] 尝试从 yfinance 下载 {len(tickers)} 只股票过去 {period} 的日线数据...")
    try:
        df = yf.download(tickers, period=period, interval="1d", progress=False)['Close']
        
        # 检查是否下载失败（数据为空）
        if df.empty or len(df.columns) == 0:
            raise ValueError("下载的数据为空")
            
        df = df.ffill().bfill()
        data_matrix = np.ascontiguousarray(df.T.values.astype(np.float64))
        print(f"[+] 真实数据清洗完成！矩阵维度: {data_matrix.shape}")
        return df.index, df.columns, data_matrix
        
    except Exception as e:
        print(f"[!] yfinance 下载失败: {e}")
        print("[!] 自动降级为【仿真数据模式】，以确保开发不被阻塞！")
        # 3年大约 756 个交易日
        return generate_mock_data(len(tickers), 756)

def run_day1_integration():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    dates, stock_names, data_matrix = fetch_real_data(tickers, period="3y")

    engine = finance_calc_core.QuantEngineSoA()
    window = 20

    print(f"[*] 正在调用 C++ 引擎计算 {window} 日均线 (OpenMP 批量算子)...")
    t0 = time.perf_counter()
    
    ma_matrix = engine.compute_ma_batch(data_matrix, window)
    
    calc_time = (time.perf_counter() - t0) * 1000
    print(f"[+] 计算完成！耗时: {calc_time:.2f} ms")

    target_stock_idx = 0
    target_stock_name = stock_names[target_stock_idx]
    
    print(f"========== 结果验证: {target_stock_name} 最近 5 天数据 ==========")
    print(f"{'日期':<15} | {'收盘价':<15} | {'C++计算的 MA20':<15}")
    print("-" * 50)
    
    for i in range(-5, 0):
        date_str = dates[i].strftime("%Y-%m-%d")
        price = data_matrix[target_stock_idx, i]
        ma_val = ma_matrix[target_stock_idx, i]
        print(f"{date_str:<15} | {price:<15.2f} | {ma_val:<15.2f}")

if __name__ == "__main__":
    run_day1_integration()
