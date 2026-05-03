import numpy as np
import pandas as pd
import time
import os
import sys

# 从你刚才跑通的 data_feeder.py 中引入获取数据的函数
from data_feeder import fetch_real_data 

current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.insert(0, build_path)
import finance_calc_core

def run_vectorized_backtest():
    print("="*60)
    print("    异构计算驱动：纯向量化回测引擎 (MVP)")
    print("="*60)

    # 1. 获取数据 (使用 Day1 的容错函数，依然测试 10 只股票)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    dates, stock_names, data_matrix = fetch_real_data(tickers, period="3y")
    
    # 2. 调用 C++ 引擎计算指标矩阵
    engine = finance_calc_core.QuantEngineSoA()
    window = 20
    
    t0 = time.perf_counter()
    # 获得 [n_stocks, n_days] 的 MA 矩阵
    ma_matrix = engine.compute_ma_batch(data_matrix, window)
    
    # ========================================================
    # 核心：纯向量化信号生成与撮合 (在 Python 端利用 NumPy 矩阵运算)
    # ========================================================
    
    # 3. 生成交易信号矩阵 (Price > MA20 为 1，否则为 0)
    # 忽略前 window 天的无效数据
    signal_matrix = np.where(data_matrix > ma_matrix, 1.0, 0.0)
    signal_matrix[:, :window] = 0.0 
    
    # 4. 计算真实的每日收益率矩阵 (今日价格 / 昨日价格 - 1)
    # 为了矩阵维度对齐，我们在最前面补一列 0
    price_df = pd.DataFrame(data_matrix.T)
    daily_returns = price_df.pct_change().fillna(0).values.T # 维度: [n_stocks, n_days]
    
    # 5. 向量化撮合：策略收益 = 昨天的信号 * 今天的真实收益
    # 使用 np.roll 将信号矩阵向右平移 1 天 (模拟 T 日收盘后发信号，T+1 日吃收益)
    shifted_signals = np.roll(signal_matrix, shift=1, axis=1)
    shifted_signals[:, 0] = 0.0
    
    strategy_returns = shifted_signals * daily_returns
    
    # 6. 计算资金曲线 (Cumulative Returns)
    # 假设每只股票分配 1 的初始资金
    cumulative_returns = np.cumprod(1 + strategy_returns, axis=1)
    
    total_time = (time.perf_counter() - t0) * 1000
    print(f"\n[+] 向量化回测完成！(含 C++ 指标计算 + 矩阵信号撮合)")
    print(f"[+] 总耗时: {total_time:.2f} ms")
    
    # 7. 打印结果
    print(f"\n========== 策略绩效预览 (MA20 突破策略) ==========")
    print(f"{'股票代码':<15} | {'买入持有总收益':<15} | {'策略总收益':<15}")
    print("-" * 50)
    
    for i in range(len(stock_names)):
        # 买入持有收益 (最后一天价格 / 第一天价格 - 1)
        bh_return = (data_matrix[i, -1] / data_matrix[i, 0]) - 1
        # 策略收益
        strat_return = cumulative_returns[i, -1] - 1
        
        print(f"{stock_names[i]:<15} | {bh_return*100:>10.2f}%    | {strat_return*100:>10.2f}%")

if __name__ == "__main__":
    run_vectorized_backtest()
