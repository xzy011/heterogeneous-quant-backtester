import numpy as np
import pandas as pd
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.insert(0, build_path)
import finance_calc_core

def generate_massive_mock_data(num_stocks=2000, num_days=1000):
    """生成全市场级别的大规模仿真数据"""
    print(f"[*] 正在生成全市场规模仿真数据: {num_stocks} 只股票 x {num_days} 天...")
    np.random.seed(42)
    # 模拟每日收益率并生成价格序列
    returns = np.random.normal(0.0005, 0.02, (num_stocks, num_days))
    prices = 100 * np.exp(np.cumsum(returns, axis=1))
    
    # 构建 Pandas DataFrame (供 Pandas 组使用)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='B')
    stock_names = [f"STOCK_{i}" for i in range(num_stocks)]
    price_df = pd.DataFrame(prices.T, index=dates, columns=stock_names)
    
    # 构建 C++ 引擎需要的 SoA 连续内存矩阵 (供 C++ 组使用)
    data_matrix = np.ascontiguousarray(prices.astype(np.float64))
    
    print(f"[+] 数据生成完毕！矩阵形状: {data_matrix.shape}")
    return price_df, data_matrix

def run_pandas_baseline(price_df, window=20):
    """传统量化分析师的写法：纯 Pandas 事件驱动/向量化回测"""
    t0 = time.perf_counter()
    
    # 1. 计算 MA 矩阵 (Pandas 内部调用 Cython)
    ma_df = price_df.rolling(window=window).mean()
    
    # 2. 生成信号 (价格 > MA)
    signal_df = (price_df > ma_df).astype(float)
    
    # 3. 信号平移与收益计算
    shifted_signal = signal_df.shift(1).fillna(0)
    daily_returns = price_df.pct_change().fillna(0)
    strategy_returns = shifted_signal * daily_returns
    
    # 4. 计算等权组合收益
    port_returns = strategy_returns.mean(axis=1)
    
    calc_time = (time.perf_counter() - t0) * 1000
    return calc_time, port_returns

def run_cpp_engine(data_matrix, window=20):
    """你的硬核写法：C++ OpenMP 算子 + NumPy 零拷贝矩阵撮合"""
    engine = finance_calc_core.QuantEngineSoA()
    
    # 预热一下 C++ 引擎（防止动态库初次加载的开销干扰测试）
    _ = engine.compute_ma_batch(data_matrix[:10, :100], window)
    
    t0 = time.perf_counter()
    
    # 1. C++ 引擎瞬间算出全市场 MA 矩阵
    ma_matrix = engine.compute_ma_batch(data_matrix, window)
    
    # 2. NumPy 纯矩阵生成信号
    signal_matrix = np.where(data_matrix > ma_matrix, 1.0, 0.0)
    signal_matrix[:, :window] = 0.0 
    
    # 3. NumPy 矩阵平移与收益计算
    shifted_signals = np.roll(signal_matrix, shift=1, axis=1)
    shifted_signals[:, 0] = 0.0
    
    # 模拟 pct_change (今日价格/昨日价格 - 1)
    shifted_prices = np.roll(data_matrix, shift=1, axis=1)
    shifted_prices[:, 0] = data_matrix[:, 0] # 防止除以 0
    daily_returns = (data_matrix / shifted_prices) - 1
    daily_returns[:, 0] = 0.0
    
    strategy_returns = shifted_signals * daily_returns
    
    # 4. 计算等权组合收益
    port_returns = np.mean(strategy_returns, axis=0)
    
    calc_time = (time.perf_counter() - t0) * 1000
    return calc_time, port_returns

def run_day4_benchmark():
    print("="*70)
    print("    🔥 终极性能压测：传统 Pandas 架构 VS C++ 异构引擎架构")
    print("="*70)
    
    window = 20
    # 生成 2000 只股票，1000 天的数据
    price_df, data_matrix = generate_massive_mock_data(num_stocks=2000, num_days=1000)
    
    print("\n[*] 正在运行 传统 Pandas 回测基准...")
    pandas_time, _ = run_pandas_baseline(price_df, window)
    print(f"[+] Pandas 耗时: {pandas_time:.2f} ms")
    
    print("\n[*] 正在运行 C++ 异构引擎回测...")
    cpp_time, _ = run_cpp_engine(data_matrix, window)
    print(f"[+] C++ 引擎耗时: {cpp_time:.2f} ms")
    
    print("\n" + "="*70)
    print("    🏆 压测结果揭晓")
    print("="*70)
    print(f"数据规模: 2000 只股票 × 1000 个交易日 (共 2,000,000 个数据点)")
    print(f"传统 Pandas 耗时: {pandas_time:>8.2f} ms")
    print(f"C++ 异构引擎耗时: {cpp_time:>8.2f} ms")
    print("-" * 70)
    
    speedup = pandas_time / cpp_time
    print(f"🚀 终极加速比: 你的系统比传统 Pandas 快了 {speedup:.2f} 倍！")
    print("="*70)

if __name__ == "__main__":
    run_day4_benchmark()
