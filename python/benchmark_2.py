import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))

import finance_calc_core
import numpy as np
import pandas as pd
import time

def benchmark():
    # 1. 准备 1000 万条数据，测试 i9 极限性能
    N = 10_000_000
    window = 20
    print(f"--- 性能压测开始 (数据规模: {N}, 窗口大小: {window}) ---")
    
    engine = finance_calc_core.QuantEngineSoA()
    engine.reserve(N)
    data_list = [float(i % 100) for i in range(N)]
    for d in data_list:
        engine.add_bar(0, 0, 0, d, 0)
    
    close_np = np.array(data_list)
    close_pd = pd.Series(data_list)

    # --- 实验 1: Pandas (工业标准) ---
    start = time.perf_counter()
    res_pd = close_pd.rolling(window=window).mean()
    end = time.perf_counter()
    t_pd = (end - start) * 1000
    print(f"Pandas rolling mean 耗时: {t_pd:.2f} ms")

    # --- 实验 2: NumPy (向量化，但 rolling 较弱) ---
    start = time.perf_counter()
    # NumPy 通常使用 cumsum 实现 rolling
    ret = np.cumsum(close_np, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    res_np = ret[window - 1:] / window
    end = time.perf_counter()
    t_np = (end - start) * 1000
    print(f"NumPy cumsum 优化耗时: {t_np:.2f} ms")

    # --- 实验 3: 我的 C++ 优化算子 ---
    start = time.perf_counter()
    res_cpp = engine.compute_ma(window)
    end = time.perf_counter()
    t_cpp = (end - start) * 1000
    print(f"C++ SoA 优化算子耗时: {t_cpp:.2f} ms")

    # --- 结论分析 ---
    print(f"\n结论: C++ 相比 Pandas 加速比: {t_pd/t_cpp:.2f}x")
    print(f"结论: C++ 相比 NumPy 加速比: {t_np/t_cpp:.2f}x")

if __name__ == "__main__":
    benchmark()
