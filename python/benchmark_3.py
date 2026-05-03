import numpy as np
import time
import os
import sys
# 确保路径正确
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
import finance_calc_core

def benchmark_parallel():
    num_stocks = 1000
    data_len = 100_000
    window = 20
    
    print(f"准备数据: {num_stocks} 只股票, 每只 {data_len} 条记录...")
    # 生成 1 亿个 float64 数据点
    matrix_data = np.random.rand(num_stocks, data_len).astype(np.float64)
    engine = finance_calc_core.QuantEngineSoA()

    # --- 实验 1: 强制单线程运行 ---
    # 通过环境变量限制 OpenMP 只使用 1 个核心
    os.environ["OMP_NUM_THREADS"] = "1"
    start = time.perf_counter()
    # 注意：这里调用的还是同一个函数，但环境限制了它只能用单核
    res_single = engine.compute_ma_batch(matrix_data, window)
    t_single = (time.perf_counter() - start) * 1000
    print(f"单线程批量计算耗时: {t_single:.2f} ms")

    # --- 实验 2: 全核并行运行 ---
    # 开启你的 i9 核心数（建议设为 14 或不设，默认全开）
    os.environ["OMP_NUM_THREADS"] = "6"
    start = time.perf_counter()
    res_parallel = engine.compute_ma_batch(matrix_data, window)
    t_parallel = (time.perf_counter() - start) * 1000
    
    print(f"OpenMP 并行计算耗时: {t_parallel:.2f} ms")
    print(f"结果形状: {res_parallel.shape}")

    # --- 计算加速比 ---
    if t_parallel > 0:
        print(f"--- 结论 ---")
        print(f"i9 处理器并行加速比: {t_single / t_parallel:.2f}x")

if __name__ == "__main__":
    benchmark_parallel()
