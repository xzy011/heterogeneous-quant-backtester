import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

# 路径绑定
current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.append(build_path)

import finance_calc_core

def run_simd_comparison():
    # 准备大规模数据：1000 只股票, 10 万条记录 (约 800MB)
    n_stocks = 1000
    n_days = 100000
    window = 30
    data = np.random.randn(n_stocks, n_days).astype(np.float64)
    
    engine = finance_calc_core.QuantEngineSoA()
    
    # 我们固定线程数为 6 (你的 P-Core 甜点位)，专注于指令级优化的对比
    os.environ["OMP_NUM_THREADS"] = "14"
    print(f"--- SIMD 性能对比测试 (线程数: 14, 数据规模: {n_stocks}x{n_days}) ---")

    # --- 1. 测试标量版本  ---
    engine.compute_ma_batch(data, window) # 预热
    t0 = time.perf_counter()
    for _ in range(5):
        engine.compute_ma_batch(data, window)
    scalar_time = ((time.perf_counter() - t0) / 5) * 1000
    print(f"标量版 (Scalar + OMP) 平均耗时: {scalar_time:.2f} ms")

    # --- 2. 测试 SIMD 版本  ---
    # 注意：确保你的 C++ 导出名是 compute_ma_batch_simd
    engine.compute_ma_batch_simd(data, window) # 预热
    t1 = time.perf_counter()
    for _ in range(5):
        engine.compute_ma_batch_simd(data, window)
    simd_time = ((time.perf_counter() - t1) / 5) * 1000
    print(f"向量化 (SIMD + OMP) 平均耗时: {simd_time:.2f} ms")

    # --- 3. 结果分析 ---
    improvement = (scalar_time - simd_time) / scalar_time * 100
    speedup_vs_scalar = scalar_time / simd_time
    print(f"[结论] SIMD 优化提升了 {improvement:.2f}% 的性能")
    print(f"相对于标量版加速比: {speedup_vs_scalar:.2f}x")

    # --- 4. 生成对比图 ---
    labels = ['Scalar + OMP', 'SIMD + OMP']
    means = [scalar_time, simd_time]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, color=['#fb8072', '#8dd3c7'], width=0.5)
    plt.ylabel('Latency (ms)')
    plt.title('Instruction Level Optimization: Scalar vs SIMD')
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}ms', ha='center', va='bottom')
    
    plt.savefig('simd_comparison.png')
    print(f"[成功] 对比图表已生成: simd_comparison.png")

if __name__ == "__main__":
    run_simd_comparison()
