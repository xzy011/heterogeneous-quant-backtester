import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

# 1. 动态路径绑定：确保 Python 能找到 build 目录下的 .so 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本在 python/ 目录下，build 在根目录下
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.append(build_path)

try:
    import finance_calc_core
    print(f"[成功] 已加载模块: {finance_calc_core.__file__}")
except ImportError as e:
    print(f"[错误] 无法加载 finance_calc_core。请检查 build 路径: {build_path}")
    print(f"详细错误: {e}")
    sys.exit(1)

def run_benchmark():
    # --- 数据准备 ---
    n_stocks = 1000
    n_days = 100000
    window = 30
    print(f"正在准备压测数据: {n_stocks} 只股票, 每只 {n_days} 条记录 (约 800MB)...")
    
    # 生成随机数据并确保是 float64
    data = np.random.randn(n_stocks, n_days).astype(np.float64)

    # --- 引擎实例化 ---
    # 根据你的 dir() 结果，compute_ma_batch 是 QuantEngineSoA 的成员方法
    try:
        engine = finance_calc_core.QuantEngineSoA()
        print("[成功] QuantEngineSoA 引擎实例化完成")
    except AttributeError:
        print("[错误] 模块中未找到 QuantEngineSoA 类，请检查 C++ 导出名。")
        sys.exit(1)

    # --- 压测配置 ---
    threads_to_test = [1, 2, 4, 6, 8, 10, 12, 14]
    results = []

    print(f"{'线程数':<10} | {'平均耗时(ms)':<15} | {'加速比':<10}")
    print("-" * 45)

    # 1. 单线程基准测试
    os.environ["OMP_NUM_THREADS"] = "1"
    # 预热算子：排除冷启动影响并触发 CPU 睿频
    engine.compute_ma_batch(data, window)
    
    start_base = time.perf_counter()
    engine.compute_ma_batch(data, window)
    base_time = (time.perf_counter() - start_base) * 1000
    results.append(base_time)
    print(f"{1:<10} | {base_time:<15.2f} | {1.0:<10.2f}x")

    # 2. 多线程循环测试
    for t in threads_to_test[1:]:
        os.environ["OMP_NUM_THREADS"] = str(t)
        
        # 多次测量取平均值
        iters = 3
        latencies = []
        for _ in range(iters):
            start = time.perf_counter()
            engine.compute_ma_batch(data, window)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(latencies) / iters
        speedup = base_time / avg_time
        results.append(avg_time)
        print(f"{t:<10} | {avg_time:<15.2f} | {speedup:<10.2f}x")

    # --- 3. 绘制加速比曲线 ---
    plt.figure(figsize=(12, 7))
    speedups = [base_time / t for t in results]
    
    # 实际加速比
    plt.plot(threads_to_test, speedups, marker='o', markersize=8, linewidth=2, color='#1f77b4', label='Actual Speedup')
    # 理想线性加速比
    plt.plot(threads_to_test, threads_to_test, linestyle='--', color='#d62728', alpha=0.6, label='Ideal Linear Speedup')
    
    # 寻找性能甜点位
    max_idx = np.argmax(speedups)
    plt.annotate(f'Sweet Spot: {threads_to_test[max_idx]} Cores{speedups[max_idx]:.2f}x Speedup',
                 xy=(threads_to_test[max_idx], speedups[max_idx]),
                 xytext=(threads_to_test[max_idx]-2, speedups[max_idx]+1),
                 bbox=dict(boxstyle="round", fc="w"),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.title('OpenMP Parallel Performance: QuantEngineSoA (MA Operator)', fontsize=14)
    plt.xlabel('Number of Threads (i9-12900H P/E Cores)', fontsize=12)
    plt.ylabel('Speedup Factor (relative to 1-Core)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 保存图片
    plt.savefig('performance_curve.png', dpi=300)
    print(f"[完成] 加速比曲线已生成: {os.getcwd()}/performance_curve.png")

if __name__ == "__main__":
    run_benchmark()
