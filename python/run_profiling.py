import numpy as np
import os
import sys

# ✅ Profiling 阶段：强制单线程（排除并行噪音）
os.environ["OMP_NUM_THREADS"] = "1"

# ✅ 确保能找到 C++ 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.append(build_path)

import finance_calc_core

def profile_run():
    n_stocks = 2000     # 比 benchmark 大，保证运行时间
    n_days = 100000
    window = 30

    print("[Profiling] Preparing data...")
    data = np.random.randn(n_stocks, n_days).astype(np.float64)

    engine = finance_calc_core.QuantEngineSoA()

    print("[Profiling] Running compute_ma_batch loop...")
    for i in range(10):   # 多次循环，方便 perf 采样
        engine.compute_ma_batch(data, window)
        print(f"Iteration {i + 1} done")

if __name__ == "__main__":
    profile_run()
