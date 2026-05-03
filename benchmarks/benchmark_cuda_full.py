import numpy as np
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
build_path = os.path.join(project_root, 'build')
sys.path.insert(0, build_path)
sys.path.insert(0, project_root)

try:
    import finance_calc_core
    print("[系统] 成功加载 finance_calc_core 模块")
except ImportError:
    print("[错误] 未找到编译后的模块")
    sys.exit(1)

def run_full_benchmark():
    n_stocks = 2000
    n_days = 100000
    window = 30

    print(f"\n" + "="*70)
    print(f"    AI Infra 异构计算全模式对比（含传统CPU标量版）")
    print(f"    数据规模：{n_stocks} 股票 x {n_days} 天")
    print(f"    GPU: RTX 3060 Laptop (6GB)")
    print("="*70)

    data = np.random.randn(n_stocks, n_days).astype(np.float64)
    engine = finance_calc_core.QuantEngineSoA()

    
    # ========== 1. 真正单线程 ==========
    print(f"\n[*] CPU 标量单线程（OMP_NUM_THREADS=1）...", end="", flush=True)
    
    engine.compute_ma_batch(data, window)  # 预热
    t = time.perf_counter()
    engine.compute_ma_batch(data, window)  # 只跑1次
    cpu_scalar = (time.perf_counter() - t) * 1000
    print(f" {cpu_scalar:.2f} ms")

    # ========== 2. CPU 标量 + OpenMP（批量）==========
    print(f"[*] CPU 标量 + OpenMP（批量）...", end="", flush=True)
    engine.compute_ma_batch(data, window)
    t = time.perf_counter()
    for _ in range(3):
        engine.compute_ma_batch(data, window)
    cpu_omp = ((time.perf_counter() - t) / 3) * 1000
    print(f" {cpu_omp:.2f} ms")

    # ========== 3. CPU SIMD + OpenMP ==========
    print(f"[*] CPU SIMD + OMP...", end="", flush=True)
    engine.compute_ma_batch_simd(data, window)
    t = time.perf_counter()
    for _ in range(3):
        engine.compute_ma_batch_simd(data, window)
    cpu_simd = ((time.perf_counter() - t) / 3) * 1000
    print(f" {cpu_simd:.2f} ms")

    # ========== 4. CUDA 传统 ==========
    print(f"[*] CUDA 传统（每次malloc+同步传输）...", end="", flush=True)
    engine.compute_ma_cuda(data, window)
    t = time.perf_counter()
    for _ in range(3):
        engine.compute_ma_cuda(data, window)
    cuda_trad = ((time.perf_counter() - t) / 3) * 1000
    print(f" {cuda_trad:.2f} ms")

    # ========== 5. CUDA Pinned ==========
    print(f"[*] CUDA Pinned（页锁定内存+异步传输）...", end="", flush=True)
    engine.compute_ma_cuda_pinned(data, window)
    t = time.perf_counter()
    for _ in range(3):
        engine.compute_ma_cuda_pinned(data, window)
    cuda_pinned = ((time.perf_counter() - t) / 3) * 1000
    print(f" {cuda_pinned:.2f} ms")

    # ========== 6. CUDA 常驻内存 ==========
    print(f"[*] CUDA 常驻（上传一次，零拷贝计算）...", end="", flush=True)
    engine.upload_to_gpu(data)
    
    engine.compute_ma_gpu_resident(window)
    t = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        res = engine.compute_ma_gpu_resident(window)
    cuda_resident = ((time.perf_counter() - t) / n_iters) * 1000
    print(f" {cuda_resident:.2f} ms")
    engine.free_gpu_memory()

    # ========== 结果汇总 ==========
    print(f"\n" + "-"*70)
    print(f"{'版本':<40} | {'耗时(ms)':>10} | {'相对加速':>10}")
    print("-"*70)
    print(f"{'CPU 标量单线程（最基础）':<40} | {cpu_scalar:>10.2f} | {'1.00x':>10}")
    print(f"{'CPU 标量 + OpenMP':<40} | {cpu_omp:>10.2f} | {cpu_scalar/cpu_omp:>9.2f}x")
    print(f"{'CPU SIMD + OMP':<40} | {cpu_simd:>10.2f} | {cpu_scalar/cpu_simd:>9.2f}x")
    print(f"{'CUDA 传统（每次malloc+同步）':<40} | {cuda_trad:>10.2f} | {cpu_scalar/cuda_trad:>9.2f}x")
    print(f"{'CUDA Pinned（页锁定内存）':<40} | {cuda_pinned:>10.2f} | {cpu_scalar/cuda_pinned:>9.2f}x")
    print(f"{'CUDA 常驻内存（零拷贝）':<40} | {cuda_resident:>10.2f} | {cpu_scalar/cuda_resident:>9.2f}x")
    print("-"*70)

    print(f"\n[关键洞察]")
    print(f"1. OpenMP 并行加速：{cpu_scalar/cpu_omp:.2f}x")
    print(f"2. SIMD 向量化再加速：{cpu_omp/cpu_simd:.2f}x")
    print(f"3. GPU 传输开销占比：{(cuda_trad-cuda_resident)/cuda_trad*100:.1f}%")
    print(f"4. 终极加速比（常驻 vs 标量单线程）：{cpu_scalar/cuda_resident:.2f}x")

if __name__ == "__main__":
    run_full_benchmark()