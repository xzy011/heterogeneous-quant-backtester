import numpy as np
import pandas as pd
import time
import os
import sys

# 导入无头渲染，防止 WSL 报错
import matplotlib
matplotlib.use('Agg')

current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.insert(0, build_path)
import finance_calc_core

# 引入终极压测的数据生成器 (不用 yfinance 了，直接造 200 万数据点压测 GPU)
from ultimate_benchmark import generate_massive_mock_data

# =====================================================================
# 模块一：策略抽象层 (支持 GPU)
# =====================================================================
class BaseStrategy:
    def generate_signals(self, data_matrix, engine, use_gpu=False):
        raise NotImplementedError()

class MACrossStrategy(BaseStrategy):
    def __init__(self, window=20):
        self.window = window
        
    def generate_signals(self, data_matrix, engine, use_gpu=False):
        # 🌟 核心分发：根据引擎类型调用不同的底层算子
        if use_gpu:
            # 调用 GPU 常驻内存算子 (数据已经在初始化时上传)
            ma_matrix = engine.compute_ma_gpu_resident(self.window)
        else:
            # 调用 CPU OpenMP 算子
            ma_matrix = engine.compute_ma_batch(data_matrix, self.window)
            
        signal_matrix = np.where(data_matrix > ma_matrix, 1.0, 0.0)
        signal_matrix[:, :self.window] = 0.0 
        return signal_matrix

# =====================================================================
# 模块二：异构回测引擎层 (CPU/GPU 自由切换)
# =====================================================================
class HeterogeneousBacktester:
    def __init__(self, data_matrix, engine_type='CPU', commission=0.0003, slippage=0.001):
        self.data_matrix = data_matrix
        self.commission = commission
        self.slippage = slippage
        self.engine_type = engine_type.upper()
        self.engine = finance_calc_core.QuantEngineSoA()
        
        # 🌟 如果是 GPU 引擎，初始化时一次性将数据上传至显存 (Data Residency)
        if self.engine_type == 'GPU':
            print(f"[*] 正在将 {data_matrix.shape} 数据矩阵上传至 GPU 显存...")
            t_up = time.perf_counter()
            self.engine.upload_to_gpu(data_matrix)
            print(f"[+] 数据驻留完成！耗时: {(time.perf_counter() - t_up)*1000:.2f} ms")
            
    def run(self, strategy):
        t0 = time.perf_counter()
        
        # 1. 策略生成信号 (传入 use_gpu 标志位)
        use_gpu = (self.engine_type == 'GPU')
        signal_matrix = strategy.generate_signals(self.data_matrix, self.engine, use_gpu)
        
        # 2. 向量化收益计算与摩擦扣除 (复用之前的逻辑)
        shifted_prices = np.roll(self.data_matrix, shift=1, axis=1)
        shifted_prices[:, 0] = self.data_matrix[:, 0]
        daily_returns = (self.data_matrix / shifted_prices) - 1
        daily_returns[:, 0] = 0.0
        
        shifted_signals = np.roll(signal_matrix, shift=1, axis=1)
        shifted_signals[:, 0] = 0.0
        
        position_changes = np.abs(np.diff(shifted_signals, axis=1, prepend=0))
        friction_costs = position_changes * (self.commission + self.slippage)
        
        strategy_returns = (shifted_signals * daily_returns) - friction_costs
        port_returns = np.mean(strategy_returns, axis=0)
        
        calc_time = (time.perf_counter() - t0) * 1000
        return calc_time, port_returns
        
    def __del__(self):
        # 🌟 析构时清理显存
        if hasattr(self, 'engine_type') and self.engine_type == 'GPU':
            self.engine.free_gpu_memory()

# =====================================================================
# 主程序：全市场级 CPU vs GPU 终极对决
# =====================================================================
def run_gpu_benchmark():
    print("="*75)
    print("    🔥 异构计算巅峰对决：全市场回测 CPU (OpenMP) vs GPU (CUDA)")
    print("="*75)
    
    # 生成 2000 只股票，1000 天数据 (200万数据点)
    _, data_matrix = generate_massive_mock_data(num_stocks=2000, num_days=1000)
    my_strategy = MACrossStrategy(window=20)
    
    # ---------- 测试 1: CPU 引擎 ----------
    print("" + "-"*40)
    print("    [引擎模式：CPU OpenMP]")
    print("-"*40)
    cpu_backtester = HeterogeneousBacktester(data_matrix, engine_type='CPU')
    
    # 预热并运行
    cpu_backtester.run(my_strategy) 
    cpu_time, _ = cpu_backtester.run(my_strategy)
    print(f"[+] CPU 回测全链路耗时: {cpu_time:.2f} ms")
    
    # ---------- 测试 2: GPU 引擎 ----------
    print("" + "-"*40)
    print("    [引擎模式：CUDA GPU 常驻内存]")
    print("-"*40)
    gpu_backtester = HeterogeneousBacktester(data_matrix, engine_type='GPU')
    
    # 预热并运行
    gpu_backtester.run(my_strategy)
    gpu_time, _ = gpu_backtester.run(my_strategy)
    print(f"[+] GPU 回测全链路耗时: {gpu_time:.2f} ms")
    
    # ---------- 结果揭晓 ----------
    print("" + "="*75)
    print(f"🏆 压测结论 (数据规模: 2,000,000 数据点)")
    print("="*75)
    print(f"CPU OpenMP 耗时: {cpu_time:>8.2f} ms")
    print(f"CUDA GPU 耗时  : {gpu_time:>8.2f} ms")
    
    if gpu_time < cpu_time:
        print(f"🚀 GPU 加速比: 较 CPU 进一步提速 {cpu_time/gpu_time:.2f} 倍！")
    else:
        print(f"⚠️ GPU 未能加速。原因诊断：MA 算子计算密度过低 (Memory-bound)，")
        print(f"   导致 GPU 算数单元空载，耗时主要由 Python 层矩阵撮合主导。")
    print("="*75)

if __name__ == "__main__":
    run_gpu_benchmark()
