import numpy as np
import pandas as pd
import time
import os
import sys

# 导入无头渲染，防止 WSL 报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.insert(0, build_path)
import finance_calc_core
from data_feeder import fetch_real_data 

# =====================================================================
# 模块一：策略抽象层 (Strategy Layer)
# =====================================================================
class BaseStrategy:
    """策略基类：所有用户自定义策略都必须继承此类"""
    def generate_signals(self, data_matrix, engine):
        raise NotImplementedError("策略必须实现 generate_signals 方法！")

class MACrossStrategy(BaseStrategy):
    """具体的策略实现：MA20 均线突破策略"""
    def __init__(self, window=20):
        self.window = window
        
    def generate_signals(self, data_matrix, engine):
        # 1. 调用底层的 C++ 算子计算指标
        ma_matrix = engine.compute_ma_batch(data_matrix, self.window)
        
        # 2. 向量化生成信号矩阵 (Price > MA 为 1，否则为 0)
        signal_matrix = np.where(data_matrix > ma_matrix, 1.0, 0.0)
        
        # 3. 清除前 window 天的无效信号
        signal_matrix[:, :self.window] = 0.0 
        return signal_matrix

# =====================================================================
# 模块二：回测引擎层 (Backtest Engine Layer)
# =====================================================================
class VectorizedBacktester:
    """纯向量化回测引擎，支持交易摩擦与异构计算加速"""
    def __init__(self, data_matrix, commission=0.0003, slippage=0.001):
        """
        commission: 交易手续费率 (默认万分之三)
        slippage: 滑点损耗率 (默认千分之一)
        """
        self.data_matrix = data_matrix
        self.commission = commission
        self.slippage = slippage
        self.engine = finance_calc_core.QuantEngineSoA()
        
    def run(self, strategy):
        t0 = time.perf_counter()
        
        # 1. 策略生成目标仓位信号
        signal_matrix = strategy.generate_signals(self.data_matrix, self.engine)
        
        # 2. 计算标的资产每日真实收益率
        shifted_prices = np.roll(self.data_matrix, shift=1, axis=1)
        shifted_prices[:, 0] = self.data_matrix[:, 0] # 防止除以0
        daily_returns = (self.data_matrix / shifted_prices) - 1
        daily_returns[:, 0] = 0.0
        
        # 3. 信号平移 (T日收盘发信号，T+1日开盘执行)
        shifted_signals = np.roll(signal_matrix, shift=1, axis=1)
        shifted_signals[:, 0] = 0.0
        
        # -------------------------------------------------------------
        # 🌟 核心升级：纯向量化计算交易摩擦 (手续费 + 滑点)
        # -------------------------------------------------------------
        # 利用 np.diff 计算仓位变化。如果今天信号是1，昨天是0，diff就是1 (买入)
        # 如果今天是0，昨天是1，diff就是-1 (卖出)。取绝对值就是交易发生点。
        position_changes = np.abs(np.diff(shifted_signals, axis=1, prepend=0))
        
        # 每次调仓的摩擦成本 = 交易金额 * (手续费率 + 滑点率)
        friction_costs = position_changes * (self.commission + self.slippage)
        
        # 4. 扣除摩擦成本后的真实策略收益
        strategy_returns = (shifted_signals * daily_returns) - friction_costs
        
        # 5. 计算等权组合资金曲线
        port_returns = np.mean(strategy_returns, axis=0)
        cum_returns = np.cumprod(1 + port_returns)
        
        calc_time = (time.perf_counter() - t0) * 1000
        
        print(f"[+] 面向对象回测完成！(含交易摩擦计算)")
        print(f"[+] 引擎全链路耗时: {calc_time:.2f} ms")
        print(f"[+] 设定的交易摩擦: 手续费 {self.commission*10000}‱, 滑点 {self.slippage*1000}‰")
        
        return cum_returns

# =====================================================================
# 主程序运行入口
# =====================================================================
def main():
    print("="*70)
    print("    🔥 工业级重构：面向对象框架 + 真实交易摩擦")
    print("="*70)
    
    # 1. 获取数据
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    dates, stock_names, data_matrix = fetch_real_data(tickers, period="3y")
    
    # 2. 实例化策略
    my_strategy = MACrossStrategy(window=20)
    
    # 3. 实例化回测引擎 (加入真实的万三手续费和千一滑点)
    backtester = VectorizedBacktester(data_matrix, commission=0.0003, slippage=0.001)
    
    # 4. 运行回测
    cum_returns = backtester.run(my_strategy)
    
    # 5. 简单打印最终收益验证
    final_return = (cum_returns[-1] - 1) * 100
    print(f"[!] 扣除手续费和滑点后，策略最终收益率: {final_return:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
