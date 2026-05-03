import numpy as np
import finance_calc_core
import time

# =====================================================================
# 回测引擎层 (Backtest Engine Layer)
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
        # 纯向量化计算交易摩擦 (手续费 + 滑点)
        # -------------------------------------------------------------
        # 利用 np.diff 计算仓位变化。如果今天信号是1，昨天是0，diff就是1 (买入)
        # 如果今天是0，昨天是1，diff就是-1 (卖出)。取绝对值就是交易发生点。
        position_changes = np.abs(np.diff(shifted_signals, axis=1, prepend=0))
        
        # 每次调仓的摩擦成本 = 交易金额 * (手续费率 + 滑点率)
        friction_costs = position_changes * (self.commission + self.slippage)
        
        # 4. 扣除摩擦成本后的真实策略收益
        strategy_returns = (shifted_signals * daily_returns) - friction_costs
        
        # 5. 计算等权组合资金曲线
        port_returns = np.mean(strategy_returns, axis=0)# 这是一个 1D 数组 [n_days]
        #cum_returns = np.cumprod(1 + port_returns) 如果只想返回收益率序列，这里可以先不累乘
        
        calc_time = (time.perf_counter() - t0) * 1000
        
        print(f"[+] 面向对象回测完成！(含交易摩擦计算)")
        print(f"[+] 引擎全链路耗时: {calc_time:.2f} ms")
        print(f"[+] 设定的交易摩擦: 手续费 {self.commission*10000}‱, 滑点 {self.slippage*1000}‰")
        
        return port_returns # 确保返回的是这个 1D 数组