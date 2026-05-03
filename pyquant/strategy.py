import numpy as np

# =====================================================================
# 模块：策略抽象层 (Strategy Layer)
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