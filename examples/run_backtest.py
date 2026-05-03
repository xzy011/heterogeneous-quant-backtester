import os
import sys
import numpy as np

# 1. 环境变量注入（解决 Pylance 报错和找不到 C++ 模块的问题）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
build_path = os.path.join(project_root, 'build')

sys.path.insert(0, build_path)      # 让 Python 能找到 finance_calc_core.so
sys.path.insert(0, project_root)    # 让 Python 能找到 pyquant 文件夹

# 2. 从量化包中导入模块
from pyquant.data import fetch_real_data
from pyquant.strategy import MACrossStrategy
from pyquant.engine import VectorizedBacktester
from pyquant.metrics import plot_portfolio_performance

def main():
    print("="*70)
    print("    🔥 工业级重构：面向对象框架 + 真实交易摩擦")
    print("="*70)
    
    # 步骤 A：获取数据
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    dates, stock_names, data_matrix = fetch_real_data(tickers, period="3y")
    
    # 步骤 B：实例化策略与回测引擎
    my_strategy = MACrossStrategy(window=20)
    backtester = VectorizedBacktester(data_matrix, commission=0.0003, slippage=0.001)
    
    # 步骤 C：执行回测
    port_strat_returns = backtester.run(my_strategy)
    
    # 步骤 D：计算基准组合收益 (买入持有等权组合)
    price_df = data_matrix.T
    daily_returns = (price_df[1:] / price_df[:-1]) - 1
    daily_returns = np.vstack([np.zeros((1, daily_returns.shape[1])), daily_returns])
    bench_returns = np.mean(daily_returns, axis=1)  # 等权基准
    
    # 步骤 E：计算绩效指标并输出可视化图表
    plot_dir = os.path.join(project_root, 'docs', 'plots')
    plot_portfolio_performance(dates, port_strat_returns, bench_returns, save_dir=plot_dir)

if __name__ == "__main__":
    main()
