import numpy as np
import pandas as pd
import time
import os
import sys

# 【关键防御】强制使用无头渲染后端，防止 WSL 报错
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 引入 Day1 的数据获取模块
from data_feeder import fetch_real_data 

current_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.insert(0, build_path)
import finance_calc_core

# ... 下方的 calculate_metrics 和 run_day3_visualization 函数保持不变 ...

def calculate_metrics(daily_returns):
    """
    纯向量化计算核心绩效指标
    daily_returns: 1D numpy array, 每日收益率序列
    """
    # 1. 累计收益率
    cum_returns = np.cumprod(1 + daily_returns)
    total_return = cum_returns[-1] - 1
    
    # 2. 年化收益率 (假设每年 252 个交易日)
    n_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # 3. 年化波动率
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)
    
    # 4. 夏普比率 (假设无风险利率为 0.02)
    risk_free_rate = 0.02
    if annualized_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
    # 5. 最大回撤 (Max Drawdown)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1
    max_drawdown = np.min(drawdowns)
    
    # 6. 胜率 (盈利天数 / 总交易天数)
    active_days = daily_returns[daily_returns != 0]
    if len(active_days) == 0:
        win_rate = 0
    else:
        win_rate = np.sum(active_days > 0) / len(active_days)
        
    return total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate, cum_returns, drawdowns

def run_day3_visualization():
    print("="*60)
    print("    异构回测引擎：绩效分析与可视化 (Day 3)")
    print("="*60)

    # 1. 获取数据与计算信号 (复用 Day2 逻辑)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    dates, stock_names, data_matrix = fetch_real_data(tickers, period="3y")
    
    engine = finance_calc_core.QuantEngineSoA()
    window = 20
    
    t0 = time.perf_counter()
    ma_matrix = engine.compute_ma_batch(data_matrix, window)
    
    signal_matrix = np.where(data_matrix > ma_matrix, 1.0, 0.0)
    signal_matrix[:, :window] = 0.0 
    
    price_df = pd.DataFrame(data_matrix.T)
    stock_daily_returns = price_df.pct_change().fillna(0).values.T 
    
    shifted_signals = np.roll(signal_matrix, shift=1, axis=1)
    shifted_signals[:, 0] = 0.0
    
    strategy_returns = shifted_signals * stock_daily_returns
    calc_time = (time.perf_counter() - t0) * 1000
    print(f"[+] 底层矩阵撮合完成！耗时: {calc_time:.2f} ms")

    # ========================================================
    # 核心：投资组合层面的绩效统计 (Portfolio Metrics)
    # ========================================================
    # 假设我们等权重(Equal-weight)持有这10只股票
    port_strat_returns = np.mean(strategy_returns, axis=0) # 策略组合每日收益
    port_bh_returns = np.mean(stock_daily_returns, axis=0) # 基准组合每日收益(一直持有)
    
    # 计算策略与基准的指标
    s_tot, s_ann, s_sharpe, s_mdd, s_win, s_cum, s_dd = calculate_metrics(port_strat_returns)
    b_tot, b_ann, b_sharpe, b_mdd, b_win, b_cum, b_dd = calculate_metrics(port_bh_returns)
    
    print(f"========== 等权投资组合绩效报告 ==========")
    print(f"{'指标':<20} | {'MA20 策略':<15} | {'买入持有 (基准)':<15}")
    print("-" * 55)
    print(f"{'总收益率':<20} | {s_tot*100:>14.2f}% | {b_tot*100:>14.2f}%")
    print(f"{'年化收益率':<20} | {s_ann*100:>14.2f}% | {b_ann*100:>14.2f}%")
    print(f"{'夏普比率':<20} | {s_sharpe:>14.2f}  | {b_sharpe:>14.2f} ")
    print(f"{'最大回撤':<20} | {s_mdd*100:>14.2f}% | {b_mdd*100:>14.2f}%")
    print(f"{'日胜率':<20} | {s_win*100:>14.2f}% | {b_win*100:>14.2f}%")

    # ========================================================
    # 核心：专业级可视化输出 (保存为图片)
    # ========================================================
    print(f"[*] 正在生成可视化图表...")
    
    # 创建 plots 文件夹
    plot_dir = os.path.join(current_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 设置画图风格 (避免 WSL 下中文字体乱码，图表全用英文标签)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上图：资金曲线对比
    ax1.plot(dates, s_cum, label=f'MA20 Strategy (Ann Ret: {s_ann*100:.1f}%)', color='#e74c3c', linewidth=2)
    ax1.plot(dates, b_cum, label=f'Buy & Hold (Ann Ret: {b_ann*100:.1f}%)', color='#34495e', linewidth=1.5, alpha=0.8)
    ax1.set_title('Portfolio Cumulative Returns: Strategy vs Benchmark', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Equity', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    
    # 下图：最大回撤图
    ax2.fill_between(dates, s_dd, 0, color='#e74c3c', alpha=0.3, label=f'Strategy Drawdown (Max: {s_mdd*100:.1f}%)')
    ax2.fill_between(dates, b_dd, 0, color='#34495e', alpha=0.2, label=f'Benchmark Drawdown (Max: {b_mdd*100:.1f}%)')
    ax2.set_title('Portfolio Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(plot_dir, 'portfolio_equity_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] 图表已成功保存至: {save_path}")

if __name__ == "__main__":
    run_day3_visualization()
