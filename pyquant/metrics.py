import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def calculate_metrics(daily_returns):
    """纯向量化计算核心绩效指标"""
    cum_returns = np.cumprod(1 + daily_returns)
    total_return = cum_returns[-1] - 1
    n_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)
    
    risk_free_rate = 0.02
    sharpe_ratio = 0 if annualized_volatility == 0 else (annualized_return - risk_free_rate) / annualized_volatility
        
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1
    max_drawdown = np.min(drawdowns)
    
    active_days = daily_returns[daily_returns != 0]
    win_rate = 0 if len(active_days) == 0 else np.sum(active_days > 0) / len(active_days)
        
    return total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate, cum_returns, drawdowns

def plot_portfolio_performance(dates, strat_returns, bench_returns, save_dir):
    """封装好的专业级可视化输出函数"""
    # 计算画图所需的指标
    s_tot, s_ann, s_sharpe, s_mdd, s_win, s_cum, s_dd = calculate_metrics(strat_returns)
    b_tot, b_ann, b_sharpe, b_mdd, b_win, b_cum, b_dd = calculate_metrics(bench_returns)

    print(f"========== 等权投资组合绩效报告 ==========")
    print(f"{'指标':<20} | {'MA20 策略':<15} | {'买入持有 (基准)':<15}")
    print("-" * 55)
    print(f"{'年化收益率':<20} | {s_ann*100:>14.2f}% | {b_ann*100:>14.2f}%")
    print(f"{'最大回撤':<20} | {s_mdd*100:>14.2f}% | {b_mdd*100:>14.2f}%")
    print(f"{'夏普比率':<20} | {s_sharpe:>14.2f}  | {b_sharpe:>14.2f} ")

    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(dates, s_cum, label=f'Strategy (Ann Ret: {s_ann*100:.1f}%)', color='#e74c3c', linewidth=2)
    ax1.plot(dates, b_cum, label=f'Benchmark (Ann Ret: {b_ann*100:.1f}%)', color='#34495e', linewidth=1.5)
    ax1.set_title('Portfolio Cumulative Returns', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    
    ax2.fill_between(dates, s_dd, 0, color='#e74c3c', alpha=0.3, label=f'Strategy MDD: {s_mdd*100:.1f}%')
    ax2.fill_between(dates, b_dd, 0, color='#34495e', alpha=0.2, label=f'Benchmark MDD: {b_mdd*100:.1f}%')
    ax2.set_title('Portfolio Drawdown', fontsize=12)
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'portfolio_equity_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] 图表已成功保存至: {save_path}")
