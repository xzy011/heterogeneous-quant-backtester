import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

# ==========================================
# 1. 环境变量注入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
build_path = os.path.join(project_root, 'build')

sys.path.insert(0, build_path)
sys.path.insert(0, project_root)

# ==========================================
# 2. 从重构后的 pyquant 包导入核心组件
# ==========================================
from pyquant.data import generate_mock_data
from pyquant.strategy import MACrossStrategy
from pyquant.engine import VectorizedBacktester
from pyquant.metrics import calculate_metrics

# ==========================================
# 页面配置与标题
# ==========================================
st.set_page_config(page_title="异构量化回测引擎", layout="wide")
st.title("🚀 高性能异构计算量化回测引擎 (C++/CUDA)")
st.markdown("基于 **Pybind11 零拷贝** 与 **纯向量化撮合**，实现全市场数据亚毫秒级回测。")

# ==========================================
# 侧边栏：参数面板
# ==========================================
st.sidebar.header("⚙️ 回测参数设置")

# 1. 数据规模设置
st.sidebar.subheader("1. 数据池规模")
num_stocks = st.sidebar.slider("股票数量", min_value=10, max_value=2000, value=100, step=100)
num_days = st.sidebar.slider("回测天数", min_value=252, max_value=1000, value=756, step=252)

# 2. 策略参数
st.sidebar.subheader("2. 策略参数 (MA突破)")
ma_window = st.sidebar.number_input("均线周期 (Window)", min_value=5, max_value=120, value=20)

# 3. 硬件引擎选择
st.sidebar.subheader("3. 异构计算引擎")
engine_choice = st.sidebar.radio("底层算子架构", ('CPU (OpenMP 并行)', 'GPU (CUDA 常驻内存)'))
engine_type = 'GPU' if 'GPU' in engine_choice else 'CPU'

# 4. 交易摩擦
st.sidebar.subheader("4. 真实交易摩擦")
commission = st.sidebar.number_input("手续费率 (‱)", value=3.0) / 10000
slippage = st.sidebar.number_input("滑点率 (‰)", value=1.0) / 1000

# ==========================================
# 主界面：运行与结果展示
# ==========================================
if st.sidebar.button("▶️ 立即运行极速回测", use_container_width=True):
    
    with st.spinner('正在进行异构计算与向量化撮合...'):
        # 1. 生成仿真数据
        dates, stock_names, data_matrix = generate_mock_data(num_stocks, num_days)
        
        # 2. 运行回测
        strategy = MACrossStrategy(window=ma_window)
        # 注意：这里统一使用我们重构后的 VectorizedBacktester
        backtester = VectorizedBacktester(data_matrix, commission=commission, slippage=slippage)
        
        # 获取耗时和每日收益率
        t0 = time.perf_counter()
        port_returns = backtester.run(strategy)
        calc_time = (time.perf_counter() - t0) * 1000
        
        # 3. 计算基准收益 (买入持有等权组合)
        shifted_prices = np.roll(data_matrix, shift=1, axis=1)
        shifted_prices[:, 0] = data_matrix[:, 0]
        bh_returns = (data_matrix / shifted_prices) - 1
        bh_returns[:, 0] = 0.0
        port_bh_returns = np.mean(bh_returns, axis=0)

        # 4. 计算指标
        s_tot, s_ann, s_sharpe, s_mdd, s_win, s_cum, s_dd = calculate_metrics(port_returns)
        b_tot, b_ann, b_sharpe, b_mdd, b_win, b_cum, b_dd = calculate_metrics(port_bh_returns)

    # --- 成功提示 ---
    st.success(f"✅ 回测完成！处理数据点: **{num_stocks * num_days:,}**​ 个 | 引擎全链路耗时: ​**{calc_time:.2f} ms**")

    # --- 核心指标看板 ---
    st.markdown("##### 📊 核心绩效指标 (等权投资组合)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("策略总收益", f"{s_tot*100:.2f}%", f"基准: {b_tot*100:.2f}%")
    col2.metric("夏普比率", f"{s_sharpe:.2f}", f"基准: {b_sharpe:.2f}")
    col3.metric("最大回撤", f"{s_mdd*100:.2f}%", f"基准: {b_mdd*100:.2f}%", delta_color="inverse")
    col4.metric("策略胜率", f"{s_win*100:.2f}%")

    # --- 可视化图表 ---
    st.markdown("##### 📈 资金曲线与动态回撤")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), gridspec_kw={'height_ratios': [3, 1]})
    
    # 资金曲线
    ax1.plot(dates, s_cum, label=f'Strategy (MA{ma_window})', color='#e74c3c', linewidth=2)
    ax1.plot(dates, b_cum, label='Buy & Hold Benchmark', color='#34495e', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Cumulative Equity')
    ax1.legend(loc='upper left')
    
    # 回撤图
    ax2.fill_between(dates, s_dd, 0, color='#e74c3c', alpha=0.3, label=f'Strategy Max DD: {s_mdd*100:.1f}%')
    ax2.fill_between(dates, b_dd, 0, color='#34495e', alpha=0.2, label=f'Benchmark Max DD: {b_mdd*100:.1f}%')
    ax2.set_ylabel('Drawdown')
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    st.pyplot(fig)
