# 专题五：纯向量化撮合 —— 杜绝未来函数与真实交易摩擦计算

**核心价值**：展示软件架构解耦能力，以及对金融工程严谨性（防未来函数、滑点/手续费处理）的掌控。

## 1. 架构改变：摒弃事件驱动 (Event-Driven)

市面上主流的开源回测框架（如 Backtrader）大多采用事件驱动架构，通过 Python 的 `for` 循环逐日遍历历史数据。在处理全市场级别数据时，Python 层的循环开销会成为致命瓶颈。
为此，本项目在 Python 应用层彻底摒弃了循环，设计了**纯向量化（Vectorized）​**的矩阵撮合架构。

## 2. 核心难点：防未来函数 (Look-ahead Bias) 

向量化回测最容易犯的错误就是“用到未来的数据”。为了严格模拟真实的交易物理时间线（T日收盘发信 -> T+1日开盘执行），我巧妙利用了 NumPy 的矩阵错位操作：

```python
# 信号平移 (严格模拟 T日收盘发信号，T+1日开盘执行)
shifted_signals = np.roll(signal_matrix, shift=1, axis=1)
shifted_signals[:, 0] = 0.0 # 清洗首日脏数据
通过 np.roll，我们在亚毫秒级完成了全市场 2000 只股票的时间线对齐，从数学底层杜绝了未来函数。

## 3. 金融严谨性：零循环的交易摩擦计算
真实的量化回测必须包含手续费和滑点。如何在没有 for 循环的情况下定位到发生交易的那一天？
我利用了矩阵的一阶差分（np.diff）特性：
```python
# 利用 np.diff 计算仓位变化。1 为买入，-1 为卖出，绝对值即为调仓点
position_changes = np.abs(np.diff(shifted_signals, axis=1, prepend=0))

# 向量化扣除万三手续费与千一滑点
friction_costs = position_changes * (commission + slippage)
这段逻辑瞬间定位了 200 万个数据点中的所有调仓节点，并在对应的位置精准扣除摩擦成本，全链路耗时依然死死压制在 20 毫秒级别。
## 4.全栈交付：Streamlit 交互式前端
为了让底层引擎具备产品级的可用性，我利用 Streamlit 开发了交互式 Web UI (examples/app.py)。支持动态调整数据池规模、MA 周期、交易摩擦，并在前端实时渲染等权投资组合的资金曲线与最大回撤（Max Drawdown）图表，实现了从底层算力到前台产品的端到端（End-to-End）交付。