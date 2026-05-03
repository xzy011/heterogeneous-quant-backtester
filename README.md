# Heterogeneous Quant Backtesting Engine

[![Language](https://img.shields.io/badge/language-C++%2FPython-blue.svg)]()
[![Performance](https://img.shields.io/badge/Speedup-6.86x-green.svg)]()

一个基于 C++/CUDA 异构算力与纯向量化架构的高性能量化回测引擎。该项目摒弃了传统 Pandas/Event-driven 的性能瓶颈，通过 **SoA 内存布局**、**Pybind11 零拷贝通信** 与 **异构并行调度**，实现了全市场级别回测的毫秒级响应。

## 🚀 核心压测数据 (Performance Benchmarks)

在 2000 只股票 × 1000 个交易日（共 2,000,000 个数据点）的全市场仿真压测下，系统表现如下：

| 测试架构 | 耗时 (ms) | 加速比 | 备注 |
| :--- | :--- | :--- | :--- |
| 传统 Pandas (基准) | 181.92 ms | 1.00x | 底层调用的已经是 Cython 优化的 C 代码 |
| **本项目 (C++ 异构引擎)​** | **26.53 ms** | **6.86x** | **突破了 Python 层的性能天花板** |

> **业务价值洞察**：在真实的量化投研中，网格搜索调参通常需要上万次回测。单次 160 毫秒的性能差，在 10,000 次迭代下将节省近半小时的投研等待时间。

## 🏗️ 系统架构设计

系统采用严格的职责分离设计（Separation of Concerns），分为底层计算引擎与上层业务逻辑：

- **Data Layer**: 实现 SoA (Structure of Arrays) 连续内存布局，配合 Pybind11 `Buffer Protocol` 实现跨语言零拷贝，消除跨界通信延迟。
- **Signal Layer**: 底层 C++ 引擎内置 SIMD (AVX2) 与 OpenMP 优化算子，支持动态策略扩展。
- **Execution Layer**: 纯向量化撮合架构，利用 `np.roll` 进行时间错位撮合杜绝未来函数，利用 `np.diff` 瞬间定位调仓节点并计算真实交易摩擦（手续费与滑点）。

## 🔬 深度性能剖析 (Profiling & Hardware Insight)

在项目开发过程中，通过 `perf` 采样与微架构分析，得出了以下关键结论：

1. **多核并行甜点位 (Sweet Spot)​**：
   在 i9-12900H 平台的压测中，发现单纯增加线程数并不能带来线性提升。6 核（P-Core）为最佳性能甜点位，超过 6 核后 E-Core 的介入反而因访存延迟和 L3 Cache 竞争触发了“木桶效应”。这验证了在 AI Infra 开发中，**硬件感知调度 (Hardware-aware Scheduling)​** 的重要性。
   
2. **异构算力对决与 Roofline 模型验证**：
   对比了 CPU SIMD 与 CUDA 算子。在标准的 MA 指标（Memory-bound）下，PCIe 的数据搬运开销远大于计算开销；但当引入重型非线性数学算子（Compute-bound）时，GPU 的 SFU 单元优势彻底爆发。为此，引擎设计了常驻显存（Resident Memory）模式，实现了 `0.02ms` 的零拷贝极速计算。

## 📚 技术演进笔记 (Optimization Journey)

本项目并非一蹴而就，而是经历了从“底层算力榨取”到“上层金融业务架构”的完整演进。在此过程中，我记录了详尽的工程笔记：
**​【底层架构与算力极限】​**
- 📖 **​[Phase 1] [架构基石：零拷贝通信与 SoA 内存布局演进](docs/01_zero_copy_and_soa.md)​**
- 📖 **​[Phase 2] [并发诊断：多核并行的陷阱与 Memory-bound 瓶颈分析](docs/02_memory_bound_and_openmp.md)​**
- 📖 **​[Phase 3] [单核极限：从标量到向量 —— SIMD 优化实录](docs/03_simd_optimization.md)​**
*(注：以上为纯 CPU 极限优化阶段的工程记录，正是基于这些对 Memory-bound 的深刻诊断，最终推演出了本项目的最终形态——基于 CUDA 常驻内存的异构计算架构。)*
- 📖 [Phase 4] [突破内存墙：CUDA 异构计算与 PCIe 传输开销优化](docs/04_cuda_and_memory_hierarchy.md)
**​【金融业务与系统交付】​**
- 📖 [Phase 5] [纯向量化撮合：杜绝未来函数与真实交易摩擦计算](docs/05_vectorized_backtesting.md)

## 🚀 快速开始

### 1. 环境依赖
- 编译器: `g++ >= 9.4`, `CMake >= 3.18`
- 异构支持 (可选): `CUDA Toolkit >= 11.8`, `OpenMP`
- Python: `Python >= 3.9`

### 2. 编译与安装
```bash
git clone https://github.com/yourusername/finance_project.git
cd finance_project

mkdir build && cd build
cmake .. 
make -j

### 3. 运行回测与压测
# 运行完整的量化策略回测（含真实数据获取、撮合与可视化）
python3 examples/run_backtest.py

# 运行性能压测基准
python3 benchmarks/ultimate_benchmark.py
python3 benchmarks/benchmark_sweetpoint.py
python3 benchmarks/benchmark_cuda_full.py


### 4. 启动交互式 Web 界面 (Streamlit)
本项目内置了基于 Streamlit 的可视化前端，支持动态调整数据规模、策略参数、异构引擎类型以及交易摩擦，实现亚毫秒级的实时回测交互：
```bash
pip install streamlit
streamlit run examples/app.py

