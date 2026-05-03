# 专题一：零拷贝通信与 SoA 内存布局演进

**核心价值**：展示对存储层次结构（Memory Hierarchy）和跨语言通信开销的深刻理解。

## 1. 痛点：Python/Pandas 的“隐藏开销”

在处理亿级量化数据时，Pandas 的 `DataFrame` 虽然好用，但直接将其传入 C++ 引擎存在两个致命的性能陷阱：

*   **内存布局不连续 (Memory Layout)​**：Pandas 默认的块存储格式（BlockManager）在进行滑动窗口（Moving Average）计算时，会导致缓存命中率（Cache Hit Rate）急剧下降。
*   **通信拷贝 (Serialization Overhead)​**：传统的 Pybind11 `py::list` 或 `py::array_t` 转换方式，本质上会触发一次完整的数据拷贝。在千万级数据规模下，拷贝本身的时间可能比计算时间还要长。

## 2. 方案：SoA (Structure of Arrays) 架构设计

我们摒弃了传统的 AoS（数组对象，即 `struct Bar {double o, h, l, c;}`），在 C++ 后端通过 `std::vector<double>` 独立存储每个字段，确保：

*   **空间局部性 (Spatial Locality)​**：计算特定指标（如 MA）时，内存访问在物理上是完全连续的，最大化利用 CPU 的预取器（Hardware Prefetcher）。
*   **SIMD 友好**：连续内存布局是编译器进行向量化（Vectorization）的先决条件。

## 3. 突破：基于 Buffer Protocol 的零拷贝映射

这是本项目在工程实现上的核心亮点。利用 `pybind11::buffer_protocol`，我们将 C++ 的内存指针直接暴露给 NumPy，实现了真正的**​“零拷贝（Zero-copy）”​**。

**实现代码示例**：
```cpp
// 将 C++ 的 std::vector 包装为 NumPy 数组，不发生任何拷贝
py::array_t<double> QuantEngineSoA::get_close_prices() {
    return py::array_t<double>(
        { (ssize_t)close.size() },      // 形状
        { sizeof(double) },             // 步长
        close.data(),                   // 底层 C++ 指针
        py::cast(this)                  // 绑定对象生命周期
    );
}
