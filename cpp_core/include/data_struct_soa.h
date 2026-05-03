#pragma once
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class QuantEngineSoA {
public:
    // SoA 布局：将不同字段分开存储在连续内存中
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<long long> volume;

    // 预留内存，减少 realloc 开销（Infra 优化点）
    void reserve(size_t n);

    // 添加数据
    void add_bar(double o, double h, double l, double c, long long v);

    // 核心：零拷贝导出到 NumPy
    // 通过 Buffer Protocol 直接将 C++ vector 指针暴露给 Python
    py::array_t<double> get_close_prices();

    // 优化后的 MA 计算算子
    py::array_t<double> compute_ma(int window);
    py::array_t<double> compute_ma_from_data(py::array_t<double> input_data, int window);

    // 用于极限压测与 Roofline 模型验证的重型算子 (Compute-bound)
    py::array_t<double> compute_heavy_indicator_batch(py::array_t<double> input_matrix, int window);
    py::array_t<double> compute_heavy_indicator_cuda(py::array_t<double> input_matrix, int window);

    // 批量并行接口：改为接收 2D Matrix
    py::array_t<double> compute_ma_batch_matrix(py::array_t<double> input_matrix, int window);
    void compute_ma_batch_simd(py::array_t<double> input_array, int window);
    // 单线程版本（无 OpenMP）
    py::array_t<double> compute_ma_batch_scalar_single(py::array_t<double> input_matrix, int window);
    // GPU 优化版 MA 计算    
    py::array_t<double> compute_ma_cuda(py::array_t<double> input_matrix, int window);
    py::array_t<double> compute_ma_cuda_pinned(py::array_t<double> input_matrix, int window); // Pinned版 
    

    // GPU 常驻内存
    void upload_to_gpu(py::array_t<double> input_matrix);
    py::array_t<double> compute_ma_gpu_resident(int window);
    void free_gpu_memory();

private:    
    std::vector<double> results;    // ... 其他成员 ...
    void launch_ma_kernel(const double* h_in, double* h_out, int num_stocks, int data_len, int window);
    int gpu_num_stocks = 0;
    int gpu_data_len = 0;
};
