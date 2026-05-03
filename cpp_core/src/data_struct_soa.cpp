#include "data_struct_soa.h"
#include <numeric>
#include <omp.h>
#include <iostream>
#include <immintrin.h>

// ========== 外部 CUDA 函数声明（C 链接，避免名字修饰）==========
extern "C" {
    void launch_ma_kernel_impl(const double* h_in, double* h_out, 
                               int num_stocks, int data_len, int window);
    void launch_ma_kernel_pinned(const double* h_in, double* h_out, 
                                int num_stocks, int data_len, int window);
    void launch_heavy_kernel_impl(const double* h_in, double* h_out, 
                                int num_stocks, int data_len, int window);
    void gpu_upload_data(const double* h_data, int num_stocks, int data_len);
    void gpu_compute_resident(double* h_out, int num_stocks, int data_len, int window);
    void gpu_free_memory();
}



void QuantEngineSoA::reserve(size_t n) {
    open.reserve(n); high.reserve(n); low.reserve(n);
    close.reserve(n); volume.reserve(n);
}

void QuantEngineSoA::add_bar(double o, double h, double l, double c, long long v) {
    open.push_back(o); high.push_back(h); low.push_back(l);
    close.push_back(c); volume.push_back(v);
}

py::array_t<double> QuantEngineSoA::get_close_prices() {
    return py::array_t<double>(
        { (ssize_t)close.size() },
        { sizeof(double) },
        close.data(),
        py::cast(this)
    );
}

py::array_t<double> QuantEngineSoA::compute_ma(int window) {
    size_t n = close.size();
    py::array_t<double> result(n);
    auto r = result.mutable_unchecked<1>();
    
    for (size_t i = 0; i < (size_t)window - 1; ++i) r(i) = 0.0;
    if (n < (size_t)window) return result;

    double current_sum = 0.0;
    for (size_t i = 0; i < (size_t)window; ++i) current_sum += close[i];
    r(window - 1) = current_sum / window;

    for (size_t i = window; i < n; ++i) {
        current_sum += (close[i] - close[i - window]);
        r(i) = current_sum / window;
    }
    return result;
}

py::array_t<double> QuantEngineSoA::compute_ma_from_data(py::array_t<double> input_data, int window) {
    auto r_in = input_data.unchecked<1>();
    size_t n = input_data.size();
    py::array_t<double> result(n);
    auto r_out = result.mutable_unchecked<1>();

    for (size_t i = 0; i < (size_t)window - 1; ++i) r_out(i) = 0.0;
    if (n < (size_t)window) return result;

    double current_sum = 0.0;
    for (size_t i = 0; i < (size_t)window; ++i) current_sum += r_in(i);
    r_out(window - 1) = current_sum / window;

    for (size_t i = window; i < n; ++i) {
        current_sum += (r_in(i) - r_in(i - window));
        r_out(i) = current_sum / window;
    }
    return result;
}

py::array_t<double> QuantEngineSoA::compute_ma_batch_matrix(py::array_t<double> input_matrix, int window) {
    auto r_in = input_matrix.unchecked<2>();
    ssize_t num_stocks = r_in.shape(0);
    ssize_t data_len = r_in.shape(1);

    py::array_t<double> result({num_stocks, data_len});
    auto r_out = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(static)
        for (ssize_t i = 0; i < num_stocks; ++i) {
            double current_sum = 0.0;
            
            for (ssize_t j = 0; j < (ssize_t)window - 1; ++j) r_out(i, j) = 0.0;
            
            if (data_len >= (ssize_t)window) {
                for (ssize_t j = 0; j < (ssize_t)window; ++j) current_sum += r_in(i, j);
                r_out(i, window - 1) = current_sum / window;

                for (ssize_t j = window; j < data_len; ++j) {
                    double val = r_in(i, j) - r_in(i, j - window);
                      
                    current_sum += val;
                    r_out(i, j) = current_sum / window;
                }
            }
        }
    }
    return result;
}
// 专门用于验证 Compute-bound 瓶颈的重型算子
py::array_t<double> QuantEngineSoA::compute_heavy_indicator_batch(py::array_t<double> input_matrix, int window) {
    auto r_in = input_matrix.unchecked<2>();
    ssize_t num_stocks = r_in.shape(0);
    ssize_t data_len = r_in.shape(1);

    py::array_t<double> result({num_stocks, data_len});
    auto r_out = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(static)
        for (ssize_t i = 0; i < num_stocks; ++i) {
            double current_sum = 0.0;
            for (ssize_t j = 0; j < (ssize_t)window - 1; ++j) r_out(i, j) = 0.0;
            
            if (data_len >= (ssize_t)window) {
                for (ssize_t j = 0; j < (ssize_t)window; ++j) current_sum += r_in(i, j);
                r_out(i, window - 1) = current_sum / window;

                for (ssize_t j = window; j < data_len; ++j) {
                    double val = r_in(i, j) - r_in(i, j - window);
                    
                    // 核心差异：引入重型数学运算，强行拉高 Arithmetic Intensity
                    // 模拟深度学习激活函数或极其复杂的量化因子
                    for(int k = 0; k < 8; ++k) {        
                        val = std::tan(std::exp(val)) * std::sqrt(std::abs(val) + 1.0);    
                    }        
                    
                    current_sum += val;
                    r_out(i, j) = current_sum / window;
                }
            }
        }
    }
    return result;
}


void QuantEngineSoA::compute_ma_batch_simd(py::array_t<double> input_array, int window) {
    auto buf = input_array.request();
    int n_stocks = buf.shape[0];
    int n_days = buf.shape[1];
    
    double* __restrict__ ptr = (double*)buf.ptr;

    if (results.size() != n_stocks * n_days) {
        results.assign(n_stocks * n_days, 0.0);
    }
    double* __restrict__ res_ptr = results.data();

    double inv_window = 1.0 / (double)window;

    #pragma omp parallel for
    for (int s = 0; s < n_stocks; ++s) {
        double* __restrict__ stock_data = ptr + s * n_days;
        double* __restrict__ stock_res = res_ptr + s * n_days;

        double sum = 0;
        for (int i = 0; i < window; ++i) {
            sum += stock_data[i];
        }
        stock_res[window - 1] = sum * inv_window;

        #pragma GCC ivdep
        for (int i = window; i < n_days; ++i) {
            double val = stock_data[i] - stock_data[i - window];
            sum += val;
            stock_res[i] = sum * inv_window;
        }
    }
}

// ========== CUDA 接口实现（纯 C++，调用外部 C 函数）==========

py::array_t<double> QuantEngineSoA::compute_ma_cuda(py::array_t<double> input_matrix, int window) {    
    auto r_in = input_matrix.unchecked<2>();    
    int num_stocks = r_in.shape(0);    
    int data_len = r_in.shape(1);
    
    py::array_t<double> result({num_stocks, data_len});    
    auto r_out = result.mutable_unchecked<2>();
    
    launch_ma_kernel_impl(r_in.data(0, 0), r_out.mutable_data(0, 0), num_stocks, data_len, window);
    return result;
}

py::array_t<double> QuantEngineSoA::compute_ma_cuda_pinned(py::array_t<double> input_matrix, int window) {    
    auto r_in = input_matrix.unchecked<2>();    
    int num_stocks = r_in.shape(0);    
    int data_len = r_in.shape(1);
    
    py::array_t<double> result({num_stocks, data_len});    
    auto r_out = result.mutable_unchecked<2>();
    
    launch_ma_kernel_pinned(r_in.data(0, 0), r_out.mutable_data(0, 0), num_stocks, data_len, window);
    return result;
}

// ========== GPU 常驻内存接口 ==========

void QuantEngineSoA::upload_to_gpu(py::array_t<double> input_matrix) {
    auto r_in = input_matrix.unchecked<2>();
    gpu_num_stocks = r_in.shape(0);
    gpu_data_len = r_in.shape(1);
    
    gpu_upload_data(r_in.data(0, 0), gpu_num_stocks, gpu_data_len);
}

py::array_t<double> QuantEngineSoA::compute_ma_gpu_resident(int window) {
    if (gpu_num_stocks == 0 || gpu_data_len == 0) {
        throw std::runtime_error("GPU memory not initialized. Call upload_to_gpu first.");
    }
    
    py::array_t<double> result({gpu_num_stocks, gpu_data_len});
    auto r_out = result.mutable_unchecked<2>();
    
    gpu_compute_resident(r_out.mutable_data(0, 0), gpu_num_stocks, gpu_data_len, window);
    
    return result;
}

void QuantEngineSoA::free_gpu_memory() {
    gpu_free_memory();
    gpu_num_stocks = 0;
    gpu_data_len = 0;
}
// 真正的单线程版本（无 OpenMP）
py::array_t<double> QuantEngineSoA::compute_ma_batch_scalar_single(py::array_t<double> input_matrix, int window) {
    auto r_in = input_matrix.unchecked<2>();
    ssize_t num_stocks = r_in.shape(0);
    ssize_t data_len = r_in.shape(1);

    py::array_t<double> result({num_stocks, data_len});
    auto r_out = result.mutable_unchecked<2>();

    // 故意不用 OpenMP，纯串行
    for (ssize_t i = 0; i < num_stocks; ++i) {
        double current_sum = 0.0;
        
        for (ssize_t j = 0; j < (ssize_t)window - 1; ++j) r_out(i, j) = 0.0;
        
        if (data_len >= (ssize_t)window) {
            for (ssize_t j = 0; j < (ssize_t)window; ++j) current_sum += r_in(i, j);
            r_out(i, window - 1) = current_sum / window;

            for (ssize_t j = window; j < data_len; ++j) {
                double val = r_in(i, j) - r_in(i, j - window);     
                current_sum += val;
                r_out(i, j) = current_sum / window;
            }
        }
    }
    return result;
}
py::array_t<double> QuantEngineSoA::compute_heavy_indicator_cuda(py::array_t<double> input_matrix, int window) {    
    auto r_in = input_matrix.unchecked<2>();    
    int num_stocks = r_in.shape(0);    
    int data_len = r_in.shape(1);
    
    py::array_t<double> result({num_stocks, data_len});    
    auto r_out = result.mutable_unchecked<2>();
    
    launch_heavy_kernel_impl(r_in.data(0, 0), r_out.mutable_data(0, 0), num_stocks, data_len, window);
    return result;
}


// ========== 模块绑定 ==========
PYBIND11_MODULE(finance_calc_core, m) {
    py::class_<QuantEngineSoA>(m, "QuantEngineSoA")
        .def(py::init<>())
        .def("reserve", &QuantEngineSoA::reserve)
        .def("add_bar", &QuantEngineSoA::add_bar)
        .def("get_close_prices", &QuantEngineSoA::get_close_prices)
        .def("compute_ma", &QuantEngineSoA::compute_ma)
        .def("compute_ma_from_data", &QuantEngineSoA::compute_ma_from_data)
        .def("compute_ma_batch", &QuantEngineSoA::compute_ma_batch_matrix)
        .def("compute_ma_batch_simd", &QuantEngineSoA::compute_ma_batch_simd)
        .def("compute_ma_batch_scalar_single", &QuantEngineSoA::compute_ma_batch_scalar_single)
        .def("compute_ma_cuda", &QuantEngineSoA::compute_ma_cuda)
        .def("compute_ma_cuda_pinned", &QuantEngineSoA::compute_ma_cuda_pinned)
        .def("upload_to_gpu", &QuantEngineSoA::upload_to_gpu)
        .def("compute_ma_gpu_resident", &QuantEngineSoA::compute_ma_gpu_resident)
        .def("free_gpu_memory", &QuantEngineSoA::free_gpu_memory)
        .def("compute_heavy_indicator_batch", &QuantEngineSoA::compute_heavy_indicator_batch)
        .def("compute_heavy_indicator_cuda", &QuantEngineSoA::compute_heavy_indicator_cuda);
}