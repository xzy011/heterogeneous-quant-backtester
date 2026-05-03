#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>

// ========== CUDA Kernel ==========
__global__ void ma_kernel(const double* input, double* output, 
                          int num_stocks, int data_len, int window) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_stocks) {
        const double* stock_in = input + s * data_len;
        double* stock_out = output + s * data_len;

        double current_sum = 0.0;
        for (int j = 0; j < window - 1; ++j) stock_out[j] = 0.0;

        if (data_len >= window) {
            for (int j = 0; j < window; ++j) current_sum += stock_in[j];
            stock_out[window - 1] = current_sum / window;

            for (int j = window; j < data_len; ++j) {
                double val = stock_in[j] - stock_in[j - window];
                current_sum += val;
                stock_out[j] = current_sum / window;
            }
        }
    }
}
// 专门用于压测 GPU 浮点计算吞吐量的 Heavy Kernel
__global__ void heavy_kernel(const double* input, double* output, int num_stocks, int data_len, int window) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_stocks) {
        const double* stock_in = input + s * data_len;
        double* stock_out = output + s * data_len;

        double current_sum = 0.0;
        for (int j = 0; j < window - 1; ++j) stock_out[j] = 0.0;

        if (data_len >= window) {
            for (int j = 0; j < window; ++j) current_sum += stock_in[j];
            stock_out[window - 1] = current_sum / window;

            for (int j = window; j < data_len; ++j) {
                double val = stock_in[j] - stock_in[j - window];
                
                // 强行拉高计算密度，GPU 的 SFU 单元会在这里发力
                #pragma unroll
                for(int k = 0; k < 8; ++k) {        
                    val = tan(exp(val)) * sqrt(fabs(val) + 1.0);    
                }
                
                current_sum += val;
                stock_out[j] = current_sum / window;
            }
        }
    }
}


// ========== 1. 传统版 ==========
extern "C" void launch_ma_kernel_impl(const double* h_in, double* h_out, 
                                      int num_stocks, int data_len, int window) {
    double *d_in, *d_out;
    size_t size = (size_t)num_stocks * data_len * sizeof(double);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_stocks + threadsPerBlock - 1) / threadsPerBlock;
    ma_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_stocks, data_len, window);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// ========== 2. Pinned Memory 版 ==========
class PinnedMemoryPool {
public:
    double* h_in = nullptr;
    double* h_out = nullptr;
    double* d_in = nullptr;
    double* d_out = nullptr;
    size_t current_size = 0;
    bool initialized = false;

    void ensure(size_t size) {
        if (!initialized || current_size < size) {
            if (initialized) {
                cudaFreeHost(h_in); cudaFreeHost(h_out);
                cudaFree(d_in); cudaFree(d_out);
            }
            cudaMallocHost(&h_in, size);
            cudaMallocHost(&h_out, size);
            cudaMalloc(&d_in, size);
            cudaMalloc(&d_out, size);
            current_size = size;
            initialized = true;
        }
    }
    
    ~PinnedMemoryPool() {
        if (initialized) {
            cudaFreeHost(h_in); cudaFreeHost(h_out);
            cudaFree(d_in); cudaFree(d_out);
        }
    }
};

static PinnedMemoryPool g_pinned_pool;

extern "C" void launch_ma_kernel_pinned(const double* h_in, double* h_out, 
                                        int num_stocks, int data_len, int window) {
    size_t size = (size_t)num_stocks * data_len * sizeof(double);
    g_pinned_pool.ensure(size);

    memcpy(g_pinned_pool.h_in, h_in, size);
    
    cudaMemcpy(g_pinned_pool.d_in, g_pinned_pool.h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_stocks + threadsPerBlock - 1) / threadsPerBlock;
    ma_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        g_pinned_pool.d_in, g_pinned_pool.d_out, num_stocks, data_len, window);
    cudaDeviceSynchronize();

    cudaMemcpy(g_pinned_pool.h_out, g_pinned_pool.d_out, size, cudaMemcpyDeviceToHost);
    memcpy(h_out, g_pinned_pool.h_out, size);
}

// ========== 3. GPU 常驻内存版 ==========
class GPUMemoryPool {
public:
    double* d_in = nullptr;
    double* d_out = nullptr;
    size_t current_size = 0;
    bool initialized = false;

    void ensure(size_t size) {
        if (!initialized || current_size < size) {
            if (initialized) {
                cudaFree(d_in);
                cudaFree(d_out);
            }
            cudaMalloc(&d_in, size);
            cudaMalloc(&d_out, size);
            current_size = size;
            initialized = true;
        }
    }
    
    void free() {
        if (initialized) {
            cudaFree(d_in);
            cudaFree(d_out);
            initialized = false;
            current_size = 0;
        }
    }
};

static GPUMemoryPool g_gpu_pool;

extern "C" void gpu_upload_data(const double* h_data, int num_stocks, int data_len) {
    size_t size = (size_t)num_stocks * data_len * sizeof(double);
    g_gpu_pool.ensure(size);
    cudaMemcpy(g_gpu_pool.d_in, h_data, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

extern "C" void gpu_compute_resident(double* h_out, int num_stocks, int data_len, int window) {
    if (!g_gpu_pool.initialized) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_stocks + threadsPerBlock - 1) / threadsPerBlock;
    ma_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        g_gpu_pool.d_in, g_gpu_pool.d_out, num_stocks, data_len, window);
    cudaDeviceSynchronize();

    size_t size = (size_t)num_stocks * data_len * sizeof(double);
    cudaMemcpy(h_out, g_gpu_pool.d_out, size, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_free_memory() {
    g_gpu_pool.free();
}
// 对应的启动函数
extern "C" void launch_heavy_kernel_impl(const double* h_in, double* h_out, int num_stocks, int data_len, int window) {
    double *d_in, *d_out;
    size_t size = (size_t)num_stocks * data_len * sizeof(double);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_stocks + threadsPerBlock - 1) / threadsPerBlock;
    heavy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_stocks, data_len, window);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
