cmake_minimum_required(VERSION 3.18)
project(finance_calc_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== 1. 编译选项 (Options) ==========
option(USE_CUDA "Enable CUDA acceleration" ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_AVX2 "Enable AVX2 SIMD instructions" ON)

# ========== 2. 查找依赖 ==========
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 86) # 对应你的 RTX 30 系列 / i9 平台 GPU
endif()

# ========== 3. 定义 Pybind11 模块 ==========
# 注意：这里使用相对路径 cpp_core/src/...
pybind11_add_module(finance_calc_core cpp_core/src/data_struct_soa.cpp)
target_include_directories(finance_calc_core PRIVATE cpp_core/include)

# ========== 4. 平台特化与 SIMD ==========
if(MSVC)
    target_compile_options(finance_calc_core PRIVATE /bigobj /MP /utf-8 /O2)
    if(USE_AVX2)
        target_compile_options(finance_calc_core PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(finance_calc_core PRIVATE -fvisibility=hidden -O3)
    if(USE_AVX2)
        target_compile_options(finance_calc_core PRIVATE -mavx2 -march=native -ffast-math)
    endif()
endif()

if(APPLE)
    target_link_options(finance_calc_core PRIVATE -undefined dynamic_lookup)
endif()

# ========== 5. 链接 OpenMP ==========
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(finance_calc_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# ========== 6. 链接 CUDA (如果启用) ==========
if(USE_CUDA)
    # 编译 CUDA 核函数为静态库
    add_library(cuda_kernels STATIC cpp_core/src/ma_kernel.cu)
    set_target_properties(cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(cuda_kernels PRIVATE cpp_core/include)
    
    # 链接 CUDA 库并保留符号
    target_link_libraries(finance_calc_core PRIVATE cuda_kernels)
    
    if(NOT MSVC)
        target_link_options(finance_calc_core PRIVATE 
            "-Wl,--whole-archive" 
            $<TARGET_FILE:cuda_kernels>
            "-Wl,--no-whole-archive"
        )
    endif()
    
    # 注入宏定义，让 C++ 代码知道 CUDA 已启用
    target_compile_definitions(finance_calc_core PRIVATE HAS_CUDA=1)
endif()
