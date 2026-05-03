import sys
import os
# 确保能找到 build 目录下的 .so 文件
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))

import finance_calc_core
import numpy as np
import time

def run_benchmark():
    engine = finance_calc_core.QuantEngineSoA()
    n_elements = 1_000_000
    
    print(f"正在准备 {n_elements} 条数据...")
    engine.reserve(n_elements)
    for i in range(n_elements):
        engine.add_bar(10.0, 11.0, 9.0, float(i % 100), 1000)

    # 测试零拷贝
    start = time.perf_counter()
    prices = engine.get_close_prices()
    end = time.perf_counter()
    
    print(f"零拷贝获取 {n_elements} 条数据耗时: {(end - start)*1000:.4f} ms")
    print(f"数据类型: {type(prices)}, 前 5 个值: {prices[:5]}")

    # 修改 Python 端的 prices，验证是否直接修改了 C++ 内存（慎用，仅作原理验证）
    # prices[0] = 888.8
    # 验证内存共享：修改 Python 侧，观察 C++ 侧（或再次获取）的结果
    print(f"修改前第一个值: {prices[0]}")
    prices[0] = 999.9
    # 再次从 engine 获取，看是否同步变化
    prices_new = engine.get_close_prices()
    print(f"再次获取后第一个值: {prices_new[0]}")


if __name__ == "__main__":
    run_benchmark()
