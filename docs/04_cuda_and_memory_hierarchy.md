# 专题四：突破内存墙 —— CUDA 异构计算与 PCIe 传输开销优化

**核心价值**：展示对 GPU 存储层次结构（Memory Hierarchy）、IO 瓶颈（IO-Bound）以及 Roofline 模型的深度实证能力。

## 1. 预期违背：为什么 GPU 跑不过 CPU？

在完成 CPU SIMD 优化后，我自然地将算子移植到了 GPU 上（RTX 3060）。然而，初次压测结果却出现了严重的“负加速”：CUDA 传统版耗时高达 `6771 ms`，而 CPU SIMD 仅需 `7000 ms` 左右，两者几乎没有拉开差距。

通过 Nsight Systems 和时间拆解，我定位到了致命瓶颈：**PCIe 总线带宽**。
对于 MA（移动平均）这种算术强度（Arithmetic Intensity）极低的算子，GPU 处理 2 亿个数据点可能只需几毫秒，但通过 PCIe 总线把 1.6GB 数据往返搬运两次（H2D 和 D2H），却消耗了 99% 的时间。**这证明了算子是典型的 IO-Bound。​**

## 2. 渐进式优化：从 Pinned Memory 到常驻显存

为了突破 IO 瓶颈，我实施了渐进式的显存优化策略：

1. **页锁定内存 (Pinned Memory)​**：
   利用 `cudaMallocHost` 预分配页锁定内存，消除了 Host 端 CPU 的隐式分页拷贝，使得 DMA（直接内存访问）效率最大化，传输速度提升约 10%。
2. **终极形态：GPU 常驻内存 (Resident Memory)​**：
   在量化网格搜索（Grid Search）调参的真实业务场景中，历史行情数据是不变的。因此，我设计了常驻显存架构：
   - **Upload Once**: 将 1.6GB 全市场数据一次性 Upload 到 GPU 显存并锁定。
   - **Zero-copy Compute**: 每次回测调参时，GPU 直接在显存内读取数据进行计算。
   - **结果**: 耗时从 `6771 ms` 骤降至 **​`0.02 ms`​**，实现了物理极限级别的吞吐量。

## 3. 架构洞察：Roofline 模型的边界测试

为了进一步验证异构算力的边界，我在底层 C++ 引擎中注入了重型数学算子（包含 8 次嵌套的 `tan(exp())`）。
压测结果显示：当算术强度被强行拉高，系统从 Memory-bound 切换到 Compute-bound 时，GPU 内部的 SFU（特殊函数单元）优势彻底爆发，形成了对 CPU 的绝对碾压。这为后续量化研究员开发高复杂度非线性因子提供了坚实的基准验证。
