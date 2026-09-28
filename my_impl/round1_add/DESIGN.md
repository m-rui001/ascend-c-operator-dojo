# Round 1: Add 算子 —— 我的设计（写代码前）

> 规则：本文件在阅读任何社区实现代码之前完成，只基于已读的官方文档概览。

## 1. 算子分析

- 数学表达式：`z = x + y`（逐元素）
- 输入：x, y，shape (8, 2048)，float32，ND
- 输出：z，shape 同输入
- 计算逻辑：Global Memory → Local Memory 搬入 → 矢量加法 → 搬出

## 2. 并行设计

- SPMD 模型：8 个核（AIV），每核分到 8*2048/8 = 2048 个元素
- 每核内部再切块（tile）做三级流水：CopyIn → Compute → CopyOut
- 开 double buffer：每个队列 2 块 buffer，搬入/搬出与计算重叠

## 3. Tiling 参数

- totalLength（总元素数）、tileNum（每核 tile 数，取 8）
- blockLength = totalLength / GetBlockNum()
- tileLength = blockLength / tileNum / BUFFER_NUM（再除 2 因 double buffer 逻辑上分两半）

## 4. 使用的 API

- GlobalTensor<float>::SetGlobalBuffer（按核偏移）
- TPipe + TQue<VECIN/VECOUT, 2>，InitBuffer
- DataCopy、Add、AllocTensor/FreeTensor、EnQue/DeQue

## 5. 我不确定的点（留给对比环节验证）

1. 核函数上是否需要显式声明只用 AIV 核（我记得有个 KERNEL_TASK_TYPE 宏，但记不清用法）
2. tiling 参数怎么传给核函数：是值传结构体，还是放 GM 上的 TilingData 指针？
3. 循环次数应该是 tileNum * BUFFER_NUM 还是 tileNum？（教程里 double buffer 的进度索引怎么对应）
4. 尾块不是 32B 对齐时 DataCopy 怎么办
5. 真实工程里 host 侧是走 aclnn 单算子 API + TilingFunc，还是手写 aclrtMalloc 这套
