# Round 59: AdaptiveAvgPool3D —— 我的设计（A 节前置）

> 语义：自适应池化——每个输出元素的窗口边界由比例公式计算（变长窗口、相邻窗口重叠）。
> 已读：`adaptive_avg_pool3d/op_kernel/`（multi_w/split_w/split_c 三变体 + common 的 6D Index 结构）。

## A. CHECKLIST 设计必答

1. **窗口索引预计算**：startWIndex/endWIndex 先算好存进 **indexBuf（int64 tensor）**，主循环 GetValue 取——**索引预计算到张量**（控制量 tensor 化的窗口版）。
2. **重叠窗口共享载入**：一次 CopyIn（blockCount=wend−wstart 行）喂**多个窗口累加器**（sumBufLocal[offset]，offset+=cLengthAligned）——相邻窗口重叠是 adaptive 的主要收益点，共享载入省重复读。
3. **逐窗口除数**：MEAN 的除数 = d_len×h_len×w_len **逐窗口不同**——factor 在窗口循环内标量算 + Muls。
4. **变体按形状分派**：multi_w（W 窗口多）/ split_w（W 超长切分）/ split_c（通道切分）——**阻塞轴按 shape 选**（R40 变体分类学）。
5. **6D Index 结构**（d,h,w + nstride/dstride/hstride/wstride + GetOffset/GetIndexFromBuffer）——偏移数学结构体化。

## 1. 我的实现（1D 自适应均值池化 mini：预计算窗口 + 共享载入 + 逐窗口除数）
