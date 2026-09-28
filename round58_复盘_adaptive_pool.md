# Round 58 复盘：AdaptiveAvgPool3D（自适应窗口 + 重叠共享）

> 我的实现：`my_impl/round58_adaptive_pool/`（DESIGN.md / adaptive_pool_custom.h，1D mini）
> 对比对象：`pooling/adaptive_avg_pool3d/op_kernel/`（multi_w/split_w/split_c 三变体 + common 6D Index）

---

## 一、自适应池化的结构要点

1. **窗口索引预计算到张量**（startWIndexBuf/endWIndexBuf，int64）：主循环 `GetValue` 取边界——控制量张量化（B3）的窗口版；避免每输出点重算比例公式。
2. **重叠窗口共享载入**：multi_w 变体一次 CopyIn（blockCount=wend−wstart）喂**多个窗口累加器**（sumBufLocal[offset] 逐窗推进）——adaptive 的核心收益：相邻窗口大量重叠，共享载入省重复读。
3. **逐窗口除数**：MEAN 除数 = d_len×h_len×w_len **逐窗口不同**——窗口循环内标量算 factor + Muls（自适应与固定池化的本质区别）。
4. **变体按形状分派**：multi_w / split_w（W 超长）/ split_c（通道切分）——阻塞轴按 shape 选（R40 分类学）。
5. **6D Index 结构体**（d/h/w + 四个 stride + GetOffset/GetIndexFromBuffer）——偏移数学结构体化，生产在 common 头共享。

## 二、我的差距与改进点

1. 我的 1D mini 逐元素 GetValue 求窗口和（O(窗口长) 标量）——**宽窗口应改前缀和差分**（prefix sum：acc 前缀和一次算好，窗口和 = pref[we]−pref[ws] 两次读），或生产式多窗口共享载入；
2. 生产窗口边界由 tiling 下发（host 算好），我核内算——R16"能 host 算的都 host 算"的边界案例（边界依赖 inputLen/outLen 是静态的，应 host 算）；
3. 3D 版的 D/H/W 三层循环 + blockCount DataCopyPad 的组合是"窗口块访存"的完整形态——mini 只做 1D。

## 三、CHECKLIST 增量

- **B50（新）**：自适应窗口（比例边界、相邻重叠）→ 窗口索引预计算到张量 + 重叠窗口共享载入（一次 CopyIn 喂多累加器）+ 逐窗口除数（MEAN 除数随窗口变）；宽窗口改前缀和差分；变体按阻塞轴分派（multi_w/split_w/split_c 式）。
- **C5 边界案例**：窗口边界虽是"派生量"，但若静态（shape 决定）→ host 算好下发；若数据依赖 → 核内预计算到张量。

## 下一轮候选

conv2d_transpose_v2（转置卷积 dim 推导）或 mse_loss_grad（loss 反向）。
