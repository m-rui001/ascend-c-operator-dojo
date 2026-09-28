# Round 53: FA Grad Post —— 我的设计（A 节前置）

> 语义：主 grad 核把 fp32 部分和落 workspace（dq/dk/dv 三段），post 阶段做 聚合+rescale+cast+布局转换 → 最终输出。
> 已读：`flash_attention_score_grad_post.h`（553 行）Process/NZ2ND/NZVecClc。

## A. CHECKLIST 设计必答

1. **Post 独立阶段模式（本轮核心）**：确定性聚合的主核部分（双 SyncAll，R52）之后，**rescale+cast+布局转换拆成独立 post 核**——主核保持并行纯计算，post 串行处理收尾。判据：收尾操作（cast/scale/布局）与主计算耦合会破坏流水。
2. **NZ→ND 布局转换**：Matmul NZ 输出经 `Copy`（UB→UB，CopyRepeatParams 四步长参数，c0×c1×n 三层循环）重排为 ND——**NZ 是 Cube 的母语，ND 是 Vector 的母语，转换在 post**。
3. **cast 写出对齐随 dtype**：fp32 输出对齐 8 元素、fp16 对齐 16 元素（均 32B）——`(n+7)/8*8` vs `(n+15)/16*16`。
4. **workspace 三段偏移**：dq/dk/dv 部分和各占一段（Init 里 offset 推导）。

## 1. 我的实现（post mini：读 workspace 部分和 → rescale → cast → 写出）
