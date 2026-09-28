# Round 18: foreach_norm —— 我的设计（CHECKLIST 前置）

> 目标算子：foreach_norm（每张量 L2 范数，输出每张量一个标量）。走 v2 reduce 工厂。
> 已读：`op_kernel_v2/kernel_foreach_reduce.h`（Process/Init 主体）。

## A. CHECKLIST 设计必答

1. **API 选型**：计算=Power(平方)→ReduceSum→Sqrt；工厂=KernelForeadhReduce + Predicate。查 ReduceMax/ReduceSum 配套（R3 已销案），无更高层封装。
2. **产品矩阵**：v2 层用 `#if __CCE_AICORE__ == 220` 守卫。
3. **封装边界**：两段归约的跨核同步由工厂做（CrossCore flag 对）；我的 Predicate 只定义"单段内怎么归约"。
4. **瓶颈**：访存型（读全量、写 N 个标量）——MTE2 瓶颈，BUFFER_NUM=2。
5. **中间量路径**：**partial 落 workspace 不落 GM 输出**（stage1 写 workspace，stage2 才写输出）——融合收益的列表版。
6. **原子 vs 两段归约**：标量输出、要确定性 → **两段归约**（R10 教训，正是本工厂的形态）。
7. **UB 预算**：dataQueue 2×(inputsTensorUbSize×COPY_SPACE_MULTIPLE)（fp16 路径额外 float32Queue 存 cast 后 fp32）；outQueue 只要 32B（输出就是标量！）；calcBuf 给 ReduceSum 工作区。

## 1. 工厂结构的理解（要验证的模型）

- **Stage1**：每核扫自己 [tensorStart, tensorEnd] 的张量份额，每张量份额归约成一个 partial 标量（fp32），写到 workspace 的 `coreMiddleOffset + i - tensorStart` 位置（按 [核][核内张量序] 布局）。
- **同步**：`CrossCoreSetFlag<0, PIPE_MTE3>(1)` + `WaitFlag(1)` 一对旗（PIPE_MTE3 保证 workspace 写可见）。
- **Stage2**：核 b 跨步认领输出张量 i = b, b+needCoreNum, ...；从 workspace 读该张量全部 partial（`tensorMiddleStartList/CountList` 是 [张量→workspace 区间] 的映射），二次归约 + Sqrt，写输出标量。
- **零长度张量**：OutputZero 兜底。
- **我的 Predicate**（NormPred）：
  - stage1：`Mul(x,x)→ReduceSum`（fp32 域）→ partial = Σx²
  - round2：partial 们再 ReduceSum → **Sqrt 一次**（sqrt 只能出现在最后，否则不满足范数可加性）

## 2. //?? 清单

1. tiling 数组 `tensorMiddleStartList/tensorMiddleCountList/coreMiddleOffset` 的精确语义（写读两侧的 workspace 索引换算）。
2. fp16 路径的 `COPY_SPACE_MULTIPLE` 与 float32Queue 的申请/常驻方式（Alloc 一次 EnQue 全程持有？）。
3. OutputZero 的实现（DataCopyZero? 置零后拷出）。
4. round2 的归约输入是 workspace 读回的 partial 数组——用哪个接口归约（ReduceSum count=partials 数）？
5. foreach_norm 生产实例的 Predicate 长什么样（是否还处理 mean/整除）。
