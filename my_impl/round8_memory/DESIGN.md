# Round 8: 内存语义专题（TBufPool / 内存复用 / L2 cache）—— 我的设计（写代码前，未看社区代码）

## 0. 前置功课收获

1. **TBufPool**（样例 README 文档）：大场景内存不够一次装下时，把 TPipe 用 `InitBufPool` 划出子资源池；**两个池可声明互相复用**（共享起始地址与长度），分别用于两个计算阶段——阶段切换即内存复用，无需重复申请。
2. **L2 cache**（官方 API）：GlobalTensor 上 `SetL2CacheHint(CacheMode)`，默认 `CACHE_MODE_NORMAL`（使能 L2）；**一次流过的数据（读一次不再用 / 无核间复用）应 DISABLE，避免把 L2 污染掉**。另有 `DataCacheCleanAndInvalid` 用于标量写回场景的缓存控制。
3. **地址重叠约束**（Round 4 已读）：单次迭代内源目 100% 重叠允许；Add/Sub 等部分双目指令在特定条件下允许 dst 与第二源重叠（in-place 计算）。

## 1. 本轮练习算子设计

两阶段逐元素算子（模拟"资源不足以一次装下"的场景）：

```
输入 x, y（各 totalLength）
阶段1：前半 z1 = x1 + y1   （Add，且用 in-place：dst 复用 x 的缓冲）
阶段2：后半 z2 = x2 - y2   （Sub）
输出 z（totalLength）
```

单核分两轮：round1 用 pool1，round2 用 pool2，**pool2 与 pool1 复用同一块 UB**。

## 2. 三项技术的应用决策

| 技术 | 我的设计决策 | 预期收益 |
| --- | --- | --- |
| TBufPool | TPipe 先给常驻队列，剩余空间划 pool1/pool2 且二者复用；两轮计算共享同一物理 UB | UB 容量近似减半，同等 UB 能跑更大单批 |
| 内存复用（tensor 级） | 阶段1 Add 直接 `Add(dst=xLocal, src0=xLocal, src1=yLocal)`（in-place，依赖 Round 4 读到的重叠约束） | 省一个输出缓冲（3 份→2 份） |
| L2 cache hint | x/y/z 都是"一次流过、无复用"→ 三个 GlobalTensor 都 `SetL2CacheHint(CACHE_MODE_DISABLE)` | 不污染 L2，其他核/L2 复用数据不受损 |

## 3. 不确定点（//?? 清单）

1. 划池/建池的 API 准确形态：是 `pipe.InitBufPool(pool, ...)` 还是独立 TBufPool 类型构造？池如何声明"与另一池复用"？
2. TBufPool 里分配 tensor 的方式：还用 AllocTensor/FreeTensor 队列语义，还是 TBuf 的 Get 语义？
3. 池复用后，前一轮的 tensor 是否需要显式"释放/失效"，还是直接被覆盖即可？
4. L2 hint 的合理粒度：DISABLE 是 per-GlobalTensor 的，那"读禁用、写使能"这类混合策略怎么表达？CACHE_MODE 有哪些枚举值？
5. in-place Add 的条件：dst 与第二源重叠 + stride 为 0——我的场景满足吗（同一片连续 UB）？
6. 两阶段各自的队列要不要也省掉（阶段内串行搬-算-搬，直接 TBuf 而非 TQue）？
