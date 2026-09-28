# Round 40: DynamicQuantUpdateScatter —— 我的设计（A 节前置）

> 语义：updates 动态量化后按 indices 散写进 var 表，varScaleOut 记录每行 scale（R28 量化的散写对偶：quant 是过程不是输出）。
> 已读：base 头（ComputeQuantByOne/ByEle/Abs+ReduceMax）+ TilingMode 扇出（8 变体）。

## A. CHECKLIST 设计必答

1. **量化粒度两态**：`ComputeQuantByOne`（每行一个 scale：Abs+ReduceMax → 127/max → Muls 广播 → cast）与 `ComputeQuantByEle`（逐元素 scale，R28 的 by-ele 路线）——granularity 是 TilingMode 的一维。
2. **变体爆炸点**：TILING_MODE = AXIS_NEG_2 × {base, LARGE_BATCH, LARGE_ELE_LITTLE_QUANT, LARGE_ELE_LARGE_QUANT, LARGE_BATCH_LITTLE/LARGE_QUANT, MOD_64}——**batch/element 尺寸 × 量化粒度 × 对齐方式的组合扇出**。R24（6 变体）之后又一扇出峰值；变体命名即分类学。
3. **确定性偏移映射**：`GetDetOffsetNeg2LargeEle`——rank2 索引（batch,idx）在确定性模式下计算唯一目标偏移，避免原子。
4. smoothScales 可选融合（量化前逐元素乘）。

## 1. 我的实现（by-one 量化 + 散写 mini）

updates 行 → Abs → ReduceMax → scale=127/max → Muls(1/scale) → cast int8 → DataCopy 写 var[idx]；scale 写 varScaleOut[idx]。
