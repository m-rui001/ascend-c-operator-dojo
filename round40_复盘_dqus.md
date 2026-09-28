# Round 40 复盘：DynamicQuantUpdateScatter（量化过程化 + 变体扇出峰值）

> 我的实现：`my_impl/round40_dqus/`（DESIGN.md / dqus_custom.h，by-one 量化 + 散写 mini）
> 对比对象：`index/dynamic_quant_update_scatter/op_kernel/`（base + 8 变体，1351 行）

---

## 一、量化从"输出"变"过程"（与 R28 对偶）

R28 的 dynamic quant：quant 结果是**输出**（y1 int8 + scale fp32 出口）。本轮：quant 是**散写路径上的变换**——updates 量化后写入 var 表，scale 写 varScaleOut。同一数学（Abs→max→127/max→广播除→int8），两种生命周期：
- 出口型：int8+scale 是对外契约（R28）；
- 过程型：int8 只写进内部表，契约是 var+varScale 的联合一致性（R40）。

`ComputeQuantByOne`（每行一 scale）vs `ComputeQuantByEle`（逐元素 scale）——量化粒度是 TilingMode 的一维。

## 二、变体扇出峰值（变体分类学收官观察）

TILING_MODE = AXIS_NEG_2 × {base, LARGE_BATCH, LARGE_ELE_LITTLE/LARGE_QUANT, LARGE_BATCH_LITTLE/LARGE_QUANT, MOD_64}——8 变体按 **batch 尺寸 × element 尺寸 × 量化粒度 × 对齐方式** 四维组合。结合前几轮的变体族观察：

| 算子 | 变体维度 |
| --- | --- |
| rms_norm_grad（R24） | 归约轴 × 整行/分段 × 精度 = 6 |
| dynamic_quant_update_scatter（本轮） | 轴 × batch × ele × 量化粒度 = 8 |
| foreach_copy（R30） | dtype = 12 |

**变体维度数 = 算子的自由度数**；TilingKey 的位宽预算要提前规划（dtype 12 个已接近 4bit）。

## 三、确定性偏移映射

`GetDetOffsetNeg2LargeEle`：rank2 索引（batch, idx）在确定性模式下计算**唯一目标偏移**（bsIdx×stride + …）——散写场景的确定性 = 让每个更新有唯一归属核/位置，与 R25 的 core-0 串行聚合是两种实现（此处靠索引结构保证唯一，无需聚合）。

## 四、CHECKLIST 增量

- **B37（新）**：量化作为散写路径变换时，契约 = var(int8)+varScale 联合一致；by-one（每行 scale）vs by-ele（逐元素）粒度是 TilingMode 一维；确定性可由索引结构保证唯一偏移（免原子免聚合）。
- **A3 补充**：TilingKey 位宽预算 = 变体维度组合数，提前规划（12 dtype 已近 4bit 上限）。

## 下一轮候选

40 轮节点的第二个元轮候选（CHECKLIST 决策树增补索引/散写分支）或 foreach_lerp_list。
