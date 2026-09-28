# Round 25 复盘：dgamma 跨核聚合（rms_norm_grad split_d 的两路输出）

> 我的实现：`my_impl/round25_dgamma_agg/`（dgamma_two_stage.h，自创"跨步聚合"mini 版）
> 对比对象：`rms_norm_grad_split_d.h` 的 `CopyDgammaOutWorkspace / CopyDgammaOut / AddDgamma / InitOutput`
> 本轮性质：销案轮——解决 R24 遗留的"dgamma 跨核如何聚合"。

---

## 一、销案：dgamma 跨核聚合的两条生产路径

split_d 按行分核 → 每核得到 dgamma 的**部分和**（列全宽），聚合有两条路（`fixed_output` 旗标选择）：

| 路径 | 机制 | 性质 |
| --- | --- | --- |
| 默认（快） | **`SetAtomicAdd<float>()` → `DataCopyPad(dgammaGm, partial)` → `SetAtomicNone()`**——每核原子直写 GM 输出 | 无 workspace、无 SyncAll；**求和顺序不定** |
| 确定性（fixed_output） | 部分和 → **每核 workspace 槽** → `SyncAll` → **core-0 串行聚合**（`if (GetBlockIdx()!=0) return;` 后按 列段×核 双层循环 DataCopyPad+Add，顺序固定） | 可复现；聚合串行化（dgamma 只有 rowSize 元素，代价可接受） |

配套机制：
- **`InitOutput<float>(dgammaGm, col_val, 0)`**：GM 输出区清零 API——原子路径的前置条件（GM 必须从 0 开始累加）；
- **`SyncAll()` 在 InitOutput 之后**：保证清零对所有核可见，原子加才不丢；
- 确定性路径的聚合循环里 `PipeBarrier<PIPE_ALL>()` 全管线屏障成对出现（GM 读→UB→Add 的链）。

R10 的"确定性与原子互斥"教训在此落到**两个具体的出口函数**——同一份部分和，选不同写出函数即选了确定性语义。

## 二、我的自创方案对照（跨步聚合未被生产采用）

我写的"Stage2 核按列段跨步、累加所有核槽"是第三种可能——生产没选它。推测原因：确定性路径干脆串行到 core-0（dgamma 仅 rowSize 元素，聚合开销相对反向主计算可忽略），跨步并行化省的时间不值 SyncAll 双次与代码复杂度。**教训：聚合策略的成本要对照主计算量级评估，小输出不需要并行聚合。**

我的实现的另一个不严谨处：Stage2 的 `Add(sum, sum, sum)` 自加是占位错误（需要双 buffer 轮转读入），生产 `AddDgamma` 用 dgamma/tmpBuf 两 buffer + DataCopyPad 读入——B14 配对表再次应验。

## 三、CHECKLIST 增量

- **B28（新）**：跨核聚合三态：①按归约轴分核→分区即终值（无聚合）；②原子直写（SetAtomicAdd+DataCopyPad+SetAtomicNone，快但乱序）；③workspace 每核槽+SyncAll+单核串行聚合（确定性）。选择 = 确定性需求 vs 聚合成本（输出尺寸小则串行可接受）。前置：原子路径需 InitOutput 清零 + SyncAll 保证可见。

## 四、R24 //?? 台账更新

| 遗留项 | 状态 |
| --- | --- |
| dgamma 跨核聚合 | ✅ 销案（两路出口函数，见上） |
| high_precision 变体差异 | 仍开放（跨轮滚动） |
| 二叉树 Add 适用边界 | 部分销案（与 B28 的分段求和互补） |

## 下一轮候选

foreach_unary_v2 / foreach_copy（搬运工厂）或 high_precision 变体销案。
