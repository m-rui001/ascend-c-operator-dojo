# Round 47: MSELossV2 —— 我的设计（A 节前置）

> 语义：loss = reduction(mean/sum)((x−y)²)，全量归约输出**单个标量**。
> 已读：`mse_loss_v2_base.h`（双 dtype 特化 base）+ `mse_loss_v2_sum.h` 的 ReduceSumBisect/CopyOut。

## A. CHECKLIST 设计必答

1. **全量标量输出的跨核两段（R25 B28 模式在单标量上的最简实例）**：每核部分和（8 float 对齐槽）→ 写 workspace → `SyncAll()` → core0 读全部槽 → ReduceSum → cast → DataCopyPad 标量。
2. **ReduceSumBisect（R26 折半求和的块对齐变体）**：`while(len>8): offset=Ceil(Ceil(len,8)/2)*8; Add(src, src, src[offset], len-offset); len=offset`——**每次折半保持 8 元素（32B）块对齐**，收口后 Muls(scale)。
3. **mode 即子类**：Base → Sum → Mean（继承叠加归约模式，scale 参数区分）——vs TilingKey 分派（R24）：模式少且静态时子类继承更直接。
4. **scale=1/N**：mean 的除法在归约收尾一次性乘（host 传 1/N），非逐块。

## 1. 我的实现（fp32 mini）

diff(x−y)→Mul 平方→ReduceSumBisect 部分和→workspace+SyncAll+core0 聚合。
