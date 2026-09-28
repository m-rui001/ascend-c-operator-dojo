# Round 34: ScatterAddWithSorted —— 我的设计（A 节前置）

> 语义：updates 按 sorted_index 散写累加进 output；pos 记录原位置。**索引已排序** → 重复索引连续 → 段聚合。
> 已读：`scatter_with_sorted.h`（207 行，value 聚合核心）+ int 变体。

## A. CHECKLIST 设计必答

1. **关键决策：排序消原子**。R25 的三态（原子/workspace/分区）之外——**索引排序使重复段连续，聚合退化为段内求和+段尾一次写**，无原子、无 SyncAll、确定性。宿主侧（框架/上游）负责提供排序索引与 pos。
2. **跨核段归属（生产独特模式）**：段可能跨核边界——**段首所在核独占前向扫描**：core0 起步，逐索引比较发现段边界；段跨到下一核时，owning core **继续读下一核的 GM 索引区**直到段尾，聚合后一次写。无跨核同步（GM 全局可读），代价是 owning core 多读邻居数据。
3. **标量边界扫描的正当性**：段边界检测是数据相关分支（R23 B26 的 kernel 内分支），逐索引 GetValue 是 O(索引数) 标量读——此处被生产接受（索引元素少、分支本质标量），与 B3 不冲突（B3 禁的是数据计算走标量）。
4. **UB 预算**：indices/pos 两个小队列 + updates 段缓冲。

## 1. 我的实现（段聚合 mini，单核语义）

```
读 indices+pos 段 → 扫描边界：
  段 [s..e) 同索引 idx：updates[idxPos] 段内 DataCopy+Add 累加 → 一次写 output[idx]
（跨核 handoff 模式进复盘，不在 mini 中复刻）
```
