# Round 35: EmbeddingDenseGradV2 —— 我的设计（A 节前置）

> 语义：embedding 反向——相同 indice 的 grad 行累加到 out[id]（numWeights×dim）。输入 grad + sortIndices + posIdx（上游已排序，R34 同款契约）。
> 已读：`embedding_dense_grad_v2.h`（306 行）主变体 + determinist/scale/small_dim 变体清单。

## A. CHECKLIST 设计必答

1. **与 R34 的同族对偶**：同为 sorted 段聚合，但 R34 的 updates 是"源"（只读），embedding-grad 的 grad 行是**要累加的**（多行→1 行）。生产解法：**UB 常驻累加器行**——同索引连续行用 **`AtomicAddInUb`（UB 内原子加）**累加，索引变化时一次写 output[currentId] 并清零重来。**免 GM 读改写**：普通 Add 累加需对 output 读-改-写三次 GM 流量；UB 累加只在段尾写一次。
2. **dim 超 UB**：embedding 维度大 → dimJ 循环分段（former/tail），累加器按段重置——**R22 的"沿非归约维 chunk"在散写场景的对偶（沿归约维分段驻留）**。
3. **变体族**：determinist（去 UB 原子，纯 Add 累加）/ scale（scaleGradByFreq 词频缩放）/ small_dim（dim 小单遍）。
4. **输出零初始化**：未被索引覆盖的行保持 0——框架 InitOutput 预清零（R25 B28 前置的实例）。

## 1. 我的实现（单核段语义 mini）

```
accRow(UB, 清零)：
for idx in sortIndices:
    if idx == last: AtomicAddInUb(accRow, grad_row)
    else: 写出 accRow→out[lastId]；Duplicate(accRow,0)；AtomicAddInUb(accRow, grad_row)
```
