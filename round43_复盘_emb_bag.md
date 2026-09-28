# Round 43 复盘：EmbeddingBag（变长 bag 聚合 + 簿记输出）

> 我的实现：`my_impl/round43_emb_bag/`（DESIGN.md / embedding_bag_custom.h，三 mode mini）
> 对比对象：`index/embedding_bag/op_kernel/embedding_bag_fp16.h`（460 行）

---

## 一、变长段聚合（与 R34/R35 的段模型对表第三例）

| 段模型 | 段边界 | 累加器 |
| --- | --- | --- |
| R34 sorted 散写 | 重复索引连续，比较发现 | 每段重读 |
| R35 embedding-grad | 同索引连续（sorted） | UB 常驻跨行累加 |
| **R43 bag（本轮）** | **offsets 张量定界（变长）** | 驻留跨块（长 bag 分块不重置） |

bag 的段长任意、由 offsets 数组驱动——段边界读两次 GetValue，段内索引逐个标量读（稀疏访存正当）。**offsets 驱动的变长段**是排序段的泛化。

## 二、三 mode 共存与簿记输出

1. **SUM/MEAN/MAX 单 kernel 三路**：MEAN=SUM 后 ÷bagSize（惰性，bagSize>0 才除）；MAX=−inf 初始化 + 逐行 Max；生产 MAX 路还维护 **maxIndices（argmax 追踪）**——记录最大值来自哪个索引，需在 Max 的同时比较更新（向量级 argmax）。
2. **簿记输出增量维护**：offset2bag 首块流式写出、bagSize 逐 bag 计数、maxIndices 随行更新——主聚合循环顺带维护，**无第三次扫描**（R27 xOut 流式写出同型）。
3. **长 bag 分块**：indicesMaxMoveLength 分块读索引，累加器跨块驻留——块间不重置（与 R35 同）。
4. 生产用 `intToFloatBits`（union 位转换读累加器）与四组 SyncXxxY 事件对封装（M2toV/VtoM3/VtoS/M3toS）——**事件对封装成具名函数**是大型算子的工程化形态（B14 的可读性进化）。

## 三、CHECKLIST 增量

- **B38（新）**：变长段聚合（offsets 驱动）= 边界 GetValue ×2 + 段内索引标量驱动；MEAN 惰性除；MAX 用 −inf 初始化 + argmax 追踪需同步簿记；簿记输出（bagSize/offset2bag）主循环增量维护，禁第三遍扫描。
- **B14 工程化**：大型算子把事件对封装为具名函数（SyncM2toV 等），配对表以函数名为单位自查。

## 下一轮候选

top_k 之外的"选择类"——masked_select_v3（位图压缩实战，与 R33 GatherMask 呼应）或 CHECKLIST E/A 节对表小元轮。
