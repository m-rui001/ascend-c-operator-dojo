# Round 35 复盘：EmbeddingDenseGradV2（UB 累加器 + 原子内化）

> 我的实现：`my_impl/round35_emb_grad/`（DESIGN.md / embedding_grad_custom.h）
> 对比对象：`index/embedding_dense_grad_v2/op_kernel/embedding_dense_grad_v2.h`（306 行）+ determinist/scale/small_dim 变体

---

## 一、与 R34 的同族对偶（sorted 段聚合的两兄弟）

| | scatter_add_with_sorted（R34） | embedding_dense_grad_v2（本轮） |
| --- | --- | --- |
| 数据角色 | updates 是**源**（只读搬） | grad 行是**被累加的**（多行→1 行） |
| 聚合载体 | 每段 DataCopy+Add（段内重读源） | **UB 常驻累加器行** + UB 内原子加 |
| GM 流量 | 段内逐行读源 | grad 每行读一次；**output 只在段尾写一次**（免 GM 读改写） |
| 段边界处理 | 标量扫描跨核 handoff | 变索引 flush+清零（连续同索引天然分段） |

**核心新知：UB 内原子加（AtomicAddInUb）**——`SetAtomicAdd` 使 V 加法以原子语义累加进常驻行，同索引多行免 GM 读改写。R25 的 SetAtomicAdd 是 GM 出口；本轮是 UB 累加语义（//??：准确 API 形态待销案，生产注释"atomic add in ub"）。确定性要求高时用 determinist 变体（去原子纯 Add）。

## 二、结构要点

1. **dim 超 UB 的分段驻留**：dimJ 循环（former/tail）——归约维太大装不下 → **累加器沿归约维分段**，每段独立完成"累加-写出"（R22"沿非归约维 chunk"的对偶：散写场景沿归约维 chunk）；
2. **变体族三件**：determinist（去原子）/ scale（scaleGradByFreq 词频缩放，需频次统计前处理）/ small_dim（dim 小单遍）——语义旗标变体（R13 B11 的延续）；
3. 输出行未被索引覆盖须为 0 → 框架 InitOutput 预清零（B28 前置的实例）；
4. 索引队列双缓冲、grad 队列双缓冲——**累加器驻留 + 双缓冲输入流**的组合是"流式聚合"的标准形态。

## 三、CHECKLIST 增量

- **B34（新）**：多行累加→1 行的场景（embedding 反向、段求和）优先 UB 常驻累加器 + UB 原子加（SetAtomicAdd 族），段尾一次写出，免 GM 读改写；确定性要求高换纯 Add 累加（determinist 变体模式）。
- **A6 补充**：归约维超 UB → 累加器沿归约维分段驻留（散写场景的对偶 chunk）。

## 下一轮候选

top_k_v3（选择/排序结构）或 determinist 变体对比（UB 原子 vs 纯 Add 的取舍细节）。
