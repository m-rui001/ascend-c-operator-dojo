# Round 43: EmbeddingBag —— 我的设计（A 节前置）

> 语义：bag i = indices[offset[i]..offset[i+1])，聚合 weight[idx] 行：SUM / MEAN（÷bagSize）/ MAX（附 maxIndices argmax 追踪）。
> 辅助输出：offset2bag（流式）、bagSize、maxIndices。
> 已读：`embedding_bag_fp16.h`（460 行）Process/Compute/MoveAndCompute。

## A. CHECKLIST 设计必答

1. **变长段聚合（与 R34/R35 的等长/连续段对偶）**：段边界由 offsets 张量驱动（每 bag 两次 GetValue），段长任意 → 累加器跨"块"驻留（长 bag 用 indicesMaxMoveLength 分块，块间累加器不重置）。
2. **三 mode 共存单 kernel**：SUM=Add 累加；MEAN=SUM 后 ÷bagSize；MAX=首行初始化 + 逐行 Max + **maxIndices argmax 追踪**（记录最大值来自哪个索引）。mode 分支运行时（tiling 常量）。
3. **辅助簿记输出增量维护**：offset2bag 首块流式写出、bagSize 计数、maxIndices 随 Max 更新——主循环顺带维护，无额外扫描。
4. **索引逐个标量读**（GetValue per index，O(bag size)）+ weight 行按索引取——稀疏访存正当（B3 控制流/索引语义）。
5. perSampleWeights 可选乘（SUM 模式）。

## 1. 我的实现（三 mode mini，fp32）

驻留累加器 y + 分块扫描 bag 索引 → SUM/MEAN/MAX 三路 → bagSize/maxIndices 簿记 → 段尾写出。
