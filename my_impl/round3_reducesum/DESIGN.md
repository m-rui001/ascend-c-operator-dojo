# Round 3: ReduceSum 算子 —— 我的设计（写代码前，未看社区实现）

## 0. API 选型（Round 2 教训先行）

按 last-dim 归约的场景，候选接口：

| 候选 | 我的理解 | 取舍 |
| --- | --- | --- |
| `ReduceSum<T>(dst, src, mask, count)` 基础 API | 逐片归约，输出小 tensor，需要自己跨片累加 | 可行但碎；跨片状态管理是我 Round 2 踩过的坑 |
| `WholeReduceSum<T>(dst, src, mask, count, repeat, params, ...)` | 一次把多 repeat 整块归约成每 repeat 一个值 | 待确认签名，直觉上这才是整块归约的主力 |
| 高阶 API（ReduceSum/Reduce 高阶封装） | 我**猜**存在类似 SoftMax 的高阶封装带 Tiling 结构 | 设计假设：优先用它；若无则用 WholeReduceSum |

结论：设计按"WholeReduceSum 整块归约 + 跨块标量累加"写；如果对比环节发现存在高阶 API，记一条教训。

## 1. 算子分析

- 数学：`y[m] = sum_n x[m][n]`（last dim 归约，fold 前导维为 M 行）
- 输入 (M, N) fp16/fp32；输出 (M,) 或标量（本实现按每行一个输出，支持 keepdim 语义讨论但不展开）
- 输出量极小（M 个元素），瓶颈在**读 x 的带宽**，写出开销可忽略

## 2. 与前两轮不同的新问题

1. **归约输出的对齐**：输出每行 1 个元素，但 DataCopy 写 GM 最小 32B——多行输出要不要攒一批再写？
2. **计算强度极低**：读 1 次算 1 次写 0.0x 次，纯粹带宽型算子，double buffer 收益最大化的场景。
3. fp16 归约精度：整行直接 fp16 累加会掉精度，倾向 fp32 累加。

## 3. 设计

- 核间：按行大小核切分（沿用 Round 2）
- 核内：每行 `WholeReduceSum` 整块；行 > UB 时跨片累加（fp32 标量累加，最后一行结束转 T 写出）
- 输出：每核把本核所有行的结果攒在一片小 UB 里，行循环结束后一次 DataCopy 写出（32B 对齐用 pad 思路，具体待验证）
- UB 预算：x 队列 2 份 × BUFFER_NUM + 输出暂存 1 小片
