# Round 2: Softmax 算子 —— 我的设计（写代码前，未看社区实现）

> 应用 Round 1 教训：泛化 shape、尾块处理、UB 预算表、uint64、模板 dtype。

## 1. 算子分析

- 数学表达式：`out[i][j] = exp(x[i][j] - max_j(x[i][j])) / sum_j(exp(x[i][j] - max_j(x[i][j])))`
- 输入：x，shape (M, N)，last dim 连续，ND；输出同 shape
- 数值稳定：先减行最大值再 exp（防溢出）
- 行内归约：max 和 sum 都是行方向（last dim）归约

## 2. 与 Add 的本质区别（设计新难点）

1. **跨 tile 依赖**：一行可能大于一片 UB，需要多片扫描。max/sum 是行内跨片的累加状态，tile 之间不再独立。
2. **三次扫描**：x→max、(x-max)→exp&sum、exp/sum→out。要么读三遍 x（省 UB），要么缓存整行（费 UB）。我选**读三遍 x**（从 GM 重读），行状态（max、sum）各保一个标量/一元 tensor。
3. **多核划分**：按行切核（每核若干整行），行是归约的原子单位，不能把一行劈给两个核。

## 3. UB 预算表（每核，fp16 为例）

| 用途 | 份数 | 大小 |
| --- | --- | --- |
| inQueue X（double buffer） | 2 | ubTile * 2B |
| outQueue Y（double buffer） | 2 | ubTile * 2B |
| 中间 exp 结果（第三遍要除 sum） | 1 | ubTile * 2B（用 TBuf，不进队列） |
| 行 max 广播 / 行 sum 广播 | 2 | 32B 对齐的 1 元素级（用 ReduceMax/ReduceSum 得到 per-tile 局部值后与行状态合并） |

ubTile：取 `UB / (2+2+1)份 / BUFFER_NUM` 向下取 32B。fp16 时约几 KB 元素量级。

## 4. Tiling 参数（host 动态计算）

- rowsPerCore / big/small core 行数（沿用 Round 1 大小核思想，但以"行"为单位）
- colTileNum：每行切几片、ubTile：每片列数、尾片列数 tailColNum
- rowTailNum：最后多余行的行数

## 5. kernel 伪代码

```
for r in rows_of_this_core:
    rowMax = -inf; rowSum = 0
    for tile in row:  rowMax = max(rowMax, ReduceMax(x[tile]))
    for tile in row:  e = Exp(x[tile]-rowMax); rowSum += ReduceSum(e)  # 同时可存 e 到 GM/UB
    for tile in row:  y[tile] = Exp(x[tile]-rowMax) / rowSum → DataCopy out
```

## 6. 我不确定的点（留给对比）

1. ReduceMax/ReduceSum 的输出形状/用法：是不是产出一个 per-call 标量型 LocalTensor？跨 tile 合并 max 用 Max 接口逐元素更新一元 tensor 可以吗？
2. 社区是否会用高阶 API（我记得 Ascend C 有 SoftMax 高阶接口），还是全部手搓三遍扫描？
3. exp 的接口名（Exp）与精度（fp16 直接算还是转 fp32 累加 sum？累加精度问题我担心 fp16 求和误差，倾向中间用 fp32）
4. 尾片列数不足 32B 时 DataCopy 怎么办（Add 那轮的遗留问题）
5. 社区如何组织"行状态"：TBuf？寄存器标量？还是 WholeReduce 系列接口一步到位？
