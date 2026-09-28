# Round 5: Broadcast 二元算子 —— 我的设计（写代码前，未看社区实现）

## 0. 问题定义

二元逐元素算子（如 Mul/Add），但 x 与 y 的 shape 允许 NumPy 广播语义：维度从右对齐，size 为 1 的维自动扩展。例：x (8,2048) + y (1,2048)，或 x (8,2048) + y (8,1)。

## 1. API 选型（延续 Round 4 方法）

| 候选 | 结论 |
| --- | --- |
| 基础 API 二元计算（Add/Mul 等） | 计算可以用，但**广播语义要自己展开**——broadcast 不发生在计算层 |
| 高维切分形态 + repeatStride=0 | Round 4 前置功课的启发：repeatStride=0 反复读同一块 = 广播读！可作为 y 广播维的实现手段 |
| 专用 Broadcast 搬运 API（印象中有 Broadcast/DataCopy 带 stride 的扩展形态） | 不确定，设计按"分场景手写"方案，对比环节销案 |

## 2. 我的核心设计思路：把"哪个输入在广播"分类

设 fold 后 x 是 (M, N)，y 是 (mY, nY)，其中 mY∈{1,M}、nY∈{1,N}（广播只允许 1 或相等）：

- **nY == N 且 mY == M**：普通逐元素，Round 4 的 elementwise 直接用
- **nY == N 且 mY == 1**（行广播，如 bias）：x 按行切核；y 只有 1 行，搬一次进 UB 常驻（TBuf），每行计算 `Add(z, xRow, yRow, N)`——y 的搬运量 O(1)，是免费的性能优势
- **nY == 1 且 mY == M**（列广播）：每行内 y 只有 1 个值。两个手段：
  a. `Adds(z, xRow, yScalar, N)`——但 yScalar 从哪来？UB→标量 GetValue 后用标量接口（SetValue/GetValue 标量通路，Round 3 教训说慢，但这里 y 本来就只有 M 个值，标量读 M 次 O(M) 可接受）
  b. 或先把 y 广播进 UB 一行再用普通 Add——多一次搬运，不如 (a)
- **mY==1 且 nY==1**（纯标量）：等价 Adds，退化情况

## 3. 多核与 tiling

- 按 x 的行切分核（行是 y 语义的原子，列广播时一行只依赖 1 个 y 值，行切分无冲突）
- 行数 < 核数：host 直接算出 blockDim（Round 4 销案的模式 A）
- 尾行/尾块：沿用大小核 + 前置功课的"前 n 个"形态

## 4. UB 预算（行广播场景，fp16）

x 队列 2 份 × BUFFER_NUM + y 常驻 1 份 + z 队列 2 份 × BUFFER_NUM = 7 份

## 5. 不确定点（//??）

1. Ascend C 是否有专门的 Broadcast 数据搬运 API（免展开）？
2. 列广播用 Adds + 标量通路，社区会不会有更好的 tensor 通路做法？
3. 广播维超过 2 维（如 (2,3,4) + (1,3,1)）的 fold 是否该在 host 做"维度收缩"（把 size=1 维合并进步长），kernel 只处理 (M,N) 两轴？
4. in-place（z 与 x 同地址）场景 Round 4 读到的"地址重叠约束"（100% 重叠才允许）对广播算子是否有额外限制？
