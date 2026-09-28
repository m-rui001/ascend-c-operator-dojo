# Round 11: RmsNorm 复合算子 —— 我的设计（写代码前，未读 kernel 代码）

## 0. 算子分析（来自 README/docs）

- 公式：`y = x / sqrt(mean(x²) + eps) * gamma`；`rstd = 1/sqrt(mean(x²)+eps)` 作为第二输出（fp32）
- 输入 x/gamma（fp16/bf16/fp32，ND，gamma shape = x 的后缀维），attr epsilon(double, 默认 1e-6，host 校验 <0)
- 与 Softmax 同类：行内归约 + 两遍使用 x——第 2 轮知识的直接迁移场
- 生产目录文件名透露按行长度分策略：single_row / merge_n / split_d / whole_reduce_sum——**行能不能装进 UB** 是第一分岔

## 1. API 选型

| 候选 | 结论 |
| --- | --- |
| 手搓：Mul(x,x) → WholeReduceSum → Muls/Sqrt → Mul | 基础路线 |
| 高阶 API（RmsNorm 如有） | //?? 本轮不确定是否存在，对比环节销案 |
| Softmax 范式回apply：块内一次算完 | 结构参考 |

选基础路线（练习综合运用），若存在高阶 API 记教训。

## 2. 设计

- **fold**：x 前导维合并为 M，后 gamma 维数为 N
- **行内精度**：x² 在 fp16 下会溢出（|x|>256），故 **Cast 到 fp32 后 Mul + WholeReduceSum**（Round 2/3 教训：fp32 中间量）；rstd 输出天然 fp32
- **两条路径**：
  - N ≤ ubTile（行装得下）：行一次进 UB，平方和归约 → rstd → 直接 Mul 出 y（x 读 1 遍）；gamma 行常驻（Round 5 行广播思想）
  - N > ubTile：两遍扫描（第一遍平方和，第二遍逐 tile 归一化，x 读 2 遍）——对照生产 split_d 变体
- **四件检查项**（Round 7 流程修正）：
  1. 核间：按行大小核切分 + 行偏移（含 tailRows 折算进偏移的 `min(idx,tailRows)` 修正）
  2. 尾块：行尾（tailRows）；列尾仅 split 路径（colTail 用"前 n 个"安全形态）
  3. 早退：host 计算 blockDim=有效核数（模式A），kernel 无需早退；rstd 输出同样按行核偏移
  4. workspace：0（无跨核数据交换）
- **DataCopy 对齐检查**（Round 10 新增第 5 项）：N≤ubTile 路径整行搬运按 32B 对齐处理（N*sizeof(T) 向上取整 pad，用 DataCopyPad 思路）；split 路径尾 tile 用 count 精确 + pad

## 3. 不确定点（//?? 清单）

1. 是否存在 RmsNorm 高阶 API？
2. Sqrt + Reciprocal 还是 Rsqrt 一步？精度/指令数权衡
3. gamma 与 y 相乘的顺序：y = (x*rstd)*gamma 两次 Mul vs gamma 先乘 rstd 广播——数值等价但指令数不同
4. rstd 输出按行 1 元素写 GM：Round 3 的"标量写 GM 要攒批"问题（32B 对齐）——本设计攒 rowsPerCore 个 rstd 一次写出
5. 生产的 whole_reduce_sum 变体暗示 WholeReduce 系列在归约中的特殊用法——对比环节销案
