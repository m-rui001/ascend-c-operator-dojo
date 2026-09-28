# Round 26 复盘：high_precision 变体销案（RmsNormGrad 精度路径对比）

> 方法：`diff` 两变体的函数集合 → 精读 `ReduceSumHalfInterval`（reduce_common.h）与 high_precision 的 Compute。
> 对比对象：`rms_norm_grad_split_n.h`（865 行）vs `rms_norm_grad_split_n_high_precision.h`（542 行）。

---

## 一、销案结论：high_precision = 成对折半求和 + rstd 张量广播（不是 fp64/Kahan）

| 差异点 | 默认版 split_n | high_precision 版 |
| --- | --- | --- |
| mean 归约 | 常规 ReduceSum（跨段顺序累加） | **`ReduceSumHalfInterval`：折半树形求和**——`Add(src, src, src[bodyCount], tail)` 后 bodyCount 减半迭代，深度 O(log n)，最后单次 WholeReduceSum 收口 |
| rstd 作用到行 | 标量 `GetValue` + `Muls` | **`BroadCast<float, DIM_NUM, DIM_D>(row, rstdLocal, dstNDShape, srcN1Shape, sharedTmp)`**——R5 学的广播 API 把每行 rstd 张量化铺开，全程 tensor 通路 |
| 代码组织 | 按 dtype 拆 8 个函数（ComputeMainFp16/Bf16/SmallD…） | 单一 Compute + `constexpr if` dtype 分支（542 行 vs 865 行） |

**"high precision"的实质 = 降低归约的舍入累积误差**（成对求和误差 O(log n) vs 顺序 O(n)），不动数值格式。R24 的遗留猜测（fp64/Kahan）证伪。

## 二、可迁移 idiom

1. **折半树形求和（CHECKLIST B29）**：`findPowerTwo(count)` 找不超过 count 的 2 幂 → 尾段 `Add(self, self[body], tailCount)` → 循环 `body/=2; Add(self, self[body], body)` → 收口 WholeReduceSum。**大行数归约的精度手段**，代价是 O(n) 次 Add（带宽多一遍 UB 内流量）——默认版不用它说明常规精度够用，仅"高精模式"开启。
2. **per-row 标量广播回行**：反向算子常见的"每行一个系数作用到整行"，张量化方案是 BroadCast API + sharedTmp（R5 API 三件套的第三例：配套 GetBroadCastMaxMinTmpSize → tmpSize 进 tiling）。
3. **代码组织反差**：默认版按 dtype 手工特化（8 函数），high_precision 用 constexpr 统一——同一族内两种组织并存，新写变体优先 constexpr。

## 三、台账更新

| R24 遗留 | 状态 |
| --- | --- |
| dgamma 跨核聚合 | ✅ R25 销案 |
| high_precision 差异 | ✅ 本轮销案（折半树形求和 + BroadCast） |
| 二叉树 Add 边界 | ✅ 并案（ReduceSumHalfInterval 即其通用化封装，已被生产抽成公共头） |

## CHECKLIST 增量

- **B29（新）**：大行数归约精度不足时用折半树形求和（Add self+offset 减半迭代 + WholeReduceSum 收口），误差 O(log n)；常规精度够用时不开启（UB 流量翻倍）。
- **A1 补充**：per-row 标量作用到行 = BroadCast API 张量化（配套 tmpSize 三件套），优于标量 Muls 循环。

## 下一轮候选

foreach_unary_v2 / foreach_copy（搬运工厂 v2）；或 CannOps `norm/add_rms_norm`（正反向+残差加融合，R11 RmsNorm 的进阶）。
