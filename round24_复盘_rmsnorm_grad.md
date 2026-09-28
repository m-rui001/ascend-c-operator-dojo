# Round 24 复盘：RmsNormGrad（反向传播导数链）

> 我的实现：`my_impl/round24_rmsnorm_grad/`（DESIGN.md / rms_norm_grad_custom.h，fp32 整行驻留简化版）
> 对比对象：`rms_norm_grad/op_kernel/`（6 变体 4303 行：whole_reduce_n/d、split_n/d、split_n/d_high_precision + reduce_common）

---

## 一、反向算子的变体族（本轮结构性收获）

生产用 **2×3 变体矩阵**：归约轴（N=行方向整归约 / D=norm 轴分段）× 整行/分段 ×（+ high_precision 两变体），TilingKey 分派。反向算子的变体维度比正向多一层的根源：**输入多一份（dy）且输出两份（dx 逐元素 + dgamma 跨行归约）**，行长的约束同时压在两条输出通路上。

反向公式链（kernel 化）：
```
xNorm = x·rstd;  dgRow = xNorm·dy;  meanY = mean(dgRow·γ)
dx = (dy·γ − xNorm·meanY)·rstd;   dgamma = Σ_rows(dgRow)
```
关键结构点：meanY 是**行内归约**、dgamma 是**跨行归约**——一前一侧，同一个 kernel 里两方向归约共存。

## 二、我的实现自查抓到的 bug（B14 配对表纪律的必要性实证）

`wLocal`（γ 载体）在 meanY 计算时被 `Mul(wLocal, tLocal, wLocal)` **覆写**，导致后续 `dx = dy·γ` 用的是乘积而非 γ——**典型的"缓冲清单未做配对表"事故**，B14（R14）预警的场景在 R24 现场复现。修正需要独立 γ 常驻槽。CHECKLIST B14 加注：**多输入反向算子是配对表事故高发区，Init 前必须先画缓冲生命周期表**。

## 三、生产变体的三个可迁移点

1. **跨行 dgamma 的 UB 累加**：行循环内 `Add(dgBuf, dgBuf, dgRow)` 累加、循环外一次写出——与我的设计一致（B5 攒批的累加变体）。跨核聚合（各行核的部分和再归约）在 host/后续算子层完成（//?? 未完全销案，tiling 有 workspace 线索）。
2. **二叉树 Add 跨段归约**（whole_reduce_n 内）：`Add(dySum[i], dySum[i], dySum[i+colSplitNum/2])` 成对折半——当行被 colSplitNum 分段后，**用向量 Add 做折半树归约**比循环标量累加快；配合 `avgFactor` 标量乘收尾。
3. **rstd 由正向传入**（不重算 Rms）——反向算子输入复用正向中间量，**省一次整行归约**；"正向留什么中间量给反向"是算子对设计的联立决策（对应 rstd 输出接口存在于 R21/R22 的 LayerNorm 前向）。

## 四、//?? 销案状态

1. 二叉树 Add：确认存在且用于跨段和（见上），比 WholeReduceSum 灵活（可处理超 mask 宽度的分段和）；
2. dgamma 跨核：部分销案（生产分 N/D 变体，D 变体按 dgamma 轴分核+聚合，细节留后续轮）；
3. high_precision 变体：未销案（4303 行只精读了 N 变体骨架）——跨轮滚动；
4. gamma 参数：确认存在（dx 与 dg 公式都含 g）。

## 五、CHECKLIST 增量

- **B27（新）**：双输出反向算子 = 逐元素通路 + 跨行归约通路并存；dgamma 用 UB 累加器（行循环 Add，循环外一次写出）；跨段和可用二叉树向量 Add 折半。
- **B14 加注**：反向/多输入算子写 Init 前先画缓冲生命周期表（γ/β 等被多次使用的常驻量独立开槽，禁止复用载体）。
- **A3 精化**：正反向算子对设计——正向留中间量（rstd/mean）作反向输入，可省反向一次归约。

## 下一轮候选

rms_norm_grad 的 D 变体（dgamma 跨核聚合细节，销案 #2/#3）或 `foreach_copy`（搬运工厂）。
