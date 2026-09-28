# Round 22: LayerNorm transpose 策略（many-rows 块状两遍方差）—— 我的设计（A 节前置）

> 已读：`layer_norm_v4_transpose.h` Process/ProcessBasicBlock/CalcGeneralParams。
> 场景：行数多、行长较短（与 R21 single-read 的"整矩阵驻留"相对）。按 ubFormer 行一块循环处理。

## A. CHECKLIST 设计必答

1. **API 选型**：归约 ReduceSum；转置用 DoTranspose/DoReshape（v4 内部封装的硬件转置）。
2. **方差路径修正（R21 结论的边界）**：**块驻留 UB 时两遍方差零额外 GM 读**——`DoSub(x−mean) 原地 → Mul → Σcoef`；R21 的 single-read 用 E[x²]−mean² 是"整矩阵驻留、少一次 UB 重算"的取舍。两变体都不重读 GM，差别在 UB 指令数与精度。**A6 精化：两遍/单遍的选择先问"块是否驻留"，再问带宽**。
3. **倒数**：生产用 `Duplicate(1)+Div(ones, temp)` 求 rstd（不用 Reciprocal 指令）——精度/习惯待对比记录。
4. **核间**：former/tail 循环块大小核（R1 模式在循环粒度复现）。
5. **UB 预算**：xLocalFp32 驻留（块）+ tmpBuf 复用（转置临时/乘法临时）+ mean/rstd 小队列。

## 1. 我的实现设计（简化版：不做硬件转置）

- 块 = nRowPerBlock 行 × rowSize（行对齐 rowAlign），2D pad load 驻留；
- Pass1：逐行 WholeReduceSum → mean（掩码计数 or V_S 均可，用 R21 精化的 GetAccVal 路线）→ 攒批 meanLocal；
- Pass2（块内原地）：`Adds(block, −meanPerRow)`（逐行）→ `Mul(block, block)` 整块一次 → `Muls(coef)` 整块 → 逐行 ReduceSum → var → rstd=1/sqrt(var+eps)（标量除法或 Duplicate+Div）→ 攒批 rstdLocal；
- Pass3：γ/β 广播乘加（R21 藏载）→ 攒批写 y。

## 2. //?? 清单

1. DoTranspose/DoReshape 的接口层实现（自定义指令？还是 UG 级 API？）——预测为指令封装，kernel 外不可直接用。
2. 生产 rstd 用 Div(ones,x) 而非 Reciprocal 的原因（精度模式/历史习惯）。
3. 转置后 DoReduce 的归约形态（按列连续归约 → 跨块行和如何拼回）。
4. mean/rstd 输出的 CopyOutMeanOrRstd 走什么粒度（每块 C0_SIZE 个值 pad 写）。
