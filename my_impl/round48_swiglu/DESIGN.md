# Round 48: SwiGLU —— 我的设计（A 节前置）

> 语义：y = SiLU(gate) ⊙ up，SiLU(x) = x·sigmoid(x)。两输入（gate/up）逐元素门控乘。
> 已读：`swi_glu.cpp`（64 行纯分派）+ gelu_quant_base（GELU tanh 多项式生产形态）。

## A. CHECKLIST 设计必答

1. **库模板类层级（本轮新知）**：生产 kernel 只有分派，计算体 `SwigluVector<inT, computeT, outT, bufferNum>` 来自**随 CANN 发布的 lib 头**（不在开源库）——API 层级在"高阶 API"与"手写"之间还有 **2.5 层：库参数化模板类**（算子只做实例化分派）。
2. **分派结构**：tiling 字段 `isDoubleBuffer` × TilingKey(dtype) × 220 守卫 = 6 实例化——**运行时配置（buffer 数）与编译期配置（dtype）分离**：前者用 tiling 字段 if，后者用 TilingKey。
3. **SiLU 数学**（无专用指令）：sigmoid(x) = 1/(1+exp(−x)) → `Muls(−x)→Exp→Adds(1)→Reciprocal或Div(1)` → `Mul(x, sig)`；fp16 升 fp32 域（B8，sig 的 exp 敏感）。
4. **GELU 多项式生产确认**（R12 闭环）：`x²→x³→exp(−√(8/π)(x+0.044715x³))` 的 tanh-approximate 展开在 gelu_quant_base 中逐条 Mul/Exp——R12 的手拼形态与生产一致。

## 1. 我的实现

分派骨架（复刻生产）+ 我自己的 SiLU 组合（fp32 域：Muls(-1)→Exp→Adds(1)→Div(1/·)→Mul gate×up）。
