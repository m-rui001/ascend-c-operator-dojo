# Round 49: DequantSwigluQuant —— 我的设计（A 节前置）

> 语义：int8 gate/up（+weight_scale 通道级 × activation_scale token 级 + bias）→ dequant → SiLU×Mul → ÷quantScale → int8 y + scale_gm。
> 已读：`dequant_swiglu_quant_static_base.hpp`（InitCommon/dequant/swiglu/processCompute）。

## A. CHECKLIST 设计必答

1. **数值契约链全实例（R27/R40 的收官）**：链两端都是 int8 截断——入端 dequant（Cast→Mul(weightScale)→Muls(actScale)→bias 加），出端 quant（÷quantScale→Cast int8）。中间 fp32 域算 swiglu。**每端截断点都是契约**：dequant 的乘法顺序、bias 的 fp32 加、sigmoid 的域都要与拆开实现逐位一致。
2. **scale 三来源**：常量（1.0）/输入张量（quant_scale）/动态计算（R28 输出型）。**倒数预计算于 Init**：`quant_scale = 1/GetValue(0)` 一次，逐块 Muls（B8 标量除数→乘倒数 + 初始化预计算）。
3. **双维 scale**：weight_scale（通道级/列）× activation_scale（token 级/行）——两次乘法覆盖二维缩放。
4. **.hpp 惯例**：CANN 库模板类用 .hpp（头即实现）。
5. **列块循环 × 行循环**：colLoops 外层（weightScale/bias 逐块载入）× 行内层（CopyIn 两半/dequant/swiglu/CopyOut）——二维 scale 的生命周期决定循环序（外层换 scale，内层扫行）。

## 1. 我的实现（static 路径 mini）

int8 载入 → dequant（cast→×weightScale→×actScale）→ sigmoid 组合 → gate×up → ÷precomputed 1/quantScale → cast int8 → 写 y+scale。
