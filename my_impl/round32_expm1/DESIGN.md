# Round 32: foreach_expm1 —— 我的设计（A 节前置）

> 语义：y_i = e^(x_i) − 1。已读：foreach_expm1.cpp——`Exp + Adds(−1)` 两指令组合，无专用指令。

## A. CHECKLIST 设计必答

1. **API 选型（R31 三级递进的组合层实证）**：expm1 无专用指令 → **基础指令组合**：`Exp` + `Adds(−1)`；仍无需手拼多项式。
2. **隐式输出原地性**：`Adds(dstLocal, srcLocal, -1)` 在 ImplictOutput 流中 dst/src 同物理缓冲——减 1 原地落（B5）。
3. **精度取舍个案**：fp16 实例化在 **half 域直算**（paramsCount=1，无 cast 升级）；bf16 才升 fp32（`<bfloat16_t, float, Expm1Adapter<float>>`）。与 R31 log 对比：**是否升 fp32 按"函数敏感度 + 指令可用性"个案决定，非一刀切**。
4. bf16 的 `__CCE_AICORE__ == 220` 守卫（R16 E4）。
