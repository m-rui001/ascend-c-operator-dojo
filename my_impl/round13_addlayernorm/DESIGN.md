# Round 13: AddLayerNorm 融合算子 —— DESIGN（CHECKLIST 逐项过）

## A. 设计必答

1. **API 选型**：融合算子（A1 精化条目实战）——查无 AddLayerNorm 高阶 API（LayerNorm 高阶 API 存在但不含 Add 前置与本算子的 4 输出形态，//?? 对比销案）→ **手拼**，保留 x_added 中间量通路（要输出 x 且续接归一化）。
2. **产品支持**：手拼基础 API，天然跨芯片 ✓。
3. **封装边界**：手拼，全部自管 ✓。
4. **瓶颈**：读 2 份输入 + 1 份输出 + 归一化算术（两次归约）——MTE 与 V 均衡型，BUFFER_NUM=2。
5. **融合/物化**：**本算子本体就是融合**——x_added 不落 GM（除非 additionalOut=true 才写出）；归一化直接消费 UB 里的 x_added。收益：省 2 次 GM 往返（x 写出再读入）。
6. **原子/同步**：无 ✓。
7. **UB 预算**（按 fp32 中间量，R12 精化）：x1 队列 2×2 + x2 队列 2×2 + y 队列 2×2 + x_added fp32 常驻 1 + 归约临时 2 = 按 13 份估（fp32 常驻决定单遍可行性，R11 教训）。

## B. kernel 必检

1. 四件：行大小核 + 行偏移折算、列/行尾块、host 模式A blockDim、workspace=0 ✓
2. DataCopy 对齐：整行/尾片 32B 锚定 + 尾块 DataCopyPad 字节收尾（R12 已落实的模式复用）✓
3. 标量通路禁令：mean/var/rstd 的后续 Muls/Adds/Sqrt 全部**按行批矢量化**（R11 教训，列为本轮重点自查项）✓
4. 2D 块：行装不下时按列片两遍；gamma/beta 行常驻或滚动 ✓
5. 内存工具箱：x_added 用 fp32 TBuf 常驻；ReinterpretCast 覆盖候选 ✓
6. L2 hint：x1/x2 流过 DISABLE；y 流过 DISABLE ✓
7. BUFFER_NUM=2 ✓
8. 精度：加法与归一化全 fp32 中间量；**方差用 E[x²]−mean² 单遍还是两遍**？→ 两遍（减均值后再平方），数值稳定优先（fp16 输入平方大数风险），//?? 对比生产选择
9. 队列外依赖链：x_added 复用链逐对 PipeBarrier ✓

## C. host 必检

1. 三元组 ✓ 2. 32B 锚点 ✓ 3. attr：epsilon（仅 1e-5）+ additionalOut(bool→TilingKey 或 tiling 字段) ✓ 4. 无变体路由（单 kernel 双路径）✓ 5. bias shape 二义性（N 或 M×N）→ host 判定传 tiling 标志 ✓

## D. 流程纪律

//?? 清单：① LayerNorm 高阶 API 是否适配本算子 ② 方差单遍/两遍 ③ bias 广播实现（Adds vs Mul 广播）④ 生产 less_tensor 变体含义——对比销案。
