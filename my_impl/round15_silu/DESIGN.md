# Round 15: Silu 独立算子 —— DESIGN（CHECKLIST 精简过单）

- **A1 API 选型**：独立算子 → **高阶 API `Silu(dst, src, count)`**（激活函数类，api_0003.md 424 行确认存在）。R12 精化条目适用：若生产对比发现融合场景手拼，则验证"上下文依赖"结论。
- **A4 瓶颈**：exp+除法 计算量中等，V 瓶颈倾向，BUFFER_NUM=2。
- **A7 UB 预算**：4 份队列（x/y 各 2×2）；API 内部临时量未知（//?? 是否有配套 TmpSize）。
- **B1 四件**：沿用 R12 模板（大小核/32B 锚定/host blockDim/无 workspace）✓
- **B2 对齐**：尾块 DataCopyPad 字节收尾 ✓
- **B6 L2**：x/y 流过 DISABLE ✓
- **B8 精度**：API 内部处理（//??）；生产 swi_glu 手拼时用 fp32 中间量+beta 缩放防溢出（对比销案）
- **C1-C5**：dtype 长度表/32B 锚点/无 attr/TilingKey 不需要/host 无标量 ✓

//?? 清单：① Silu 签名与内部临时量要求 ② 生产 swi_glu 是否用 Silu API（预测：手拼，R12 结论）③ 1/x 用 Reciprocal 还是 Div 全1向量
