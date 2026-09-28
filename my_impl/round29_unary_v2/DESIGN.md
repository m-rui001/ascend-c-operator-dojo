# Round 29: foreach_unary_v2 —— 我的设计（A 节前置）

> 目标：对比 v1/v2 工厂代际，迁移 R16 的 foreach_add_scalar 到 v2 形态。
> 已读：`op_kernel_v2/kernel_foreach_base_v2.h`（129 行）+ `foreach_unary_v2.h`（155 行）结构。

## A. CHECKLIST 设计必答

1. **API 选型**：v2 工厂（`ForeachUnaryV2` + `KernelForeachElewise` 中间层）；运算符仍为函数指针模板参数。
2. **v1→v2 代际差异（侦察结论，实现验证）**：
   - **Tiling 泛型化**：`KernelForeachBaseV2<T, Tiling>` 模板参数化 tiling 结构（v1 硬编码 ForeachCommonTilingData）——elewise 与 **reduce 共用同一基类**（R18 的 reduce 工厂就是 v2 系）；
   - **TPipe 内置**：v1 需外部传 `TPipe*`（BareMix/Matmul 风格），v2 基类自持 `TPipe pipe`；
   - **模板旗标扩展**：新增 `needTempBuf`（v1 只有 needCopyOut）；
   - 中间量 workspace 字段（coreMiddleOffset/tensorMiddle*List）**上提到基类**——reduce 与 elewise 共享寻址。
3. **迁移设计**：R16 的 `KernelForeachAddScalar` 改写为 `MyForeachAddScalarV2 : KernelForeachElewise<T, ...>`：Init 只做标量读取，Compute 挂 Adds，钩子最小化。

## 1. //?? 清单

1. `KernelForeachElewise` 相比 BaseV2 多承担什么（预测：队列对与 CopyIn/CopyOut 主循环）。
2. needTempBuf=true 时缓冲从哪来（基类 TBuf？）。
3. v2 是否还有 ProcessPlusInLoop 钩子（标量列表场景）。
