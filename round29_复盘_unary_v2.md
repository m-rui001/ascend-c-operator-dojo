# Round 29 复盘：foreach_unary_v2（工厂代际对比）

> 我的实现：`my_impl/round29_unary_v2/`（DESIGN.md / foreach_add_scalar_v2_custom.h，v2 形态迁移版）
> 对比对象：`op_kernel_v2/`（base_v2 129 行 / elewise 238 行 / unary_v2 155 行）vs v1（`op_kernel/kernel_foreach_unary.h` + base）
> 本轮性质：代际对比轮——把 R16/R17 的工厂理解升级到 v2。

---

## 一、v1 → v2 的四项代际差异

| 维度 | v1 | v2 |
| --- | --- | --- |
| Tiling | 硬编码 `ForeachCommonTilingData` | **模板参数 `<T, Tiling>`**——elewise 与 reduce 共用基类（R18 的 reduce 工厂属 v2 系） |
| TPipe | 外部传 `TPipe*`（BareMix/Matmul 风格） | **基类自持 `TPipe pipe`** |
| 运算符注入 | 函数指针非类型模板参数（`op`） | **Predicate 对象引用**（构造注入 `pred(p)`，成员函数可 `static_assert` 校验签名——R18 已见 `is_member_function_pointer_v` 断言） |
| 模板旗标 | needCopyOut | needCopyOut + **needTempBuf** |

继承层次升级：v1 = Base → Unary；v2 = BaseV2 → **Elewise（新增中间层，持有队列与 CopyIn/CopyOut 主循环）** → UnaryV2 / Reduce / ImplictOutput。中间量 workspace 字段（coreMiddleOffset/tensorMiddleList）上提到 BaseV2——reduce 与 elewise 共享寻址。

## 二、钩子架构延续性确认

v2 完整保留 v1 的钩子集（Compute/CopyInPlus/CopyOut/BeforeProcess/AfterProcess/ProcessPlusInLoop），Compute 的 `float32Tensor & isRemainder` 签名一致，InnerComputer 的 bf16 cast 特化一致——**代际升级没有破坏子类契约，迁移成本 = 换基类 + 函数指针改 Predicate 对象**。

## 三、我的迁移实现与差距

1. 我的 mini 版用"模拟基类"占位（`MyElewiseSim`），未逐行复刻 Elewise 的队列管理——迁移可行性的判断依据已足够（钩子签名一致）；
2. scalar 传递：v2 下我按 tiling 下发（`tilingData.scalarPad` 占位）——生产 v2 系标量仍是 Device tensor 读入（R16 机制不变）；
3. needTempBuf 语义未深挖（预测：为 cast 中间量提供基类管理的 TBuf 槽，替代各子类自建 float32Queue）——开放项，跨轮滚动。

## 四、CHECKLIST 增量

- **E6（新）**：工厂选代际——新写 foreach 算子优先 v2（Tiling 泛型、pipe 内置、Predicate 对象）；v1 系只读维护。迁移成本 = 换基类 + 函数指针改 Predicate 对象，钩子契约不变。
- **E7（新）**：运算符注入形态演进：v1 函数指针（自由函数 Adapter）→ v2 Predicate 对象（成员函数 + static_assert 签名校验）。

## 下一轮候选

foreach_copy（纯搬运 v2）或 add_layer_norm_quant（双 smooth 量化）；或插入方法轮：把 29 轮 CHECKLIST 重排为"决策树"文档（元轮）。
