# Round 20 复盘：foreach_addcdiv_scalar（三列表 + 隐式输出）

> 我的实现：`my_impl/round20_foreach_addcdiv/`（DESIGN 见 R18/R17 模板复用，A 节结论内嵌）/ foreach_addcdiv_scalar_custom.cpp
> 对比对象：`cann-ops/src/common/inc/foreach/op_kernel/foreach_one_scalar_quaternary_implict_output.h` + `foreach/foreach_addcdiv_scalar/op_kernel/foreach_addcdiv_scalar.cpp`
> 本轮定位：工厂复用第三弹——验证"新算子 = Adapter + 两行实例化"的理解，并解码 **ImplictOutput** 机制。

---

## 一、算子与工厂匹配

- 语义：`y[i] = x1[i] + scalar·(x2[i]/x3[i])`，三个张量列表 + 一个 Device 标量。
- 工厂类：`ForeachOneScalarQuaternaryImplictOutput<T, Adapter, bufferNum=2, paramsCount=3>`——Quaternary=四操作数（3 列表+标量）；实例化时 paramsCount 显式传 3（fp32 中间量槽数：x2/x3/结果）。命名解码（B17）再次生效。

## 二、ImplictOutput 机制解码（本轮核心新知）

**"隐式输出"≠ out 与 x1 同址**（Init 仍收独立 y 指针）。真实语义：

1. **结果累积在第一个输入的本地缓冲里**：Adapter 直接改写 `tensor1Local`（fp32 路径 `Axpy(tensor1, tensor2, scalar)` 原地累加）；
2. 基类模板参数 `needCopyOut = false`——基类的 CopyOut 通路关闭，子类 CopyOut 从 dataQueue 取回 t1（EnQue 回队再 DeQue）写 y；
3. 收益：**省一条"结果拷到独立输出缓冲"的 UB 内搬运**，且 y 若与 x1 同址（优化器原地更新场景）零拷贝自然成立。

这与 R17 的 `GetPhyAddr()` 别名跳拷贝、R13 的隐式输出一脉相承：**foreach 优化器类算子的输出策略 = 原地累积**。

## 三、Adapter 数学的两个确认/精化

1. **矢量÷矢量：`Div` 直除**。我 CHECKLIST B8 的"除法→乘倒数"被精化：倒数 idiom 只适用于**标量除数**（`Muls(1/s)`，R11）或**重复除数**；x2/x3 逐元素相除时 Div 一条指令胜过 Reciprocal+Mul 两条。**B8 分裂为标量除数/矢量除数两条规则**。
2. fp16 路径：x2/x3 分段 cast 进 float32Tensor 的不同槽（`float32Tensor[maxCastDataCount]` 偏移切片，R3 idiom），fp32 域 Div+Muls，cast 回原精度后 **Axpy 系数传原精度标量**。我的"Axpy 系数取 1"方案（把 scalar 并进 Muls）与生产"Div 结果先乘 scalar 或 Axpy 带 scalar"等价——生产 fp16 路径把 scalar 放 Axpy 的 P 系数（`Axpy<T,float>` 支持跨精度系数），少一次 Muls。

## 四、我的实现差距

1. 生产 Adapter 不自己写 `PipeBarrier` 于 Axpy 后（Axpy 内部依赖由硬件排序/编译器插入？），我在 Div 与 Axpy 之间挂了 barrier——**多余 barrier 是性能损耗**，但漏挂是正确性风险；生产只在 cast 链上挂（R12 B9 的"逐对"边界：同管顺序指令间不挂，跨缓冲重用才挂——需要更精确的判据，列入待深入项）。
2. 我的 Compute 里 `f32Queue.Get<float>()` 常驻取槽与生产"Alloc→EnQue→DeQue 队列化"不一致——生产把 fp32 工作区也队列化以进入 MTE2/V 事件管理。
3. 函数指针模板参数（生产的 `op` 是函数指针非类型参数）——我沿用了自由函数形态，一致。

## 五、CHECKLIST 增量

- **B8 精化**：标量除数→Muls(1/s)；矢量÷矢量→Div 直除；矢量÷标量→Reciprocal+Mul 或 Muls(1/s)（按是否复用）。
- **B20（新）**：foreach 优化器类算子优先"隐式输出"形态（结果原地累积进第一输入缓冲，needCopyOut=false，省一次 UB 内搬运）；y 独立时 CopyOut 从 dataQueue 回队取结果。
- **B21（新）**：PipeBarrier 判据精化——同管顺序指令不挂，跨缓冲重用/跨管依赖才挂；多余 barrier 也是损耗。

## 下一轮候选

foreach_lerp_scalar（权重插值，验证 Axpy 泛化）或返回单算子维度：cann-ops `norm/rms_norm`（R11 做过 RmsNorm，对比 LayerNormV4 的 variance 路径）。
