# Round 57: Conv Backprop 的 Fixpipe 出口 —— 我的设计（A 节前置）

> 对比源：`conv/conv2d_backprop_filter_v2/op_kernel/convolution_backprop/impl/dav_v220/conv_bp_sub_func.h` 的 `LoadL0c2Gm`。
> Conv 族都是 2000+ 行的 Intf 引擎架构——本轮聚焦其**出口模式**（L0C→GM 的 fixpipe/原子/顺序写）。

## A. CHECKLIST 设计必答

1. **Fixpipe 直写（Cube 出口第三种）**：L0C(CO1 分形) → **`Fixpipe`** 直达 GM，`FixpipeParams<L0cT>`（nDim/mDim/nSize/mSize/srcNStride/dstNStride，C0 单位）——绕过 UB，Cube 结果直出。之前学的出口：V 管写出（R1）、UB攒批（R11）、现在 Fixpipe（R6 的 IterateAll 内部机制的显式版）。
2. **出口的确定性开关**：`enAtomic` → `SetAtomicAdd<DstT>()` 原子累加（filter 梯度跨 batch 累加）vs `enSequentialWrite` 顺序写（确定性）——R25 三态在 conv 出口的变体（原子 fixpipe vs 顺序布局）。
3. **格式即模板配置**：`Intf::Config::dType::format == CubeFormat::NC1HWC0 || FRACTALZ_C04` if-constexpr 分派——**布局是 Intf Config 的编译期成员**。
4. **Intf/ctx 引擎架构**：conv kernel 用 `Intf` 类持有 `ctx`（curNL0Idx_/mIter_/baseN...）+ 静态循环函数（LoadL0c2Gm(self, ...)）——**上下文结构体+自由函数引擎**，与 foreach 类层次、Matmul 对象并列第三种 kernel 组织。
5. **分形偏移数学**：`dstOffset = (nL0Idx%stepN)*baseN*Cout + (mL0Idx%mIter)*baseM*16`——NC1HWC0 的 N、M 两级 base 块偏移。

## 1. 我的实现（出口模式 mini：分形偏移 + 原子/顺序双路分派）
