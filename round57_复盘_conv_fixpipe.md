# Round 57 复盘：Conv Backprop 的 Fixpipe 出口（Intf 引擎架构观察）

> 我的实现：`my_impl/round57_conv_fixpipe/`（DESIGN.md / conv_store_custom.h，出口模式 mini）
> 对比对象：`conv/conv2d_backprop_filter_v2/op_kernel/convolution_backprop/impl/dav_v220/conv_bp_sub_func.h` 的 `LoadL0c2Gm`

---

## 一、Cube 出口的第三种路径：Fixpipe 直写

| 出口 | 路径 | 场景 |
| --- | --- | --- |
| V 管写出（R1） | Cube→GM→Vec | 一般融合 |
| UB 攒批（R11） | Cube→UB→处理→GM | 后处理 |
| **Fixpipe（本轮）** | **L0C→GM 直写** | 无 UB 介入的 Cube 出口（conv/反向 filter） |

`FixpipeParams<L0cT>`（nDim/mDim/nSize/mSize/双 stride，C0 单位）描述分形搬运；`enAtomic` 旗标在 Fixpipe 前挂 `SetAtomicAdd` —— **反向 filter 的跨 batch 累加在出口原子化**；`enSequentialWrite` 顺序写为确定性替代（各 batch 写独立 workspace 段，post 聚合）。

## 二、Intf/ctx 引擎架构（kernel 组织第三种）

conv 族用 `Intf` 类持有 `ctx`（所有循环游标/tiling 派生量）+ **静态自由函数引擎**（`LoadL0c2Gm(self, ...)` 传 self）——与 foreach 类层次（R16）、Matmul 对象（R6）并列第三种 kernel 组织。**ctx 结构体让引擎函数无状态化**，配置（format/dtype）在 `Intf::Config` 编译期成员（if-constexpr 分派）。

## 三、格式即编译期配置

`Intf::Config::dType::format == CubeFormat::NC1HWC0 || FRACTALZ_C04`——布局是 Config 成员而非运行时参数，**布局分派零运行时开销**（R16"架构守卫编译期化"同型的格式版）。

## 四、CHECKLIST 增量

- **B49（新）**：Cube 出口三路径（V 管 / UB 攒批 / **Fixpipe 直写**，FixpipeParams 分形描述）；反 向 filter 类用出口原子（SetAtomicAdd+Fixpipe）或 enSequentialWrite 顺序写做确定性；NC1HWC0 分形偏移 = (nL0%stepN)·baseN·Cout + (mL0%mIter)·baseM·16。
- **E12（新）**：kernel 组织第三种——Intf/ctx 引擎（上下文结构体+静态自由函数，Config 编译期配置），conv 族专属；三种组织按"循环引擎复杂度"选择。

## 下一轮候选

conv2d_transpose_v2（转置卷积的 dim 推导）或 pooling 目录（池化的窗口滑动访存）。
