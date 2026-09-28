# Round 16 复盘：foreach 系列算子（多张量批量处理）

> 我的实现：`my_impl/round16_foreach/`（DESIGN.md / foreach_add_scalar_custom.cpp / op_host/）
> 对比对象：`cann-ops/src/foreach/foreach_add_scalar/` + 公共层 `src/common/inc/foreach/`（kernel_foreach_base/unary、foreach_one_scalar_binary、foreach_tiling_def、foreach_proto_utils）

---

## 一、结构发现：foreach 是"算子工厂"，不是"一个算子"

生产 foreach 系列的形态出乎意料地彻底：

- **host = 一行宏**：`FOREACH_OPDEF(HOST_CONFIG, BINARY_SCALAR_TENSOR, AddScalar, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16)`——OpDef、TilingFunc、TilingKey 全由宏按"模式"生成；
- **kernel = 21 行**：按 TilingKey 实例化 `ForeachOneScalarBinary<half, half, Adds, 1>`——**运算符本身是模板参数（函数对象 `Adds`）**；
- **三层模板架构**：`KernelForeachBase`（张量列表寻址/遍历）→ `KernelForeachUnary`（流水骨架 CopyIn/Compute/CopyOut）→ 具体算子层（只提供 Compute 语义 + bf16 特化的 InnerComputer：Cast 到 fp32 → op → Cast 回）。

约 30 个 foreach 算子共享这套骨架。**新增一个 foreach 算子的成本 = 一个函数对象 + 两行实例化。** 这把前 15 轮学到的所有"单算子工程"结构（OpDef 注册/TilingFunc/kernel 类）重构成"按模式分类的模板族"——算子开发的终极 DRY。

## 二、//?? 清单销案：张量列表的机制（本轮最大悬念）

1. **描述表怎么进 kernel？我的 workspace 方案错得彻底**——根本不需要 host 写任何表。launch 传入的张量列表指针本身就指向**框架在 GM 上建好的地址表**：`GetTensorAddr(index, tensorPtr)` 用指针算术走表——`*dataAddr` 是地址数组偏移，`(retPtr + index)` 取第 index 个张量地址。框架契约替我解决了最大设计难题。
2. **多核-张量映射**：社区在 **host tiling 里预计算每个核的 [tensorStart, tensorEnd) 与首尾张量内偏移**（tiling 数组 `tensorStartList/tensorEndList/tensorStartOffsetList/tensorEndOffsetList`，定长 `MAX_CORE_CONT`）。我的"kernel 运行时贪心扫块"多做了运行时工作且逐块 locate 浪费标量循环——host 算一次、kernel 纯执行才是对的（CHECKLIST C5"能在 host 算的都在 host 算"的直接延伸到列表结构）。
3. **变长列表 vs 定长数组**：tiling 用定长数组（`MAX_TENSOR_CONT`/`MAX_CORE_CONT`）截断即可，我的"放 workspace 才能变长"顾虑不成立。
4. **Device 标量读取**：`inScalarGM.SetGlobalBuffer(...,1)` 后读入——我写的 MTE2_S 事件对方向正确但过于繁琐，社区封装在基类。
5. **dtype 分发**：TilingKey(1/2/3/4) + `#if __CCE_AICORE__ == 220`——**产品支持矩阵在编译期落地**（int32/bf16 只在 220 架构编译），Round 4 教训的机制化形态。
6. **非连续**：文档声明支持但本层未处理——框架/上层已保证连续性（我的怀疑对象是 aclnn 前置处理）。

## 三、我的实现缺陷清单

1. **P0（设计级）**：我发明的"host 把描述表写进 workspace"在 tiling 阶段**根本不可能**（tiling 时设备内存内容不可写），整个 Init 的读表逻辑建立在不存在的机制上——这正是"猜 API"风险的极致案例，对比环节暴露得非常及时。
2. kernel 运行时扫块 locate（应 host 预分配核区间）。
3. out 地址我偷懒复用 x 的（foreach_add_scalar 是 out-of-place，out 是独立列表）。
4. 描述表读入后的逐元素访问 `dLocal(i)` 语法是猜的，未验证。

## 四、经验教训（进 CHECKLIST）

1. **识别"算子族"信号**：当文档/接口名带 foreach/list 前缀、host 参数是 aclTensorList 时，先找**公共模板层**（common/inc/foreach），不要按单算子工程从头写——对题量大的算子族，华为的答案是宏生成 host + 运算符模板参数化 kernel。
2. **框架契约优先**：张量列表的地址表由 launch 机制自带（GM 指针走表），任何"host 手工搬运元数据到设备"的设计都要先质疑——launch 契约可能已经解决。
3. **多核分配的归属地**：列表类结构的核-张量映射归 host tiling（数组下发区间），kernel 保持"纯执行"。
4. **架构支持矩阵编译期化**：`#if __CCE_AICORE__ == 220` 把 Round 4 的"查支持矩阵"变成编译期守卫——新架构相关分支的标准写法。
5. 复盘方法增量：本轮 //?? 全部围绕"机制归属"（谁负责、在哪一层），销案结果全数指向"框架已有契约"——**设计新结构类时第一问应是"launch/框架契约里有没有现成的表"**。

## 下一轮候选

- `10_communicate_compute_fused`（通信计算融合，全新领域，samples 有代码）；
- cann-ops `foreach_add_list`（列表+列表，验证三层架构在另一模式下的复用）——低成本验证"算子工厂"理解是否完整。
