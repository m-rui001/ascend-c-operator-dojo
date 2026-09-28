# Round 48 复盘：SwiGLU（库模板类层级 + 分派型 kernel）

> 我的实现：`my_impl/round48_swiglu/`（DESIGN.md / swiglu_custom.h，SiLU 基础 API 组合 mini）
> 对比对象：`activation/swi_glu/op_kernel/swi_glu.cpp`（64 行纯分派）+ `gelu_quant/op_kernel/gelu_quant_base.h`（GELU 多项式生产形态）

---

## 一、本轮核心发现：API 层级还有 2.5 层——库模板类

swi_glu.cpp 只有 64 行**纯分派**：`isDoubleBuffer（tiling 字段）× TilingKey(dtype) × 220 守卫` = 6 个实例化，计算体 `SwigluVector<inT, computeT, outT, bufferNum>` 来自**随 CANN 发布的 lib 头**（`lib/activation/`，不在开源仓库）。

API 层级修正为四级半：

1. 高阶 API（完整算子语义 + 配套 tiling）
2. **库模板类（本轮新知）**：参数化模板类随安装目录发布，算子只做实例化分派
3. 基础指令 API（含组合）
4. 手拼多项式

**判断"计算体在不在开源库"的方法**：kernel 文件 include 的头不在仓库内 → 即库模板类；此时算子开发=分派+tiling。

## 二、分派结构：运行时/编译期配置分离

- `isDoubleBuffer` 是 **tiling 字段**（运行时 if）——因为它是性能配置非语义分支；
- dtype 是 **TilingKey**（编译期）——语义分支；
- 220 守卫编译期。
**运行时性能配置用 tiling 字段 if，语义/精度分支用 TilingKey**——两种 if 的职责分界（此前各轮未显式区分）。

## 三、GELU 多项式生产确认（R12 闭环）

`gelu_quant_base.h`：`x²→x³→Mul(0.044715)→Muls(√(8/π))→Exp→...` 的 tanh-approximate 展开逐条与 R12 手拼一致——**R12 的手拼形态即生产形态**，当时缺的只是"先查有无现成 API"的前置检查。

## 四、我的自查 bug（第 N 次：B14 配对表）

SiLU 组合里我先 `Muls(g, −1)` 就地取负，后面 `Mul(g, sig)` 时 g 已是负值——**就地修改破坏原始输入**，需要先备份原 g 或用 Muls 到临时槽。多步组合里"哪一步破坏哪个缓冲"必须在动手前列表（B14 第 N 次现场验证）。

## 五、CHECKLIST 增量

- **A1 终版**：API 层级四级半——高阶 API / 库模板类（lib 头，算子只分派）/ 基础指令 API（含组合）/ 手拼多项式。
- **B43（新）**：分派型 kernel 的两种 if：运行时性能配置（buffer 数）→ tiling 字段 if；语义/精度分支 → TilingKey 编译期。
- **B14 持续验证**：多步组合的缓冲破坏清单先行（本轮 SiLU 的 g 负值事故）。

## 下一轮候选

mse_loss_grad（loss 反向对偶）或 dequant_swiglu_quant（量化+门控的四级组合——数值契约链验证）。
