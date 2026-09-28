# Round 49 复盘：DequantSwigluQuant（数值契约链收官）

> 我的实现：`my_impl/round49_dequant_swiglu/`（DESIGN.md / dequant_swiglu_custom.h）
> 对比对象：`activation/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_static_base.hpp`（.hpp，static 路径）

---

## 一、数值契约链全实例（R27→R40→R49 三轮收官）

dequant_swiglu_quant 是"量化链"的完整形态：**两端 int8 截断 + 中间 fp32 域计算**：

```
int8 [gate|up] → dequant: Cast→×weightScale(通道)→×actScale(token)[+bias] → SiLU(gate)⊙up(fp32) → ×(1/quantScale) → int8 y + scaleOut
```

三轮契约演进：R27（单端回环，y 的 T 精度）→ R40（量化作过程，var+scale 联合）→ R49（两端截断 + 双维 scale）。**融合算子的数值契约 = 枚举所有截断点并逐位复现**——链越长、截断点越多，越必须以拆开实现为金标准。

## 二、本轮工程要点

1. **scale 三来源定式**：常量 1.0 / 输入张量（quant_scale，Init 时 `1/GetValue(0)` **预计算倒数**）/ 动态计算（R28 出口型）。倒数在 Init 算一次，逐块 Muls——标量除法归零（B8 终态）。
2. **双维 scale 的乘法序**：weightScale（列，常驻）× actScale（行，标量）两次乘覆盖二维缩放；**循环序由 scale 生命周期决定**（外层列块换 weightScale，内层行扫 actScale）。
3. **.hpp 惯例**：CANN 库模板类头文件即实现用 .hpp 扩展（R48 库模板类的文件名特征）。
4. **静态/动态量化分派**：static（quant_scale 输入，本类）vs dynamic（scale 输出，R28 类）——同一算子的两个方向各成类。

## 三、我的自查（B14 纪律的正面案例）

本轮动手前先列了 fp32 槽配对表（f0=gate dequant / f1=up dequant / f2=取负+sigmoid+积 / f3=全 1 向量），**首次实现未发生缓冲覆写事故**（对比 R24/R48 的事故）——配对表先行从"事后教训"转为"事前流程"，CHECKLIST 的作用闭环。

## 四、CHECKLIST 增量

- **B44（新）**：量化链两端截断（int8 in/out）+ 中间 fp32 域；scale 三来源（常量/输入张量 Init 预计算倒数/动态输出）；双维 scale（通道×token）的循环序由生命周期决定；库模板类用 .hpp。

## 五、"数值契约链"小节弧线总结（R27/R40/R49）

契约的三个层次：单点回环（R27）→ 联合一致性（R40）→ 两端截断+多维 scale（R49）。判据统一为：**以拆开实现为金标准，枚举截断点，round-trip 复现**。

## 下一轮候选

50 轮整元轮（CHECKLIST 去重 + DECISION_FLOW 契约链分支入树 + 50 轮阶段总结）。
