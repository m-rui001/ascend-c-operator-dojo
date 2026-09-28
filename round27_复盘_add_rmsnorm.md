# Round 27 复盘：AddRmsNorm（三输出融合 + 数值等价回环）

> 我的实现：`my_impl/round27_add_rmsnorm/`（DESIGN.md / add_rms_norm_custom.h，half 输入演示版）
> 对比对象：`add_rms_norm/op_kernel/add_rms_norm.h`（331 行）+ base 头

---

## 一、本轮核心发现：融合算子的数值等价性约束（CHECKLIST A5 重大精化）

AddRmsNorm 把 Add 融进 RmsNorm，但 Add 的结果**有第二个消费者**（残差链路，xOut 输出）。生产在归一化后做了一段"看似多余"的操作：

```cpp
Muls(x_fp32, x_fp32, rstdValue, numCol);   // 归一化（fp32）
Cast(yLocal, x_fp32, CAST_RINT, numCol);   // fp32 → T（降精度！）
Cast(x_fp32, yLocal, CAST_NONE, numCol);   // T → fp32（读回！）
Mul(x_fp32, x_fp32, gamma_fp32, numCol);   // 再乘 γ
```

原因：拆开实现时，Add 算子把 x 写成 **T 精度**，RmsNorm 读回的是 T 精度的 xNorm。融合版若全程 fp32 乘 γ，结果与两算子串行**不逐位一致**，残差链路的下游（下一层 Add）会发散。所以融合必须**先把中间量量化到"写出精度"，再做后续计算**——舍入路径逐位复现。

**CHECKLIST A5 精化（融合/物化决策的第三维）**：融合不只是搬运优化，还承担**数值契约**——凡是拆开实现会经过 GM 精度截断的中间量，融合版必须显式重建同样的截断点（round-trip cast）。这解释了为什么 add_rms_norm/add_layer_norm/add_rms_norm_quant 是一整族（quant 变体还多一层量化截断）。

## 二、其余确认与 idiom 复现

1. **三输出**：yOut / rstdOut / xOut——xOut 是残差写出（T 精度），等于把 Add 算子的输出义务接过来；
2. rstd 倒数：`Duplicate(1)+Div` 张量通路（R22 B8 三现）+ **V_S→S_V 成对往返**（B18 实录：GetValue 读出、SetValue 写回 rstdLocal 攒批）；
3. `Cast(sq, gammaLocal)` —— γ 的 fp32 槽**复用 sqx 缓冲**（B5 工具箱的复用粒度），配合 buffer 计划；
4. `avgFactor` host 下发（C5）；γ 一次载入常驻（B25 小系数规则）；
5. base 头封装 `ReduceSumCustom`（整块收口+掩码恢复），把 R26 的折半求和做成公共件。

## 三、我的差距

1. **自抓 bug**：演示代码里 `Duplicate(rb, 1.0f, 1)` 会覆写 sqrt 结果——单槽做 Div(1/sqrt) 不够，需要 sqrt 结果与全 1 向量两个槽（生产 `reduce_buf_local` 独立开槽）。B14 配对表纪律第三次被现场验证；
2. 我用标量 `1.0f/rb.GetValue(0)` 演示倒数，生产走 Div 张量通路（B8/B28 已记）；
3. 生产 CopyIn 用 outQueueY 的槽位借放 x2（队列槽复用），我的演示同型但事件对配对未逐对展开——多队列复用时事件号管理（FetchEventID 每依赖一号）需更严。

## 四、CHECKLIST 增量

- **A5 精化（本轮最重要）**：融合算子承担数值契约——拆开实现经过 GM 精度截断的中间量，融合版必须用 round-trip cast 重建同样截断点，保证逐位一致。
- **B30（新）**：三输出融合（主输出+残差写出+统计量攒批）的缓冲计划：残差写出用主输入槽位、γ 的 fp32 槽复用计算缓冲、rstd 独立小队列。

## 下一轮候选

add_rms_norm_quant（+量化截断的第四输出，验证"数值契约"链）或 foreach_unary_v2。
