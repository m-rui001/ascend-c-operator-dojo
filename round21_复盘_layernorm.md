# Round 21 复盘：LayerNorm single-read（variance 路径）

> 我的实现：`my_impl/round21_layernorm/`（DESIGN.md / layer_norm_custom.h，fp32 single-read 单遍方差）
> 对比对象：`cann-ops/src/norm/layer_norm_v4/op_kernel/layer_norm_v4_single_read.h`（373 行）
> 本轮主题：variance 的数学路径选择 + 三个新 idiom。

---

## 一、variance 路径的决策（本轮主题）

LayerNorm 的 var 有两条数学路：

| 路径 | 公式 | 代价 | 精度 |
| --- | --- | --- | --- |
| 两遍 two-pass | 先 mean，再 E[(x−mean)²] | 重读一遍 x（GM 带宽 ×2）或缓存原始 x 两份 UB | 稳定 |
| **单遍 single-read**（生产选择） | E[x²]−mean²，从**同一份缓存数据**上 Mul 得平方项 | 读一遍 x | 有相消误差，**靠 fp32 中间量兜住**（B8） |

生产文件名单刀直入叫 `single_read`——**带宽换精度风险，再用 fp32 中间量把风险买回来**，这是 B8"低精度→fp32 中间量"在方差场景的具体化。R11 的 RmsNorm 天然只要 E[x²]，LayerNorm 多出的 mean 项正是两种路径的分叉点。

执行序（生产）：载入整块 nRow×rowSize（2D pad load）→ cast fp32 → `Muls(y, x, coef)` 逐元素缩放 → **行循环 ReduceSum 求 mean 攒批** → 全块 `Mul(y,y)+Muls(coef)` → 行循环 ReduceSum 得方差项 → rstd=1/√(var+eps)（标量）→ 逐行 `Adds(x,−mean)·rstd` → γ/β 乘加 → 攒批写 y/mean/rstd。

## 二、三个新 idiom（本轮最大收获）

1. **掩码计数模式读累加器（CHECKLIST B10 重大精化）**：
   ```cpp
   AscendCUtils::SetMaskCount<float>();  SetVectorMask<float>(0, rowSize);
   ReduceSum(y, y, y, 1);                 // repeatTimes=1
   uint64_t acc = GetAccVal();            // 从累加器寄存器直读
   float v = *reinterpret_cast<float*>(&acc);
   SetMaskNorm();                          // 恢复普通掩码模式
   ```
   归约→标量不再走"UB 结果 tensor + V_S 事件对 + GetValue"四步，**累加器寄存器直达**。逐行小归约（行内标量）场景全面替代旧写法；V_S/GetValue 仍适用大块归约结果 tensor 化场景。
2. **γ/β 隐藏加载**：γ 在 rowIdx==0 的计算后载入、β 在 rowIdx==1 后——用 `FetchEventID(HardEvent::V_MTE2/MTE2_V)` 事件对把**权重搬运藏进行归约循环**（B14 配对表的实战形态）。我的"循环前一次载入"串行了 2 次 MTE2。要点：**归一化系数只在 normalize 阶段才需要，前段算 mean/var 的窗口正好预取**。
3. **ReinterpretCast 双半区布局**：`xLocal.ReinterpretCast<Tfm>()[(sizeof(Tfm)==2)*tileLength]`——同一 buffer 低半区放 fp32、高半区放原精度原始数据，DataCopyPad 落高半区、Cast 原地转低半区。**cast 场景"一份 buffer 两种 dtype"的标准布局**。

## 三、我的差距

1. **掩码计数 idiom 未掌握**（B10 精化）：我逐行 WholeReduceSum+V_S+GetValue，生产 SetMaskCount+GetAccVal；
2. **批量化粒度**：生产的 Mul(y,y) 在**行循环外整块一次**，我的平方 Mul 在行循环内逐行——指令发射数差 nRow 倍（B4 批行意识的回归，重犯记录+1：逐行处理 R2/R11→R21）；
3. γ/β 加载时机串行（见上）；
4. eps 硬编码（C3 违反：应 attr→tiling 下发）——自查发现，重犯记录+1；
5. `REGISTER_TILING_DEFAULT` 放错位置（应在 kernel 入口）——初稿毛刺。

## 四、CHECKLIST 增量

- **B10 精化**：逐行/小归约标量读用 SetMaskCount+GetAccVal 直读累加器（免 UB 往返与 V_S）；大块归约结果 tensor 化仍走 B3。
- **B22（新）**：normalize 系数（γ/β/缩放）的 GM 加载藏在 mean/var 归约循环内，FetchEventID+V_MTE2/MTE2_V 事件对预取。
- **B23（新）**：cast 布局——ReinterpretCast 双半区，一份 buffer 容两种 dtype，DataCopyPad 落原精度区、Cast 原地转 fp32 区。

## 重犯记录

| 教训 | 首犯 | 重犯 |
| --- | --- | --- |
| 逐行处理 vs 批行 | R2 | R11, **R21**（平方 Mul 应提到行循环外整块做） |
| 标量/系数进常量 | R4 | **R21**（eps 硬编码，自查发现） |

## 下一轮候选

layer_norm_v4_transpose（914 行，转置布局策略——"行太长装不下时换数据布局"的对偶问题）或 foreach_lerp_scalar。
