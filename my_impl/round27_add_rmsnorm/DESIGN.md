# Round 27: AddRmsNorm —— 我的设计（A 节前置）

> 语义：x = x1+x2；yOut = RmsNorm(x)·γ；rstdOut = Rms(x)；**xOut = x（残差和写出）**——三输出融合。
> 已读：aclnn 文档 + add_rms_norm.h（331 行）Process/SubProcess/Compute。

## A. CHECKLIST 设计必答

1. **API 选型**：Rms 倒数 = Duplicate(1)+Div（R22/B8）；归约 ReduceSumCustom（base.h 封装）。
2. **融合等价性（本轮核心）**：Add 的结果**同时**是 RmsNorm 的输入和残差链路的输出（xOut）。拆开实现时 RmsNorm 读到的是 **T 精度**的 x；融合后若一直用 fp32 的 x 算 y，**与两算子串行结果不逐位一致**。生产的解法：归一化后 `Cast(yLocal, CAST_RINT)` 降到 T 再 `Cast` 回 fp32——**先量化到写出精度，再做 γ 乘**。CHECKLIST A5 精化：融合算子必须逐位复现拆开实现的舍入路径（输出精度回环）。
3. **UB 预算**：xFp32 驻留 + sqx（γ 槽复用）+ reduceBuf。
4. **事件对**：V_S 读 rstd → S_V 写 rstdLocal（B18 成对往返实录）；V_MTE2 收尾。
5. **avgFactor** host 算（C5）。

## 1. 我的实现要点

- 三输出：yOut（γ 乘后）、xOut（残差和，T 精度）、rstdOut（fp32 攒批）；
- x1/x2 cast fp32 相加 → **cast 回 T 写 xOut**（残差输出精度=T）→ fp32 域 Σx²·avgFactor+eps → Sqrt → Div(1) 倒数 → V_S/S_V → Muls 归一化 → **量化回环（CAST_RINT 降 T 再升回）** → 乘 γ → cast 写 yOut。
