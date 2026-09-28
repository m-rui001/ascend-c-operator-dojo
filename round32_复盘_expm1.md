# Round 32 复盘：foreach_expm1（基础 API 组合层实证）

> 我的实现：`my_impl/round32_expm1/`（DESIGN.md / foreach_expm1_custom.cpp）
> 对比对象：`foreach/foreach_expm1/op_kernel/foreach_expm1.cpp`（与我的实现逐字级同构）

---

## 一、销案：R31 三级递进的"组合层"实证

expm1 无专用指令 → 生产用 **`Exp + Adds(-1)` 两指令组合**（隐式输出同缓冲原地减）。R31 的三级递进补全中间层：

| 层级 | 例 |
| --- | --- |
| 高阶 API | SoftMax |
| 基础指令直接映射 | Log（R31） |
| **基础指令组合（本轮）** | expm1 = Exp + Adds |
| 手拼多项式 | Gelu |

**选型顺序不变：能组合绝不多项式。**

## 二、精度取舍的个案性（与 R31 对比）

| 函数 | fp16 实例化 | bf16 实例化 |
| --- | --- | --- |
| log（R31） | half 域直算 | **升 fp32 域**（`<bfloat16_t, float, LogAdapter<float>>`） |
| expm1（本轮） | half 域直算 | 升 fp32 域 |

fp16 的 log/expm1 都在 half 域直算（无 cast 升级）——**是否升 fp32 是按"指令可用性 + 函数误差敏感度"的个案决策，不存在统一规则**；bf16 因硬件 bf16 指令少，普遍升 fp32。CHECKLIST B8 不再泛化"低精度一律升 fp32"，改为"**默认升 fp32；有对应 half 指令且函数不敏感时可留在 half 域（以生产为准）**"。

## 三、隐式输出原地性确认

`Adds(dstLocal, srcLocal, -1, u)` 在 ImplictOutput 流中 dst/src 同物理缓冲——减 1 原地落，零额外搬运（B5/R20 的第 N 次复现，已成为肌肉记忆级 idiom）。

## 四、我的实现与生产逐字级一致

本轮首次出现实现与生产几乎同构（仅函数名差异）——工厂中间带的算子已进入"复制粘贴成本 < 设计成本"区间。**循环练习的边际收益信号**：同族算子（foreach 一元）的轮次价值下降，下一轮起转向未覆盖的结构类。

## CHECKLIST 增量

- **B8 再精化（R32）**：低精度→fp32 非绝对规则；有对应 half 指令且函数不敏感可留 half 域（expm1/log 的 fp16 实例化实证），bf16 普遍升 fp32（指令少）。以生产为准。
- **A1 补充**：三级递进的"组合层"实证：Exp+Adds 组合 expm1，无需手拼多项式。

## 下一轮候选

add_layer_norm_quant（双 smooth 量化收尾）或结构性新题：cann-ops `index` 目录（index_put/gather 类，索引访存——全新访存模式）。
