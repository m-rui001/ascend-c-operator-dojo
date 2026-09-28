# Round 15 复盘：Silu 独立算子（高阶 API 路线）

> 我的实现：`my_impl/round15_silu/`（DESIGN.md / silu_custom.cpp / op_host/）——R12 模板复用轮
> 对比对象：`cann-ops/src/activation/swi_glu/`（Swish 门控融合算子，含 silu 计算）

---

## 一、//?? 销案结果

| //?? | 结果 |
| --- | --- |
| 1. Silu 高阶 API 签名 | API 列表确认存在（激活函数类）；精确签名未取到（同 R12），但独立算子选高阶 API 的路线成立 |
| 2. 生产是否用 Silu API | **手拼**。swi_glu 的 silu 部分：`Muls(βx) → Exp → Adds(1) → Div → Mul(x)`——与 R12 gelu_quant 的结论一致：**融合算子手拼**，A1 精化条目第二次实证 |
| 3. 1/x 的实现 | **Div 配全 1 向量**：Init 时 `Duplicate(tempLocal, 1.0, tileLength)` 一次性预置，逐 tile `Div(dst, ones, src)`——没用 Reciprocal API（推测：Div 走除法部件、Reciprocal 是近似倒数，精度要求高时选前者）|

## 二、生产 swi_glu 的三个可学细节

1. **全 1 向量预置**：Init 里 Duplicate 一次，全算子生命周期复用——"常量张量预置在 Init、循环外零成本"的模式（对照 R11 gamma fp32 化复用，同族）。
2. **beta 缩放内置**：silu 先 `Muls(x, beta)`（swish 公式的 β 参数），融合算子把数学变体的参数吸收进计算链，而不是靠调用方预处理。
3. **PipeBarrier 全程覆盖**：与 R12 一致，队列外的临时缓冲依赖链逐对屏障——swi_glu 的 Compute 里**每一步**都有 PipeBarrier，确认 R12 的 B9 条目是普适纪律而非 gelu 特例。

## 三、我的实现自评

1. **流程**：本轮是 R12 模板的复用轮——DESIGN 精简过单、kernel 结构/尾块/L2/四件全部照单落实，**复用成本低到可以直接拷改**，这正是 CHECKLIST 固化的收益：第 12 轮花了整轮打磨的模板，第 15 轮变成半小时的填空。
2. 未验证点：Silu 高阶 API 的真实签名/临时量要求仍需在真机环境编译确认（//?? 保留）。
3. 结构性反思：到第 15 轮，独立 elementwise 算子的"标准模板"已完全稳定（Init 大小核 → 双队列 → 循环三段 → 尾块 Pad），后续同类算子的边际价值趋零——**算子线应转向模板覆盖不了的结构**（多输出、动态 shape、通信融合、Cube 融合）。

## 四、经验教训

1. A1 精化条目（独立→高阶 API / 融合→手拼）连续两轮被生产证据支持，已可视为定律。
2. "常量预置于 Init" 与 "参数吸收进计算链" 是融合算子的两个通用细节。
3. 练习的边际收益递减信号：当模板复用率超过 80%，应主动换题目类别而不是继续堆算子数量。

## 十五轮总览

| 轮 | 主题 | 关键增量 |
| --- | --- | --- |
| 1-14 | （见前几轮） | — |
| 15 | Silu/模板复用 | 全 1 向量 Div 求倒数；常量预置；模板边际收益递减信号 |

**下一轮候选**：换结构——`10_communicate_compute_fused`（通信计算融合，全新领域）或 `foreach` 系列算子（cann-ops foreach 目录，多张量批量处理结构）。
