# Round 12 复盘：Gelu（CHECKLIST 首轮实战）

> 我的实现：`my_impl/round12_gelu/`（DESIGN.md 按 CHECKLIST A/B/C 逐项过 + gelu_custom.cpp + op_host/）
> 对比对象：`cann-ops/src/activation/gelu_quant/`（Gelu+量化融合，7 个变体头文件）

---

## 一、//?? 销案结果

| //?? | 结果 |
| --- | --- |
| 1. Gelu API 签名/approximate | 高阶 API `Gelu` 确认存在（激活函数类，ascendc-api-adv 库），但见下条 |
| 4. **生产用高阶 API 还是手拼** | **手拼**。`ComputeGeluErf`：7 系数多项式逼近 erf（先 `Maxs/Mins` 把输入钳到 [ERF_MIN, ERF_MAX] 防爆），`ComputeGeluTanh`：exp 展开——全部基础 API + tensor 通路，零标量 |
| 2. 配套 TmpSize/isReuseSource | 高阶 API 自身的机制，但生产融合场景未用它（保留中间量控制权） |
| 3. 产品支持 | gelu_quant 覆盖 训练/推理/A2 全系，手拼路径天然跨芯片 |

## 二、本轮最重要的认知修正：API 选型的答案依赖上下文

CHECKLIST A1 写的是"有高阶 API 就用"。生产证据给出精化：

- **独立 elementwise 算子** → 高阶 API `Gelu` 是正解（我的 DESIGN 判断正确）；
- **融合算子**（Gelu+quant）→ 手拼多项式，因为 erf 多项式的中间量（tempRes/xSquared）要**直接续接量化计算**，不经过任何搬运；且 `Maxs/Mins` 钳位这种数值稳定技巧要自己嵌进去。

**A1 条目精化为：独立算子优先高阶 API；融合算子评估手拼以保留中间量通路控制权。**

## 三、生产代码的三个新知识点

1. **erf 的数值稳定逼近**：钳位 + 7 参数多项式（x² 嵌套乘加），全部 tensor 化——超越函数在矢量核上的实现套路（与 Round 4 int8 的"高精度中间类型"是同族 idiom）。
2. **PipeBarrier<PIPE_V> 高密度使用**：每对依赖矢量指令之间都加管内屏障——在队列之外复用临时缓冲（castFp32/tempRes/xSquared 跨操作复用）时，依赖序要自己保证。Round 10 见过单次实例，这里是系统性用法：**队列外的张量运算链 = 显式屏障链**。
3. **UB 预算按算法临时量算**：erf 路径需要 3 个 fp32 临时张量（castFp32/tempRes/xSquared），不止进出队列 4 份——CHECKLIST A7 再精化：**临时量数量由算法决定，多项式逼近类的临时量是大头**。

## 四、我的实现自评

1. **流程层面**：CHECKLIST 首轮实战，A/B/C 三节 20 项逐项过了，Round 10/11 重犯的尾块对齐（B2）本轮用 DataCopyPad 字节粒度收尾落实——**检查单生效**。
2. 设计缺陷：kernel 里 TilingKey 1/2 两个分支体完全相同（占位）——真实切换需要两条 Compute 路径（生产的 ComputeGeluErf/Tanh 分函数），我的占位写法在对比中暴露。
3. L2 hint（B6）、四件脚手架（B1）、dtype 长度表（C1）、attr→TilingKey（C3/C4）全部按单落实，与前 11 轮相比是完成度最高的一轮。

## 五、CHECKLIST 增补（本轮产生）

- A1 精化：独立算子→高阶 API；融合算子→评估手拼保留中间量通路。
- A7 精化：UB 预算 = 进出队列 + **算法临时量**（多项式/归约中间量）。
- B3 补充：队列外复用缓冲的依赖矢量指令链，逐对 `PipeBarrier<PIPE_V>`。
- B8 补充：超越函数 = 钳位 + 多项式/exp 逼近的 tensor 化套路。

## 十二轮总览

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1-11 | （见前几轮） | — |
| 12 | Gelu/CHECKLIST 首战 | API 选型依赖上下文（独立 vs 融合）；队列外依赖链要显式屏障 |

**下一轮候选**：AddLayerNorm 融合（检验 A1 精化条目）；或 2_features/10_communicate_compute_fused（通信计算融合，全新领域）。
